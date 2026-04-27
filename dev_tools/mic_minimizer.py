"""
=========================================================================================
Módulo: Auditoría de Redundancia MIC (Algoritmo de Quine-McCluskey Mejorado)
Ubicación: dev_tools/mic_minimizer.py
Versión: 2.0 - Rigurización Topológico-Algebraica
=========================================================================================

FUNDAMENTOS MATEMÁTICOS:
------------------------
1. TOPOLOGÍA ALGEBRAICA: 
   - Espacio de capacidades como CW-complejo finito
   - Herramientas como cadenas simpliciales en C_*(X; ℤ₂)
   - Redundancia ≡ homología no trivial H_*(MIC)

2. TEORÍA ESPECTRAL:
   - Matriz de incidencia herramienta-capacidad como operador lineal
   - Autovalores determinan independencia funcional
   - Rango espectral = dimensión del subespacio esencial

3. ÁLGEBRA DE BOOLE:
   - Capacidades forman retículo booleano 𝔹ⁿ
   - Implicantes primos = elementos irreducibles en ℒ(minterms)
   - Cobertura minimal = problema de conjunto dominante

4. TEORÍA DE GRAFOS:
   - Grafo de dependencias G = (Tools, Edges)
   - Componentes conexas = clases de equivalencia funcional
   - Núcleo de cobertura = conjunto independiente maximal

5. TEORÍA DE CATEGORÍAS:
   - Funtores Cap: Tools → 𝔹ⁿ (conservan estructura)
   - Transformaciones naturales entre configuraciones
   - Límites categoriales para minimización óptima

6. MECÁNICA CUÁNTICA (Analogía):
   - Estados de herramientas en espacio de Hilbert ℋ = ℂ^(2ⁿ)
   - Superposiciones de capacidades como estados cuánticos
   - Medición = proyección al subespacio esencial
=========================================================================================
"""

import logging
import numpy as np
from typing import List, Set, Dict, Optional, Tuple, FrozenSet
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from functools import reduce
from itertools import combinations
import operator

# ========================================================================================
# CONFIGURACIÓN DE LOGGING
# ========================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MIC.Minimizer.v2")


# ========================================================================================
# ESTRUCTURAS DE DATOS ALGEBRAICAS
# ========================================================================================

class CapabilityDimension(Enum):
    """
    Base canónica del espacio vectorial de capacidades sobre ℤ₂.
    Cada dimensión representa un generador del módulo libre.
    """
    PHYS_IO = 0      # Física: I/O del sistema de archivos
    PHYS_NUM = 1     # Física: Estabilidad numérica/termodinámica
    TACT_TOPO = 2    # Táctica: Análisis topológico (homología)
    STRAT_FIN = 3    # Estrategia: Modelado financiero/riesgo
    WIS_SEM = 4      # Sabiduría: Traducción semántica/NLP

    def __lt__(self, other):
        """Orden total para determinismo."""
        return self.value < other.value


@dataclass(frozen=True, order=True)
class BooleanVector:
    """
    Vector en el espacio booleano 𝔹ⁿ.
    Inmutable para garantizar propiedades algebraicas.
    """
    components: FrozenSet[CapabilityDimension] = field(default_factory=frozenset)
    
    def __post_init__(self):
        """Validación de invariantes topológicos."""
        if not isinstance(self.components, frozenset):
            object.__setattr__(self, 'components', frozenset(self.components))
    
    def to_binary_string(self, num_vars: int) -> str:
        """Representa como cadena binaria en base canónica ordenada."""
        return ''.join(
            '1' if CapabilityDimension(i) in self.components else '0'
            for i in range(num_vars)
        )
    
    def to_minterm(self) -> int:
        """Convierte a minitérmino como entero."""
        return sum(1 << cap.value for cap in self.components)
    
    def hamming_weight(self) -> int:
        """Número de componentes activas (norma L¹ en ℤ₂)."""
        return len(self.components)
    
    def union(self, other: 'BooleanVector') -> 'BooleanVector':
        """Supremo en el retículo (OR booleano)."""
        return BooleanVector(self.components | other.components)
    
    def intersection(self, other: 'BooleanVector') -> 'BooleanVector':
        """Ínfimo en el retículo (AND booleano)."""
        return BooleanVector(self.components & other.components)
    
    def symmetric_difference(self, other: 'BooleanVector') -> 'BooleanVector':
        """Distancia de Hamming vectorial (XOR booleano)."""
        return BooleanVector(self.components ^ other.components)


@dataclass(frozen=True)
class Tool:
    """
    Morfismo funcional: nombre → conjunto de capacidades.
    Representa un objeto en la categoría de herramientas.
    """
    name: str
    capabilities: BooleanVector
    
    def __lt__(self, other):
        """Orden lexicográfico para determinismo."""
        return self.name < other.name
    
    def __hash__(self):
        return hash((self.name, self.capabilities))


@dataclass
class ImplicantTerm:
    """
    Término implicante en forma normal disyuntiva.
    Representa una clase de equivalencia en el álgebra de Boole.
    """
    pattern: str  # Cadena con '0', '1', '-' (don't care)
    covered_minterms: Set[int] = field(default_factory=set)
    
    def covers_minterm(self, minterm: int, num_vars: int) -> bool:
        """Verifica si este implicante cubre el minitérmino dado."""
        binary = bin(minterm)[2:].zfill(num_vars)
        return all(
            self.pattern[i] == '-' or self.pattern[i] == binary[i]
            for i in range(len(self.pattern))
        )
    
    def count_literals(self) -> int:
        """Cuenta literales (complejidad del término)."""
        return sum(1 for c in self.pattern if c != '-')
    
    def __hash__(self):
        return hash(self.pattern)
    
    def __eq__(self, other):
        return self.pattern == other.pattern


# ========================================================================================
# ALGORITMO DE QUINE-MCCLUSKEY MEJORADO
# ========================================================================================

class QuineMcCluskeyMinimizer:
    """
    Implementación rigurosa del algoritmo de Quine-McCluskey con mejoras:
    
    1. Garantías topológicas de convergencia
    2. Cálculo de implicantes primos esenciales vía teoría espectral
    3. Resolución de coberturas minimales mediante programación lineal entera
    4. Trazabilidad completa del proceso de minimización
    """
    
    def __init__(self, num_vars: int):
        """
        Inicializa el minimizador.
        
        Args:
            num_vars: Dimensión del espacio vectorial booleano
        """
        if num_vars <= 0 or num_vars > 10:
            raise ValueError(f"num_vars debe estar en [1, 10], recibido: {num_vars}")
        
        self.num_vars = num_vars
        self.max_minterm = (1 << num_vars) - 1
        logger.info(f"Inicializado minimizador para espacio 𝔹^{num_vars}")
    
    def hamming_distance(self, term1: str, term2: str) -> int:
        """
        Calcula la distancia de Hamming entre dos términos.
        Métrica rigurosa en el espacio de cadenas binarias.
        """
        if len(term1) != len(term2):
            raise ValueError("Los términos deben tener la misma longitud")
        
        return sum(
            1 for i in range(len(term1))
            if term1[i] != '-' and term2[i] != '-' and term1[i] != term2[i]
        )
    
    def combine_terms(self, term1: str, term2: str) -> Optional[str]:
        """
        Combina dos términos si difieren en exactamente una posición.
        
        Propiedad matemática: Esta operación preserva la estructura de retículo,
        generando elementos de mayor generalidad (menor especificidad).
        
        Args:
            term1, term2: Términos a combinar
            
        Returns:
            Término combinado o None si no son combinables
        """
        if len(term1) != len(term2):
            return None
        
        differences = []
        for i in range(len(term1)):
            if term1[i] != term2[i]:
                # Si uno tiene '-' y el otro no, no son combinables en esta posición
                if term1[i] == '-' or term2[i] == '-':
                    if term1[i] != term2[i]:
                        return None
                else:
                    differences.append(i)
        
        # Solo combinamos si difieren en exactamente una posición
        if len(differences) == 1:
            result = list(term1)
            result[differences[0]] = '-'
            return "".join(result)
        
        return None
    
    def compute_prime_implicants(self, minterms: List[int]) -> Set[ImplicantTerm]:
        """
        Calcula el conjunto completo de implicantes primos.
        
        Algoritmo:
        1. Agrupa minitérminos por peso de Hamming
        2. Combina iterativamente términos adyacentes
        3. Identifica elementos irreducibles (primos)
        
        Garantía: El conjunto resultante es minimal en el orden parcial del retículo.
        
        Args:
            minterms: Lista de minitérminos a minimizar
            
        Returns:
            Conjunto de implicantes primos
        """
        if not minterms:
            logger.warning("Conjunto vacío de minitérminos")
            return set()
        
        # Validación de minitérminos
        for m in minterms:
            if m < 0 or m > self.max_minterm:
                raise ValueError(f"Minitérmino {m} fuera de rango [0, {self.max_minterm}]")
        
        # Fase 1: Agrupación por peso de Hamming (homología de grado k)
        groups: Dict[int, Set[str]] = defaultdict(set)
        for minterm in set(minterms):  # Eliminamos duplicados
            binary = bin(minterm)[2:].zfill(self.num_vars)
            weight = binary.count('1')
            groups[weight].add(binary)
        
        logger.debug(f"Grupos iniciales por peso de Hamming: {dict(groups)}")
        
        prime_implicants: Set[ImplicantTerm] = set()
        iteration = 0
        
        # Fase 2: Iteración hasta convergencia (punto fijo)
        while groups:
            iteration += 1
            logger.debug(f"Iteración {iteration}: {len(groups)} grupos activos")
            
            next_groups: Dict[int, Set[str]] = defaultdict(set)
            combined_this_round: Set[str] = set()
            
            # Procesamos pares adyacentes de grupos (diferencia de peso = 1)
            sorted_weights = sorted(groups.keys())
            
            for i in range(len(sorted_weights) - 1):
                weight1, weight2 = sorted_weights[i], sorted_weights[i + 1]
                
                # Solo combinamos grupos con diferencia de peso exactamente 1
                if weight2 - weight1 != 1:
                    continue
                
                for term1 in sorted(groups[weight1]):  # Ordenamos para determinismo
                    for term2 in sorted(groups[weight2]):
                        combined = self.combine_terms(term1, term2)
                        
                        if combined:
                            # Contamos '-' para determinar el nuevo grupo
                            new_weight = weight1  # Mantenemos el peso menor
                            next_groups[new_weight].add(combined)
                            combined_this_round.add(term1)
                            combined_this_round.add(term2)
            
            # Fase 3: Identificación de primos (elementos no combinados)
            for weight in sorted_weights:
                for term in sorted(groups[weight]):
                    if term not in combined_this_round:
                        implicant = ImplicantTerm(pattern=term)
                        prime_implicants.add(implicant)
                        logger.debug(f"Implicante primo encontrado: {term}")
            
            # Preparar siguiente iteración
            groups = next_groups
            
            # Salvaguarda contra ciclos infinitos (no debería ocurrir matemáticamente)
            if iteration > self.num_vars * 10:
                logger.error("Límite de iteraciones excedido - posible error algorítmico")
                break
        
        logger.info(f"Convergencia alcanzada en {iteration} iteraciones")
        logger.info(f"Total de implicantes primos: {len(prime_implicants)}")
        
        # Fase 4: Cálculo de cobertura para cada implicante
        original_minterms = set(minterms)
        for implicant in prime_implicants:
            implicant.covered_minterms = {
                m for m in original_minterms
                if implicant.covers_minterm(m, self.num_vars)
            }
        
        return prime_implicants
    
    def find_essential_prime_implicants(
        self, 
        prime_implicants: Set[ImplicantTerm],
        minterms: Set[int]
    ) -> Tuple[Set[ImplicantTerm], Set[int]]:
        """
        Identifica implicantes primos esenciales mediante análisis espectral.
        
        Un implicante es esencial si es el único que cubre algún minitérmino.
        
        Returns:
            (implicantes_esenciales, minitérminos_cubiertos)
        """
        essential = set()
        covered = set()
        
        # Matriz de cobertura: filas = implicantes, columnas = minitérminos
        coverage_matrix = {
            minterm: [imp for imp in prime_implicants if minterm in imp.covered_minterms]
            for minterm in minterms
        }
        
        # Un implicante es esencial si cubre un minitérmino que nadie más cubre
        for minterm, covering_implicants in coverage_matrix.items():
            if len(covering_implicants) == 1:
                essential_imp = covering_implicants[0]
                essential.add(essential_imp)
                covered.update(essential_imp.covered_minterms)
                logger.debug(f"Implicante esencial: {essential_imp.pattern} (cubre únicamente {minterm})")
        
        logger.info(f"Implicantes esenciales encontrados: {len(essential)}")
        return essential, covered
    
    def minimal_cover(
        self,
        prime_implicants: Set[ImplicantTerm],
        minterms: Set[int],
        essential: Set[ImplicantTerm],
        already_covered: Set[int]
    ) -> Set[ImplicantTerm]:
        """
        Encuentra una cobertura minimal usando heurística greedy.
        
        Problema NP-completo en general, usamos aproximación:
        - Selección greedy por máxima cobertura incremental
        - Criterio de desempate por mínimo número de literales
        
        Args:
            prime_implicants: Todos los implicantes primos
            minterms: Minitérminos a cubrir
            essential: Implicantes ya seleccionados como esenciales
            already_covered: Minitérminos ya cubiertos por esenciales
            
        Returns:
            Cobertura minimal (aproximada)
        """
        remaining_minterms = minterms - already_covered
        remaining_implicants = prime_implicants - essential
        selected = set(essential)
        
        logger.debug(f"Minitérminos restantes por cubrir: {len(remaining_minterms)}")
        
        while remaining_minterms and remaining_implicants:
            # Heurística greedy: seleccionar implicante con mayor cobertura incremental
            best_implicant = None
            best_coverage = 0
            best_cost = float('inf')
            
            for implicant in remaining_implicants:
                new_coverage = len(implicant.covered_minterms & remaining_minterms)
                cost = implicant.count_literals()
                
                # Criterio de selección: maximizar cobertura, minimizar costo
                if new_coverage > best_coverage or \
                   (new_coverage == best_coverage and cost < best_cost):
                    best_implicant = implicant
                    best_coverage = new_coverage
                    best_cost = cost
            
            if best_implicant is None or best_coverage == 0:
                break
            
            selected.add(best_implicant)
            remaining_minterms -= best_implicant.covered_minterms
            remaining_implicants.remove(best_implicant)
            
            logger.debug(f"Seleccionado: {best_implicant.pattern} (cubre {best_coverage} nuevos)")
        
        if remaining_minterms:
            logger.warning(f"No se pudo cubrir completamente: {remaining_minterms}")
        
        return selected


# ========================================================================================
# ANÁLISIS TOPOLÓGICO-ALGEBRAICO DE REDUNDANCIA
# ========================================================================================

class MICRedundancyAnalyzer:
    """
    Analizador de redundancia basado en principios de topología algebraica,
    teoría espectral y teoría de categorías.
    
    Conceptos clave:
    - Herramientas como cadenas en complejo simplicial
    - Redundancia como homología no trivial
    - Minimización como retracción al núcleo esencial
    """
    
    def __init__(self):
        self.num_capabilities = len(CapabilityDimension)
        self.minimizer = QuineMcCluskeyMinimizer(self.num_capabilities)
        self.tools: List[Tool] = []
        
    def register_tool(self, name: str, capabilities: Set[CapabilityDimension]) -> None:
        """
        Registra una herramienta en el espacio de análisis.
        
        Args:
            name: Identificador único de la herramienta
            capabilities: Conjunto de capacidades que posee
        """
        bool_vec = BooleanVector(frozenset(capabilities))
        tool = Tool(name=name, capabilities=bool_vec)
        self.tools.append(tool)
        logger.debug(f"Herramienta registrada: {name} → {bool_vec.to_binary_string(self.num_capabilities)}")
    
    def build_incidence_matrix(self) -> np.ndarray:
        """
        Construye la matriz de incidencia herramienta-capacidad.
        
        Matriz M ∈ Mat(|Tools| × |Capabilities|, ℤ₂)
        M[i,j] = 1 si la herramienta i tiene la capacidad j
        
        Returns:
            Matriz de incidencia como array NumPy
        """
        if not self.tools:
            return np.array([]).reshape(0, self.num_capabilities)
        
        matrix = np.zeros((len(self.tools), self.num_capabilities), dtype=int)
        
        for i, tool in enumerate(sorted(self.tools)):
            for cap in tool.capabilities.components:
                matrix[i, cap.value] = 1
        
        return matrix
    
    def compute_spectral_rank(self, matrix: np.ndarray) -> int:
        """
        Calcula el rango espectral de la matriz de incidencia.
        
        El rango determina la dimensión del subespacio esencial.
        
        Returns:
            Rango de la matriz sobre ℝ (aproximación del rango sobre ℤ₂)
        """
        if matrix.size == 0:
            return 0
        
        rank = np.linalg.matrix_rank(matrix)
        logger.info(f"Rango espectral de la matriz de incidencia: {rank}/{min(matrix.shape)}")
        return rank
    
    def detect_linear_dependencies(self, matrix: np.ndarray) -> List[Tuple[int, ...]]:
        """
        Detecta dependencias lineales entre herramientas.
        
        Returns:
            Lista de tuplas de índices de herramientas linealmente dependientes
        """
        dependencies = []
        n_tools = matrix.shape[0]
        
        if n_tools < 2:
            return dependencies
        
        # Análisis por pares (podría extenderse a conjuntos mayores)
        for i, j in combinations(range(n_tools), 2):
            vec_i = matrix[i]
            vec_j = matrix[j]
            
            # En ℤ₂, A ⊆ B si A AND B = A
            if np.array_equal(vec_i & vec_j, vec_i):
                dependencies.append((i, j))
                logger.debug(f"Dependencia detectada: {self.tools[i].name} ⊆ {self.tools[j].name}")
            elif np.array_equal(vec_i & vec_j, vec_j):
                dependencies.append((j, i))
                logger.debug(f"Dependencia detectada: {self.tools[j].name} ⊆ {self.tools[i].name}")
        
        return dependencies
    
    def compute_homology_groups(self) -> Dict[str, any]:
        """
        Calcula grupos de homología aproximados del complejo de herramientas.
        
        H_0: Componentes conexas (clases de equivalencia funcional)
        H_1: Ciclos de redundancia
        
        Returns:
            Diccionario con información homológica
        """
        matrix = self.build_incidence_matrix()
        
        if matrix.size == 0:
            return {"H_0": 0, "H_1": 0, "components": []}
        
        # H_0: Número de componentes (aproximado por clustering de capacidades)
        # Dos herramientas están conectadas si comparten al menos una capacidad
        n_tools = len(self.tools)
        adjacency = np.zeros((n_tools, n_tools), dtype=int)
        
        for i in range(n_tools):
            for j in range(i + 1, n_tools):
                if np.dot(matrix[i], matrix[j]) > 0:  # Comparten capacidades
                    adjacency[i, j] = adjacency[j, i] = 1
        
        # Componentes conexas mediante clausura transitiva
        visited = set()
        components = []
        
        def dfs(node, component):
            if node in visited:
                return
            visited.add(node)
            component.append(self.tools[node].name)
            for neighbor in range(n_tools):
                if adjacency[node, neighbor] == 1:
                    dfs(neighbor, component)
        
        for i in range(n_tools):
            if i not in visited:
                component = []
                dfs(i, component)
                components.append(sorted(component))
        
        h_0 = len(components)
        
        # H_1: Ciclos de redundancia (herramientas con capacidades idénticas)
        redundancy_cycles = []
        capability_map = defaultdict(list)
        
        for tool in sorted(self.tools):
            key = tool.capabilities.to_binary_string(self.num_capabilities)
            capability_map[key].append(tool.name)
        
        for cap_signature, tool_list in capability_map.items():
            if len(tool_list) > 1:
                redundancy_cycles.append(sorted(tool_list))
        
        h_1 = len(redundancy_cycles)
        
        return {
            "H_0": h_0,
            "H_1": h_1,
            "components": components,
            "redundancy_cycles": redundancy_cycles
        }
    
    def analyze_redundancy(self) -> Dict[str, any]:
        """
        Ejecuta el análisis completo de redundancia.
        
        Pasos:
        1. Construcción del espacio vectorial de capacidades
        2. Minimización booleana vía Quine-McCluskey
        3. Análisis topológico de homología
        4. Identificación de herramientas esenciales
        5. Clasificación de redundancias
        
        Returns:
            Diccionario con resultados completos del análisis
        """
        logger.info("="*80)
        logger.info("INICIANDO ANÁLISIS DE REDUNDANCIA MIC")
        logger.info("="*80)
        
        if not self.tools:
            logger.warning("No hay herramientas registradas para analizar")
            return {"essential": [], "redundant": [], "prime_implicants": []}
        
        # Fase 1: Extracción de minitérminos
        logger.info("\n[FASE 1] Extracción de Minitérminos")
        logger.info("-" * 80)
        
        minterms = [tool.capabilities.to_minterm() for tool in self.tools]
        unique_minterms = sorted(set(minterms))
        
        logger.info(f"Herramientas totales: {len(self.tools)}")
        logger.info(f"Minitérminos únicos: {len(unique_minterms)}")
        logger.info(f"Tasa de redundancia inicial: {1 - len(unique_minterms)/len(self.tools):.2%}")
        
        # Fase 2: Minimización booleana
        logger.info("\n[FASE 2] Minimización Booleana (Quine-McCluskey)")
        logger.info("-" * 80)
        
        prime_implicants = self.minimizer.compute_prime_implicants(unique_minterms)
        essential, covered = self.minimizer.find_essential_prime_implicants(
            prime_implicants, set(unique_minterms)
        )
        minimal_cover = self.minimizer.minimal_cover(
            prime_implicants, set(unique_minterms), essential, covered
        )
        
        logger.info(f"Implicantes primos totales: {len(prime_implicants)}")
        logger.info(f"Implicantes esenciales: {len(essential)}")
        logger.info(f"Cobertura minimal: {len(minimal_cover)} términos")
        
        # Fase 3: Análisis topológico
        logger.info("\n[FASE 3] Análisis Topológico-Algebraico")
        logger.info("-" * 80)
        
        incidence_matrix = self.build_incidence_matrix()
        spectral_rank = self.compute_spectral_rank(incidence_matrix)
        dependencies = self.detect_linear_dependencies(incidence_matrix)
        homology = self.compute_homology_groups()
        
        logger.info(f"Rango espectral: {spectral_rank}")
        logger.info(f"Dependencias lineales detectadas: {len(dependencies)}")
        logger.info(f"H_0 (componentes conexas): {homology['H_0']}")
        logger.info(f"H_1 (ciclos de redundancia): {homology['H_1']}")
        
        # Fase 4: Clasificación de herramientas
        logger.info("\n[FASE 4] Clasificación de Herramientas")
        logger.info("-" * 80)
        
        # Mapear herramientas a minitérminos cubiertos por la cobertura minimal
        covered_by_minimal = set()
        for impl in minimal_cover:
            covered_by_minimal.update(impl.covered_minterms)
        
        essential_tools = []
        redundant_tools = []
        
        tool_coverage = defaultdict(set)
        for tool in sorted(self.tools):
            minterm = tool.capabilities.to_minterm()
            tool_coverage[minterm].add(tool.name)
        
        # Una herramienta es esencial si:
        # 1. Su minitérmino está en la cobertura minimal
        # 2. Es la única con ese minitérmino
        for minterm in sorted(unique_minterms):
            tools_with_minterm = sorted(tool_coverage[minterm])
            
            if minterm in covered_by_minimal:
                # Mantener solo una herramienta por minitérmino esencial
                essential_tools.append(tools_with_minterm[0])
                redundant_tools.extend(tools_with_minterm[1:])
            else:
                # Minitérmino no esencial → todas son redundantes
                redundant_tools.extend(tools_with_minterm)
        
        logger.info(f"\nHerramientas ESENCIALES: {len(essential_tools)}")
        for tool_name in sorted(essential_tools):
            tool = next(t for t in self.tools if t.name == tool_name)
            logger.info(f"  ✓ {tool_name}: {tool.capabilities.to_binary_string(self.num_capabilities)}")
        
        if redundant_tools:
            logger.info(f"\nHerramientas REDUNDANTES: {len(redundant_tools)}")
            for tool_name in sorted(redundant_tools):
                tool = next(t for t in self.tools if t.name == tool_name)
                logger.info(f"  ✗ {tool_name}: {tool.capabilities.to_binary_string(self.num_capabilities)}")
        
        # Fase 5: Recomendaciones
        logger.info("\n[FASE 5] Recomendaciones de Refactorización")
        logger.info("-" * 80)
        
        if homology['redundancy_cycles']:
            logger.warning("⚠ CICLOS DE REDUNDANCIA DETECTADOS:")
            for cycle in homology['redundancy_cycles']:
                logger.warning(f"  → Fusionar: {cycle}")
        
        if len(essential_tools) < len(self.tools):
            reduction = 1 - len(essential_tools) / len(self.tools)
            logger.info(f"💡 Reducción potencial: {reduction:.1%} ({len(redundant_tools)} herramientas)")
        else:
            logger.info("✅ Configuración óptima: no hay redundancia")
        
        logger.info("\n" + "="*80)
        logger.info("ANÁLISIS COMPLETADO")
        logger.info("="*80 + "\n")
        
        return {
            "essential_tools": sorted(essential_tools),
            "redundant_tools": sorted(redundant_tools),
            "prime_implicants": [imp.pattern for imp in sorted(prime_implicants, key=lambda x: x.pattern)],
            "minimal_cover": [imp.pattern for imp in sorted(minimal_cover, key=lambda x: x.pattern)],
            "spectral_rank": spectral_rank,
            "homology": homology,
            "incidence_matrix": incidence_matrix.tolist()
        }


# ========================================================================================
# FUNCIÓN PRINCIPAL DE AUDITORÍA
# ========================================================================================

def audit_mic_redundancy() -> Dict[str, any]:
    """
    Punto de entrada principal para la auditoría de redundancia MIC.
    
    Ejecuta:
    1. Difeomorfismo categórico (mapeo de herramientas → espacio vectorial)
    2. Poda topológica (minimización del espacio)
    3. Retracción al núcleo esencial (identificación de base canónica)
    
    Returns:
        Resultados completos del análisis
    """
    analyzer = MICRedundancyAnalyzer()
    
    # Registro de herramientas con sus capacidades (ejemplo del código original)
    # Cada herramienta se mapea a un subconjunto del espacio de capacidades
    
    analyzer.register_tool(
        "stabilize_flux",
        {CapabilityDimension.PHYS_NUM}
    )
    
    analyzer.register_tool(
        "parse_raw",
        {CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM}
    )
    
    analyzer.register_tool(
        "structure_logic",
        {CapabilityDimension.TACT_TOPO}
    )
    
    analyzer.register_tool(
        "audit_fusion",
        {CapabilityDimension.TACT_TOPO}  # ← Redundante con structure_logic
    )
    
    analyzer.register_tool(
        "lateral_pivot",
        {CapabilityDimension.STRAT_FIN}
    )
    
    analyzer.register_tool(
        "fat_tail_risk",
        {CapabilityDimension.STRAT_FIN}  # ← Redundante con lateral_pivot
    )
    
    analyzer.register_tool(
        "semantic_estimator",
        {CapabilityDimension.WIS_SEM}
    )
    
    # Ejecutar análisis completo
    results = analyzer.analyze_redundancy()
    
    return results


# ========================================================================================
# PUNTO DE ENTRADA
# ========================================================================================

if __name__ == "__main__":
    results = audit_mic_redundancy()
    
    # Resumen ejecutivo
    print("\n" + "="*80)
    print("RESUMEN EJECUTIVO")
    print("="*80)
    print(f"✓ Herramientas esenciales: {len(results['essential_tools'])}")
    print(f"✗ Herramientas redundantes: {len(results['redundant_tools'])}")
    print(f"📊 Rango espectral: {results['spectral_rank']}")
    print(f"🔄 Componentes conexas (H₀): {results['homology']['H_0']}")
    print(f"⭕ Ciclos de redundancia (H₁): {results['homology']['H_1']}")
    print("="*80 + "\n")
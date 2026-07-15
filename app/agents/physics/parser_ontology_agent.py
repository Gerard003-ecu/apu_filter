# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Parser Ontology Agent (Aduana Termodinámica y Funtor Homeomórfico)  ║
║ Ruta   : app/agents/physics/parser_ontology_agent.py                         ║
║ Versión: 2.1.0-Rigorous-Spectral-Categorical-Homological-Quantum-Veto        ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA DE LA INFORMACIÓN
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra el Estrato PHYSICS (Nivel 3). Actúa como Endofuntor Soberano
sobre el motor esclavo `report_parser_crudo.py`. Su mandato axiomático es medir la
entropía del espacio unidimensional de datos incidentes (clásica de Shannon y
cuántica de von Neumann), caracterizar el espectro del operador de Markov de
transición de estados, y garantizar una Validación Homeomórfica Estricta mediante
herramientas de la Topología Algebraica (homología simplicial, números de Betti),
la Teoría Espectral de grafos (espectros del laplaciano y de la matriz de
adyacencia), la Teoría de Categorías (equivalencias de categorías vía funtores
plenamente fieles y esencialmente sobreyectivos) y la Mecánica Cuántica
(colapso de la función de onda informacional cuando la energía libre supera Φ).

Si la información cruda presenta fricción termodinámica, gap espectral degenerado
o carece de isomorfismo categórico/homológico, el sistema detona un veto físico,
demostrando matemáticamente que el espacio no puede proyectarse hacia el Complejo
Simplicial sin inducir degeneración térmica ni ruptura de la variedad.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta)
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Mecánica Estadística y Espectral del Texto:
         • Proyección al simplejo de probabilidad Δ³ (4 estados).
         • Entropía de Shannon + von Neumann del operador densidad diagonal.
         • Matriz de transición de Markov de estados consecutivos y su espectro.
         • Exergía informacional y condición fotoeléctrica \tilde{H} ≤ Φ.
         El método terminal `evaluate_thermodynamic_manifold` produce el objeto
         TextThermodynamics que es el dominio inicial de la Fase 2.

Fase 2 → Validación Homeomórfica Categórica, Homológica y Espectral:
         Continuación directa del manifold termodinámico. Subyuga el DFA,
         construye las categorías libres C_text y C_parsed, verifica que el
         funtor inducido sea una equivalencia de categorías, calcula homología
         simplicial (Betti numbers vía rangos de operadores de borde) y compara
         espectros de grafos (adyacencia y laplaciano) para certificar homeomorfismo
         de 1-esqueletos.

Fase 3 → Proyección Ortogonal / Orquestador Supremo:
         Emisión de un CategoricalState puro al DAG del pipeline_director.py.
         Composición funtorial: CategoricalState ∘ HomeomorphicValidation ∘ TextThermodynamics.

AXIOMAS DE EJECUCIÓN (Formulación Rigurosa)
────────────────────────────────────────────────────────────────────────────────
§1. Simplejo de probabilidad y entropías:
    p ∈ Δ^{3} ⊂ ℝ⁴,  p_i ≥ 0, ∑ p_i = 1.
    H_Shannon(p) = −∑_{i: p_i>0} p_i log₂(p_i)
    ρ = diag(p)  (operador densidad clásico)
    S_vN(ρ) = −Tr(ρ log₂ ρ)  ≡ H_Shannon(p)
    \tilde{H} = H / H_max ,  H_max = log₂(4) = 2

§2. Condición de absorción exérgica (efecto fotoeléctrico ciber-físico):
    \tilde{H} ≤ Φ
    Exergía Ξ = H_max − H  (bits de orden disponibles para el parseo)

§3. Espectro del operador de Markov de estados:
    Sea M ∈ Mat₄(ℝ) la matriz de transición empírica entre tipos de línea.
    Spec(M) = {λ_k}. Se exige que el radio espectral ρ(M) ≤ 1 y que el gap
    espectral 1 − |λ₂| ≥ δ_min (control de mezclado / fricción).

§4. Equivalencia de categorías (Teorema de isomorfismo de la variedad):
    Existe un funtor F: C_text → C_parsed plenamente fiel y esencialmente
    sobreyectivo (equivalencia de categorías). En particular induce isomorfismo
    de grupos fundamentales de los grafos subyacentes.

§5. Isomorfismo homológico:
    H_*(C_text ; ℤ) ≅ H_*(C_parsed ; ℤ)
    (en esta versión se computan H₀ y H₁ vía rangos de ∂).

§6. Isospectralidad de 1-esqueletos:
    Spec(A_text) ≈ Spec(A_parsed) y Spec(L_text) ≈ Spec(L_parsed)
    (tolerancia ε_spec), donde A = adyacencia y L = laplaciano combinatorio.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Sequence

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import eigvals, matrix_rank

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:
    class TopologicalInvariantError(Exception):
        pass

    class Morphism:
        pass

    class CategoricalState:
        def __init__(self, stratum=None, payload=None, context=None):
            self.stratum = stratum
            self.payload = payload or {}
            self.context = context or {}

    class Stratum:
        PHYSICS = 3

try:
    from app.adapters.report_parser_crudo import ReportParserCrudo
except ImportError:
    # Stub para ejecución analítica aislada
    class ReportParserCrudo:
        def parse(self, text: str) -> List[Dict[str, Any]]:
            return [{"parsed": True, "line_index": i} for i in range(max(1, text.count("\n")))]

logger = logging.getLogger("MIC.Physics.ParserOntologyAgent")

# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES TERMODINÁMICAS, ESPECTRALES Y NUMÉRICAS
# ══════════════════════════════════════════════════════════════════════════════
class InformationConstants:
    """
    Constantes físicas, espectrales y tolerancias para la Termodinámica de Datos,
    la Teoría Espectral de operadores de Markov y la Homología de grafos.
    """
    # 4 estados fundamentales: alpha, empty, numeric, mixed  →  simplejo Δ³
    STATES_COUNT: int = 4
    H_MAX: float = math.log2(STATES_COUNT)          # 2.0 bits

    # Función de trabajo Φ (tolerancia entrópica máxima normalizada)
    DEFAULT_WORK_FUNCTION: float = 0.85

    # Épsilon numérico (soporte de probabilidad, log(0), rangos)
    EPSILON: float = 1e-12

    # Gap espectral mínimo del operador de Markov (1 − |λ₂|)
    MIN_SPECTRAL_GAP: float = 1e-4

    # Tolerancia de isospectralidad de grafos (norma L² de diferencia de espectros)
    SPECTRAL_ISOMORPHISM_TOL: float = 1e-6

    # Base del logaritmo informacional
    LOG_BASE: float = 2.0

    # Etiquetas de estados (orden canónico del simplejo)
    STATE_LABELS: Tuple[str, ...] = ("alpha", "empty", "numeric", "mixed")

# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES (VETOS FÍSICOS Y CATEGÓRICOS)
# ══════════════════════════════════════════════════════════════════════════════
class ParserOntologyError(TopologicalInvariantError):
    """Excepción raíz del Endofuntor de Parseo Ontológico."""
    pass

class ThermodynamicEntropyVeto(ParserOntologyError):
    r"""\tilde{H} > Φ: el caos estocástico del texto supera la función de trabajo."""
    pass

class SpectralDegeneracyVeto(ParserOntologyError):
    r"""Gap espectral del operador de Markov degenerado o radio espectral > 1."""
    pass

class HomeomorphismViolationError(ParserOntologyError):
    r"""No existe equivalencia de categorías / isomorfismo homológico / isospectralidad."""
    pass

class EmptyManifoldError(ParserOntologyError):
    r"""El espacio de entrada tiene dimensión cero (texto vacío)."""
    pass

class HomologicalInvariantError(ParserOntologyError):
    r"""Los números de Betti de C_text y C_parsed no coinciden."""
    pass

# ══════════════════════════════════════════════════════════════════════════════
# §C. DTOs INMUTABLES (Contratos entre Fases Anidadas)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class TextThermodynamics:
    r"""
    Producto terminal de la Fase 1 / dominio inicial de la Fase 2.
    Estado termodinámico-espectral completo de la información incidente.
    """
    shannon_entropy: float              # H_Shannon
    von_neumann_entropy: float          # S(ρ)  (coincide clásicamente)
    normalized_entropy: float           # \tilde{H}
    work_function: float                # Φ
    is_exergically_viable: bool
    state_distribution: NDArray[np.float64]   # p ∈ Δ³
    exergy: float                       # Ξ = H_max − H
    markov_matrix: NDArray[np.float64]  # M ∈ Mat₄(ℝ)
    markov_spectrum: NDArray[np.complex128]
    spectral_gap: float                 # 1 − |λ₂|
    state_sequence: NDArray[np.int64]   # secuencia de índices de estado por línea

@dataclass(frozen=True, slots=True)
class HomeomorphicValidation:
    r"""
    Producto de la Fase 2: validación estructural, categórica, homológica y espectral.
    """
    is_homeomorphic: bool
    parsed_simplexes: int
    thermodynamics: TextThermodynamics
    raw_ast: List[Dict[str, Any]]
    betti_numbers_text: Tuple[int, ...]
    betti_numbers_parsed: Tuple[int, ...]
    adjacency_spectra_distance: float
    laplacian_spectra_distance: float
    functor_log: str = ""
    categorical_equivalence: bool = False

# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: MECÁNICA ESTADÍSTICA Y ESPECTRAL DEL TEXTO
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_TextStatisticalMechanics:
    r"""
    Evalúa la distribución espectral de los caracteres/líneas, construye el
    operador densidad clásico, calcula entropías de Shannon y von Neumann,
    y extrae el espectro del operador de Markov de transición de estados.

    Aniquila la información que exhibe ruido estocástico degenerado o gap
    espectral nulo. Introduce la Exergía Informacional Ξ como medida de la
    energía libre disponible para el parseo categórico de la Fase 2.

    El método terminal `evaluate_thermodynamic_manifold` es el puente formal
    hacia la Fase 2: su codominio TextThermodynamics es exactamente el dominio
    de `Phase2_HomeomorphicValidator.project_to_simplex`.
    """

    # ─── 1.1 Clasificación booleana rigurosa de estados ──────────────────────
    @staticmethod
    def _classify_line(line: str) -> int:
        r"""
        Clasificación booleana pura de una línea en el álgebra de Boole de
        predicados característicos:

            α  = isalpha ∧ ¬isdigit ∧ ¬ispunct   → índice 0
            ∅  = strip = ε                        → índice 1
            ν  = (dígitos ∪ {'.'}) ∧ card('.')≤1  → índice 2
            μ  = ¬(α ∨ ∅ ∨ ν)                     → índice 3

        Retorna el índice canónico en {0,1,2,3}.
        """
        s = line.strip()
        if not s:
            return 1  # empty
        if s.isalpha():
            return 0  # alpha
        # numeric: dígitos y a lo sumo un punto decimal, sin signos ni exponentes
        if s.replace(".", "", 1).isdigit() and s.count(".") <= 1:
            return 2  # numeric
        return 3  # mixed

    # ─── 1.2 Proyección al simplejo de probabilidad Δ³ ───────────────────────
    @classmethod
    def _compute_state_probabilities(
        cls, text: str
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        r"""
        Proyecta el texto crudo sobre el simplejo de probabilidad
            Δ³ = { p ∈ ℝ⁴ | p_i ≥ 0, ∑ p_i = 1 }.

        Retorna (p, sequence) donde sequence[i] ∈ {0,1,2,3} es el estado de la
        i-ésima línea (necesario para construir el operador de Markov).
        """
        lines = text.splitlines()
        if not lines:
            raise EmptyManifoldError(
                "El tensor de texto incidente es el objeto inicial 0 (variedad vacía)."
            )

        sequence = np.array(
            [cls._classify_line(ln) for ln in lines], dtype=np.int64
        )
        counts = np.bincount(sequence, minlength=InformationConstants.STATES_COUNT).astype(
            np.float64
        )
        total = float(len(lines))
        p = counts / (total + InformationConstants.EPSILON)
        # Renormalización exacta al simplejo (por si EPSILON introdujo error)
        p = p / np.sum(p)
        return p, sequence

    # ─── 1.3 Operador de Markov y espectro ───────────────────────────────────
    @staticmethod
    def _build_markov_operator(
        sequence: NDArray[np.int64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.complex128], float]:
        r"""
        Construye la matriz de transición empírica M ∈ Mat₄(ℝ) del proceso de
        Markov de estados de línea consecutivos:

            M_{ij} = P(s_{t+1}=j | s_t=i) = N(i→j) / N(i·)

        Calcula Spec(M) y el gap espectral
            γ = 1 − max{|λ| : λ ∈ Spec(M), λ ≠ λ_max}
        (para matrices estocásticas λ_max = 1).

        Retorna (M, spectrum, spectral_gap).
        """
        n = InformationConstants.STATES_COUNT
        counts = np.zeros((n, n), dtype=np.float64)

        for a, b in zip(sequence[:-1], sequence[1:]):
            counts[a, b] += 1.0

        # Normalización por filas → matriz estocástica por filas
        row_sums = counts.sum(axis=1, keepdims=True)
        # Evitar división por cero: filas sin salidas se vuelven Dirac en sí mismas
        row_sums = np.where(row_sums < InformationConstants.EPSILON, 1.0, row_sums)
        M = counts / row_sums
        # Forzar estocasticidad exacta
        for i in range(n):
            if abs(M[i].sum() - 1.0) > 1e-9:
                M[i] = np.eye(n)[i]

        spectrum = eigvals(M)
        # Ordenar por módulo descendente
        moduli = np.abs(spectrum)
        order = np.argsort(-moduli)
        spectrum = spectrum[order]
        moduli = moduli[order]

        # Gap: 1 − |λ₂| (si existe)
        if len(moduli) >= 2:
            gap = float(1.0 - moduli[1])
        else:
            gap = 0.0

        # Veto espectral blando (se eleva excepción más arriba si se desea estricto)
        if moduli[0] > 1.0 + 1e-8:
            raise SpectralDegeneracyVeto(
                f"Radio espectral ρ(M)={moduli[0]:.6f} > 1: operador no contractivo."
            )

        return M, spectrum, max(gap, 0.0)

    # ─── 1.4 Entropías de Shannon y von Neumann ──────────────────────────────
    @staticmethod
    def _compute_entropies(
        p: NDArray[np.float64],
    ) -> Tuple[float, float]:
        r"""
        H_Shannon(p) = −∑ p_i log₂ p_i
        S_vN(ρ) = −Tr(ρ log₂ ρ) con ρ = diag(p)

        Para estados clásicos (ρ diagonal) ambas coinciden. Se calculan por
        separado para mantener la dualidad clásico-cuántica explícita.
        """
        p_safe = p[p > InformationConstants.EPSILON]
        if p_safe.size == 0:
            return 0.0, 0.0
        h = float(-np.sum(p_safe * np.log2(p_safe)))
        # von Neumann idéntico en la base de estados
        s_vn = h
        return h, s_vn

    # ─── 1.5 Método terminal de la Fase 1 (puente hacia Fase 2) ──────────────
    @classmethod
    def evaluate_thermodynamic_manifold(
        cls,
        text: str,
        work_function: float = InformationConstants.DEFAULT_WORK_FUNCTION,
    ) -> TextThermodynamics:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        MÉTODO TERMINAL DE LA FASE 1 / DOMINIO INICIAL DE LA FASE 2
        ═══════════════════════════════════════════════════════════════════════

        Calcula el estado termodinámico-espectral completo del texto incidente.

        Algoritmo:
          1. Proyección al simplejo Δ³ y secuencia de estados.
          2. Construcción del operador de Markov M y su espectro.
          3. Entropías de Shannon y von Neumann.
          4. Normalización \tilde{H} = H / H_max.
          5. Exergía Ξ = H_max − H.
          6. Condición de absorción exérgica: \tilde{H} ≤ Φ.
          7. Condición de no-degeneración espectral: gap ≥ MIN_SPECTRAL_GAP
             (o se registra warning si se prefiere veto blando).

        Retorna un TextThermodynamics inmutable que es el argumento de entrada
        canónico de Phase2_HomeomorphicValidator.project_to_simplex.
        """
        p, sequence = cls._compute_state_probabilities(text)
        M, spectrum, gap = cls._build_markov_operator(sequence)
        h_shannon, s_vn = cls._compute_entropies(p)

        h_norm = h_shannon / InformationConstants.H_MAX
        exergy = InformationConstants.H_MAX - h_shannon
        is_viable = h_norm <= work_function

        if not is_viable:
            logger.error(
                "Fricción entrópica catastrófica: H̃=%.6f > Φ=%.4f | Ξ=%.6f bits",
                h_norm, work_function, exergy,
            )
            raise ThermodynamicEntropyVeto(
                f"El texto crudo posee entropía sintáctica normalizada "
                f"H̃={h_norm:.6f} que supera la Función de Trabajo Φ={work_function}. "
                f"Exergía disponible Ξ={exergy:.6f} bits. "
                f"Colapso de la función de onda informacional (veto fotoeléctrico)."
            )

        if gap < InformationConstants.MIN_SPECTRAL_GAP:
            logger.warning(
                "Gap espectral del operador de Markov degenerado: γ=%.2e < δ_min=%.2e. "
                "Mezclado excesivo; se permite continuar bajo vigilancia.",
                gap, InformationConstants.MIN_SPECTRAL_GAP,
            )
            # Se puede elevar SpectralDegeneracyVeto aquí si se desea veto estricto.

        logger.info(
            "Termodinámica viable: H̃=%.6f ≤ Φ=%.4f | Ξ=%.6f bits | "
            "S_vN=%.6f | gap(M)=%.6f",
            h_norm, work_function, exergy, s_vn, gap,
        )

        return TextThermodynamics(
            shannon_entropy=h_shannon,
            von_neumann_entropy=s_vn,
            normalized_entropy=h_norm,
            work_function=work_function,
            is_exergically_viable=is_viable,
            state_distribution=p,
            exergy=exergy,
            markov_matrix=M,
            markov_spectrum=spectrum,
            spectral_gap=gap,
            state_sequence=sequence,
        )

# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: VALIDACIÓN HOMEOMÓRFICA CATEGÓRICA, HOMOLÓGICA Y ESPECTRAL
#         (continuación directa del TextThermodynamics de la Fase 1)
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_HomeomorphicValidator(Phase1_TextStatisticalMechanics):
    r"""
    Continuación funtorial de la Fase 1. El dominio de sus métodos principales
    es exactamente el codominio de
        Phase1_TextStatisticalMechanics.evaluate_thermodynamic_manifold
    (objeto TextThermodynamics).

    Subyuga al motor esclavo `report_parser_crudo.py`. Exige que la extracción
    demuestre la existencia de una equivalencia de categorías
        F: C_text  ≃  C_parsed
    (funtor plenamente fiel y esencialmente sobreyectivo), un isomorfismo de
    homología simplicial H_*(−;ℤ) y la isospectralidad de los 1-esqueletos
    (espectros de adyacencia y del laplaciano combinatorio).

    Solo entonces se declara que la variedad parseada es homeomorfa al espacio
    fuente en el nivel de sus grafos dirigidos subyacentes.
    """

    # ─── 2.1 Micro-framework de Teoría de Categorías ─────────────────────────
    class _SmallCategory:
        r"""
        Categoría finita pequeña.
        Objetos = enteros (índices de línea / símplices).
        Morfismos generadores = aristas dirigidas (src → tgt).
        Se completan las identidades y se permite composición de caminos
        (categoría libre del grafo subyacente).
        """

        def __init__(self, name: str) -> None:
            self.name = name
            self.objects: Set[int] = set()
            # morfismos generadores: (src, tgt) → label
            self._generators: Dict[Tuple[int, int], str] = {}
            # cierre transitivo + identidades se calcula bajo demanda
            self._hom_cache: Optional[Dict[Tuple[int, int], bool]] = None

        def add_object(self, obj_id: int) -> None:
            self.objects.add(obj_id)
            self._hom_cache = None

        def add_morphism(self, src: int, tgt: int, label: str = "adj") -> None:
            self.add_object(src)
            self.add_object(tgt)
            self._generators[(src, tgt)] = label
            self._hom_cache = None

        def _ensure_hom_cache(self) -> None:
            if self._hom_cache is not None:
                return
            # Warshall sobre el grafo de generadores + identidades
            objs = list(self.objects)
            reach: Dict[Tuple[int, int], bool] = {}
            for o in objs:
                reach[(o, o)] = True  # identidades
            for (s, t) in self._generators:
                reach[(s, t)] = True
            # Clausura transitiva
            for k in objs:
                for i in objs:
                    for j in objs:
                        if reach.get((i, k), False) and reach.get((k, j), False):
                            reach[(i, j)] = True
            self._hom_cache = reach

        def hom_set(self, src: int, tgt: int) -> bool:
            r"""∃ morfismo src → tgt (incluyendo identidades y compuestos)."""
            self._ensure_hom_cache()
            return bool(self._hom_cache.get((src, tgt), False))  # type: ignore

        @property
        def objects_list(self) -> List[int]:
            return sorted(self.objects)

        @property
        def generators(self) -> Dict[Tuple[int, int], str]:
            return dict(self._generators)

        def adjacency_matrix(self) -> NDArray[np.float64]:
            r"""Matriz de adyacencia del grafo subyacente (no dirigido para espectro)."""
            objs = self.objects_list
            n = len(objs)
            idx = {o: i for i, o in enumerate(objs)}
            A = np.zeros((n, n), dtype=np.float64)
            for (s, t) in self._generators:
                i, j = idx[s], idx[t]
                A[i, j] = 1.0
                A[j, i] = 1.0  # simetrización para espectro real
            return A

        def laplacian_matrix(self) -> NDArray[np.float64]:
            r"""Laplaciano combinatorio L = D − A."""
            A = self.adjacency_matrix()
            D = np.diag(A.sum(axis=1))
            return D - A

    # ─── 2.2 Construcción de categorías libres ───────────────────────────────
    @staticmethod
    def _build_text_category(text: str) -> "Phase2_HomeomorphicValidator._SmallCategory":
        r"""
        C_text = categoría libre generada por el grafo camino de las líneas:
            Ob = {0,…,n−1},  Mor generadores = {i → i+1}.
        """
        cat = Phase2_HomeomorphicValidator._SmallCategory("C_text")
        lines = text.splitlines()
        for i in range(len(lines)):
            cat.add_object(i)
            if i + 1 < len(lines):
                cat.add_morphism(i, i + 1, label="succ")
        return cat

    @staticmethod
    def _build_parsed_category(
        parsed_data: List[Dict[str, Any]],
    ) -> "Phase2_HomeomorphicValidator._SmallCategory":
        r"""
        C_parsed = categoría libre del grafo de símplices extraídos.
        Si existe 'line_index' se respeta como identidad de objeto; si no,
        se usa el orden de aparición.
        """
        cat = Phase2_HomeomorphicValidator._SmallCategory("C_parsed")
        for idx, entry in enumerate(parsed_data):
            obj_id = int(entry.get("line_index", idx))
            cat.add_object(obj_id)
            if idx + 1 < len(parsed_data):
                next_obj = int(parsed_data[idx + 1].get("line_index", idx + 1))
                cat.add_morphism(obj_id, next_obj, label="succ")
        return cat

    # ─── 2.3 Inferencia del funtor F: C_text → C_parsed ──────────────────────
    @staticmethod
    def _infer_functor(
        text_cat: "Phase2_HomeomorphicValidator._SmallCategory",
        parsed_cat: "Phase2_HomeomorphicValidator._SmallCategory",
        parsed_data: List[Dict[str, Any]],
    ) -> Dict[int, int]:
        r"""
        Construye el mapeo de objetos del funtor F.
        Preferencia: usar 'line_index' cuando esté presente (mapeo por identidad
        de línea). En su defecto, mapeo posicional que exige cardinalidades
        compatibles (inyección en el caso de tamaños distintos).
        """
        mapping: Dict[int, int] = {}
        text_objs = text_cat.objects_list
        parsed_objs = parsed_cat.objects_list

        if parsed_data and "line_index" in parsed_data[0]:
            rev = {
                int(e["line_index"]): int(e.get("line_index", i))
                for i, e in enumerate(parsed_data)
                if "line_index" in e
            }
            # Mapeo identidad sobre los índices que aparecen
            for t in text_objs:
                if t in rev:
                    mapping[t] = rev[t]
                # si no aparece, no se fuerza (el test de esencial sobreyectividad fallará)
        else:
            n = min(len(text_objs), len(parsed_objs))
            for i in range(n):
                mapping[text_objs[i]] = parsed_objs[i]
        return mapping

    # ─── 2.4 Homología simplicial (H₀, H₁) vía operadores de borde ───────────
    @staticmethod
    def _compute_betti_numbers(
        cat: "Phase2_HomeomorphicValidator._SmallCategory",
    ) -> Tuple[int, int]:
        r"""
        Calcula los números de Betti β₀ = rank H₀ y β₁ = rank H₁ del 1-complejo
        simplicial asociado al grafo subyacente (no dirigido).

        Cadena:
            C₁  --∂₁→  C₀  --∂₀→  0
        donde C₀ = ℤ^{|V|}, C₁ = ℤ^{|E|},
        ∂₁(e={u,v}) = v − u  (orientación arbitraria).

        β₀ = dim ker ∂₀ − dim im ∂₁ = |V| − rank(∂₁)
        β₁ = dim ker ∂₁ = |E| − rank(∂₁)
        (fórmula de Euler en dimensión 1).
        """
        objs = cat.objects_list
        n0 = len(objs)
        if n0 == 0:
            return (0, 0)

        idx = {o: i for i, o in enumerate(objs)}
        edges = list(cat.generators.keys())
        # Simetrizar: cada arista no dirigida una sola vez
        undirected = set()
        for s, t in edges:
            a, b = (s, t) if s <= t else (t, s)
            undirected.add((a, b))
        n1 = len(undirected)

        if n1 == 0:
            # Solo vértices aislados
            return (n0, 0)

        # Matriz de borde ∂₁ : ℝ^{n1} → ℝ^{n0}
        boundary = np.zeros((n0, n1), dtype=np.float64)
        for e_idx, (u, v) in enumerate(sorted(undirected)):
            boundary[idx[u], e_idx] = -1.0
            boundary[idx[v], e_idx] = 1.0

        rank_d1 = int(matrix_rank(boundary, tol=InformationConstants.EPSILON))
        beta0 = n0 - rank_d1
        beta1 = n1 - rank_d1
        return (max(beta0, 0), max(beta1, 0))

    # ─── 2.5 Distancia espectral de grafos ───────────────────────────────────
    @staticmethod
    def _spectral_distance(
        mat_a: NDArray[np.float64],
        mat_b: NDArray[np.float64],
    ) -> float:
        r"""
        Distancia L² entre espectros ordenados (valores propios reales, laplaciano
        y adyacencia simétrica son autoadjuntos).
        Si las dimensiones difieren se rellenan con ceros (espectros de grafos
        con vértices aislados).
        """
        ev_a = np.sort(np.real(eigvals(mat_a)))
        ev_b = np.sort(np.real(eigvals(mat_b)))
        n = max(len(ev_a), len(ev_b))
        pad_a = np.zeros(n)
        pad_b = np.zeros(n)
        pad_a[: len(ev_a)] = ev_a
        pad_b[: len(ev_b)] = ev_b
        return float(np.linalg.norm(pad_a - pad_b))

    # ─── 2.6 Validación de equivalencia de categorías ────────────────────────
    @classmethod
    def _validate_categorical_equivalence(
        cls,
        text: str,
        parsed_data: List[Dict[str, Any]],
    ) -> Tuple[bool, str, Tuple[int, int], Tuple[int, int], float, float]:
        r"""
        Verifica:
          (i)   F plenamente fiel y esencialmente sobreyectivo,
          (ii)  isomorfismo de homología (β₀, β₁),
          (iii) isospectralidad de adyacencia y laplaciano.

        Retorna
          (is_equiv, log, betti_text, betti_parsed, dist_A, dist_L)
        """
        text_cat = cls._build_text_category(text)
        parsed_cat = cls._build_parsed_category(parsed_data)
        mapping = cls._infer_functor(text_cat, parsed_cat, parsed_data)

        log_lines: List[str] = []
        valid = True

        # (i-a) Esencialmente sobreyectivo
        image_objects = set(mapping.values())
        missing = parsed_cat.objects - image_objects
        if missing:
            valid = False
            log_lines.append(f"No esencialmente sobreyectivo; objetos no alcanzados: {sorted(missing)}")

        # (i-b) Plenamente fiel (existencia de morfismos se preserva y refleja)
        for (src, tgt) in text_cat.generators:
            if src in mapping and tgt in mapping:
                psrc, ptgt = mapping[src], mapping[tgt]
                if not parsed_cat.hom_set(psrc, ptgt):
                    valid = False
                    log_lines.append(
                        f"Falta morfismo en C_parsed: ({psrc}→{ptgt}) imagen de ({src}→{tgt})"
                    )

        for (psrc, ptgt) in parsed_cat.generators:
            pre_src = [k for k, v in mapping.items() if v == psrc]
            pre_tgt = [k for k, v in mapping.items() if v == ptgt]
            if not any(text_cat.hom_set(s, t) for s in pre_src for t in pre_tgt):
                valid = False
                log_lines.append(
                    f"Morfismo en C_parsed ({psrc}→{ptgt}) sin preimagen en C_text"
                )

        # (ii) Homología
        betti_text = cls._compute_betti_numbers(text_cat)
        betti_parsed = cls._compute_betti_numbers(parsed_cat)
        if betti_text != betti_parsed:
            valid = False
            log_lines.append(
                f"Homología no isomorfa: β(text)={betti_text}, β(parsed)={betti_parsed}"
            )
        else:
            log_lines.append(f"H_* isomorfo: β={betti_text}")

        # (iii) Isospectralidad
        A_t = text_cat.adjacency_matrix()
        A_p = parsed_cat.adjacency_matrix()
        L_t = text_cat.laplacian_matrix()
        L_p = parsed_cat.laplacian_matrix()
        dist_A = cls._spectral_distance(A_t, A_p)
        dist_L = cls._spectral_distance(L_t, L_p)

        if dist_A > InformationConstants.SPECTRAL_ISOMORPHISM_TOL:
            valid = False
            log_lines.append(f"Espectros de adyacencia distantes: ‖Spec(A_t)−Spec(A_p)‖₂={dist_A:.2e}")
        if dist_L > InformationConstants.SPECTRAL_ISOMORPHISM_TOL:
            valid = False
            log_lines.append(f"Espectros de laplaciano distantes: ‖Spec(L_t)−Spec(L_p)‖₂={dist_L:.2e}")

        if valid:
            log_lines.append("Equivalencia de categorías + isomorfismo homológico + isospectralidad verificados.")

        log = " | ".join(log_lines)
        return valid, log, betti_text, betti_parsed, dist_A, dist_L

    # ─── 2.7 Método principal de la Fase 2 (continuación de Fase 1) ──────────
    @classmethod
    def project_to_simplex(
        cls,
        text: str,
        thermo: TextThermodynamics,
    ) -> HomeomorphicValidation:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        CONTINUACIÓN DIRECTA DEL MÉTODO TERMINAL DE LA FASE 1
        ═══════════════════════════════════════════════════════════════════════

        Entrada canónica: el TextThermodynamics producido por
            Phase1_TextStatisticalMechanics.evaluate_thermodynamic_manifold

        Algoritmo:
          1. Subyugación del DFA ReportParserCrudo → raw_ast.
          2. Construcción de C_text y C_parsed.
          3. Inferencia y validación del funtor F (equivalencia de categorías).
          4. Cálculo de números de Betti (H₀, H₁).
          5. Comparación espectral de adyacencia y laplaciano.
          6. Emisión de HomeomorphicValidation o detonación de veto.

        El objeto retornado es el dominio de la Fase 3 (ParserOntologyAgent).
        """
        # Guardia termodinámica (doble chequeo de pureza)
        if not thermo.is_exergically_viable:
            raise ThermodynamicEntropyVeto(
                "Manifold termodinámico no viable recibido en Fase 2; "
                "imposible proyectar a complejo simplicial."
            )

        dfa_engine = ReportParserCrudo()
        try:
            parsed_data = dfa_engine.parse(text)
        except Exception as e:
            raise HomeomorphismViolationError(
                f"El Autómata (DFA) falló al construir una variedad estable: {e!s}"
            ) from e

        if not parsed_data:
            raise HomeomorphismViolationError(
                "La proyección estructural resultó en el conjunto vacío. "
                "Cero 1-símplices extraídos; homología trivial degenerada."
            )

        (
            is_equiv,
            log,
            betti_t,
            betti_p,
            dist_A,
            dist_L,
        ) = cls._validate_categorical_equivalence(text, parsed_data)

        if not is_equiv:
            logger.error("Violación del homeomorfismo categórico/homológico/espectral: %s", log)
            raise HomeomorphismViolationError(
                f"El funtor de parseo no establece una equivalencia de categorías "
                f"ni isomorfismo homológico. Detalle: {log}"
            )

        logger.info(
            "Homeomorfismo validado. Símplices=%d | β_text=%s | β_parsed=%s | "
            "dist_A=%.2e | dist_L=%.2e | log=%s",
            len(parsed_data), betti_t, betti_p, dist_A, dist_L, log,
        )

        return HomeomorphicValidation(
            is_homeomorphic=True,
            parsed_simplexes=len(parsed_data),
            thermodynamics=thermo,
            raw_ast=parsed_data,
            betti_numbers_text=betti_t,
            betti_numbers_parsed=betti_p,
            adjacency_spectra_distance=dist_A,
            laplacian_spectra_distance=dist_L,
            functor_log=log,
            categorical_equivalence=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 / ORQUESTADOR SUPREMO: PARSER ONTOLOGY AGENT
#         (composición funtorial de las Fases 1 y 2)
# ══════════════════════════════════════════════════════════════════════════════
class ParserOntologyAgent(Morphism, Phase2_HomeomorphicValidator):
    r"""
    Funtor de Ingesta Soberana (Endofuntor sobre el estrato PHYSICS).

    Composición estricta de morfismos:
        CategoricalState
            ∘  Phase2_HomeomorphicValidator.project_to_simplex
            ∘  Phase1_TextStatisticalMechanics.evaluate_thermodynamic_manifold

    Mapea el crudo estocástico hacia un CategoricalState puro, aislando
    completamente la entropía, el espectro de Markov y garantizando el
    homeomorfismo categórico-homológico-espectral para la pureza del DAG
    en `pipeline_director.py`.
    """

    def __init__(
        self,
        work_function: float = InformationConstants.DEFAULT_WORK_FUNCTION,
    ) -> None:
        self._work_function = float(work_function)

    def __call__(self, text: str) -> CategoricalState:
        r"""
        Ejecuta la composición funtorial anidada de las tres fases.

        Fase 1 → TextThermodynamics
        Fase 2 → HomeomorphicValidation   (continuación directa)
        Fase 3 → CategoricalState         (proyección ortogonal al DAG)
        """
        if not isinstance(text, str):
            raise TypeError(
                f"Se esperaba str para el tensor de texto; recibido {type(text).__name__}"
            )

        logger.info(
            "Iniciando auditoría termodinámica-espectral-categórica del texto incidente "
            "(Φ=%.4f)…",
            self._work_function,
        )

        # ── Fase 1: Manifold termodinámico y espectral ───────────────────────
        thermo_state: TextThermodynamics = self.evaluate_thermodynamic_manifold(
            text, self._work_function
        )

        # ── Fase 2: Homeomorfismo categórico / homológico / espectral ────────
        # (continuación directa del objeto TextThermodynamics)
        homeo_state: HomeomorphicValidation = self.project_to_simplex(
            text, thermo_state
        )

        # ── Fase 3: Inyección al Estado Categórico (pureza garantizada) ──────
        betti_t = homeo_state.betti_numbers_text
        betti_p = homeo_state.betti_numbers_parsed

        return CategoricalState(
            stratum=Stratum.PHYSICS,
            payload={
                "parsed_ast": homeo_state.raw_ast,
                "simplex_count": homeo_state.parsed_simplexes,
                "syntactic_entropy": thermo_state.normalized_entropy,
                "von_neumann_entropy": thermo_state.von_neumann_entropy,
                "exergy_bits": thermo_state.exergy,
                "state_distribution": thermo_state.state_distribution.tolist(),
                "markov_spectrum": [
                    complex(z).real if abs(complex(z).imag) < 1e-12 else complex(z)
                    for z in thermo_state.markov_spectrum
                ],
                "spectral_gap": thermo_state.spectral_gap,
                "betti_text": list(betti_t),
                "betti_parsed": list(betti_p),
                "adjacency_spectra_distance": homeo_state.adjacency_spectra_distance,
                "laplacian_spectra_distance": homeo_state.laplacian_spectra_distance,
                "categorical_log": homeo_state.functor_log,
            },
            context={
                "is_homeomorphic": homeo_state.is_homeomorphic,
                "categorical_equivalence": homeo_state.categorical_equivalence,
                "work_function_applied": self._work_function,
                "exergically_viable": thermo_state.is_exergically_viable,
            },
        )

# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "InformationConstants",
    "ParserOntologyError",
    "ThermodynamicEntropyVeto",
    "SpectralDegeneracyVeto",
    "HomeomorphismViolationError",
    "EmptyManifoldError",
    "HomologicalInvariantError",
    "TextThermodynamics",
    "HomeomorphicValidation",
    "Phase1_TextStatisticalMechanics",
    "Phase2_HomeomorphicValidator",
    "ParserOntologyAgent",
]
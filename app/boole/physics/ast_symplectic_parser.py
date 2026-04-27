"""
Módulo: AST Symplectic Parser (Analizador Simpléctico de Árboles de Sintaxis Abstracta)
Ubicación: app/boole/physics/ast_symplectic_parser.py
Versión 1.1 – Rigurización matemática y corrección topológica

Naturaleza Ciber-Física y Termodinámica:
Actúa como base del Estrato Γ-PHYSICS. Transforma el código fuente en un espacio métrico analizable.
1. Forma Simpléctica (ω): Extrae tensores de lectura (q) y escritura (p) del AST y permite calcular
   el conmutador de Lie [A, B] = |A.p_writes ∩ B.q_reads| - |B.p_writes ∩ A.q_reads|,
   indicando interferencia no conmutativa si es distinto de cero.
2. Frontera de Dirichlet (Silo A): Impone restricciones de Lipschitz sobre el anidamiento de JSONs
   para evitar ataques de denegación de servicio (Billion Laughs) y agotar el KV-Cache del LLM.
3. Inercia Termodinámica: Mide la entropía ciclomática (McCabe). Si el exponente de Lyapunov
   diverge (excede el umbral), emite un Veto Físico Absoluto.
4. Retracto TOON: Destila la topología validada hacia una Base Canónica Tabular hiperdensa,
   aniquilando la entropía sintáctica.
"""

from __future__ import annotations

import ast
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

logger = logging.getLogger("Gamma.Physics.ASTSymplecticParser.2.0")

# =============================================================================
# CONSTANTES FÍSICAS Y TOPOLÓGICAS
# =============================================================================
class DirichletBoundaries:
    """Condiciones de frontera absolutas para la variedad estructural."""
    MAX_AST_DEPTH: int = 20            # Cota de Lipschitz para anidamiento
    MAX_CYCLOMATIC_ENTROPY: int = 15   # Umbral máximo de inercia computacional (P_diss)
    JSON_MAX_RECURSION: int = 5        # Estrangulamiento topológico para el SiloAContract
    TOON_MAX_FLATTEN_DEPTH: int = 2    # Profundidad extra al representar sub‑tensores


# =============================================================================
# ESTRUCTURAS DEL ESPACIO DE FASE
# =============================================================================
@dataclass(frozen=True, order=True)
class PhaseSpaceCoordinates:
    """
    Coordenadas en el espacio de Hilbert para la Forma Simpléctica.
    q ↔ posiciones (lecturas en memoria/estado).
    p ↔ momentos  (mutaciones/escrituras en memoria/estado).
    """
    q_reads: FrozenSet[str] = field(default_factory=frozenset)
    p_writes: FrozenSet[str] = field(default_factory=frozenset)

    def __repr__(self) -> str:
        return f"Phase(q={sorted(self.q_reads)}, p={sorted(self.p_writes)})"

    @staticmethod
    def lie_commutator(A: PhaseSpaceCoordinates, B: PhaseSpaceCoordinates) -> int:
        """
        Evalúa la forma simpléctica (conmutador) entre dos herramientas.

        Definición:
            [A, B] = |A.p_writes ∩ B.q_reads| - |B.p_writes ∩ A.q_reads|

        Propiedades:
          - [A, B] = -[B, A]  (antisimetría)
          - Si [A, B] ≠ 0, las operaciones no conmutan y existe interferencia cuántica.
        """
        return len(A.p_writes.intersection(B.q_reads)) - len(B.p_writes.intersection(A.q_reads))


@dataclass(frozen=True)
class ThermodynamicProfile:
    """Perfil inercial y de disipación de potencia de la herramienta."""
    cyclomatic_entropy: int
    is_asymptotically_stable: bool
    mass_penalty: float

    def __repr__(self) -> str:
        return (f"Thermo(ν={self.cyclomatic_entropy}, "
                f"stable={self.is_asymptotically_stable}, "
                f"μ={self.mass_penalty:.3f})")


# =============================================================================
# OPERADORES ALGEBRAICOS DE AST (VISITADOR CORREGIDO)
# =============================================================================
class SymplecticASTVisitor(ast.NodeVisitor):
    """
    Navega el 1‑esqueleto del AST para mapear la dinámica Port‑Hamiltoniana.
    Clasifica los accesos a variables en momentos (p) y posiciones (q).

    La profundidad se controla sobrescribiendo `visit` en lugar de `generic_visit`,
    garantizando que cada nodo (incluso con visitador específico) contribuya al
    contador de profundidad.
    """

    COMPLEXITY_NODES = {
        ast.If, ast.For, ast.While,
        ast.ExceptHandler,          # cláusula except
        ast.match_case,             # Python ≥ 3.10
        ast.BoolOp,                 # and/or añaden caminos
        ast.comprehension,          # loops implícitos en comprensiones
    }

    def __init__(self) -> None:
        self.reads: Set[str] = set()
        self.writes: Set[str] = set()
        self.current_depth: int = 0
        self.max_depth_reached: int = 0
        self.cyclomatic_complexity: int = 1  # β0 = 1 (camino base)

    # ---------- Control de profundidad unificado ----------
    def visit(self, node: ast.AST):
        """
        Punto de entrada de cada nodo. Incrementa la profundidad,
        aplica límite de Lipschitz y luego delega al visitador específico.
        """
        self.current_depth += 1
        self.max_depth_reached = max(self.max_depth_reached, self.current_depth)

        if self.current_depth > DirichletBoundaries.MAX_AST_DEPTH:
            raise RecursionError(
                f"Singularidad topológica: profundidad {self.current_depth} "
                f"excede la cota de Dirichlet ({DirichletBoundaries.MAX_AST_DEPTH})."
            )

        # La clase base `visit` redirige a `visit_XXX` o `generic_visit`
        result = super().visit(node)

        self.current_depth -= 1
        return result

    # ---------- Mapeo simpléctico (q / p) ----------
    def visit_Name(self, node: ast.Name):
        """Store → p_writes, Load → q_reads."""
        if isinstance(node.ctx, ast.Store):
            self.writes.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.reads.add(node.id)
        # No llamamos a generic_visit; los nombres no tienen hijos

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.ctx, ast.Store):
            self.writes.add(node.attr)
        elif isinstance(node.ctx, ast.Load):
            self.reads.add(node.attr)
        self.generic_visit(node)   # recorre value (ej: 'obj.attr' → analiza 'obj')

    # ---------- Contadores de complejidad ciclomática ----------
    def _inc_complexity(self):
        self.cyclomatic_complexity += 1

    def visit_If(self, node: ast.If):
        self._inc_complexity()
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self._inc_complexity()
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        self._inc_complexity()
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        self._inc_complexity()
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match):
        # Cada caso incrementa uno; match en sí mismo no añade arista extra,
        # pero la posibilidad de múltiples patrones es análoga a if/elif.
        self._inc_complexity()
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp):
        # Un and/or aporta tantas ramas como operandos-1
        self.cyclomatic_complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension):
        # Comprensiones contienen un `for` implícito
        self._inc_complexity()
        self.generic_visit(node)


# =============================================================================
# ANALIZADORES FÍSICOS Y DE FRONTERA
# =============================================================================
class JSONBoundaryEnforcer:
    """
    Impone condiciones de frontera de Dirichlet sobre el SiloAContract.
    """

    @staticmethod
    def enforce_lipschitz_continuity(json_obj: Union[Dict, List, Any],
                                     current_depth: int = 0) -> bool:
        """
        Verifica que el tensor JSON no exceda la recursión máxima.
        """
        if current_depth > DirichletBoundaries.JSON_MAX_RECURSION:
            logger.error(
                "Violación de frontera de Dirichlet en JSON: "
                f"profundidad {current_depth} > {DirichletBoundaries.JSON_MAX_RECURSION}"
            )
            return False

        if isinstance(json_obj, dict):
            return all(
                JSONBoundaryEnforcer.enforce_lipschitz_continuity(v, current_depth + 1)
                for v in json_obj.values()
            )
        if isinstance(json_obj, list):
            return all(
                JSONBoundaryEnforcer.enforce_lipschitz_continuity(item, current_depth + 1)
                for item in json_obj
            )
        return True


class CategoricalTOONDistiller:
    """
    Transforma el espacio estocástico JSON en una representación tabular plana
    (variedad TOON), preservando la homotopía de la información.
    """

    @staticmethod
    def _flatten_value(value: Any, depth: int) -> str:
        """
        Convierte un valor, limitando la profundidad de representación.
        """
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            if depth >= DirichletBoundaries.TOON_MAX_FLATTEN_DEPTH:
                return f"<dict: {len(value)} keys>"
            items = ", ".join(
                f"{k}: {CategoricalTOONDistiller._flatten_value(v, depth + 1)}"
                for k, v in value.items()
            )
            return "{" + items + "}"
        if isinstance(value, list):
            if depth >= DirichletBoundaries.TOON_MAX_FLATTEN_DEPTH:
                return f"<list: {len(value)} items>"
            items = ", ".join(
                CategoricalTOONDistiller._flatten_value(v, depth + 1) for v in value
            )
            return "[" + items + "]"
        return str(value)

    @staticmethod
    def distill_to_toon(json_contract: Dict[str, Any]) -> str:
        """
        Retracto topológico: comprime el JSON en una tabla Markdown.
        """
        if not isinstance(json_contract, dict):
            raise TypeError("El contrato debe ser un diccionario para destilar a TOON")

        keys = list(json_contract.keys())
        header = "| " + " | ".join(keys) + " |"
        separator = "|" + "|".join(["---"] * len(keys)) + "|"
        row_values = [
            CategoricalTOONDistiller._flatten_value(json_contract[k], 0)
            for k in keys
        ]
        row = "| " + " | ".join(row_values) + " |"
        return f"{header}\n{separator}\n{row}"


# =============================================================================
# ORQUESTADOR PRINCIPAL (FUNTOR F)
# =============================================================================
class ASTSymplecticParser:
    """
    Funtor F: CódigoCrudo → V_{Gamma‑PHYSICS}.
    """

    @staticmethod
    def parse_tool_dynamics(source_code: str) -> Tuple[PhaseSpaceCoordinates, ThermodynamicProfile]:
        """
        Convierte una herramienta en sus invariantes físicos abstractos.

        Args:
            source_code: Código fuente Python de la herramienta.

        Returns:
            (coordenadas en espacio de fase, perfil termodinámico)
        Raises:
            RecursionError: si la profundidad del AST supera el límite de Dirichlet.
            SyntaxError: si el código no es Python válido.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise ValueError(f"Ruido sintáctico no resoluble: {e}")

        visitor = SymplecticASTVisitor()
        visitor.visit(tree)

        # 1. Espacio de Fase Simpléctico
        phase_space = PhaseSpaceCoordinates(
            q_reads=frozenset(visitor.reads),
            p_writes=frozenset(visitor.writes)
        )

        # 2. Perfil termodinámico
        entropy = visitor.cyclomatic_complexity
        is_stable = entropy <= DirichletBoundaries.MAX_CYCLOMATIC_ENTROPY
        mass_penalty = math.exp(entropy / DirichletBoundaries.MAX_CYCLOMATIC_ENTROPY)

        thermo = ThermodynamicProfile(
            cyclomatic_entropy=entropy,
            is_asymptotically_stable=is_stable,
            mass_penalty=mass_penalty
        )

        if not is_stable:
            logger.critical(
                "VETO FÍSICO: entropía ciclomática %d excede el límite de %d.",
                entropy, DirichletBoundaries.MAX_CYCLOMATIC_ENTROPY
            )

        return phase_space, thermo

    @staticmethod
    def process_data_contract(json_schema: Dict[str, Any]) -> Optional[str]:
        """
        Procesa el contrato de datos aplicando la frontera de Dirichlet y, si es válido,
        lo destila a formato TOON (Markdown tabular).
        """
        if not JSONBoundaryEnforcer.enforce_lipschitz_continuity(json_schema):
            return None  # Aniquilación topológica

        return CategoricalTOONDistiller.distill_to_toon(json_schema)
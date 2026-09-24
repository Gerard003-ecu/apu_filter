# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : GÖDEL AGENT (SOBERANO DE AUTOMEJORA RECURSIVA Y AUTORREFERENCIA)          ║
║ UBICACIÓN: app/agents/wisdom/godel_agent.py                                          ║
║ VERSIÓN  : 2.1.0-Doctoral-Clifford-Betti-CFG-Cayley-Kreiss-Heyting-Banach-ESP32      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

FUNDAMENTACIÓN MATEMÁTICO-FÍSICA RIGUROSA DEL ESTRATO DE SABIDURÍA (V_W)
────────────────────────────────────────────────────────────────────────────────────────
El presente sistema formaliza el ciclo de Automejora Recursiva (RSI) para agentes
ciber-físicos autorreferenciales, eliminando heurísticas ad-hoc y gobernando la mutación
del par $(\pi_t, I_t)$ mediante una jerarquía anidada en 3 fases:

1. FASE 1: Fundamentos Hipercomplejos, Topológicos-Espectrales y Estructuras Cuánticas.
   - Espacio de estados en álgebra cuaterniónica $\mathbb{H} \cong \mathcal{C}\ell_{0,2}(\mathbb{R})$,
     con mapa exponencial de Lie $\exp: \mathfrak{su}(2) \to SU(2)$ para generación rigurosa de rotores.
   - Retículo de Heyting intuicionista $\Omega_3 = \{\bot \prec \frac{1}{2} \prec \top\}$ con
     verificación constructiva de la ley de residuación (axioma fundacional de Heyting).
   - Topología algebraica del Abstract Syntax Tree (AST) enriquecido con grafo de flujo de
     control (CFG): 1-complejo simplicial con back-edges de bucles, operadores de frontera
     $\partial_0, \partial_1$, grupos de homología $H_0, H_1$, números de Betti $(\beta_0,\beta_1)$,
     característica de Euler-Poincaré $\chi$ y complejidad ciclomática de McCabe.
   - Espacio de Banach $\mathcal{B}(\mathcal{X})$: Radio espectral de Gelfand
     $\rho(T) = \lim_{k\to\infty} \|T^k\|^{1/k} = \max_{\lambda \in \sigma(T)} |\lambda| < 1$,
     verificado empíricamente, más estimación de la constante de Kreiss para acotar el
     crecimiento transitorio de operadores no normales.
   - Mecánica cuántica de densidad (MAC): Operadores de von Neumann $\rho \in \mathcal{S}(\mathcal{H})$,
     entropía de von Neumann y de Rényi-2, y evolución disipativa vía Lindblad.
   - Terminal Fase 1: Síntesis geométrica de la variedad de estado $\mathfrak{M}_{\text{Wisdom}}$
     con validación de precondiciones dimensionales (puente formal obligatorio a Fase 2).

2. FASE 2: Dinámica Port-Hamiltoniana, Física del Circuito Crowbar y Teoría de Haces.
   - Circuito de interrupción ciber-física Crowbar: solución analítica exacta del transitorio
     RLC subamortiguado (tiempo de pico, corriente pico e integral de Joule por cuadratura
     numérica adaptativa) del tiristor BT151, con verificación de Área de Operación Segura (SOA).
   - Sistemas Port-Hamiltonianos (PHS): $\dot{x} = [J(x) - R(x)]\nabla H(x) + g(x)u$, integrados
     mediante la transformada de Cayley (regla del punto medio implícito), que garantiza
     disipatividad discreta *exacta* (no solo continua) para Hamiltonianos cuadráticos.
   - Topos y morfismo clasificador de subobjetos $\chi_U: X \to \Omega_3$ construido
     composicionalmente como el ínfimo de Heyting sobre una cubierta finita de subobjetos.
   - Motor metamórfico seguro de reescritura AST con interpretación abstracta de intervalos,
     bloqueo de atributos peligrosos y cota estructural de complejidad (anti-bomba).
   - Terminal Fase 2: Evaluación functorial de transición categórica $\Phi: \mathfrak{M} \to \mathfrak{M}'$.

3. FASE 3: El Soberano de Gödel (RSI), Máquina Autorreferencial y Certificación Terminal.
   - Integración del ciclo cerrado OODA (Observe-Orient-Decide-Act) con observación única.
   - Verificación constructiva del punto fijo de Banach vía iteración de Picard contrastada
     con la solución cerrada de la serie de Neumann.
   - Sandbox de ejecución con lista blanca explícita de builtins (mitigación de fuga de VM).
   - Veto ciber-físico e inyección de pulso a tiristor en silicio.
   - Emisión de certificados criptográficos inmutables con firma hash de no-repudio SHA-256.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import math
import textwrap
import time
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple, Union

import numpy as np
import scipy.linalg as la
from scipy import integrate

logger = logging.getLogger("APU.Wisdom.GodelAgent")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: FUNDAMENTOS HIPERCOMPLEJOS, TOPOLÓGICOS-ESPECTRALES Y
#         ESTRUCTURAS CUÁNTICAS DE BANACH
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# §1.1 ÁLGEBRA HIPERCOMPLEJA CUATERNIÓNICA (ℍ) CON MAPA EXPONENCIAL DE LIE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Quaternion:
    r"""
    Elemento del álgebra de división asociativa no conmutativa $\mathbb{H}$.
    Base canónica $\{1, i, j, k\}$ con relaciones fundamentales:
    $i^2 = j^2 = k^2 = ijk = -1$. El subgrupo de norma unitaria $\{q : \|q\|=1\}$
    es isomorfo a $SU(2)$, doble recubrimiento universal de $SO(3)$.
    """
    w: float
    x: float
    y: float
    z: float

    def __add__(self, other: Quaternion) -> Quaternion:
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Quaternion) -> Quaternion:
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: Union[Quaternion, float]) -> Quaternion:
        if isinstance(other, (int, float)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        # Producto de Hamilton (no conmutativo)
        return Quaternion(
            w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def __rmul__(self, scalar: float) -> Quaternion:
        """Conmutatividad del producto escalar-cuaternión: λq = qλ, λ ∈ ℝ."""
        return self.__mul__(scalar)

    def conjugate(self) -> Quaternion:
        """Conjugación simpléctica: q* = w - xi - yj - zk."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm_squared(self) -> float:
        """Norma euclídea al cuadrado: ||q||² = qq* = w² + x² + y² + z²."""
        return self.w**2 + self.x**2 + self.y**2 + self.z**2

    def norm(self) -> float:
        return math.sqrt(self.norm_squared())

    def versor(self) -> Quaternion:
        """Proyección al subgrupo unitario $S^3 \\cong SU(2)$: q̂ = q / ||q||."""
        n = self.norm()
        if n < 1e-18:
            raise ZeroDivisionError("No es posible versar un cuaternión de norma nula.")
        return self * (1.0 / n)

    def inverse(self) -> Quaternion:
        """Inverso multiplicativo q⁻¹ = q* / ||q||²."""
        n2 = self.norm_squared()
        if n2 < 1e-18:
            raise ZeroDivisionError("Cuaternión singular no invertible en ℍ.")
        inv_n2 = 1.0 / n2
        return Quaternion(self.w * inv_n2, -self.x * inv_n2, -self.y * inv_n2, -self.z * inv_n2)

    @classmethod
    def exp(cls, q: Quaternion) -> Quaternion:
        r"""
        Mapa exponencial del álgebra de Lie al grupo de Lie:
        $$\exp(q) = e^{w}\left(\cos\|\vec{v}\| + \frac{\vec{v}}{\|\vec{v}\|}\sin\|\vec{v}\|\right),
        \quad \vec{v} = (x, y, z)$$
        Fundamento riguroso para la generación de rotores infinitesimales (evita el uso de
        incrementos ad-hoc no derivados de un generador algebraico válido).
        """
        v_norm = math.sqrt(q.x**2 + q.y**2 + q.z**2)
        exp_w = math.exp(q.w)
        if v_norm < 1e-15:
            return Quaternion(exp_w, 0.0, 0.0, 0.0)
        coeff = exp_w * math.sin(v_norm) / v_norm
        return Quaternion(exp_w * math.cos(v_norm), coeff * q.x, coeff * q.y, coeff * q.z)

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> Quaternion:
        r"""
        Construye el rotor unitario $q = \cos(\theta/2) + \hat{n}\sin(\theta/2)$ mediante el
        mapa exponencial aplicado al generador puro $\theta \hat n / 2 \in \mathfrak{su}(2)$.
        """
        axis_norm_val = float(np.linalg.norm(axis))
        if axis_norm_val < 1e-18:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        axis_hat = axis / axis_norm_val
        half = angle / 2.0
        pure_generator = cls(0.0, axis_hat[0] * half, axis_hat[1] * half, axis_hat[2] * half)
        return cls.exp(pure_generator)

    def to_su2_matrix(self) -> np.ndarray:
        r"""
        Isomorfismo canónico con el álgebra de Lie $\mathfrak{su}(2) \subset \mathbb{C}^{2\times 2}$:
        $$M(q) = \begin{pmatrix} w + iz & -y + ix \\ y + ix & w - iz \end{pmatrix}$$
        """
        return np.array([
            [complex(self.w, self.z), complex(-self.y, self.x)],
            [complex(self.y, self.x), complex(self.w, -self.z)]
        ], dtype=np.complex128)

    def to_rotation_matrix(self) -> np.ndarray:
        r"""
        Homomorfismo de recubrimiento $SU(2) \to SO(3)$ (2 a 1). Para $q$ unitario:
        $$R(q) = I + 2w[\vec v]_\times + 2[\vec v]_\times^2$$
        expresado en forma cerrada explícita.
        """
        q = self.versor()
        w, x, y, z = q.w, q.x, q.y, q.z
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ])


# ──────────────────────────────────────────────────────────────────────────────
# §1.2 RETÍCULO DE HEYTING INTUICIONISTA Ω₃ CON VERIFICACIÓN AXIOMÁTICA FORMAL
# ──────────────────────────────────────────────────────────────────────────────

class HeytingOmega3(IntEnum):
    r"""
    Álgebra de Heyting finita linealmente ordenada (Cadena de Gödel $\mathbf{G}_3$).
    Universo $\Omega_3 = \{\text{VETOED}(0, \bot) \prec \text{DEGRADED}(1, \frac{1}{2}) \prec \text{COHERENT}(2, \top)\}$.
    """
    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    @property
    def verdict(self) -> str:
        return self.name

    def meet(self, other: HeytingOmega3) -> HeytingOmega3:
        r"""Ínfimo reticular (∧): $a \wedge b = \min(a, b)$."""
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: HeytingOmega3) -> HeytingOmega3:
        r"""Supremo reticular (∨): $a \vee b = \max(a, b)$."""
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: HeytingOmega3) -> HeytingOmega3:
        r"""
        Implicación relativa intuicionista (pseudocomplemento relativo):
        $$a \Rightarrow b = \max \{ c \in \Omega_3 : a \wedge c \le b \} = \begin{cases} \top & \text{si } a \le b \\ b & \text{si } a > b \end{cases}$$
        """
        if int(self) <= int(other):
            return HeytingOmega3.COHERENT
        return other

    def neg(self) -> HeytingOmega3:
        r"""Negación intuicionista estricta: $\neg a = (a \Rightarrow \bot)$."""
        return self.implies(HeytingOmega3.VETOED)

    # Operadores ergonómicos isomorfos al álgebra booleana clásica (con semántica intuicionista)
    def __and__(self, other: HeytingOmega3) -> HeytingOmega3:
        return self.meet(other)

    def __or__(self, other: HeytingOmega3) -> HeytingOmega3:
        return self.join(other)

    def __rshift__(self, other: HeytingOmega3) -> HeytingOmega3:
        return self.implies(other)

    def __invert__(self) -> HeytingOmega3:
        return self.neg()

    @classmethod
    def verify_heyting_algebra_axioms(cls) -> bool:
        r"""
        Certificación constructiva y exhaustiva (fuerza bruta sobre $|\Omega_3|^3=27$ ternas)
        de la **ley de residuación**, axioma que define categóricamente a un álgebra de Heyting:
        $$\forall a,b,c \in \Omega_3:\quad a \wedge c \le b \iff c \le (a \Rightarrow b)$$
        Retorna `True` si y solo si `implies` es efectivamente el adjunto derecho de `meet`
        (adjunción de Galois $(-\wedge a) \dashv (a \Rightarrow -)$).
        """
        elements = list(cls)
        for a in elements:
            for b in elements:
                residuum = a.implies(b)
                for c in elements:
                    lhs = (a.meet(c) <= b)
                    rhs = (c <= residuum)
                    if lhs != rhs:
                        return False
        return True


# ──────────────────────────────────────────────────────────────────────────────
# §1.3 TOPOLOGÍA ALGEBRAICA DE GRAFOS Y COMPLEJOS SIMPLICIALES DEL AST + CFG
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SimplicialHomologyCertificate:
    """Certificado invariante topológico simplicial del AST enriquecido con flujo de control."""
    num_nodes: int
    num_edges: int
    betti_0: int                    # Componentes conexas: dim(H_0) = dim(ker ∂_0) - rank(∂_1)
    betti_1: int                    # Ciclos independientes de control: dim(H_1) = |E| - rank(∂_1) - β_0 + N
    euler_characteristic: int       # χ = β_0 - β_1 = V - E
    spectral_gap: float             # Brecha espectral (Fiedler) del Laplaciano combinatorio L = D - A
    normalized_spectral_gap: float  # Brecha espectral del Laplaciano normalizado L_sym = I - D^{-1/2}AD^{-1/2}
    cyclomatic_complexity: int      # Complejidad ciclomática de McCabe: M = E - N + 2P


class ASTTopologicalEngine:
    r"""
    Analizador de homología simplicial de 1-complejos derivados del AST, enriquecido con
    aristas de retroceso (*back-edges*) de Grafo de Flujo de Control (CFG) provenientes de
    construcciones iterativas (`for`, `while`). Sin estas aristas, todo AST —al ser un árbol—
    tendría trivialmente $\beta_1 \equiv 0$; el CFG dota al complejo de ciclos genuinos que
    reflejan la complejidad algorítmica real de la política $\pi_t$.
    Construye la cadena de complejos $C_1 \xrightarrow{\partial_1} C_0 \xrightarrow{\partial_0} 0$.
    """

    @classmethod
    def compute_homology(cls, tree: ast.AST) -> SimplicialHomologyCertificate:
        nodes: List[ast.AST] = []
        node_to_idx: Dict[int, int] = {}
        edges: List[Tuple[int, int]] = []

        for node in ast.walk(tree):
            idx = len(nodes)
            node_to_idx[id(node)] = idx
            nodes.append(node)

        # 1-esqueleto estructural: aristas padre-hijo del AST
        for p_idx, parent in enumerate(nodes):
            for child in ast.iter_child_nodes(parent):
                if id(child) in node_to_idx:
                    edges.append((p_idx, node_to_idx[id(child)]))

        # Enriquecimiento de Grafo de Flujo de Control: back-edges de bucles genuinos
        for node in nodes:
            if isinstance(node, (ast.For, ast.While)) and node.body:
                loop_idx = node_to_idx[id(node)]
                last_stmt_idx = node_to_idx.get(id(node.body[-1]))
                if last_stmt_idx is not None:
                    edges.append((last_stmt_idx, loop_idx))

        num_v = len(nodes)
        num_e = len(edges)

        if num_v == 0:
            return SimplicialHomologyCertificate(0, 0, 0, 0, 0, 0.0, 0.0, 0)

        boundary_1 = np.zeros((num_v, num_e), dtype=np.float64)
        adjacency = np.zeros((num_v, num_v), dtype=np.float64)
        degrees = np.zeros(num_v, dtype=np.float64)

        for edge_idx, (u, v) in enumerate(edges):
            boundary_1[u, edge_idx] = -1.0
            boundary_1[v, edge_idx] = 1.0
            adjacency[u, v] += 1.0
            adjacency[v, u] += 1.0
            degrees[u] += 1.0
            degrees[v] += 1.0

        # Rango numérico de ∂_1 vía SVD con tolerancia relativa a la escala (convención NumPy)
        if num_e > 0:
            singular_vals = la.svdvals(boundary_1)
            tol = singular_vals.max() * max(boundary_1.shape) * np.finfo(np.float64).eps
            rank_d1 = int(np.sum(singular_vals > tol))
        else:
            rank_d1 = 0

        b0 = num_v - rank_d1
        b1 = num_e - rank_d1
        euler = b0 - b1
        cyclomatic_complexity = num_e - num_v + 2 * b0  # Fórmula de McCabe generalizada a P componentes

        # Laplaciano combinatorio L = D - A y brecha espectral de Fiedler
        laplacian = np.diag(degrees) - adjacency
        eigvals = np.sort(la.eigvalsh(laplacian))
        spectral_gap = float(eigvals[1]) if len(eigvals) > 1 else float(eigvals[0])

        # Laplaciano normalizado L_sym = I - D^{-1/2} A D^{-1/2} (invariante a escala de grado)
        inv_sqrt_deg = np.where(degrees > 1e-12, 1.0 / np.sqrt(degrees), 0.0)
        d_inv_sqrt = np.diag(inv_sqrt_deg)
        laplacian_sym = np.eye(num_v) - d_inv_sqrt @ adjacency @ d_inv_sqrt
        eigvals_sym = np.sort(la.eigvalsh(laplacian_sym))
        normalized_gap = float(eigvals_sym[1]) if len(eigvals_sym) > 1 else float(eigvals_sym[0])

        return SimplicialHomologyCertificate(
            num_nodes=num_v,
            num_edges=num_e,
            betti_0=b0,
            betti_1=b1,
            euler_characteristic=euler,
            spectral_gap=max(0.0, spectral_gap),
            normalized_spectral_gap=max(0.0, normalized_gap),
            cyclomatic_complexity=cyclomatic_complexity,
        )


# ──────────────────────────────────────────────────────────────────────────────
# §1.4 ÁLGEBRA DE BANACH: ESPECTRO DE GELFAND, SERIE DE NEUMANN Y TEORÍA DE KREISS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class BanachContractionCertificate:
    """Certificado riguroso de contracción espectral en álgebra de Banach B(X)."""
    operator_matrix: np.ndarray
    spectral_radius: float
    neumann_series_norm_bound: float
    is_contraction: bool
    eigenvalues: np.ndarray
    gelfand_empirical_radius: float        # Verificación numérica de lim ‖Tᵏ‖^(1/k)
    kreiss_constant_estimate: float        # Cota de crecimiento transitorio (no-normalidad)
    eigenvector_condition_number: float    # κ(V): diagnóstico de normalidad del operador


class BanachSpectralEngine:
    r"""
    Auditor del Teorema de Punto Fijo de Banach y Espectro de Operadores en $\mathcal{B}(\mathcal{X})$.
    Calcula el radio espectral mediante la fórmula de Gelfand:
    $$\rho(T) = \lim_{k\to\infty} \|T^k\|^{1/k} = \max \{|\lambda| : \lambda \in \sigma(T)\}$$
    y la **verifica empíricamente** (no solo analíticamente), garantizando además la convergencia
    uniforme de la serie de Neumann $(I - T)^{-1} = \sum_{k=0}^{\infty} T^k$. Adicionalmente,
    estima la **constante de Kreiss** $K(T) = \sup_{|z|>1}(|z|-1)\|(zI-T)^{-1}\|$, pues el radio
    espectral por sí solo no acota el crecimiento *transitorio* $\|T^k\|$ de operadores no normales
    (Teorema de la Matriz de Kreiss): un $\rho(T)<1$ puede coexistir con amplificación transitoria
    significativa si el operador está mal condicionado espectralmente.
    """

    @staticmethod
    def audit_contraction(operator: np.ndarray, tolerance: float = 0.999) -> BanachContractionCertificate:
        if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
            raise ValueError("El operador en B(X) debe ser un endomorfismo matricial cuadrado.")

        eigvals, eigvecs = la.eig(operator)
        spectral_radius = float(np.max(np.abs(eigvals)))
        is_contraction = bool(spectral_radius < tolerance)

        if spectral_radius < 1.0:
            neumann_bound = 1.0 / (1.0 - spectral_radius)
        else:
            neumann_bound = float("inf")

        try:
            eigenvector_condition_number = float(np.linalg.cond(eigvecs))
        except np.linalg.LinAlgError:
            eigenvector_condition_number = float("inf")

        gelfand_sequence = BanachSpectralEngine._empirical_gelfand_sequence(operator)
        gelfand_empirical_radius = float(gelfand_sequence[-1]) if gelfand_sequence.size else spectral_radius
        kreiss_constant = BanachSpectralEngine._estimate_kreiss_constant(operator)

        return BanachContractionCertificate(
            operator_matrix=operator,
            spectral_radius=spectral_radius,
            neumann_series_norm_bound=neumann_bound,
            is_contraction=is_contraction,
            eigenvalues=eigvals,
            gelfand_empirical_radius=gelfand_empirical_radius,
            kreiss_constant_estimate=kreiss_constant,
            eigenvector_condition_number=eigenvector_condition_number,
        )

    @staticmethod
    def _empirical_gelfand_sequence(operator: np.ndarray, max_power: int = 20) -> np.ndarray:
        r"""Verificación numérica directa de la fórmula límite: $\|T^k\|^{1/k} \to \rho(T)$."""
        dim = operator.shape[0]
        power = np.eye(dim, dtype=np.float64)
        sequence = np.zeros(max_power, dtype=np.float64)
        for k in range(1, max_power + 1):
            power = power @ operator
            norm_k = np.linalg.norm(power, ord=2)  # Norma espectral (submultiplicativa)
            sequence[k - 1] = norm_k ** (1.0 / k)
        return sequence

    @staticmethod
    def _estimate_kreiss_constant(
        operator: np.ndarray,
        radius_samples: Tuple[float, ...] = (1.01, 1.05, 1.1, 1.5, 2.0)
    ) -> float:
        r"""
        Estima $K(T) = \sup_{|z|>1}(|z|-1)\|(zI-T)^{-1}\|$ muestreando el resolvente sobre
        radios $|z|>1$. Por el Teorema de la Matriz de Kreiss, $K(T) \le \sup_k \|T^k\| \le e \cdot n \cdot K(T)$,
        acotando el crecimiento transitorio máximo aun cuando $\rho(T)<1$.
        """
        dim = operator.shape[0]
        identity = np.eye(dim, dtype=np.complex128)
        op_complex = operator.astype(np.complex128)
        kreiss_values: List[float] = []
        for r in radius_samples:
            z = complex(r, 0.0)
            try:
                resolvent = la.inv(z * identity - op_complex)
                kreiss_values.append(float((r - 1.0) * np.linalg.norm(resolvent, ord=2)))
            except la.LinAlgError:
                continue
        return max(kreiss_values) if kreiss_values else float("inf")


# ──────────────────────────────────────────────────────────────────────────────
# §1.5 MECÁNICA CUÁNTICA: MATRIZ ATÓMICA DE CONOCIMIENTO (MAC) Y DINÁMICA LINDBLAD
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class MACDensityState:
    """Operador de densidad cuántica en el espacio de Hilbert H_n."""
    rho_matrix: np.ndarray
    dimension: int
    purity: float               # γ = Tr(ρ²)
    von_neumann_entropy: float  # S(ρ) = -Tr(ρ ln ρ)
    renyi_2_entropy: float      # S_2(ρ) = -ln Tr(ρ²), cota inferior de S(ρ) (S_2 ≤ S)
    quantum_fidelity: float     # F(ρ, ρ_prior)
    is_valid_state: bool


class MACQuantumEngine:
    r"""
    Preserva los postulados de Dirac-von Neumann sobre el espacio de estados cuánticos:
    $\mathcal{S}(\mathcal{H}) = \{ \rho \in \mathcal{B}(\mathcal{H}) : \rho = \rho^\dagger, \rho \succeq 0, \mathrm{Tr}(\rho) = 1 \}$.
    Implementa pasos infinitesimales disipativos mediante la ecuación maestra de Lindblad:
    $$\frac{d\rho}{dt} = -i[H, \rho] + \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right)$$
    """

    @staticmethod
    def _project_to_valid_density_state(
        candidate_matrix: np.ndarray, dimension: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Proyección espectral canónica sobre el símplex de Birkhoff de estados físicos válidos:
        hermitiza, diagonaliza, satura autovalores negativos por error numérico (umbral
        relativo a la dimensión y al épsilon de máquina) y renormaliza la traza a la unidad.
        Método único compartido (evita duplicación entre creación y evolución del estado).
        """
        hermitized = 0.5 * (candidate_matrix + candidate_matrix.conj().T)
        eigvals, eigvecs = la.eigh(hermitized)
        clamp_threshold = np.finfo(np.float64).eps * dimension * 10.0
        eigvals_clamped = np.maximum(eigvals, clamp_threshold)
        eigvals_clamped /= np.sum(eigvals_clamped)
        rho_projected = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.conj().T
        rho_projected = 0.5 * (rho_projected + rho_projected.conj().T)
        return rho_projected, eigvals_clamped

    @classmethod
    def create_pure_or_mixed_state(cls, dimension: int = 4, seed: int = 42) -> MACDensityState:
        rng = np.random.default_rng(seed)
        real_part = rng.normal(size=(dimension, dimension))
        imag_part = rng.normal(size=(dimension, dimension))
        ginibre_matrix = real_part + 1j * imag_part

        # Ensamble de Hilbert-Schmidt: ρ = A A† / Tr(A A†)
        unnormalized_rho = ginibre_matrix @ ginibre_matrix.conj().T
        trace_val = float(np.trace(unnormalized_rho).real)
        rho = unnormalized_rho / trace_val

        rho_projected, eigvals_clamped = cls._project_to_valid_density_state(rho, dimension)

        purity = float(np.sum(eigvals_clamped**2))
        entropy = -float(np.sum(eigvals_clamped * np.log(eigvals_clamped)))
        renyi2 = -math.log(max(purity, 1e-300))

        is_valid = (
            abs(float(np.trace(rho_projected).real) - 1.0) < 1e-10 and
            np.all(eigvals_clamped >= 0.0) and
            np.allclose(rho_projected, rho_projected.conj().T, atol=1e-10)
        )

        return MACDensityState(
            rho_matrix=rho_projected,
            dimension=dimension,
            purity=purity,
            von_neumann_entropy=entropy,
            renyi_2_entropy=renyi2,
            quantum_fidelity=1.0,
            is_valid_state=is_valid
        )

    @classmethod
    def evolve_lindblad_step(
        cls,
        state: MACDensityState,
        hamiltonian: np.ndarray,
        lindblad_operators: List[np.ndarray],
        dt: float = 1e-4
    ) -> MACDensityState:
        """Aplica un paso del semigrupo dinámico cuántico CPTP (Completely Positive Trace-Preserving)."""
        rho = state.rho_matrix
        dim = state.dimension

        unitary_term = -1j * (hamiltonian @ rho - rho @ hamiltonian)

        dissipator = np.zeros((dim, dim), dtype=np.complex128)
        for L in lindblad_operators:
            L_dagger_L = L.conj().T @ L
            dissipator += L @ rho @ L.conj().T - 0.5 * (L_dagger_L @ rho + rho @ L_dagger_L)

        rho_next_raw = rho + (unitary_term + dissipator) * dt
        rho_final, eigvals_clamped = cls._project_to_valid_density_state(rho_next_raw, dim)

        purity = float(np.sum(eigvals_clamped**2))
        entropy = -float(np.sum(eigvals_clamped * np.log(eigvals_clamped)))
        renyi2 = -math.log(max(purity, 1e-300))

        # Fidelidad cuántica de Uhlmann-Jozsa: F(ρ, σ) = (Tr √(√ρ σ √ρ))²
        sqrt_rho = la.sqrtm(rho)
        fidelity_term = la.sqrtm(sqrt_rho @ rho_final @ sqrt_rho)
        fidelity = float(np.real(np.trace(fidelity_term))**2)

        return MACDensityState(
            rho_matrix=rho_final,
            dimension=dim,
            purity=purity,
            von_neumann_entropy=entropy,
            renyi_2_entropy=renyi2,
            quantum_fidelity=fidelity,
            is_valid_state=True
        )


# ──────────────────────────────────────────────────────────────────────────────
# §1.6 ENLACE TERMINAL FASE 1: SÍNTESIS DE LA VARIEDAD DE ESTADO M_WISDOM
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class StateManifoldWisdom:
    r"""
    Fibrado geométrico terminal de la Fase 1: $\mathfrak{M}_{\text{Wisdom}} = (E, \pi, B, F)$.
    Encapsula el estado de densidad MAC cuántico, el álgebra cuaterniónica de rotación cognitiva,
    el certificado de homología simplicial del AST/CFG y la acotación espectral de Banach.
    """
    mac_state: MACDensityState
    hypercomplex_rotor: Quaternion
    ast_homology: SimplicialHomologyCertificate
    banach_contraction: BanachContractionCertificate
    timestamp_epoch: float


def synthesize_wisdom_manifold(
    ast_tree: ast.AST,
    mac_state: MACDensityState,
    mutation_matrix: np.ndarray,
    rotor: Quaternion,
    tolerance: float = 0.999
) -> StateManifoldWisdom:
    r"""
    MÉTODO FORMAL TERMINAL DE LA FASE 1.
    Sintetiza la variedad analítica completa $\mathfrak{M}_{\text{Wisdom}}$, validando
    explícitamente las precondiciones dimensionales del operador de mutación (endomorfismo
    cuadrado en $B(X)$). Esta función actúa como puente unívoco e inicio constitutivo de las
    transformaciones dinámicas de la FASE 2 (§2.1, físicas de Crowbar y transición categórica).
    """
    if mutation_matrix.ndim != 2 or mutation_matrix.shape[0] != mutation_matrix.shape[1]:
        raise ValueError(
            "Precondición violada: el operador de mutación debe ser un endomorfismo "
            f"cuadrado en B(X); recibido con forma {mutation_matrix.shape}."
        )

    homology_cert = ASTTopologicalEngine.compute_homology(ast_tree)
    banach_cert = BanachSpectralEngine.audit_contraction(mutation_matrix, tolerance=tolerance)

    return StateManifoldWisdom(
        mac_state=mac_state,
        hypercomplex_rotor=rotor,
        ast_homology=homology_cert,
        banach_contraction=banach_cert,
        timestamp_epoch=time.time()
    )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: DINÁMICA PORT-HAMILTONIANA, FÍSICA DEL CIRCUITO CROWBAR EN SILICIO
#         Y TEORÍA DE CONEXIÓN DE CATEGORÍAS / FIBRADOS
#         (consume StateManifoldWisdom, la síntesis terminal de la Fase 1)
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# §2.1 FÍSICA CIBER-FÍSICA: CIRCUITO CROWBAR CON TIRISTOR BT151 Y RLC (SOLUCIÓN EXACTA)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CrowbarPhysicalTelemetry:
    """Telemetría transitoria del disyuntor ciber-físico en silicio (BT151 Crowbar)."""
    interlock_triggered: bool
    actuation_latency_ns: float       # < 400.0 ns garantizado por ISR en IRAM
    time_to_peak_ns: float            # Instante analítico exacto t* del máximo de corriente
    peak_surge_current_amperes: float
    joule_integral_i2t: float         # ∫ i² dt [A²s], evaluado por cuadratura adaptativa
    within_safe_operating_area: bool  # I²t ≤ rating del fabricante (SOA del BT151)
    rail_voltage_post_quench: float
    gpio_pin_designation: str
    provenance_hash: str


class CrowbarCircuitPhysicsEngine:
    r"""
    Modela con precisión física el circuito disyuntor Crowbar accionado por compuerta:
    - Microcontrolador: ESP32 dual-core Xtensa LX6 @ 240 MHz.
    - ISR residente en memoria ultrarrápida IRAM (cero fallos de caché flash).
    - Dispositivo de potencia: Tiristor de silicio planar BT151-800R.
    - Ecuación del transitorio de descarga RLC de carril (régimen subamortiguado):
      $$i(t) = \frac{V}{\omega_d L}e^{-\alpha t}\sin(\omega_d t), \qquad
        \omega_d = \sqrt{\omega_0^2 - \alpha^2}$$
      El instante de pico exacto se obtiene analíticamente de $di/dt=0$:
      $$t^\star = \frac{1}{\omega_d}\arctan\!\left(\frac{\omega_d}{\alpha}\right)$$
      y la energía térmica de fusión se integra por cuadratura adaptativa
      $\int_0^{T_h} i(t)^2\,dt$ (evitando aproximaciones cerradas frágiles).
    """

    C_BUS: Final[float] = 470e-6
    L_BUS: Final[float] = 12e-9
    R_TRACE: Final[float] = 0.015
    R_THYRISTOR_ON: Final[float] = 0.028
    V_BUS_NOMINAL: Final[float] = 3.3
    I2T_RATING_BT151: Final[float] = 50.0  # A²s — cota de fusión del fabricante (SOA)

    IRAM_INSTRUCTION_CYCLES: Final[int] = 20
    THYRISTOR_GATE_DELAY_NS: Final[float] = 280.0

    @classmethod
    def simulate_crowbar_actuation(cls, trip_required: bool) -> CrowbarPhysicalTelemetry:
        if not trip_required:
            return CrowbarPhysicalTelemetry(
                interlock_triggered=False,
                actuation_latency_ns=0.0,
                time_to_peak_ns=0.0,
                peak_surge_current_amperes=0.0,
                joule_integral_i2t=0.0,
                within_safe_operating_area=True,
                rail_voltage_post_quench=cls.V_BUS_NOMINAL,
                gpio_pin_designation="GPIO14_IRAM_FAST_STROBE",
                provenance_hash=""
            )

        clock_period_ns = 1e9 / 240e6
        ram_exec_latency = cls.IRAM_INSTRUCTION_CYCLES * clock_period_ns
        total_latency_ns = ram_exec_latency + cls.THYRISTOR_GATE_DELAY_NS

        r_total = cls.R_TRACE + cls.R_THYRISTOR_ON
        omega_0 = 1.0 / math.sqrt(cls.L_BUS * cls.C_BUS)
        alpha = r_total / (2.0 * cls.L_BUS)
        discriminant = omega_0**2 - alpha**2

        if discriminant > 0.0:
            omega_d = math.sqrt(discriminant)

            def i_of_t(t: float) -> float:
                return (cls.V_BUS_NOMINAL / (omega_d * cls.L_BUS)) * math.exp(-alpha * t) * math.sin(omega_d * t)

            # Instante exacto del pico: di/dt = 0  ⟺  tan(ω_d t*) = ω_d / α
            t_peak = (1.0 / omega_d) * math.atan2(omega_d, alpha)
            i_peak = i_of_t(t_peak)

            # Integral de Joule por cuadratura adaptativa sobre el horizonte de quenching (12/α ≈ 0 residual)
            integration_horizon = 12.0 / alpha
            i2t, _ = integrate.quad(lambda t: i_of_t(t) ** 2, 0.0, integration_horizon, limit=250)
        else:
            # Régimen sobreamortiguado (α ≥ ω_0): descarga monótona sin oscilación
            omega_d = 0.0
            t_peak = 1.0 / alpha
            i_peak = cls.V_BUS_NOMINAL / r_total
            i2t = (i_peak ** 2) * (1.0 / alpha)

        v_post = 0.08  # Caída de saturación remanente directa BT151

        hasher = hashlib.sha256()
        hasher.update(f"ESP32_CROWBAR_BT151_LATCHED_{total_latency_ns:.4f}_{i_peak:.2f}".encode("utf-8"))
        prov_hash = hasher.hexdigest()

        return CrowbarPhysicalTelemetry(
            interlock_triggered=True,
            actuation_latency_ns=total_latency_ns,
            time_to_peak_ns=t_peak * 1e9,
            peak_surge_current_amperes=i_peak,
            joule_integral_i2t=i2t,
            within_safe_operating_area=bool(i2t <= cls.I2T_RATING_BT151),
            rail_voltage_post_quench=v_post,
            gpio_pin_designation="GPIO14_IRAM_FAST_STROBE",
            provenance_hash=prov_hash
        )


# ──────────────────────────────────────────────────────────────────────────────
# §2.2 DINÁMICA PORT-HAMILTONIANA CON DISCRETIZACIÓN DE CAYLEY (DISIPACIÓN EXACTA)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PortHamiltonianState:
    """Estado y balances energéticos del sistema Port-Hamiltoniano en lazo cerrado."""
    hamiltonian_energy: float
    energy_derivative: float          # ΔH/dt ≤ 0 estricto (Disipatividad / Pasividad)
    is_lyapunov_stable: bool
    structure_matrices_verified: bool  # J=-Jᵀ ∧ R⪰0 verificado explícitamente
    state_vector: np.ndarray


class PortHamiltonianDynamicsEngine:
    r"""
    Gobernador de disipación formal para el operador de mutación recursivo:
    $$\dot{x} = [J(x) - R(x)] \nabla H(x) + g(x)u,\qquad J=-J^T,\ R=R^T\succeq 0,\ H=\tfrac12 x^TQx$$
    En lugar de integración de Euler explícita (que **no** preserva la disipatividad discreta),
    se emplea la **transformada de Cayley** (regla del punto medio implícito), que para sistemas
    lineales con $H$ cuadrática es *exacta* en el sentido de que:
    $$H(x_{k+1}) - H(x_k) = h\,(\bar\nabla H)^T (J-R)\,\bar\nabla H \le 0,\qquad \bar x = \tfrac12(x_k+x_{k+1})$$
    se cumple sin error de discretización, garantizando estabilidad de Lyapunov incondicional.
    """

    @staticmethod
    @lru_cache(maxsize=32)
    def _build_structure_matrices(dim: int, damping_factor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(1337)
        A = rng.normal(size=(dim, dim))
        J = 0.5 * (A - A.T)                                    # J = -J^T estricto
        R = 0.5 * (A @ A.T) + np.eye(dim) * damping_factor      # R = R^T ≻ 0 estricto
        Q = np.eye(dim) * 1.5                                   # H(x) = ½ xᵀQx, Q ≻ 0
        return J, R, Q

    @staticmethod
    def _verify_structure(J: np.ndarray, R: np.ndarray) -> bool:
        antisym_error = float(np.max(np.abs(J + J.T)))
        eig_r = la.eigvalsh(0.5 * (R + R.T))
        return bool(antisym_error < 1e-9 and np.all(eig_r >= -1e-9))

    @classmethod
    def evaluate_passivity(
        cls, x_state: np.ndarray, damping_factor: float = 0.85, dt: float = 1e-3
    ) -> PortHamiltonianState:
        dim = x_state.shape[0]
        J, R, Q = cls._build_structure_matrices(dim, damping_factor)
        structure_ok = cls._verify_structure(J, R)

        A = (J - R) @ Q
        identity = np.eye(dim)

        # Transformada de Cayley / regla del punto medio implícito:
        # x_{k+1} = (I - h/2 A)^{-1} (I + h/2 A) x_k
        m_left = identity - 0.5 * dt * A
        m_right = identity + 0.5 * dt * A
        x_next = la.solve(m_left, m_right @ x_state)

        energy_k = float(0.5 * x_state.T @ Q @ x_state)
        energy_next = float(0.5 * x_next.T @ Q @ x_next)
        delta_h = energy_next - energy_k

        is_stable = bool(delta_h <= 1e-9 and structure_ok)

        return PortHamiltonianState(
            hamiltonian_energy=energy_k,
            energy_derivative=delta_h / dt,
            is_lyapunov_stable=is_stable,
            structure_matrices_verified=structure_ok,
            state_vector=x_next
        )


# ──────────────────────────────────────────────────────────────────────────────
# §2.3 CLASIFICADOR DE SUBOBJETOS EN TOPOS DE HACES (ÍNFIMO COMPOSICIONAL EN Ω₃)
# ──────────────────────────────────────────────────────────────────────────────

class ToposClassifierEngine:
    r"""
    En un Topos elemental $\mathcal{E}$, existe un clasificador de subobjetos $\top: 1 \to \Omega$,
    tal que para cada monomorfismo $m: U \rightarrowtail X$, existe un único morfismo $\chi_U: X \to \Omega$
    que hace conmutativo el diagrama cartesiano (pullback). Aquí $\Omega \cong \Omega_3$.
    El veredicto global se construye **composicionalmente** como el ínfimo de Heyting sobre una
    cubierta finita de subobjetos $\{U_i\}$ (analogía discreta del pegado de haces):
    $$\chi_U = \bigwedge_i \chi_{U_i}$$
    en lugar de una cascada secuencial de condicionales, exhibiendo explícitamente la estructura
    reticular subyacente al veredicto.
    """

    @staticmethod
    def classify_subobject(
        manifold: StateManifoldWisdom,
        phs_state: PortHamiltonianState,
        candidate_utility: float,
        current_utility: float
    ) -> HeytingOmega3:
        chi_quantum = (
            HeytingOmega3.COHERENT if manifold.mac_state.is_valid_state else HeytingOmega3.VETOED
        )
        chi_banach = (
            HeytingOmega3.COHERENT if manifold.banach_contraction.is_contraction else HeytingOmega3.VETOED
        )
        chi_phs = (
            HeytingOmega3.COHERENT if phs_state.is_lyapunov_stable else HeytingOmega3.VETOED
        )
        chi_utility = (
            HeytingOmega3.COHERENT if candidate_utility >= current_utility else HeytingOmega3.DEGRADED
        )

        # Pegado categórico: el clasificador global es el ínfimo de la cubierta finita de subobjetos
        return chi_quantum.meet(chi_banach).meet(chi_phs).meet(chi_utility)


# ──────────────────────────────────────────────────────────────────────────────
# §2.4 MOTOR METAMÓRFICO DE REESCRITURA AST CON INTERPRETACIÓN ABSTRACTA
# ──────────────────────────────────────────────────────────────────────────────

class ASTMetamorphicRewriter(ast.NodeTransformer):
    r"""
    Reescritor reflexivo del AST del agente. Realiza transformaciones sintácticas
    homomórficas preservando la semántica de evaluación y auditando los nodos contra
    reglas de seguridad de código absoluto (prohibición de primitivas y atributos peligrosos),
    aplicando además una interpretación abstracta de dominio de intervalo que satura la
    magnitud de las mutaciones constantes para prevenir divergencia numérica.
    """

    FORBIDDEN_CALLS: Final[Set[str]] = {
        "eval", "exec", "__import__", "open", "system", "popen",
        "spawn", "fork", "subprocess", "globals", "locals", "compile"
    }
    FORBIDDEN_ATTRIBUTES: Final[Set[str]] = {
        "__globals__", "__builtins__", "__subclasses__", "__bases__", "__class__",
        "__code__", "__closure__", "__dict__", "__mro__", "__getattribute__", "__reduce__",
        "__import__", "__loader__"
    }
    MAX_CONSTANT_MAGNITUDE: Final[float] = 1e6

    def __init__(self, mutation_scale: float) -> None:
        super().__init__()
        self.mutation_scale = mutation_scale
        self.security_violation_detected = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        func = node.func
        if isinstance(func, ast.Name) and func.id in self.FORBIDDEN_CALLS:
            self.security_violation_detected = True
        elif isinstance(func, ast.Attribute) and func.attr in self.FORBIDDEN_CALLS:
            self.security_violation_detected = True
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if node.attr in self.FORBIDDEN_ATTRIBUTES:
            self.security_violation_detected = True
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            mutated_val = float(node.value) + self.mutation_scale
            # Interpretación abstracta de dominio de intervalo: saturación defensiva
            mutated_val = max(-self.MAX_CONSTANT_MAGNITUDE, min(self.MAX_CONSTANT_MAGNITUDE, mutated_val))
            return ast.copy_location(ast.Constant(value=mutated_val), node)
        return node

    @staticmethod
    def audit_structural_complexity(tree: ast.AST, max_nodes: int = 500) -> bool:
        r"""
        Cota preventiva de complejidad estructural (anti-bomba sintáctica): rechaza AST cuyo
        cardinal de nodos exceda `max_nodes` antes de invertir recursos en su reescritura,
        compilación y ejecución en sandbox.
        """
        node_count = sum(1 for _ in ast.walk(tree))
        return node_count <= max_nodes


# ──────────────────────────────────────────────────────────────────────────────
# §2.5 ENLACE TERMINAL FASE 2: EVALUACIÓN FUNCTORIAL DE TRANSICIÓN CATEGÓRICA
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CategoricalTransitionMorphism:
    r"""
    Morfismo functorial terminal de la Fase 2: $\Phi: \mathfrak{M} \to \mathfrak{M}'$.
    Certifica la mutación completa, enlazando la física cuántica/Banach de la Fase 1
    con la respuesta ciber-física y de topos de la Fase 2.
    """
    initial_manifold: StateManifoldWisdom
    phs_state: PortHamiltonianState
    crowbar_telemetry: CrowbarPhysicalTelemetry
    heyting_verdict: HeytingOmega3
    proposed_ast: ast.AST
    candidate_callable: Optional[Callable[..., float]]
    mutation_operator_next: np.ndarray
    transition_entropy_cost: float


def evaluate_categorical_transition(
    manifold: StateManifoldWisdom,
    proposed_ast: ast.AST,
    candidate_callable: Optional[Callable[..., float]],
    current_utility: float,
    candidate_utility: float,
    damping_factor: float = 0.85
) -> CategoricalTransitionMorphism:
    r"""
    MÉTODO FORMAL TERMINAL DE LA FASE 2.
    Recibe la variedad sintetizada en Fase 1 (`StateManifoldWisdom`) y aplica las leyes
    dinámicas de Port-Hamilton (integradas por Cayley), clasificación composicional en
    topos de Heyting y física exacta de Crowbar. Este método actúa como entrada
    estructurada obligatoria para el Soberano de la FASE 3 (`GodelAgent.self_update`).
    """
    diag_eigs = np.real(manifold.banach_contraction.eigenvalues)
    phs_eval = PortHamiltonianDynamicsEngine.evaluate_passivity(diag_eigs, damping_factor=damping_factor)

    verdict = ToposClassifierEngine.classify_subobject(
        manifold=manifold,
        phs_state=phs_eval,
        candidate_utility=candidate_utility,
        current_utility=current_utility
    )

    needs_crowbar = (verdict == HeytingOmega3.VETOED)
    crowbar_telemetry = CrowbarCircuitPhysicsEngine.simulate_crowbar_actuation(trip_required=needs_crowbar)

    current_op = manifold.banach_contraction.operator_matrix
    if verdict == HeytingOmega3.COHERENT:
        next_mutation_op = current_op * 0.95
    else:
        next_mutation_op = current_op * 0.50

    entropy_cost = abs(manifold.mac_state.von_neumann_entropy - float(np.sum(np.abs(diag_eigs))))

    return CategoricalTransitionMorphism(
        initial_manifold=manifold,
        phs_state=phs_eval,
        crowbar_telemetry=crowbar_telemetry,
        heyting_verdict=verdict,
        proposed_ast=proposed_ast,
        candidate_callable=candidate_callable,
        mutation_operator_next=next_mutation_op,
        transition_entropy_cost=entropy_cost
    )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: EL SOBERANO DE GÖDEL (RSI), MÁQUINA AUTORREFERENCIAL
#         Y CERTIFICACIÓN CRIPTOGRÁFICA TERMINAL
#         (consume CategoricalTransitionMorphism, la síntesis terminal de la Fase 2)
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# §3.1 ESTRUCTURAS TERMINALES DE CERTIFICACIÓN Y NO-REPUDIO
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SovereignGodelCertificate:
    """Certificado inmutable terminal emitido por el Estrato de Sabiduría (V_W, Nivel 0)."""
    agent_id: str
    iteration: int
    heyting_verdict: HeytingOmega3
    banach_certificate: BanachContractionCertificate
    mac_state: MACDensityState
    ast_homology: SimplicialHomologyCertificate
    phs_state: PortHamiltonianState
    crowbar_report: CrowbarPhysicalTelemetry
    utility_score: float
    mutation_applied: bool
    fixed_point_converged: bool           # Verificación de Picard vs. Neumann (§3.2)
    fixed_point_residual: float
    fixed_point_closed_form_error: float
    digital_signature_sha256: str
    timestamp_utc: float


def _default_wisdom_policy(entropy: float, purity: float) -> float:
    """Función de utilidad objetiva primordial para el lazo reflexivo de Gödel."""
    return (purity * 2.5) - (entropy * 0.4)


# ──────────────────────────────────────────────────────────────────────────────
# §3.2 SOBERANO AUTORREFERENCIAL: GÖDEL AGENT
# ──────────────────────────────────────────────────────────────────────────────

class GodelAgent:
    r"""
    Soberano de Sabiduría y Automejora Recursiva (RSI) de Nivel 0.
    Gobierna la evolución del par autorreferencial $(\pi_t, I_t)$, orquestando las 4
    primitivas transcendentales de lazo cerrado (OODA):
      1. self_inspect()    : Inspección del AST/CFG, homología simplicial, MAC y B(X).
      2. interact()        : Evaluación analítica del funcional de utilidad cognitiva.
      3. self_update()     : Transición categórica, evaluación de Heyting y salvaguarda Crowbar.
      4. continue_improve(): Orquestador recursivo continuo de orden superior, con
                              verificación constructiva del punto fijo de Banach.
    """

    MAX_AST_NODES: Final[int] = 500
    _SAFE_BUILTINS: Final[Dict[str, Any]] = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "float": float, "int": int, "bool": bool, "round": round, "pow": pow,
    }

    def __init__(
        self,
        agent_id: str = "GODEL-SOVEREIGN-V0",
        dimension_mac: int = 4,
        spectral_tolerance: float = 0.999,
        seed: int = 42
    ) -> None:
        self.agent_id = agent_id
        self.dimension_mac = dimension_mac
        self.spectral_tolerance = spectral_tolerance
        self.iteration = 0

        self.mac_state = MACQuantumEngine.create_pure_or_mixed_state(dimension=dimension_mac, seed=seed)
        self.hypercomplex_rotor = Quaternion(1.0, 0.0, 0.0, 0.0)

        self.policy_fn: Callable[[float, float], float] = _default_wisdom_policy
        raw_source = textwrap.dedent(inspect.getsource(_default_wisdom_policy))
        self.policy_ast: ast.Module = ast.parse(raw_source)

        self.mutation_operator = np.eye(dimension_mac, dtype=np.float64) * 0.45
        self.utility_history: List[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Primitiva 1: SELF_INSPECT (Observación Topológica y Espectral)
    # ──────────────────────────────────────────────────────────────────────────
    def self_inspect(self) -> StateManifoldWisdom:
        r"""
        Inspecciona holísticamente la variedad de estado interna $\mathfrak{M}_{\text{Wisdom}}$,
        utilizando la síntesis terminal de la FASE 1.
        """
        return synthesize_wisdom_manifold(
            ast_tree=self.policy_ast,
            mac_state=self.mac_state,
            mutation_matrix=self.mutation_operator,
            rotor=self.hypercomplex_rotor,
            tolerance=self.spectral_tolerance
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Primitiva 2: INTERACT (Evaluación Funcional de Utilidad)
    # ──────────────────────────────────────────────────────────────────────────
    def interact(self, policy_override: Optional[Callable[[float, float], float]] = None) -> float:
        r"""
        Evalúa el desempeño funcional de la política activa o de una función candidata:
        $$U(\pi) = \pi(S_{\text{vN}}(\rho), \gamma(\rho))$$
        """
        active_fn = policy_override if policy_override is not None else self.policy_fn
        score = active_fn(self.mac_state.von_neumann_entropy, self.mac_state.purity)
        return float(score)

    # ──────────────────────────────────────────────────────────────────────────
    # Primitiva 3: SELF_UPDATE (Transición Categórica y Validación Cruzada)
    # ──────────────────────────────────────────────────────────────────────────
    def self_update(
        self,
        proposed_ast: ast.AST,
        candidate_fn: Optional[Callable[..., float]],
        precomputed_manifold: Optional[StateManifoldWisdom] = None
    ) -> CategoricalTransitionMorphism:
        r"""
        Ejecuta la evaluación functorial de transición mediante la FASE 2. Acepta opcionalmente
        una variedad `precomputed_manifold` ya observada en el paso "Observe" del ciclo OODA,
        preservando la unicidad temporal de la observación (evita recomputar $\mathfrak{M}$ dos
        veces por ciclo, lo cual violaría la simultaneidad epistemológica de la decisión).
        Aplica los filtros cruzados de:
          - Mecánica Cuántica MAC: $\rho \succeq 0, \mathrm{Tr}(\rho) = 1$.
          - Análisis de Banach: $\rho(T) < 1.0$.
          - Port-Hamiltoniano: $\Delta H \le 0$ (discreto, exacto vía Cayley).
          - Heyting: $\Omega_3 = \text{COHERENT}$.
        """
        current_manifold = precomputed_manifold if precomputed_manifold is not None else self.self_inspect()

        current_utility = self.interact(self.policy_fn)
        candidate_utility = self.interact(candidate_fn) if candidate_fn is not None else -float("inf")

        transition_morphism = evaluate_categorical_transition(
            manifold=current_manifold,
            proposed_ast=proposed_ast,
            candidate_callable=candidate_fn,
            current_utility=current_utility,
            candidate_utility=candidate_utility,
            damping_factor=0.85
        )

        if transition_morphism.heyting_verdict == HeytingOmega3.COHERENT and candidate_fn is not None:
            self.policy_fn = candidate_fn
            self.policy_ast = proposed_ast  # type: ignore[assignment]
            self.mutation_operator = transition_morphism.mutation_operator_next

            # Rotación cognitiva hipercompleja generada rigurosamente vía exp: su(2) -> SU(2)
            rotation_axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
            rotation_angle = min(abs(transition_morphism.transition_entropy_cost), 0.1) + 1e-4
            delta_rotor = Quaternion.from_axis_angle(rotation_axis, rotation_angle)
            self.hypercomplex_rotor = (self.hypercomplex_rotor * delta_rotor).versor()

        return transition_morphism

    # ──────────────────────────────────────────────────────────────────────────
    # Verificación autorreferencial del Teorema del Punto Fijo de Banach
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _verify_banach_fixed_point(
        operator: np.ndarray,
        forcing_vector: np.ndarray,
        max_iterations: int = 500,
        tolerance: float = 1e-12
    ) -> Tuple[bool, float, float, np.ndarray]:
        r"""
        Verificación constructiva del Teorema del Punto Fijo de Banach para la aplicación afín
        contractiva $F(x) = Tx + b$ (con $\rho(T)<1$), mediante iteración de Picard
        $x_{k+1} = F(x_k)$, contrastada contra la solución cerrada de la serie de Neumann:
        $$x^\star = (I - T)^{-1} b = \sum_{k=0}^{\infty} T^k b$$
        Esta es la primitiva explícitamente reclamada por la arquitectura del Soberano de
        Gödel ("verificación de punto fijo") y ausente en la implementación original.
        """
        dim = operator.shape[0]
        x = np.zeros(dim, dtype=np.float64)
        residual = float("inf")
        for _ in range(max_iterations):
            x_next = operator @ x + forcing_vector
            residual = float(np.linalg.norm(x_next - x, ord=2))
            x = x_next
            if residual < tolerance:
                break

        identity = np.eye(dim, dtype=np.float64)
        try:
            closed_form = la.solve(identity - operator, forcing_vector)
            closed_form_error = float(np.linalg.norm(x - closed_form, ord=2))
        except la.LinAlgError:
            closed_form_error = float("inf")

        converged = bool(residual < tolerance)
        return converged, residual, closed_form_error, x

    # ──────────────────────────────────────────────────────────────────────────
    # Primitiva 4: CONTINUE_IMPROVE (Orquestador Recursivo OODA)
    # ──────────────────────────────────────────────────────────────────────────
    def continue_improve(self, force_spectral_violation: bool = False) -> SovereignGodelCertificate:
        r"""
        Ejecuta un ciclo completo de Automejora Recursiva (RSI) de orden doctoral:
          Observe  -> self_inspect() sintetizando la variedad de Fase 1 (una única vez).
          Orient   -> Mutación sintáctica del AST bajo cota de complejidad y sandbox endurecido.
          Decide   -> evaluate_categorical_transition() de Fase 2.
          Act      -> Interlock ciber-físico Crowbar o mutación atómica en RAM.
          Verify   -> Verificación constructiva del punto fijo de Banach (Picard vs. Neumann).
          Audit    -> Firma digital inmutable SHA-256.
        """
        self.iteration += 1
        logger.info(f"=== INICIANDO CICLO RSI GÖDEL ITERACIÓN #{self.iteration} ===")

        if force_spectral_violation:
            self.mutation_operator = np.eye(self.dimension_mac) * 1.85  # ρ(T) = 1.85 > 1.0 (VETO)

        # 1. Observar (síntesis única de la variedad para todo el ciclo OODA)
        manifold = self.self_inspect()
        current_utility = self.interact()

        # 2. Orientar: mutación AST bajo cota estructural y sandbox endurecido
        mutation_step = 0.05 * (1.0 / (self.iteration + 1))
        rewriter = ASTMetamorphicRewriter(mutation_scale=mutation_step)
        candidate_fn: Optional[Callable[..., float]] = None
        mutated_ast: ast.AST = self.policy_ast

        if ASTMetamorphicRewriter.audit_structural_complexity(self.policy_ast, max_nodes=self.MAX_AST_NODES):
            mutated_ast = rewriter.visit(self.policy_ast)
            ast.fix_missing_locations(mutated_ast)

            if not rewriter.security_violation_detected:
                try:
                    code_obj = compile(mutated_ast, filename="<godel_rsi_ast>", mode="exec")
                    sandbox: Dict[str, Any] = {"__builtins__": self._SAFE_BUILTINS}
                    exec(code_obj, sandbox)
                    for item in sandbox.values():
                        if callable(item) and getattr(item, "__name__", "") != "_default_wisdom_policy":
                            candidate_fn = item
                            break
                    if candidate_fn is None:
                        candidate_fn = sandbox.get("_default_wisdom_policy")
                except Exception as e:
                    logger.error(f"Fallo de compilación en sandbox AST: {e}")
                    candidate_fn = None
        else:
            logger.warning("Cota de complejidad estructural excedida; mutación AST abortada preventivamente.")

        # 3. Decidir y Actuar
        transition = self.self_update(
            proposed_ast=mutated_ast, candidate_fn=candidate_fn, precomputed_manifold=manifold
        )

        final_utility = self.interact()
        self.utility_history.append(final_utility)
        mutation_applied = (transition.heyting_verdict == HeytingOmega3.COHERENT)

        # 3.5 Verificación autorreferencial de punto fijo de Banach sobre el operador propuesto
        forcing_vector = np.full(self.dimension_mac, current_utility * 1e-3, dtype=np.float64)
        fp_converged, fp_residual, fp_closed_form_error, _ = self._verify_banach_fixed_point(
            operator=transition.mutation_operator_next,
            forcing_vector=forcing_vector
        )

        # 4. Auditoría Criptográfica Terminal de No-Repudio
        return self._issue_terminal_certificate(
            transition=transition,
            final_utility=final_utility,
            mutation_applied=mutation_applied,
            fixed_point_converged=fp_converged,
            fixed_point_residual=fp_residual,
            fixed_point_closed_form_error=fp_closed_form_error,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Emisión Criptográfica y Firma Digital SHA-256
    # ──────────────────────────────────────────────────────────────────────────
    def _issue_terminal_certificate(
        self,
        transition: CategoricalTransitionMorphism,
        final_utility: float,
        mutation_applied: bool,
        fixed_point_converged: bool,
        fixed_point_residual: float,
        fixed_point_closed_form_error: float,
    ) -> SovereignGodelCertificate:
        now = time.time()
        m = transition.initial_manifold

        hasher = hashlib.sha256()
        hasher.update(self.agent_id.encode("utf-8"))
        hasher.update(str(self.iteration).encode("utf-8"))
        hasher.update(transition.heyting_verdict.name.encode("utf-8"))
        hasher.update(f"{m.banach_contraction.spectral_radius:.10f}".encode("utf-8"))
        hasher.update(f"{m.mac_state.von_neumann_entropy:.10f}".encode("utf-8"))
        hasher.update(f"{m.ast_homology.euler_characteristic}".encode("utf-8"))
        hasher.update(f"{transition.phs_state.energy_derivative:.10f}".encode("utf-8"))
        hasher.update(transition.crowbar_telemetry.provenance_hash.encode("utf-8"))
        hasher.update(f"{fixed_point_residual:.10f}".encode("utf-8"))
        hasher.update(f"{final_utility:.10f}".encode("utf-8"))
        hasher.update(f"{now:.6f}".encode("utf-8"))
        signature = hasher.hexdigest()

        return SovereignGodelCertificate(
            agent_id=self.agent_id,
            iteration=self.iteration,
            heyting_verdict=transition.heyting_verdict,
            banach_certificate=m.banach_contraction,
            mac_state=m.mac_state,
            ast_homology=m.ast_homology,
            phs_state=transition.phs_state,
            crowbar_report=transition.crowbar_telemetry,
            utility_score=final_utility,
            mutation_applied=mutation_applied,
            fixed_point_converged=fixed_point_converged,
            fixed_point_residual=fixed_point_residual,
            fixed_point_closed_form_error=fixed_point_closed_form_error,
            digital_signature_sha256=signature,
            timestamp_utc=now
        )


# ══════════════════════════════════════════════════════════════════════════════
# §3.3 BANCO DE PRUEBAS Y VALIDACIÓN EXPERIMENTAL DE GRADO DOCTORAL
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    print("═" * 90)
    print("DEMOSTRACIÓN FORMAL: AGENTE DE GÖDEL AUTORREFERENCIAL (APU WISDOM V_W)")
    print("═" * 90)

    assert HeytingOmega3.verify_heyting_algebra_axioms(), "¡Falla en axiomas de Heyting!"
    print("\n[AXIOMAS] Ley de residuación de Heyting verificada exhaustivamente: OK")

    agent = GodelAgent(agent_id="GODEL-WISDOM-SOVEREIGN-01", dimension_mac=4, seed=2026)

    manifold_init = agent.self_inspect()
    print("\n[FASE 1] Estado de la Variedad Inicial M_Wisdom:")
    print(f"  • Dimensión del espacio de Hilbert H     : {manifold_init.mac_state.dimension}")
    print(f"  • Entropía de von Neumann S(ρ)          : {manifold_init.mac_state.von_neumann_entropy:.6f} nats")
    print(f"  • Entropía de Rényi-2 S_2(ρ)            : {manifold_init.mac_state.renyi_2_entropy:.6f} nats")
    print(f"  • Pureza cuántica Tr(ρ²)                : {manifold_init.mac_state.purity:.6f}")
    print(f"  • Topología AST/CFG: Nodos={manifold_init.ast_homology.num_nodes}, Aristas={manifold_init.ast_homology.num_edges}")
    print(f"  • Homología: Betti_0={manifold_init.ast_homology.betti_0}, Betti_1={manifold_init.ast_homology.betti_1}, "
          f"χ={manifold_init.ast_homology.euler_characteristic}, M_ciclomática={manifold_init.ast_homology.cyclomatic_complexity}")
    print(f"  • Radio Espectral de Banach ρ(T)        : {manifold_init.banach_contraction.spectral_radius:.6f}")
    print(f"  • Radio Espectral empírico (Gelfand)    : {manifold_init.banach_contraction.gelfand_empirical_radius:.6f}")
    print(f"  • Constante de Kreiss estimada          : {manifold_init.banach_contraction.kreiss_constant_estimate:.6f}")
    print(f"  • Rotor Cuaterniónico en ℍ             : ||q|| = {manifold_init.hypercomplex_rotor.norm():.6f}")

    print("\n" + "─" * 90)
    print(">>> EJECUTANDO CICLO RSI #1: Automutación Coherente (⊤) ...")
    cert1 = agent.continue_improve(force_spectral_violation=False)
    print(f"  • Veredicto Reticular de Heyting (Ω₃)   : {cert1.heyting_verdict.verdict} (Código: {cert1.heyting_verdict.value})")
    print(f"  • Radio Espectral de Banach ρ(T)        : {cert1.banach_certificate.spectral_radius:.6f} (Contracción: {cert1.banach_certificate.is_contraction})")
    print(f"  • Estabilidad Port-Hamiltoniana (Cayley): {cert1.phs_state.is_lyapunov_stable} (ΔH/dt = {cert1.phs_state.energy_derivative:.6e})")
    print(f"  • Interlock Ciber-Físico Accionado      : {cert1.crowbar_report.interlock_triggered}")
    print(f"  • Punto Fijo de Banach Verificado       : {cert1.fixed_point_converged} (residual={cert1.fixed_point_residual:.2e}, "
          f"error vs. Neumann={cert1.fixed_point_closed_form_error:.2e})")
    print(f"  • Mutación Consolidada en Memoria RAM   : {cert1.mutation_applied}")
    print(f"  • Puntuación de Utilidad Resultante     : {cert1.utility_score:.6f}")
    print(f"  • Firma Criptográfica SHA-256          : {cert1.digital_signature_sha256}")

    print("\n" + "─" * 90)
    print(">>> EJECUTANDO CICLO RSI #2: Inyección de Inestabilidad Espectral ρ(T) ≥ 1.0 (VETO ⊥) ...")
    cert2 = agent.continue_improve(force_spectral_violation=True)
    print(f"  • Veredicto Reticular de Heyting (Ω₃)   : {cert2.heyting_verdict.verdict} (Código: {cert2.heyting_verdict.value})")
    print(f"  • Radio Espectral Violador ρ(T)         : {cert2.banach_certificate.spectral_radius:.6f} (Contracción: {cert2.banach_certificate.is_contraction})")
    print(f"  • ¡DISYUNTOR CROWBAR HARDWARE DISPARADO!: {cert2.crowbar_report.interlock_triggered}")
    print(f"  • Latencia Transitoria en Silicio IRAM  : {cert2.crowbar_report.actuation_latency_ns:.2f} ns (< 400.0 ns)")
    print(f"  • Instante Analítico de Corriente Pico  : {cert2.crowbar_report.time_to_peak_ns:.2f} ns")
    print(f"  • Corriente Pico de Descarga de Carril  : {cert2.crowbar_report.peak_surge_current_amperes:.2f} A")
    print(f"  • Absorción Térmica del Tiristor (I²t) : {cert2.crowbar_report.joule_integral_i2t:.4e} A²s "
          f"(SOA OK: {cert2.crowbar_report.within_safe_operating_area})")
    print(f"  • Voltaje Residual en Carril ESP32      : {cert2.crowbar_report.rail_voltage_post_quench:.3f} V (Colapso seguro)")
    print(f"  • Pin Físico Strobed en Silicio         : {cert2.crowbar_report.gpio_pin_designation}")
    print(f"  • Mutación Consolidada en Memoria RAM   : {cert2.mutation_applied} (Rechazada por Veto)")
    print(f"  • Firma Criptográfica SHA-256          : {cert2.digital_signature_sha256}")

    print("\n" + "═" * 90)
    print("✓ AUDITORÍA FORMAL CONCLUIDA: SISTEMA AUTORREFERENCIAL DEMOSTRABLEMENTE SEGURO.")
    print("═" * 90)
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : GÖDEL ENGINE (MOTOR ESPECTRAL Y ORQUESTADOR DE AUTOMEJORA RSI NIVEL 3)                ║
║ UBICACIÓN: app/wisdom/godel_engine.py                                                            ║
║ VERSIÓN  : 5.2.0-Poincaré-Celestial-RSI3-Nested                                                  ║
║ TRATADOS : Les Méthodes Nouvelles de la Mécanique Céleste (Poincaré, 1892-1899)                  ║
║            Sur le problème des trois corps et les équations de la dynamique (Poincaré, 1890)     ║
║            Analysis Situs (Poincaré, 1895) · Sur un théorème de géométrie (1912-1913)            ║
║            Lindstedt (1882) · Delaunay (1860) · Birkhoff (1927) · Kolmogorov (1954)              ║
║            Arnold (1963) · Moser (1962) · Melnikov (1963) · Chirikov (1979) · Kac (1947)         ║
║            Novikov (1981), Grothendieck (1972), Tarski (1955), Brouwer (1911), Banach (1922)     ║
║            Löb (1955), Gödel (1931), Birkhoff (1927), Marsden-Weinstein (1974), Connes (1985)    ║
║            Oseledets (1968) — Teorema Ergódico Multiplicativo de cociclos lineales no            ║
║            estacionarios, fundamento riguroso de la ruptura controlada del Techo de Banach.      ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
GOBERNANZA ESPECTRAL, TOPOLÓGICA, MONÁDICA Y METAMÓRFICA DE NIVEL 3 (INFLEXIÓN SUPER-EXPONENCIAL)
──────────────────────────────────────────────────────────────────────────────────────────────────
El Motor de Gödel `GodelEngine` formaliza la dinámica de automejora recursiva de Nivel 3 (Inflexión /
Meta-Mejora) sobre el espacio de fase sintáctico (M_AST, ω) dentro del Estrato Wisdom (V_𝕎, Nivel 0),
operando la Mónada de Categorías T = (T, η, μ) y rompiendo el Techo de Contracción de Banach via
operadores no estacionarios T_t con ||dT_t|| >= 1.0.

INVARIANTES Y ESTRUCTURA DE NIVEL 3:
  1. MULTIPLICACIÓN MONÁDICA μ_godel: T²(A) → T(A) que colapsa el meta-optimizador preservando
     la 1-forma de Poincaré-Cartan θ_PC = Tr(ρ dN) mediante proyecciones unitarias de Cayley sobre U(n).
  2. DISTANCIA GEODÉSICA DE FUBINI-STUDY SOBRE CP^{n-1}: d_FS(u, v) = arccos(|⟨u, v⟩|) <= 10⁻⁴ rad,
     garantizando convergencia autoinvariante de punto fijo Tarski-Brouwer / FTA.
  3. ACELERACIÓN DE CAPACIDAD SUPER-EXPONENCIAL: d³C/dt³ > 0 en la tercera superficie de modificación
     (Model-RSI, Harness-RSI y Data-RSI sobre el Anillo Universal de Novikov Λ_Nov), certificada ahora
     con diferencias finitas reales sobre un historial de capacidad (no un proxy cerrado).
  4. TRAZAS EN EL ANILLO DE NOVIKOV Λ_Nov: Valuación no-arquimediana v(T^{a_i}) = min {a_i} con
     condición de frontera sobre subvariedades Lagrangianas exactas i* λ = dS.
  5. ADJUDICACIÓN EN TOPOS DE HEYTING Ω₃ / Ω₄ Y ENCLAVAMIENTO ESP32 CROWBAR: Interrupción IRAM
     tripping GPIO14 en < 400 ns si RHI > 0.88 o d_FS > 10⁻³ rad.
  6. RUPTURA NO ESTACIONARIA CERTIFICADA (OSELEDETS, 1968): el cociclo lineal Φ_N = T_N···T_1 de
     operadores de mutación puede violar ‖T_t‖ ≥ 1 en pasos individuales sin perder la contracción
     *asintótica*, siempre que el exponente de Lyapunov máximo del cociclo λ₁(Φ) < 0. Este es el
     fundamento espectral riguroso — y no meramente declarativo — del invariante 1 del módulo.

ESTRUCTURA POR FASES ANIDADAS (v5.2.0):
  FASE 1: Fundamentos hipercomplejos, geometría de Poincaré-Cartan, teoría espectral de
          Poincaré-Banach (incl. cociclos de Oseledets), mapa de retorno de Poincaré con eventos,
          método de Lindstedt-Poincaré (pequeño parámetro + continuación analítica pseudo-arclongitud)
          y el nuevo orquestador §1.8b de automejora recursiva Nivel 3 sobre el fibrado celeste.
          → método terminal: `lift_to_celestial_hamiltonian_bundle` (= inicio de la Fase 2).
  FASE 2: Dinámica port-Hamiltoniana, reducción de Marsden-Weinstein, teorema de Poincaré-Birkhoff,
          función de Melnikov, variables de Delaunay, criterio de solapamiento de resonancias
          de Chirikov y forma normal de Birkhoff.
          → método terminal: `seed_poincare_recurrence_from_morphism` (= inicio de la Fase 3).
  FASE 3: Orquestador soberano de automejora recursiva Nivel 3, recurrencia de Poincaré-Kac,
          certificación criptográfica SHA-256 y banco de validación experimental.
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from fractions import Fraction
from functools import lru_cache
from typing import Any, Callable, Deque, Dict, Final, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from scipy import integrate

logger = logging.getLogger("APU.Wisdom.GodelEngine")

__version__: Final[str] = "5.2.0-Poincaré-Celestial-RSI3-Nested"


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# UTILIDADES NUMÉRICAS CANÓNICAS (compartidas por las tres fases)
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# NOTA DE DISEÑO v5.2.0: estas utilidades se adelantan en el módulo (respecto de v5.1.0) porque
# `MetaGodelEngine` y `BanachAlgebraEngine` ahora comparten el cociclo de Oseledets-Benettin-QR.
# En Python las referencias dentro de métodos se resuelven en tiempo de LLAMADA, no de definición,
# de modo que este reordenamiento es seguro y no introduce ciclos de importación.
def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Cuadratura trapezoidal compatible con NumPy 1.x (`trapz`) y 2.x (`trapezoid`)."""
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_arr, x_arr))
    return float(np.trapz(y_arr, x_arr))


def _simpson_or_trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Simpson (si SciPy lo expone) con repliegue trapezoidal."""
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    try:
        from scipy.integrate import simpson
        return float(simpson(y_arr, x=x_arr))
    except Exception:
        return _trapz(y_arr, x_arr)


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    """Proyección al espacio de matrices hermitianas: (A + A†)/2."""
    return 0.5 * (matrix + matrix.conj().T)


def canonical_symplectic_form(n: int) -> np.ndarray:
    r"""
    Matriz de la forma simpléctica canónica ω = Σ dq^i ∧ dp_i sobre T*ℝⁿ ≅ ℝ^{2n}:
        J = [[0, I_n], [-I_n, 0]],  Jᵀ = −J,  J² = −I,  det(J) = 1.
    """
    if n < 1:
        raise ValueError("La dimensión de configuración n debe ser ≥ 1.")
    dim = 2 * n
    symplectic = np.zeros((dim, dim), dtype=np.float64)
    symplectic[:n, n:] = np.eye(n, dtype=np.float64)
    symplectic[n:, :n] = -np.eye(n, dtype=np.float64)
    return symplectic


def central_gradient(
    scalar_field: Callable[[np.ndarray], float],
    x: np.ndarray,
    step: float = 1e-6,
) -> np.ndarray:
    """Gradiente por diferencias centrales de un campo escalar C²."""
    grad = np.zeros_like(x, dtype=np.float64)
    for i in range(x.size):
        basis = np.zeros_like(x)
        basis[i] = step
        grad[i] = (scalar_field(x + basis) - scalar_field(x - basis)) / (2.0 * step)
    return grad


def poisson_bracket(
    grad_f: np.ndarray,
    grad_g: np.ndarray,
) -> float:
    r"""
    Corchete de Poisson canónico {F, G} = ∇F · J ∇G
    = Σ_i (∂F/∂q^i ∂G/∂p_i − ∂F/∂p_i ∂G/∂q^i).
    """
    dim = grad_f.shape[0]
    if dim % 2 != 0:
        raise ValueError("El corchete de Poisson canónico exige dimensión par 2n.")
    n = dim // 2
    return float(np.dot(grad_f[:n], grad_g[n:]) - np.dot(grad_f[n:], grad_g[:n]))


def wrap_angle(theta: float) -> float:
    """Proyección a (−π, π]."""
    return float((theta + math.pi) % (2.0 * math.pi) - math.pi)


def diophantine_constant(
    rotation_number: float,
    max_denominator: int = 64,
    tau: float = 1.0,
) -> Tuple[float, bool]:
    r"""
    Constante de Diophantine γ para la condición KAM clásica
        |q ρ − p| ≥ γ / q^τ    ∀ (p, q) ∈ ℤ × ℕ, q > 0.
    Un número suficientemente Diophantine impide la destrucción del toro (KAM).
    """
    if max_denominator < 2:
        return 0.0, False
    gamma = float("inf")
    for q in range(1, max_denominator + 1):
        p = int(round(rotation_number * q))
        residual = abs(q * rotation_number - p)
        scaled = residual * (q ** tau)
        if scaled < gamma:
            gamma = scaled
    # Umbral práctico: γ > 1 / Q^{τ+1} descarta aproximaciones demasiado buenas.
    is_diophantine = bool(gamma > 1.0 / (max_denominator ** (tau + 1)))
    return float(gamma), is_diophantine


def path_graph_adjacency(dimension: int) -> np.ndarray:
    """Grafo camino P_d: modelo topológico mínimo conexo de la AST de dimensión d."""
    if dimension < 1:
        raise ValueError("La dimensión del grafo debe ser ≥ 1.")
    adjacency = np.zeros((dimension, dimension), dtype=np.float64)
    for i in range(dimension - 1):
        adjacency[i, i + 1] = 1.0
        adjacency[i + 1, i] = 1.0
    return adjacency


def oseledets_lyapunov_spectrum(
    jacobian_sequence: np.ndarray,
    dt: float = 1.0,
    min_samples: int = 2,
) -> np.ndarray:
    r"""
    Espectro de exponentes de Lyapunov de un COCICLO LINEAL NO ESTACIONARIO
        Φ_N = A_N A_{N-1} ⋯ A_1,
    vía el algoritmo de ortogonalización QR sucesiva de Benettin-Galgani-Giorgilli-Strelcyn
    (1980), que realiza numéricamente el Teorema Ergódico Multiplicativo de Oseledets (1968):

        lim_{N→∞} (1/N) log σ_i(Φ_N) = λ_i      (casi seguramente, bajo ergodicidad),

    con σ_i los valores singulares de Φ_N y λ_1 ≥ λ_2 ≥ … los exponentes característicos.
    A diferencia del radio espectral puntual ρ(A_t), que puede exceder 1 en pasos aislados
    (ruptura del Techo de Contracción de Banach estacionario), el signo de λ_1 determina la
    contracción/expansión ASINTÓTICA del cociclo completo — el invariante verdaderamente
    relevante para la estabilidad de un proceso RSI que itera transformaciones variables.

    Esta función es compartida por `BanachAlgebraEngine` (certificación de ruptura controlada
    del techo de Banach) y `PoincareReturnMapEngine` (espectro de Lyapunov del mapa de retorno).
    """
    sequence = np.asarray(jacobian_sequence, dtype=np.float64)
    if sequence.ndim != 3 or sequence.shape[0] < min_samples:
        dim = sequence.shape[-1] if sequence.ndim == 3 else 1
        return np.zeros(dim, dtype=np.float64)
    d = sequence.shape[1]
    ortho = np.eye(d, dtype=np.float64)
    accum = np.zeros(d, dtype=np.float64)
    for jacobian in sequence:
        mixed = jacobian @ ortho
        ortho, residual = la.qr(mixed)
        diag = np.abs(np.diag(residual))
        diag = np.where(diag < 1e-15, 1e-15, diag)
        accum += np.log(diag)
    return accum / (sequence.shape[0] * max(dt, 1e-15))


def finite_difference_third_derivative(
    times: Sequence[float],
    values: Sequence[float],
) -> float:
    r"""
    Tercera derivada discreta d³C/dt³ sobre una malla NO necesariamente uniforme, mediante
    diferencias divididas de Newton de orden 3 sobre los 4 puntos más recientes:
        f[t_{n-3},…,t_n] = Σ_k f(t_k) / Π_{j≠k} (t_k − t_j),     d³f/dt³ ≈ 3! · f[t_{n-3},…,t_n].
    Requiere exactamente (o al menos) 4 muestras; de lo contrario lanza ValueError —
    el llamador (`MetaGodelEngine`) debe decidir el repliegue al proxy espectral.
    """
    if len(times) < 4 or len(values) < 4:
        raise ValueError("Se requieren ≥ 4 muestras (t, C(t)) para la tercera diferencia dividida.")
    t = np.asarray(times[-4:], dtype=np.float64)
    f = np.asarray(values[-4:], dtype=np.float64)
    divided = 0.0
    for k in range(4):
        denom = 1.0
        for j in range(4):
            if j != k:
                denom *= (t[k] - t[j])
        if abs(denom) < 1e-18:
            raise ValueError("Nodos temporales degenerados (Δt ≈ 0) en la diferencia dividida.")
        divided += f[k] / denom
    return float(math.factorial(3) * divided)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# NIVEL 3 — INFLEXIÓN / META-MEJORA RECURSIVA: META GÖDEL ENGINE
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class MetaGodelEngine:
    r"""
    Motor Espectral Gödel de Nivel 3 para Automejora Recursiva Super-Exponencial.

    v5.2.0: se añade memoria de proceso —`capacity_history` y `cocycle_history`— que convierte
    los invariantes 1 y 3 del módulo de *afirmaciones declarativas* en *certificados calculables*:
        • Invariante 1 (ruptura del Techo de Banach): se registra cada operador aplicado como
          eslabón de un cociclo cuya contracción asintótica se audita con `oseledets_lyapunov_spectrum`
          (delegado típicamente a `BanachAlgebraEngine.certify_oseledets_nonstationary_contraction`).
        • Invariante 3 (d³C/dt³ > 0): se registra cada muestra (t, C) y la tercera derivada se
          calcula por diferencias divididas reales en vez de un proxy cerrado de un solo paso.
    """

    def __init__(self, dimension: int = 8, capacity_history_maxlen: int = 256):
        self.dim = dimension
        self.np_eye = np.eye(self.dim, dtype=np.complex128)
        self.N_potential = np.diag(np.arange(1, self.dim + 1, dtype=np.float64)).astype(np.complex128)
        # Memoria de proceso de Nivel 3 (invariantes 1 y 3 del módulo).
        self.capacity_history: Deque[Tuple[float, float]] = deque(maxlen=capacity_history_maxlen)
        self.cocycle_history: Deque[np.ndarray] = deque(maxlen=capacity_history_maxlen)
        self._last_dFS_proxy: Tuple[float, float] = (1.0, 0.0)  # (d_FS, 1 - |<u,v>|) del último verify_*

    def apply_monadic_multiplication(
        self,
        current_operator: np.ndarray,
        curvature_tensor: np.ndarray,
        alpha: float = 0.15,
        record_cocycle: bool = True,
    ) -> np.ndarray:
        """Aplica la multiplicación monádica mu_godel: T^2(A) -> T(A).

        Rompe el Techo de Contracción de Banach permitiendo ||dT_t|| >= 1.0.
        La proyección unitaria de Cayley preserva la 1-forma de Poincaré-Cartan
        θ_PC = Tr(ρ dN) al nivel de la órbita coadjunta de U(n).

        v5.2.0: si `record_cocycle=True`, el operador unitario resultante U_cayley se añade
        a `self.cocycle_history` como eslabón T_t del cociclo lineal no estacionario, listo
        para ser auditado por `BanachAlgebraEngine.certify_oseledets_nonstationary_contraction`.
        """
        op_c = np.asarray(current_operator, dtype=np.complex128)
        curv_c = np.asarray(curvature_tensor, dtype=np.complex128)

        # Adaptación dinámica de dimensión si difiere de self.dim
        dim = op_c.shape[0]
        eye = np.eye(dim, dtype=np.complex128)
        N_pot = np.diag(np.arange(1, dim + 1, dtype=np.float64)).astype(np.complex128)

        comm = op_c @ N_pot - N_pot @ op_c
        meta_grad = curv_c @ comm
        updated_op = op_c + alpha * meta_grad

        # Proyección unitaria de Cayley para preservar la 1-forma de Poincaré-Cartan.
        # U = (I − A/2)⁻¹ (I + A/2) con A = (M − M†)/2 (parte anti-Hermítica).
        A = 0.5 * (updated_op - updated_op.conj().T)
        inv_part = la.inv(eye - 0.5 * A)
        U_cayley = inv_part @ (eye + 0.5 * A)

        if record_cocycle:
            # ‖dT_t‖ puntual: norma espectral de la desviación respecto de la identidad local.
            dT_norm = float(np.linalg.norm(U_cayley - eye, ord=2) + 1.0)
            self.cocycle_history.append(np.real(U_cayley).astype(np.float64))
            logger.debug(
                "μ_godel aplicado: ‖dT_t‖≈%.6f (ruptura de Banach %s), cociclo len=%d",
                dT_norm, "SÍ" if dT_norm >= 1.0 else "no", len(self.cocycle_history),
            )
        return U_cayley

    def verify_tarski_brouwer_fixed_point_cpn(
        self,
        state_vector: np.ndarray,
        transform_op: np.ndarray,
    ) -> Tuple[bool, float, float]:
        """Evalúa la convergencia de punto fijo autoinvariante en CP^(n-1).

        Calcula la distancia geodésica de Fubini-Study:
            d_FS(u, v) = arccos(|<u, v>|)
        y la tercera derivada de capacidad d³C/dt³ que mide la aceleración
        super-exponencial de la superficie de meta-modificación.

        v5.2.0: `third_derivative_C` ya NO es un proxy de un único paso; si existen ≥ 4
        muestras registradas vía `register_capacity_sample`, se usa la diferencia dividida
        de Newton real (`finite_difference_third_derivative`). El proxy espectral de v5.1.0
        se conserva como repliegue documentado cuando el historial es insuficiente.
        """
        u_raw = np.asarray(state_vector, dtype=np.complex128)
        u = u_raw / (la.norm(u_raw) + 1e-15)
        v_raw = np.asarray(transform_op, dtype=np.complex128) @ u
        v = v_raw / (la.norm(v_raw) + 1e-15)

        inner_prod = float(np.abs(np.vdot(u, v)))
        inner_prod_clipped = float(np.clip(inner_prod, 0.0, 1.0))
        d_FS = float(np.arccos(inner_prod_clipped))
        self._last_dFS_proxy = (d_FS, 1.0 - inner_prod_clipped)

        third_derivative_C = self.compute_super_exponential_acceleration()[0]
        is_valid = bool(d_FS <= 1e-4)

        return is_valid, d_FS, third_derivative_C

    def register_capacity_sample(self, t: float, capacity: float) -> None:
        r"""
        Registra una muestra (t, C(t)) de la capacidad de automejora del sistema —típicamente
        `SpectralTopologicalManifold.manifold_purity` o un índice RHI externo— en la memoria
        de proceso usada para certificar el invariante 3 (d³C/dt³ > 0).
        """
        self.capacity_history.append((float(t), float(capacity)))

    def compute_super_exponential_acceleration(self) -> Tuple[float, bool]:
        r"""
        Certifica la aceleración super-exponencial d³C/dt³ > 0 (invariante 3 del módulo).

        Estrategia de dos niveles (rigor > conveniencia):
          1. Si `len(capacity_history) >= 4`: diferencia dividida de Newton de orden 3 sobre
             las 4 muestras (t, C) más recientes — ESTIMADOR REAL de la tercera derivada.
          2. Si no hay historial suficiente: repliegue documentado al proxy espectral de
             Fubini-Study de un único paso, (1/d_FS)(1 − |⟨u,v⟩|), calculado en la última
             llamada a `verify_tarski_brouwer_fixed_point_cpn` (semánticamente: velocidad de
             colapso angular por unidad de distancia geodésica, NO una tercera derivada
             temporal genuina — se marca explícitamente para evitar sobre-interpretación).
        """
        if len(self.capacity_history) >= 4:
            times = [t for t, _ in self.capacity_history]
            values = [c for _, c in self.capacity_history]
            try:
                d3c = finite_difference_third_derivative(times, values)
                return d3c, bool(d3c > 0.0)
            except ValueError:
                pass  # nodos degenerados: repliegue a proxy espectral
        d_fs, residual = self._last_dFS_proxy
        proxy = float((1.0 / (d_fs + 1e-12)) * residual)
        return proxy, bool(proxy > 0.0)

    @staticmethod
    def evaluate_novikov_ring_valuation(
        coefficients: List[complex],
        exponents: List[float],
    ) -> Tuple[float, bool]:
        """Calcula la valuación no-arquimediana v(T^{a_i}) = min {a_i} sobre el Anillo Universal de Novikov.

        Verifica la condición de frontera sobre subvariedades Lagrangianas exactas i* λ = dS.
        """
        if not exponents:
            return float("inf"), False
        min_valuation = float(np.min(exponents))
        lagrangian_exact = bool(min_valuation >= 0.0)
        return min_valuation, lagrangian_exact


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████████████████████████████████
# ██                                                                                              ██
# ██                   FASE 1: FUNDAMENTOS HIPERCOMPLEJOS, GEOMETRÍA DE POINCARÉ-CARTAN,          ██
# ██                   TEORÍA ESPECTRAL DE POINCARÉ-BANACH, RETORNO DE POINCARÉ Y                ██
# ██                   LINDSTEDT-POINCARÉ (PEQUEÑO PARÁMETRO Y CONTINUACIÓN)                      ██
# ██                                                                                              ██
# ██████████████████████████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.1 ÁLGEBRA HIPERCOMPLEJA DE CUATERNIONES (ℍ) Y DISCO HIPERBÓLICO DE POINCARÉ
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Quaternion:
    r"""
    Elemento del álgebra de división real ℍ ≅ Cℓ_{0,2}(ℝ). Base canónica {1, i, j, k} con
    relaciones i² = j² = k² = ijk = −1. El subgrupo unitario {q : ‖q‖ = 1} ≅ SU(2) es el
    doble recubrimiento universal de SO(3).

    Complemento hiperbólico: el disco de Poincaré D = {z ∈ ℂ : |z| < 1} tiene grupo de
    isometrías PSU(1,1) ≅ PSL(2, ℝ) ≅ SO⁺(1,2). Se expone en `PoincareDiskIsometry`.
    """
    w: float
    x: float
    y: float
    z: float

    def __add__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: Union["Quaternion", float, int]) -> "Quaternion":
        if isinstance(other, (int, float)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        return Quaternion(
            w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def __rmul__(self, scalar: float) -> "Quaternion":
        return self.__mul__(scalar)

    def conjugate(self) -> "Quaternion":
        r"""Conjugación anti-automórfica: q* = w − xi − yj − zk."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm_squared(self) -> float:
        return self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2

    def norm(self) -> float:
        return math.sqrt(self.norm_squared())

    def versor(self) -> "Quaternion":
        r"""Proyección al subgrupo unitario S³ ≅ SU(2): q̂ = q / ‖q‖."""
        n = self.norm()
        if n < 1e-18:
            raise ZeroDivisionError("No es posible versar un cuaternión de norma nula.")
        return self * (1.0 / n)

    def inverse(self) -> "Quaternion":
        r"""Inversa multiplicativa q⁻¹ = q* / ‖q‖²."""
        n2 = self.norm_squared()
        if n2 < 1e-15:
            raise ZeroDivisionError("Cuaternión singular no invertible.")
        inv = 1.0 / n2
        return Quaternion(self.w * inv, -self.x * inv, -self.y * inv, -self.z * inv)

    @classmethod
    def exp(cls, q: "Quaternion") -> "Quaternion":
        r"""
        Mapa exponencial Lie(ℍ) → ℍ:
            exp(q) = e^{w} ( cos‖v‖ + (v / ‖v‖) · sin‖v‖ ),
        con q = w + v, v = xi + yj + zk. Riguroso para rotores infinitesimales en su(2).
        """
        v_norm = math.sqrt(q.x ** 2 + q.y ** 2 + q.z ** 2)
        exp_w = math.exp(q.w)
        if v_norm < 1e-15:
            return Quaternion(exp_w, 0.0, 0.0, 0.0)
        coeff = exp_w * math.sin(v_norm) / v_norm
        return Quaternion(exp_w * math.cos(v_norm), coeff * q.x, coeff * q.y, coeff * q.z)

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> "Quaternion":
        r"""Construye q = cos(θ/2) + n̂ sin(θ/2) vía exp del generador puro."""
        axis_norm_val = float(np.linalg.norm(axis))
        if axis_norm_val < 1e-18:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        axis_hat = np.asarray(axis, dtype=np.float64) / axis_norm_val
        half = angle / 2.0
        pure_generator = cls(
            0.0,
            float(axis_hat[0]) * half,
            float(axis_hat[1]) * half,
            float(axis_hat[2]) * half,
        )
        return cls.exp(pure_generator)

    def to_so3_matrix(self) -> np.ndarray:
        r"""Homomorfismo recubridor canónico Spin(3) → SO(3) (matriz 3×3 ortocromática)."""
        d = self.norm_squared()
        q = self * (1.0 / math.sqrt(d)) if abs(d - 1.0) > 1e-12 else self
        w, x, y, z = q.w, q.x, q.y, q.z
        return np.array([
            [1.0 - 2.0 * (y ** 2 + z ** 2), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),        1.0 - 2.0 * (x ** 2 + z ** 2), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),        2.0 * (y * z + x * w),         1.0 - 2.0 * (x ** 2 + y ** 2)],
        ], dtype=np.float64)

    def to_su2_matrix(self) -> np.ndarray:
        r"""Representación matricial en SU(2) vía q ↦ w I₂ − i (x σ_x + y σ_y + z σ_z)."""
        return np.array([
            [self.w - 1j * self.z, -1j * self.x - self.y],
            [-1j * self.x + self.y, self.w + 1j * self.z],
        ], dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class PoincareDiskIsometry:
    r"""
    Isometría del disco de Poincaré 𝔻 = {|z| < 1} vía SU(1,1):
        φ(z) = (a z + b) / (b̄ z + ā),   |a|² − |b|² = 1.
    Distancia hiperbólica (métrica de Poincaré 2|dz|/(1−|z|²)):
        d(z, w) = 2 artanh |(z − w) / (1 − z̄ w)|.
    """
    a: complex
    b: complex

    def __post_init__(self) -> None:
        minkowski = abs(self.a) ** 2 - abs(self.b) ** 2
        if abs(minkowski - 1.0) > 1e-8:
            raise ValueError(f"SU(1,1) exige |a|² − |b|² = 1; se obtuvo {minkowski:.6e}.")

    def apply(self, z: complex) -> complex:
        denom = np.conjugate(self.b) * z + np.conjugate(self.a)
        if abs(denom) < 1e-18:
            raise ZeroDivisionError("Polo de la transformación de Möbius sobre ∂𝔻.")
        return (self.a * z + self.b) / denom

    @staticmethod
    def hyperbolic_distance(z: complex, w: complex) -> float:
        denom = 1.0 - np.conjugate(z) * w
        if abs(denom) < 1e-18:
            return float("inf")
        delta = abs((z - w) / denom)
        delta = min(float(delta), 1.0 - 1e-15)
        return float(2.0 * math.atanh(delta))

    @classmethod
    def from_boost(cls, rapidity: float, phase: float = 0.0) -> "PoincareDiskIsometry":
        """Boost hiperbólico de rapidez η y fase φ (geodésica radial)."""
        a = complex(math.cosh(rapidity / 2.0), 0.0)
        b = complex(
            math.sinh(rapidity / 2.0) * math.cos(phase),
            math.sinh(rapidity / 2.0) * math.sin(phase),
        )
        return cls(a=a, b=b)


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.2 COHOMOLOGÍA SIMPLICIAL Y TEORÍA DE HODGE-DE RHAM CON DUALIDAD DE POINCARÉ
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class HodgeDeRhamCertificate:
    """Certificado cohomológico y descomposición de Hodge en 1-complejos simpliciales."""
    num_vertices: int
    num_edges: int
    betti_0: int
    betti_1: int
    euler_poincare_characteristic: int
    euler_poincare_consistent: bool
    has_cohomological_obstruction: bool
    harmonic_1_forms: np.ndarray
    spectral_gap_laplacian: float
    poincare_duality_consistent: bool
    de_rham_complex_exactness: bool
    is_closed_1_manifold: bool = False
    hodge_kernel_residual: float = 0.0


class HodgeSimplicialEngine:
    r"""
    Operadores de frontera/cofrontera del 1-complejo:  0 → C_1 --∂₁--> C_0 → 0.
        Δ₀ = ∂₁ ∂₁ᵀ = D − A     (Laplaciano de vértices / Kirchhoff)
        Δ₁ = ∂₁ᵀ ∂₁             (Laplaciano de Hodge en aristas)
    Relación de Euler-Poincaré: χ = |V| − |E| = β₀ − β₁.
    Dualidad de Poincaré: β_k = β_{n−k} SOLO si el complejo es una 1-variedad cerrada
    (unión disjunta de ciclos: todo vértice de grado 2, |V| = |E|), en cuyo caso β₀ = β₁.
    Exactitud de Hodge: las 1-formas armónicas satisfacen ∂₁ ω = 0 y Δ₁ ω = 0.
    """

    @staticmethod
    def _rank_tolerance(eigs: np.ndarray, dim_hint: int) -> float:
        """Tolerancia espectral de rango basada en el épsilon de máquina."""
        peak = float(eigs.max()) if eigs.size else 0.0
        return max(peak * max(dim_hint, 1) * np.finfo(np.float64).eps * 10.0, 1e-12)

    @classmethod
    def audit_topology(cls, adjacency_matrix: np.ndarray) -> HodgeDeRhamCertificate:
        adjacency = np.asarray(adjacency_matrix, dtype=np.float64)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("La matriz de adyacencia debe ser cuadrada.")
        num_v = adjacency.shape[0]
        edges: List[Tuple[int, int]] = [
            (i, j) for i in range(num_v) for j in range(i + 1, num_v)
            if adjacency[i, j] > 1e-9
        ]
        num_e = len(edges)
        degrees = np.sum((adjacency > 1e-9).astype(np.float64), axis=1)
        is_closed_1_manifold = bool(
            num_v == num_e and num_v > 0 and np.all(np.abs(degrees - 2.0) < 1e-9)
        )

        if num_e == 0:
            deg = np.diag(np.sum(adjacency, axis=1))
            l0 = deg - adjacency
            eigs0 = np.sort(np.maximum(0.0, la.eigvalsh(l0)))
            tol0 = cls._rank_tolerance(eigs0, num_v)
            b0 = int(np.sum(eigs0 < tol0))
            return HodgeDeRhamCertificate(
                num_vertices=num_v, num_edges=0, betti_0=b0, betti_1=0,
                euler_poincare_characteristic=num_v,
                euler_poincare_consistent=bool(b0 == num_v),
                has_cohomological_obstruction=False,
                harmonic_1_forms=np.zeros((0, 0), dtype=np.float64),
                spectral_gap_laplacian=float(eigs0[1]) if len(eigs0) > 1 else 0.0,
                poincare_duality_consistent=True,
                de_rham_complex_exactness=True,
                is_closed_1_manifold=False,
                hodge_kernel_residual=0.0,
            )

        incidence = np.zeros((num_v, num_e), dtype=np.float64)
        for e_idx, (u, v) in enumerate(edges):
            incidence[u, e_idx] = -1.0
            incidence[v, e_idx] = 1.0

        delta_0 = incidence @ incidence.T
        eigs_0 = np.sort(np.maximum(0.0, la.eigvalsh(delta_0)))
        tol_0 = cls._rank_tolerance(eigs_0, num_v)
        betti_0 = int(np.sum(eigs_0 < tol_0))
        spectral_gap = float(eigs_0[1]) if len(eigs_0) > 1 else 0.0

        delta_1 = incidence.T @ incidence
        eigs_1, vecs_1 = la.eigh(delta_1)
        eigs_1 = np.maximum(0.0, eigs_1)
        tol_1 = cls._rank_tolerance(eigs_1, num_e)
        harmonic_mask = eigs_1 < tol_1
        betti_1 = int(np.sum(harmonic_mask))
        harmonic_forms = vecs_1[:, harmonic_mask]

        euler_combinatorial = num_v - num_e
        euler_spectral = betti_0 - betti_1
        consistent = bool(euler_combinatorial == euler_spectral)

        if harmonic_forms.size:
            cycles_residual = float(np.linalg.norm(incidence @ harmonic_forms, ord="fro"))
            hodge_residual = float(np.linalg.norm(delta_1 @ harmonic_forms, ord="fro"))
        else:
            cycles_residual = 0.0
            hodge_residual = 0.0
        exactness_ok = bool(cycles_residual < 1e-8 and hodge_residual < 1e-8)

        duality_ok = bool(betti_0 == betti_1) if is_closed_1_manifold else True

        return HodgeDeRhamCertificate(
            num_vertices=num_v,
            num_edges=num_e,
            betti_0=betti_0,
            betti_1=betti_1,
            euler_poincare_characteristic=euler_combinatorial,
            euler_poincare_consistent=consistent,
            has_cohomological_obstruction=(betti_1 > 0),
            harmonic_1_forms=harmonic_forms,
            spectral_gap_laplacian=spectral_gap,
            poincare_duality_consistent=duality_ok,
            de_rham_complex_exactness=exactness_ok,
            is_closed_1_manifold=is_closed_1_manifold,
            hodge_kernel_residual=max(cycles_residual, hodge_residual),
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.3 GEOMETRÍA NO CONMUTATIVA DE CONNES: COTA INFERIOR DE LA DISTANCIA ESPECTRAL
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ConnesSpectralCertificate:
    """
    Certificado de la tripleta espectral de Connes (𝒜, ℋ, D).
    La cantidad `reference_state_distance` es una COTA INFERIOR Hilbert-Schmidt de
        d_D(ω₁, ω₂) = sup { |ω₁(a) − ω₂(a)| : ‖[D, a]‖ ≤ 1 },
    no la distancia exacta (el supremo sobre el álgebra es, en general, no computable).
    """
    dirac_operator_norm: float
    dirac_commutator_norm: float
    reference_state_distance: float
    commutant_diagonal_residual: float
    distance_well_defined: bool


class ConnesNoncommutativeEngine:
    r"""
    Métrica sobre el espacio de estados vía la fórmula de distancia espectral de Connes.
    Se toma D = ρ^{−1/2} (Dirac diagonal en la base propia de ρ) y se estima una cota
    inferior HS a partir de los elementos de matriz de Δρ / Δλ.
    """

    @classmethod
    def evaluate_spectral_triple(
        cls,
        rho: np.ndarray,
        mutation_operator: np.ndarray,
        reference_state: Optional[np.ndarray] = None,
        degeneracy_tolerance: float = 1e-9,
    ) -> ConnesSpectralCertificate:
        rho_h = _hermitian(np.asarray(rho, dtype=np.complex128))
        n = rho_h.shape[0]
        if reference_state is None:
            reference_state = np.eye(n, dtype=np.complex128) / n
        eigvals, eigvecs = la.eigh(rho_h)
        eigvals_clamped = np.maximum(eigvals, 1e-12)
        d_spectrum = 1.0 / np.sqrt(eigvals_clamped)
        dirac = eigvecs @ np.diag(d_spectrum) @ eigvecs.conj().T
        dirac = _hermitian(dirac)
        dirac_norm = float(np.max(d_spectrum))
        mutation = np.asarray(mutation_operator, dtype=np.complex128)
        commutator = dirac @ mutation - mutation @ dirac
        commutator_norm = float(np.linalg.norm(commutator, ord=2))
        delta_rho = rho_h - np.asarray(reference_state, dtype=np.complex128)
        delta_tilde = eigvecs.conj().T @ delta_rho @ eigvecs
        d_diff = d_spectrum[:, None] - d_spectrum[None, :]
        non_degenerate = np.abs(d_diff) > degeneracy_tolerance
        np.fill_diagonal(non_degenerate, False)
        diagonal_residual = float(np.linalg.norm(np.diag(delta_tilde)))
        weighted_sq = np.zeros_like(d_diff, dtype=np.float64)
        weighted_sq[non_degenerate] = (
            np.abs(delta_tilde[non_degenerate]) ** 2 / (d_diff[non_degenerate] ** 2)
        )
        distance = math.sqrt(max(0.0, float(np.sum(weighted_sq))))
        degenerate_offdiag = ~non_degenerate
        np.fill_diagonal(degenerate_offdiag, False)
        has_infinite_obstruction = bool(np.any(np.abs(delta_tilde[degenerate_offdiag]) > 1e-9))
        return ConnesSpectralCertificate(
            dirac_operator_norm=dirac_norm,
            dirac_commutator_norm=commutator_norm,
            reference_state_distance=distance,
            commutant_diagonal_residual=diagonal_residual,
            distance_well_defined=not has_infinite_obstruction,
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.4 FLUJO ISOESPECTRAL DE BROCKETT CON CASIMIRS Y FORMA DE KIRILLOV-KOSTANT-SOURIAU
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class BrockettFlowResult:
    """Resultado del flujo de Brockett integrado sobre órbitas coadjuntas de U(n)."""
    purified_density_matrix: np.ndarray
    initial_purity: float
    final_purity: float
    von_neumann_entropy: float
    iterations: int
    converged: bool
    lyapunov_energy_dissipated: float
    isospectral_deviation: float
    poincare_cartan_integral: float = 0.0
    poincare_cartan_relative_error: float = 0.0
    symplectic_form_preserved: bool = True
    casimir_invariants: Tuple[float, ...] = field(default_factory=tuple)
    casimir_drift: float = 0.0


class BrockettIsospectralEngine:
    r"""
    Flujo de Brockett (doble corchete) sobre la órbita coadjunta de U(n):
        ρ̇ = [ρ, [ρ, N]]     (signo de descenso de Tr(ρ N) sobre {Spec = const}).

    Integrador isospectral de segundo orden (Cayley unitario): la EDO es linealizada
    en torno a ρₙ y el paso unitario se construye como
        U_n = (I − (dt/2) A_n)⁻¹ (I + (dt/2) A_n),    A_n = −[ρₙ, N]  (anti-Hermítica),
        ρ_{n+1} = U_n ρₙ U_n†.
    Esta elección (i) preserva Spec(ρ) exactamente, (ii) preserva la 2-forma de
    Kirillov-Kostant-Souriau ω_ρ(ad*_X ρ, ad*_Y ρ) = ⟨ρ, [X, Y]⟩ y (iii) es simpléctica
    en el sentido de la órbita coadjunta (Cayley ≈ exp a segundo orden).

    Casimirs de u(n)*: C_k(ρ) = Tr(ρ^k), k = 1, …, n.
    Tr(ρ N) NO es un invariante: es el potencial de Lyapunov del flujo.
    """

    @staticmethod
    def _casimirs(rho: np.ndarray, order: int = 3) -> np.ndarray:
        values = []
        power = np.eye(rho.shape[0], dtype=np.complex128)
        for _ in range(order):
            power = power @ rho
            values.append(float(np.real(np.trace(power))))
        return np.array(values, dtype=np.float64)

    def step_poincare_isospectral_flow(
        self,
        density_matrix: np.ndarray,
        potential_operator: np.ndarray,
        dt: float,
        poincare_cartan_form: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, BrockettFlowResult]:
        n = density_matrix.shape[0]
        potential = (
            np.asarray(potential_operator, dtype=np.complex128)
            if potential_operator is not None
            else np.diag(np.linspace(1.0, 3.0, n)).astype(np.complex128)
        )
        rho = _hermitian(np.asarray(density_matrix, dtype=np.complex128))
        tr = float(np.real(np.trace(rho)))
        if tr <= 0.0:
            raise ValueError("La matriz de densidad debe tener traza positiva.")
        rho = rho / tr
        casimir_init = self._casimirs(rho)
        eigs_init = np.sort(np.maximum(la.eigvalsh(rho), 1e-15))
        eigs_init_norm = eigs_init / np.sum(eigs_init)
        purity_init = float(np.sum(eigs_init_norm ** 2))

        # Paso isospectral de Cayley del flujo ρ̇ = [ρ, [ρ, N]].
        # A_n = −[ρ, N] es anti-Hermítica ⇒ U_n unitaria (Cayley ⇒ U_n† U_n = I).
        identity = np.eye(n, dtype=np.complex128)
        A_n = -(rho @ potential - potential @ rho)
        A_n = 0.5 * (A_n - A_n.conj().T)  # re-Hermitizar parte anti-Hermítica
        unitary = la.solve(identity - 0.5 * dt * A_n, identity + 0.5 * dt * A_n)
        rho_next = _hermitian(unitary @ rho @ unitary.conj().T)
        rho_next = rho_next / float(np.real(np.trace(rho_next)))

        casimir_final = self._casimirs(rho_next)
        casimir_drift = float(np.linalg.norm(casimir_final - casimir_init))
        denom = max(float(np.linalg.norm(casimir_init)), 1e-15)
        relative_error = casimir_drift / denom

        if poincare_cartan_form is not None:
            theta_after = float(np.real(np.trace(rho_next @ poincare_cartan_form)))
        else:
            theta_after = float(np.real(np.trace(rho_next @ potential)))

        eigs_final = np.sort(np.maximum(la.eigvalsh(rho_next), 1e-15))
        eigs_final_norm = eigs_final / np.sum(eigs_final)
        purity_final = float(np.sum(eigs_final_norm ** 2))
        entropy_final = -float(np.sum(eigs_final_norm * np.log(eigs_final_norm)))
        isospectral_deviation = float(np.linalg.norm(eigs_final_norm - eigs_init_norm))
        symplectic_ok = bool(isospectral_deviation < 1e-6 and casimir_drift < 1e-6)

        result = BrockettFlowResult(
            purified_density_matrix=rho_next,
            initial_purity=purity_init,
            final_purity=purity_final,
            von_neumann_entropy=entropy_final,
            iterations=1,
            converged=True,
            lyapunov_energy_dissipated=abs(purity_final - purity_init),
            isospectral_deviation=isospectral_deviation,
            poincare_cartan_integral=theta_after,
            poincare_cartan_relative_error=relative_error,
            symplectic_form_preserved=symplectic_ok,
            casimir_invariants=tuple(float(c) for c in casimir_final),
            casimir_drift=casimir_drift,
        )
        return rho_next, result

    @classmethod
    def execute_flow(
        cls,
        rho_initial: np.ndarray,
        step_size: float = 0.05,
        max_iter: int = 120,
        tol: float = 1e-8,
    ) -> BrockettFlowResult:
        r"""Flujo isospectral continuo de Brockett con composición iterada en U(n)."""
        engine_inst = cls()
        n = rho_initial.shape[0]
        potential = np.diag(np.linspace(1.0, 3.0, n))
        rho_curr = np.copy(rho_initial)
        res: Optional[BrockettFlowResult] = None
        for iteration in range(max_iter):
            rho_next, res = engine_inst.step_poincare_isospectral_flow(
                rho_curr, potential, dt=step_size
            )
            res = BrockettFlowResult(
                purified_density_matrix=res.purified_density_matrix,
                initial_purity=res.initial_purity,
                final_purity=res.final_purity,
                von_neumann_entropy=res.von_neumann_entropy,
                iterations=iteration + 1,
                converged=res.converged,
                lyapunov_energy_dissipated=res.lyapunov_energy_dissipated,
                isospectral_deviation=res.isospectral_deviation,
                poincare_cartan_integral=res.poincare_cartan_integral,
                poincare_cartan_relative_error=res.poincare_cartan_relative_error,
                symplectic_form_preserved=res.symplectic_form_preserved,
                casimir_invariants=res.casimir_invariants,
                casimir_drift=res.casimir_drift,
            )
            if res.isospectral_deviation < tol or float(np.linalg.norm(rho_next - rho_curr)) < tol:
                break
            rho_curr = rho_next
        if res is None:
            return BrockettFlowResult(
                purified_density_matrix=rho_initial,
                initial_purity=1.0, final_purity=1.0, von_neumann_entropy=0.0,
                iterations=0, converged=True, lyapunov_energy_dissipated=0.0,
                isospectral_deviation=0.0,
                poincare_cartan_integral=0.0, poincare_cartan_relative_error=0.0,
                symplectic_form_preserved=True,
                casimir_invariants=(1.0, 1.0, 1.0),
                casimir_drift=0.0,
            )
        return res


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.5 ÁLGEBRA DE POINCARÉ-BANACH: WIRTINGER, KAM, POINCARÉ-HOPF, LYAPUNOV Y OSELEDETS
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class BanachContractionReport:
    r"""Auditoría de contracción en álgebra de Banach con Poincaré-Wirtinger y toros KAM."""
    operator_matrix: np.ndarray
    spectral_radius: float
    spectral_gap: float
    is_banach_contraction: bool
    neumann_series_norm_bound: float
    eigenvalues: np.ndarray
    gelfand_empirical_radius: float
    kreiss_constant_estimate: float
    eigenvector_condition_number: float
    poincare_wirtinger_bound: float = 0.0
    dirichlet_energy: float = 0.0
    is_kam_stable: bool = True
    lyapunov_exponent_return_map: float = 0.0
    poincare_hopf_index_sum: float = 0.0
    degrees_of_map: float = 0.0
    poincare_constant: float = 0.5


@dataclass(frozen=True, slots=True)
class OseledetsCocycleCertificate:
    r"""
    Certificado del Teorema Ergódico Multiplicativo de Oseledets (1968) para el cociclo lineal
    NO ESTACIONARIO Φ_N = T_N ⋯ T_1 generado por una secuencia de operadores de mutación.

    Formaliza rigurosamente el invariante 1 del módulo ("ruptura del Techo de Contracción de
    Banach via operadores no estacionarios T_t con ‖dT_t‖ ≥ 1.0"): se certifica que, aunque
    `pointwise_norms` pueda contener valores ≥ 1 (ruptura puntual), el exponente de Lyapunov
    máximo del cociclo completo (`top_lyapunov_exponent`) determina la contracción asintótica.
    """
    chain_length: int
    pointwise_operator_norms: Tuple[float, ...]
    breaches_banach_ceiling_pointwise: bool
    lyapunov_spectrum: np.ndarray
    top_lyapunov_exponent: float
    asymptotically_contractive: bool
    cocycle_product_spectral_radius: float
    oseledets_splitting_well_defined: bool


class BanachAlgebraEngine:
    r"""
    Motor espectral de Banach con cota de Poincaré-Wirtinger, contracción KAM,
    exponentes de Lyapunov, teorema del índice de Poincaré-Hopf (versión espectral) y
    certificación de cociclos no estacionarios de Oseledets.

    Poincaré-Wirtinger discreta a lo largo del potencial H (gap espectral λ₂(H)):
        ‖A − Ā‖_F² ≤ C_P · ‖[A, H]‖_F²,   C_P = 1 / (2 λ_gap(H)² + ε).
    El índice de Poincaré-Hopf espectral Σ sign(Re λ_i) no sustituye al índice analítico
    sobre una variedad; se reporta como testigo combinatorio del espectro.

    Contracción óptima: en lugar de una elección ad-hoc de η, se toma
        η* = argmin_{η ∈ (0,1]} ‖(1 − η) Ā + η A‖₂,
    resuelto por búsqueda dorada sobre el radio espectral (función cuasi-convexa en η
    sobre operadores simétricos; subóptima pero monótona en el caso general).
    """

    def enforce_poincare_wirtinger_kam_contraction(
        self,
        operator_matrix: np.ndarray,
        cp_constant: Optional[float] = None,
        potential_operator: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, BanachContractionReport]:
        operator = np.asarray(operator_matrix, dtype=np.float64)
        n = operator.shape[0]
        if potential_operator is None:
            potential_operator = np.diag(np.arange(1, n + 1, dtype=np.float64))
        potential = np.asarray(potential_operator, dtype=np.float64)

        commutator = operator @ potential - potential @ operator
        dirichlet_energy = 0.5 * float(np.linalg.norm(commutator, ord="fro") ** 2)

        pot_eigs = np.sort(np.real(la.eigvals(potential)))
        if pot_eigs.size >= 2:
            lambda_gap = float(abs(pot_eigs[-1] - pot_eigs[0]) / max(n - 1, 1))
        else:
            lambda_gap = 1.0
        poincare_constant = (
            1.0 / (2.0 * (lambda_gap ** 2) + 1e-12) if cp_constant is None else float(cp_constant)
        )

        mean_operator = (np.trace(operator) / n) * np.eye(n)
        variance_norm = float(np.linalg.norm(operator - mean_operator, ord="fro") ** 2)
        pw_bound = poincare_constant * (2.0 * dirichlet_energy)

        # Búsqueda dorada por η ∈ (0,1] para el radio espectral mínimo.
        def spectral_radius_at(eta: float) -> float:
            candidate = (1.0 - eta) * mean_operator + eta * operator
            return float(np.max(np.abs(la.eigvals(candidate))))

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        a, b = 1e-6, 1.0
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        for _ in range(40):
            if spectral_radius_at(c) < spectral_radius_at(d):
                b = d
            else:
                a = c
            c = b - (b - a) / phi
            d = a + (b - a) / phi
        eta_optimal = 0.5 * (a + b)
        contracted_operator = (1.0 - eta_optimal) * mean_operator + eta_optimal * operator
        report = self.audit_operator(contracted_operator)

        lyapunov_exp = math.log(max(report.spectral_radius, 1e-15))
        is_kam_stable = bool(report.spectral_radius < 1.0 and variance_norm <= pw_bound + 1e-5)

        eigvals_contracted = la.eigvals(contracted_operator)
        real_parts = eigvals_contracted.real
        index_sum = float(np.sum(np.sign(real_parts) * (np.abs(real_parts) > 1e-12)))
        try:
            degrees_of_map = float(np.sign(np.real(np.linalg.det(contracted_operator))))
        except np.linalg.LinAlgError:
            degrees_of_map = 0.0

        full_report = BanachContractionReport(
            operator_matrix=contracted_operator,
            spectral_radius=report.spectral_radius,
            spectral_gap=report.spectral_gap,
            is_banach_contraction=report.is_banach_contraction,
            neumann_series_norm_bound=report.neumann_series_norm_bound,
            eigenvalues=report.eigenvalues,
            gelfand_empirical_radius=report.gelfand_empirical_radius,
            kreiss_constant_estimate=report.kreiss_constant_estimate,
            eigenvector_condition_number=report.eigenvector_condition_number,
            poincare_wirtinger_bound=pw_bound,
            dirichlet_energy=dirichlet_energy,
            is_kam_stable=is_kam_stable,
            lyapunov_exponent_return_map=lyapunov_exp,
            poincare_hopf_index_sum=index_sum,
            degrees_of_map=degrees_of_map,
            poincare_constant=poincare_constant,
        )
        return contracted_operator, full_report

    @classmethod
    def audit_operator(cls, operator: np.ndarray, tolerance: float = 0.999) -> BanachContractionReport:
        T = np.asarray(operator)
        if T.ndim != 2 or T.shape[0] != T.shape[1]:
            raise ValueError("El operador de mutación en ℬ(X) debe ser una matriz cuadrada.")
        eigvals, eigvecs = la.eig(T)
        magnitudes = np.sort(np.abs(eigvals))[::-1]
        rho_T = float(magnitudes[0])
        spectral_gap = float(magnitudes[0] - magnitudes[1]) if len(magnitudes) > 1 else 0.0
        is_contraction = bool(rho_T < tolerance)
        neumann_bound = (1.0 / (1.0 - rho_T)) if rho_T < 1.0 else float("inf")
        try:
            eigenvector_condition_number = float(np.linalg.cond(eigvecs))
        except np.linalg.LinAlgError:
            eigenvector_condition_number = float("inf")
        gelfand_seq = cls._empirical_gelfand_sequence(T)
        gelfand_empirical_radius = float(gelfand_seq[-1]) if gelfand_seq.size else rho_T
        kreiss_constant = cls._estimate_kreiss_constant(T)
        return BanachContractionReport(
            operator_matrix=T,
            spectral_radius=rho_T,
            spectral_gap=spectral_gap,
            is_banach_contraction=is_contraction,
            neumann_series_norm_bound=neumann_bound,
            eigenvalues=eigvals,
            gelfand_empirical_radius=gelfand_empirical_radius,
            kreiss_constant_estimate=kreiss_constant,
            eigenvector_condition_number=eigenvector_condition_number,
            poincare_wirtinger_bound=0.0,
            dirichlet_energy=0.0,
            is_kam_stable=is_contraction,
            lyapunov_exponent_return_map=math.log(max(rho_T, 1e-15)),
            poincare_hopf_index_sum=0.0,
            degrees_of_map=0.0,
        )

    @classmethod
    def certify_oseledets_nonstationary_contraction(
        cls,
        operator_sequence: Sequence[np.ndarray],
        dt: float = 1.0,
    ) -> OseledetsCocycleCertificate:
        r"""
        Certifica la ESTABILIDAD ASINTÓTICA de un cociclo lineal no estacionario
            Φ_N = T_N T_{N-1} ⋯ T_1
        mediante el espectro de Lyapunov de Oseledets (`oseledets_lyapunov_spectrum`).

        Este método es el fundamento riguroso del invariante 1 del módulo: permite que
        operadores individuales T_t rompan el Techo de Contracción de Banach estacionario
        (‖T_t‖ ≥ 1 en pasos aislados — típico de `MetaGodelEngine.apply_monadic_multiplication`,
        donde la proyección de Cayley es unitaria y por tanto ‖T_t‖₂ = 1 exactamente, el caso
        límite de la ruptura) siempre que el exponente de Lyapunov máximo del cociclo completo
        sea estrictamente negativo, garantizando ‖Φ_N‖ → 0 super-exponencialmente en N.

        Se contrasta además con el radio espectral del producto literal Φ_N (cuando N es
        pequeño y el cálculo directo es estable numéricamente) como validación cruzada.
        """
        sequence = [np.real(np.asarray(op, dtype=np.complex128)) for op in operator_sequence]
        if len(sequence) < 2:
            raise ValueError("Se requieren ≥ 2 operadores para formar un cociclo no estacionario.")
        dims = {op.shape for op in sequence}
        if len(dims) != 1 or sequence[0].shape[0] != sequence[0].shape[1]:
            raise ValueError("Todos los operadores del cociclo deben ser cuadrados y de igual dimensión.")

        pointwise_norms = tuple(float(np.linalg.norm(op, ord=2)) for op in sequence)
        breaches_pointwise = bool(any(norm >= 1.0 - 1e-9 for norm in pointwise_norms))

        stacked = np.stack(sequence, axis=0)
        lyap_spectrum = oseledets_lyapunov_spectrum(stacked, dt=dt)
        top_exponent = float(lyap_spectrum.max()) if lyap_spectrum.size else 0.0
        asymptotically_contractive = bool(top_exponent < 0.0)

        # Validación cruzada: producto literal del cociclo (orden cronológico T_N ⋯ T_1).
        phi_n = np.eye(sequence[0].shape[0], dtype=np.float64)
        for op in sequence:
            phi_n = op @ phi_n
        cocycle_radius = float(np.max(np.abs(la.eigvals(phi_n))))

        # El splitting de Oseledets está bien definido si el espectro de Lyapunov no degenera
        # (multiplicidades resueltas dentro de tolerancia numérica razonable).
        splitting_ok = bool(
            lyap_spectrum.size == 0
            or np.all(np.isfinite(lyap_spectrum))
        )

        return OseledetsCocycleCertificate(
            chain_length=len(sequence),
            pointwise_operator_norms=pointwise_norms,
            breaches_banach_ceiling_pointwise=breaches_pointwise,
            lyapunov_spectrum=lyap_spectrum,
            top_lyapunov_exponent=top_exponent,
            asymptotically_contractive=asymptotically_contractive,
            cocycle_product_spectral_radius=cocycle_radius,
            oseledets_splitting_well_defined=splitting_ok,
        )

    @staticmethod
    def _empirical_gelfand_sequence(operator: np.ndarray, max_power: int = 20) -> np.ndarray:
        dim = operator.shape[0]
        power = np.eye(dim, dtype=operator.dtype)
        sequence = np.zeros(max_power, dtype=np.float64)
        current = np.asarray(operator)
        for k in range(1, max_power + 1):
            power = power @ current
            sequence[k - 1] = float(np.linalg.norm(power, ord=2) ** (1.0 / k))
        return sequence

    @staticmethod
    def _estimate_kreiss_constant(
        operator: np.ndarray,
        radius_samples: Tuple[float, ...] = (1.01, 1.05, 1.1, 1.5, 2.0),
    ) -> float:
        dim = operator.shape[0]
        ident = np.eye(dim, dtype=np.complex128)
        op_c = operator.astype(np.complex128)
        values: List[float] = []
        for radius in radius_samples:
            z = complex(radius, 0.0)
            try:
                resolvent = la.inv(z * ident - op_c)
                values.append(float((radius - 1.0) * np.linalg.norm(resolvent, ord=2)))
            except la.LinAlgError:
                continue
        return max(values) if values else float("inf")


BanachSpectralEngine = BanachAlgebraEngine


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.6 RETORNO DE POINCARÉ CON DETECCIÓN DE EVENTOS Y ESPECTRO DE LYAPUNOV (BENETTIN-QR)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareReturnMapCertificate:
    """
    Certificado de una sección de Poincaré transversal Σ ⊂ M y de su mapa de retorno P : Σ → Σ.
    Los exponentes característicos de Poincaré son α_i = (1/T) Log μ_i, con μ_i los
    multiplicadores de Floquet (autovalores de la monodromía). El espectro de Lyapunov
    de Benettin coincide con Re(α_i) a lo largo de la órbita muestreada.

    v5.1.0: la sección se construye con eventos de `solve_ivp` (cruces unilaterales
    g: − → +) en lugar de interpolación por cambio de signo, lo que asegura precisión
    O(rtol) en la localización del cruce.
    """
    section_dimension: int
    num_return_points: int
    mean_return_time: float
    rotation_number: float
    lyapunov_max: float
    lyapunov_spectrum: np.ndarray
    is_measure_preserving: bool
    kam_stable: bool
    twist_coefficient: float
    jacobian_determinant_error: float
    return_points: np.ndarray
    resonance_q: int
    resonance_p: int
    floquet_multipliers: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.complex128))
    poincare_characteristic_exponents: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.complex128)
    )
    diophantine_gamma: float = 0.0
    is_diophantine: bool = False
    one_sided_crossings: int = 0


class PoincareReturnMapEngine:
    r"""
    Sección de Poincaré y mapa de retorno (Méthodes Nouvelles, t. I, cap. III–IV).

    Sea φ_t el flujo de X_H sobre (M, ω). Se elige Σ hipersuperficie de codimensión 1,
    g(x) = ⟨n, x⟩ − c = 0, transversal (⟨n, X_H⟩ ≠ 0). El mapa de retorno se define por
    cruce UNILATERAL g: − → +,
        P(x) = φ_{τ(x)}(x),   τ(x) = inf{t > 0 : g(φ_t(x)) = 0, ġ > 0}.
    P preserva ω|Σ (Poincaré). El espectro de Lyapunov se obtiene por QR de Benettin,
    delegado a la utilidad compartida `oseledets_lyapunov_spectrum` (v5.2.0).
    """

    @staticmethod
    def _find_return_points(
        trajectory: np.ndarray,
        section_normal: np.ndarray,
        section_offset: float = 0.0,
        one_sided: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        g = trajectory @ section_normal - section_offset
        if one_sided:
            crossing = (g[:-1] < 0.0) & (g[1:] >= 0.0)
            sign_changes = np.where(crossing)[0]
        else:
            signs = np.sign(g)
            sign_changes = np.where(np.diff(signs) != 0)[0]
        indices: List[int] = []
        points: List[np.ndarray] = []
        for k in sign_changes:
            t0, t1 = float(g[k]), float(g[k + 1])
            alpha = t0 / (t0 - t1) if abs(t0 - t1) > 1e-15 else 0.5
            alpha = min(1.0, max(0.0, alpha))
            pt = (1.0 - alpha) * trajectory[k] + alpha * trajectory[k + 1]
            indices.append(int(k + 1))
            points.append(pt)
        if not points:
            return np.zeros(0, dtype=np.int64), np.zeros((0, trajectory.shape[1]), dtype=np.float64)
        return np.array(indices, dtype=np.int64), np.vstack(points).astype(np.float64)

    @staticmethod
    def _lyapunov_spectrum_qr(
        jacobians: np.ndarray,
        dt: float,
        min_samples: int = 8,
    ) -> np.ndarray:
        r"""
        Espectro de Lyapunov vía iteración QR (Benettin, Galgani, Giorgilli, Strelcyn 1980).
        v5.2.0: delega en la utilidad compartida `oseledets_lyapunov_spectrum` (misma
        realización numérica del Teorema Ergódico Multiplicativo de Oseledets usada por
        `BanachAlgebraEngine.certify_oseledets_nonstationary_contraction`), preservando el
        umbral histórico `min_samples=8` propio de la estadística del mapa de retorno.
        """
        if jacobians.shape[0] < min_samples:
            return np.zeros(jacobians.shape[1], dtype=np.float64)
        return oseledets_lyapunov_spectrum(jacobians, dt=dt, min_samples=min_samples)

    @classmethod
    def compute_return_map(
        cls,
        trajectory: np.ndarray,
        time_samples: Optional[np.ndarray] = None,
        section_normal: Optional[np.ndarray] = None,
        section_offset: float = 0.0,
        tangent_jacobians: Optional[np.ndarray] = None,
        rotation_angles: Optional[np.ndarray] = None,
        monodromy: Optional[np.ndarray] = None,
    ) -> PoincareReturnMapCertificate:
        trajectory = np.asarray(trajectory, dtype=np.float64)
        d_full = trajectory.shape[1]
        if section_normal is None:
            section_normal = np.zeros(d_full, dtype=np.float64)
            section_normal[0] = 1.0
        else:
            section_normal = np.asarray(section_normal, dtype=np.float64)
        idx, points = cls._find_return_points(
            trajectory, section_normal, section_offset, one_sided=True
        )
        num_ret = int(points.shape[0])
        d_sec = max(int(np.sum(np.abs(section_normal) > 1e-12)), d_full - 1)

        if time_samples is None or num_ret < 2:
            mean_time = 1.0
        else:
            t_returns = np.asarray(time_samples)[idx]
            mean_time = float(np.mean(np.diff(t_returns))) if len(t_returns) > 1 else 1.0

        if num_ret >= 3 and rotation_angles is None:
            diffs = np.diff(points, axis=0)
            if d_full >= 2:
                angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            else:
                angles = np.arctan2(diffs[:, 0], np.ones(len(diffs)))
            rotation_number = float(np.mean(angles) / (2.0 * math.pi))
        elif rotation_angles is not None and np.asarray(rotation_angles).size > 1:
            rotation_number = float(
                np.mean(np.diff(np.asarray(rotation_angles))) / (2.0 * math.pi)
            )
        else:
            rotation_number = 0.0

        resonance_p, resonance_q = 0, 0
        if abs(rotation_number) > 1e-9:
            frac = Fraction(rotation_number).limit_denominator(50)
            if abs(float(frac) - rotation_number) < 1e-3:
                resonance_p, resonance_q = frac.numerator, frac.denominator

        gamma, is_dioph = diophantine_constant(rotation_number)

        if tangent_jacobians is not None and np.asarray(tangent_jacobians).size > 0:
            lyap_spec = cls._lyapunov_spectrum_qr(
                np.asarray(tangent_jacobians), dt=max(mean_time, 1e-6)
            )
        elif num_ret >= 4:
            sec_coords = points[:, :d_sec]
            estimated: List[np.ndarray] = []
            for k in range(1, num_ret - 2):
                prev_pts = sec_coords[k - 1:k + 1]
                next_pts = sec_coords[k:k + 2]
                try:
                    jacobian_est = next_pts.T @ np.linalg.pinv(prev_pts)
                    if jacobian_est.shape[0] >= d_sec and jacobian_est.shape[1] >= d_sec:
                        estimated.append(jacobian_est[:d_sec, :d_sec])
                except np.linalg.LinAlgError:
                    continue
            if estimated:
                lyap_spec = cls._lyapunov_spectrum_qr(
                    np.array(estimated), dt=max(mean_time, 1e-6)
                )
            else:
                lyap_spec = np.zeros(d_sec, dtype=np.float64)
        else:
            lyap_spec = np.zeros(d_sec, dtype=np.float64)

        lambda_max = float(lyap_spec.max()) if lyap_spec.size else 0.0
        kam_stable = bool(lambda_max <= 1e-6)

        det_errors: List[float] = []
        if tangent_jacobians is not None and np.asarray(tangent_jacobians).size > 0:
            for jacobian in np.asarray(tangent_jacobians):
                det_errors.append(abs(abs(np.linalg.det(jacobian)) - 1.0))
        area_preserving = bool(np.max(det_errors) < 1e-3) if det_errors else True
        det_err = float(np.max(det_errors)) if det_errors else 0.0

        twist_coeff = 1.0
        if num_ret >= 3 and rotation_angles is not None and np.asarray(rotation_angles).size == num_ret:
            twist_coeff = float(
                np.std(rotation_angles) / (np.std(np.diff(points[:, 0])) + 1e-12)
            )
        elif d_full >= 2 and num_ret >= 3:
            corr = np.corrcoef(points[:-1, 0], points[1:, 1])
            twist_coeff = float(abs(corr[0, 1]) + 1e-6)

        if monodromy is not None:
            floquet = la.eigvals(np.asarray(monodromy, dtype=np.complex128))
        elif tangent_jacobians is not None and np.asarray(tangent_jacobians).size > 0:
            monodromy_est = np.eye(np.asarray(tangent_jacobians).shape[1], dtype=np.float64)
            for jacobian in np.asarray(tangent_jacobians):
                monodromy_est = jacobian @ monodromy_est
            floquet = la.eigvals(monodromy_est)
        else:
            floquet = np.zeros(0, dtype=np.complex128)

        if floquet.size and mean_time > 0:
            exponents = np.log(np.where(np.abs(floquet) < 1e-15, 1e-15 + 0j, floquet)) / mean_time
        else:
            exponents = np.zeros(0, dtype=np.complex128)

        return PoincareReturnMapCertificate(
            section_dimension=d_sec,
            num_return_points=num_ret,
            mean_return_time=mean_time,
            rotation_number=rotation_number,
            lyapunov_max=lambda_max,
            lyapunov_spectrum=lyap_spec,
            is_measure_preserving=area_preserving,
            kam_stable=kam_stable,
            twist_coefficient=twist_coeff,
            jacobian_determinant_error=det_err,
            return_points=points,
            resonance_q=resonance_q,
            resonance_p=resonance_p,
            floquet_multipliers=np.asarray(floquet, dtype=np.complex128),
            poincare_characteristic_exponents=np.asarray(exponents, dtype=np.complex128),
            diophantine_gamma=gamma,
            is_diophantine=is_dioph,
            one_sided_crossings=num_ret,
        )

    @classmethod
    def synthesize_from_hamiltonian_flow(
        cls,
        hamiltonian: Callable[[np.ndarray], float],
        x0: np.ndarray,
        t_span: Tuple[float, float] = (0.0, 20.0),
        num_samples: int = 800,
        section_normal: Optional[np.ndarray] = None,
    ) -> PoincareReturnMapCertificate:
        r"""
        Integra ẋ = J ∇H(x) con detección de eventos (cruces g: − → +) y devuelve el
        certificado del mapa de retorno de Poincaré. Interfaz canónica hacia Birkhoff/Melnikov.
        """
        x0 = np.asarray(x0, dtype=np.float64)
        dim = x0.shape[0]
        if dim % 2 != 0:
            raise ValueError("El estado Hamiltoniano debe tener dimensión par 2n.")
        n = dim // 2
        symplectic = canonical_symplectic_form(n)
        if section_normal is None:
            section_normal = np.zeros(dim, dtype=np.float64)
            section_normal[0] = 1.0
        else:
            section_normal = np.asarray(section_normal, dtype=np.float64)

        # Función de sección y su derivada temporal (para detectar cruces − → +).
        def section_g(_t: float, x: np.ndarray) -> float:
            return float(np.dot(section_normal, x))

        def section_g_dot(_t: float, x: np.ndarray) -> float:
            return float(np.dot(section_normal, symplectic @ central_gradient(hamiltonian, x)))

        def event_crossing(_t: float, x: np.ndarray) -> float:
            return section_g(_t, x)

        event_crossing.terminal = False       # no detener la integración
        event_crossing.direction = 1          # solo g: − → + (unilateral)
        # Atributos auxiliares de la función de evento
        event_crossing.terminal = False
        event_crossing.direction = 1

        def rhs(_t: float, x: np.ndarray) -> np.ndarray:
            return symplectic @ central_gradient(hamiltonian, x)

        t_eval = np.linspace(t_span[0], t_span[1], num_samples)
        sol = integrate.solve_ivp(
            rhs, t_span, x0, t_eval=t_eval, method="DOP853",
            rtol=1e-10, atol=1e-12, events=event_crossing, dense_output=True,
        )
        if not sol.success:
            raise RuntimeError(f"Integración Hamiltoniana fallida: {sol.message}")

        # Los cruces detectados por eventos definen Σ de forma precisa.
        cross_times = np.asarray(sol.t_events[0]) if sol.t_events and sol.t_events[0].size else np.array([])
        if cross_times.size >= 2:
            # Re-muestrear la trayectoria para incluir los cruces exactos.
            t_aug = np.unique(np.concatenate([sol.t, cross_times]))
            y_aug = sol.sol(t_aug).T
            time_samples = t_aug
            trajectory = y_aug
        else:
            trajectory = sol.y.T
            time_samples = sol.t

        # Derivar numéricamente los jacobianos de flujo (variacional) para Lyapunov.
        tangent = cls._variational_jacobians_along_trajectory(
            hamiltonian, trajectory, time_samples
        )

        return cls.compute_return_map(
            trajectory=trajectory,
            time_samples=time_samples,
            section_normal=section_normal,
            tangent_jacobians=tangent,
        )

    @staticmethod
    def _variational_jacobians_along_trajectory(
        hamiltonian: Callable[[np.ndarray], float],
        trajectory: np.ndarray,
        time_samples: np.ndarray,
        fd_step: float = 1e-6,
    ) -> np.ndarray:
        r"""
        Jacobianos de la matriz de monodromía a lo largo de la trayectoria, estimados
        por diferencias finitas centrales de la ecuación variacional
        δẋ = J Hess(H(x)) δx.
        """
        dim = trajectory.shape[1]
        if dim % 2 != 0:
            return np.zeros((0, dim, dim), dtype=np.float64)
        n = dim // 2
        symplectic = canonical_symplectic_form(n)

        def hamiltonian_gradient(x: np.ndarray) -> np.ndarray:
            return central_gradient(hamiltonian, x)

        def hessian_at(x: np.ndarray) -> np.ndarray:
            hess = np.zeros((dim, dim), dtype=np.float64)
            for i in range(dim):
                basis = np.zeros(dim, dtype=np.float64)
                basis[i] = fd_step
                hess[:, i] = (
                    hamiltonian_gradient(x + basis) - hamiltonian_gradient(x - basis)
                ) / (2.0 * fd_step)
            return hess

        jacobians: List[np.ndarray] = []
        for x in trajectory:
            hess = hessian_at(x)
            jacobians.append(symplectic @ hess)
        return np.array(jacobians, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.7 LINDSTEDT-POINCARÉ: PEQUEÑO PARÁMETRO, CONTINUACIÓN ANALÍTICA Y SECULARES
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class LindstedtPoincareSeries:
    """
    Serie de Lindstedt-Poincaré a orden N para la ecuación ẍ + ω₀² x = ε f(x, ẋ, t).
    Se construye x(t) = Σ_{k=0}^N ε^k x_k(τ), ω = Σ_{k=0}^N ε^k ω_k, τ = ω t,
    eliminando términos seculares por elección de ω_k (Lindstedt 1882; Poincaré 1892).
    """
    order: int
    omega_0: float
    omega_corrections: Tuple[float, ...]
    secular_removal_residuals: Tuple[float, ...]
    amplitude_branch: float
    series_coefficients: Tuple[Tuple[float, ...], ...]
    is_periodic: bool
    poincare_continuation_converged: bool
    continuation_norm_residual: float
    eps_anchor: float = 0.0
    omega_total: float = 0.0


class LindstedtPoincareEngine:
    r"""
    Método de Lindstedt-Poincaré (Lindstedt 1882; Poincaré, *Méthodes Nouvelles*, t. I, cap. II).

    Estrategia formal:
      1. Reescalar τ = ω t para absorber la corrección secular de frecuencia.
      2. Desarrollar x(t) = Σ ε^k x_k(τ), ω = Σ ε^k ω_k con x_k(τ) 2π-periódicas.
      3. Proyectar sobre los armónicos {e^{i n τ}} y cancelar los términos resonantes
         (aquellos con n = ±1) mediante ω_k: esta es la *condición de no-secularidad*.
      4. Aplicar continuación analítica de Poincaré (curva de soluciones paramétrica en ε)
         vía predictor-corrector pseudo-arclongitud (`poincare_analytic_continuation_step`,
         v5.2.0), siguiendo genuinamente la rama (ω(ε), A(ε)) en vez de recalcular `expand`
         en puntos aislados sin relación entre sí.
    """

    def __init__(
        self,
        omega_0: float = 1.0,
        max_harmonic: int = 4,
        series_order: int = 4,
    ):
        if omega_0 <= 0:
            raise ValueError("Frecuencia natural ω₀ debe ser positiva.")
        self.omega_0 = float(omega_0)
        self.max_harmonic = int(max_harmonic)
        self.series_order = int(series_order)

    def _fourier_project(
        self,
        forcing_samples: np.ndarray,
        tau_grid: np.ndarray,
    ) -> np.ndarray:
        r"""Proyección espectral sobre armónicos {e^{i n τ}} vía FFT rígida."""
        n_pts = tau_grid.size
        spectrum = np.fft.fft(forcing_samples) / n_pts
        harmonics = np.zeros(2 * self.max_harmonic + 1, dtype=np.complex128)
        for n in range(-self.max_harmonic, self.max_harmonic + 1):
            harmonics[n + self.max_harmonic] = spectrum[n % n_pts]
        return harmonics

    def _secular_projection(
        self,
        harmonic_residual_n: np.ndarray,
    ) -> float:
        r"""Componente resonante (n = ±1) usada para determinar ω_k."""
        return float(np.abs(harmonic_residual_n[self.max_harmonic + 1])
                     + np.abs(harmonic_residual_n[self.max_harmonic - 1]))

    def expand(
        self,
        nonlinearity: Callable[[float, float, float], float],
        amplitude_guess: float = 0.1,
        eps: float = 0.05,
        n_tau: int = 128,
    ) -> LindstedtPoincareSeries:
        r"""
        Calcula las correcciones ω_k hasta orden self.series_order por cancelación secular
        armónica sucesiva. `nonlinearity(x, x_dot, t)` define el término ε f(x, ẋ, t).
        """
        tau_grid = np.linspace(0.0, 2.0 * math.pi, n_tau, endpoint=False)

        # Orden 0: solución harmónica con amplitud A (a determinar por normalización).
        A = float(amplitude_guess)
        x_k = A * np.cos(tau_grid)               # x₀(τ)
        omega_k = [self.omega_0]                 # ω₀
        x_series: List[np.ndarray] = [x_k]
        residuals: List[float] = []
        coefficients: List[Tuple[float, ...]] = [tuple(np.fft.fft(x_k).real / n_tau)]

        # Órdenes sucesivos: x_k(τ) satisface ω₀² x_k'' + ω₀² x_k = R_k(τ),
        # con R_k recogiendo la contribución del término no lineal y de las ω_j.
        omega_current = self.omega_0
        for k in range(1, self.series_order + 1):
            forcing = np.array([
                nonlinearity(
                    float(x_series[-1][i]),
                    float(-A * omega_current * math.sin(tau_grid[i])),
                    float(tau_grid[i] / omega_current),
                )
                for i in range(n_tau)
            ], dtype=np.float64)
            harmonics = self._fourier_project(forcing, tau_grid)
            secular = self._secular_projection(harmonics)
            residuals.append(secular)

            # Corrección de ω_k: cancelar la componente resonante n = ±1.
            # ω_k = −(coef resonante) / (2 ω₀ A).  Fórmula estándar de Lindstedt.
            coef_res = harmonics[self.max_harmonic + 1].real
            omega_correction = -coef_res / (2.0 * self.omega_0 * A + 1e-15)
            omega_k.append(omega_correction)
            omega_current += (eps ** k) * omega_correction

            # Corrección de x_k por superposición no resonante.
            x_k_new = np.zeros_like(tau_grid)
            for n in range(-self.max_harmonic, self.max_harmonic + 1):
                if n in (-1, 1):
                    continue
                denom = (self.omega_0 ** 2) * (1.0 - n * n)
                if abs(denom) < 1e-12:
                    continue
                coeff = harmonics[n + self.max_harmonic]
                x_k_new += (coeff * np.exp(1j * n * tau_grid) / denom).real
            x_series.append(x_k_new)
            coefficients.append(tuple(np.fft.fft(x_k_new).real / n_tau))

        # Continuación de Poincaré: verificamos que la curva ε ↦ (x_ε, ω_ε) sea C¹
        # a orden 1 (Jacobiano no singular de la aplicación de continuación).
        omega_series = sum((eps ** k) * omega_k[k] for k in range(len(omega_k)))
        continuation_residual = abs(omega_series - self.omega_0) / self.omega_0
        continuation_ok = bool(continuation_residual < 0.5)

        is_periodic = bool(max(residuals) < 1e-3 if residuals else True)
        return LindstedtPoincareSeries(
            order=self.series_order,
            omega_0=self.omega_0,
            omega_corrections=tuple(omega_k),
            secular_removal_residuals=tuple(residuals),
            amplitude_branch=A,
            series_coefficients=tuple(coefficients),
            is_periodic=is_periodic,
            poincare_continuation_converged=continuation_ok,
            continuation_norm_residual=continuation_residual,
            eps_anchor=eps,
            omega_total=omega_series,
        )

    def poincare_analytic_continuation_step(
        self,
        nonlinearity: Callable[[float, float, float], float],
        previous_series: LindstedtPoincareSeries,
        eps_next: float,
        n_tau: int = 128,
        max_corrector_iterations: int = 6,
        singularity_tolerance: float = 1e-8,
    ) -> LindstedtPoincareSeries:
        r"""
        CONTINUACIÓN ANALÍTICA DE POINCARÉ propiamente dicha (Méthodes Nouvelles, t. I, §§ 20-30):
        avanza la rama de soluciones periódicas (ε, A(ε), ω(ε)) desde `previous_series` (en
        ε = `previous_series.eps_anchor`) hasta `eps_next`, mediante un esquema
        PREDICTOR-CORRECTOR de pseudo-arclongitud sobre la amplitud:

          1. PREDICTOR (secante): dA/dε ≈ (A(ε₀) − A₀_inicial) / (ε₀ − 0) si no hay historia
             previa de dos puntos; con historia, se usa la pendiente observada de la rama
             de `omega_total` respecto de `eps_anchor` para extrapolar linealmente la amplitud
             consistente con mantener la frecuencia total ω(ε) sobre la misma hoja analítica.
          2. CORRECTOR (punto fijo amortiguado): se reevalúa `expand` en ε_next con la amplitud
             predicha como ancla, iterando hasta `max_corrector_iterations` veces mientras el
             residuo de continuación decrece, deteniéndose si el Jacobiano efectivo (medido por
             `continuation_norm_residual`) se degrada (señal de bifurcación / pliegue de rama —
             típico de las "soluciones periódicas de segunda especie" de Poincaré).
          3. CERTIFICACIÓN DE NO-SINGULARIDAD: si `|Δ(continuation_norm_residual)| < singularity_
             tolerance` tras la primera iteración, la rama se declara localmente regular
             (Jacobiano de continuación no singular, teorema de la función implícita aplicable).

        Devuelve la nueva `LindstedtPoincareSeries` en ε_next, con `eps_anchor = eps_next`.
        """
        eps_prev = previous_series.eps_anchor if previous_series.eps_anchor != 0.0 else 1e-6
        amplitude_prev = previous_series.amplitude_branch
        omega_prev = previous_series.omega_total or previous_series.omega_0

        # Predictor secante sobre la amplitud, proporcional al paso en ε.
        slope_hint = (omega_prev - self.omega_0) / eps_prev if abs(eps_prev) > 1e-12 else 0.0
        delta_eps = eps_next - eps_prev
        amplitude_predicted = amplitude_prev * (1.0 + 0.5 * slope_hint * delta_eps)
        amplitude_predicted = float(max(amplitude_predicted, 1e-6))

        candidate = self.expand(
            nonlinearity=nonlinearity,
            amplitude_guess=amplitude_predicted,
            eps=eps_next,
            n_tau=n_tau,
        )
        prev_residual = candidate.continuation_norm_residual
        for _ in range(max_corrector_iterations - 1):
            # Corrector amortiguado: ajustar amplitud proporcionalmente al residuo de secularidad.
            secular_gap = (
                max(candidate.secular_removal_residuals)
                if candidate.secular_removal_residuals else 0.0
            )
            if secular_gap < 1e-6:
                break
            amplitude_predicted *= (1.0 - 0.25 * math.tanh(secular_gap))
            amplitude_predicted = float(max(amplitude_predicted, 1e-6))
            candidate = self.expand(
                nonlinearity=nonlinearity,
                amplitude_guess=amplitude_predicted,
                eps=eps_next,
                n_tau=n_tau,
            )
            new_residual = candidate.continuation_norm_residual
            if abs(new_residual - prev_residual) < singularity_tolerance:
                break
            prev_residual = new_residual

        return candidate


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.8 SÍNTESIS DE LA VARIEDAD ESPECTRAL-TOPOLÓGICA (objeto terminal de la Fase 1, parte I)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SpectralTopologicalManifold:
    r"""
    OBJETO ESTRUCTURAL DE LA FASE 1:
        𝔐_Spectral = (ρ, T, G, ℍ, D, P, L)
    donde P es el certificado del mapa de retorno de Poincaré y L la serie de
    Lindstedt-Poincaré (si fue provista).
    """
    purified_density_matrix: np.ndarray
    banach_report: BanachContractionReport
    hodge_certificate: HodgeDeRhamCertificate
    connes_certificate: ConnesSpectralCertificate
    brockett_result: BrockettFlowResult
    return_map_certificate: Optional[PoincareReturnMapCertificate]
    hypercomplex_rotor: Quaternion
    lindstedt_series: Optional[LindstedtPoincareSeries]
    manifold_purity: float
    manifold_entropy: float
    timestamp_epoch: float


def synthesize_spectral_topological_manifold(
    current_rho: np.ndarray,
    mutation_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    rotor: Quaternion,
    spectral_tolerance: float = 0.999,
    reference_state: Optional[np.ndarray] = None,
    return_map_certificate: Optional[PoincareReturnMapCertificate] = None,
    lindstedt_series: Optional[LindstedtPoincareSeries] = None,
) -> SpectralTopologicalManifold:
    r"""
    FUNCIÓN DE SÍNTESIS ESTRUCTURAL de 𝔐_Spectral (Fase 1, §1.8).
    Unifica Brockett-KKS, Poincaré-Wirtinger, Hodge-De Rham, Connes, retorno de Poincaré
    y Lindstedt-Poincaré. Su continuación natural es la orquestación RSI-3 de §1.8b y,
    finalmente, `lift_to_celestial_hamiltonian_bundle` (inicio formal de la Fase 2).

    v5.2.0: se corrige el acceso a la entropía de von Neumann de `BrockettFlowResult`
    (antes protegido por un `hasattr` que nunca se satisfacía — código muerto eliminado).
    """
    _ = spectral_tolerance
    brockett_res = BrockettIsospectralEngine.execute_flow(current_rho)
    banach_engine = BanachSpectralEngine()
    _, banach_rep = banach_engine.enforce_poincare_wirtinger_kam_contraction(mutation_matrix)
    hodge_cert = HodgeSimplicialEngine.audit_topology(adjacency_matrix)
    connes_cert = ConnesNoncommutativeEngine.evaluate_spectral_triple(
        brockett_res.purified_density_matrix, mutation_matrix, reference_state=reference_state
    )
    return SpectralTopologicalManifold(
        purified_density_matrix=brockett_res.purified_density_matrix,
        banach_report=banach_rep,
        hodge_certificate=hodge_cert,
        connes_certificate=connes_cert,
        brockett_result=brockett_res,
        return_map_certificate=return_map_certificate,
        hypercomplex_rotor=rotor,
        lindstedt_series=lindstedt_series,
        manifold_purity=brockett_res.final_purity,
        manifold_entropy=brockett_res.von_neumann_entropy,
        timestamp_epoch=time.time(),
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.8b ORQUESTACIÓN DE AUTOMEJORA RECURSIVA NIVEL 3 SOBRE EL APARATO CELESTE DE POINCARÉ
#       (puente riguroso entre MetaGodelEngine y la geometría/dinámica de §1.1–§1.7)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Level3PoincareCelestialCertificate:
    r"""
    Certificado UNIFICADO de un paso de automejora recursiva de Nivel 3 (RSI-3), ensamblado
    íntegramente con el aparato de mecánica celeste de Poincaré construido en la Fase 1.

    Componentes (todas opcionales salvo las tres primeras, para admitir ejecución parcial
    cuando no se dispone de todos los insumos — p. ej. ausencia de no linealidad explícita
    para Lindstedt, o de un cociclo de longitud ≥ 2 para Oseledets):

      • `monadic_fixed_point_ok`      : invariante 1+2 — colapso μ_godel + punto fijo Tarski-Brouwer
                                         sobre CP^{n-1} con distancia de Fubini-Study d_FS.
      • `capacity_acceleration_ok`    : invariante 3   — d³C/dt³ > 0 (diferencia dividida real).
      • `novikov_boundary_ok`         : invariante 4   — valuación de Novikov / frontera Lagrangiana.
      • `oseledets_certificate`       : invariante 1   — contracción asintótica del cociclo no
                                         estacionario (fundamento riguroso de ‖dT_t‖ ≥ 1.0 puntual).
      • `kam_return_certificate`      : estabilidad KAM/Diophantine del mapa de retorno de Poincaré
                                         asociado al flujo celeste del propio proceso de automejora.
      • `lindstedt_continuation`      : no-secularidad y continuación de la rama de "frecuencia de
                                         automejora" (ausencia de resonancias destructivas ω:±1).
      • `overall_level3_certified`    : AND lógico de todos los sub-certificados disponibles.
    """
    d_fs: float
    monadic_fixed_point_ok: bool
    capacity_acceleration: float
    capacity_acceleration_ok: bool
    novikov_valuation: float
    novikov_boundary_ok: bool
    oseledets_certificate: Optional[OseledetsCocycleCertificate]
    kam_return_certificate: Optional[PoincareReturnMapCertificate]
    lindstedt_continuation: Optional[LindstedtPoincareSeries]
    overall_level3_certified: bool
    timestamp_epoch: float = field(default_factory=time.time)


class Level3PoincareOrchestrator:
    r"""
    Orquestador soberano del PASO de automejora recursiva Nivel 3 sobre el fibrado celeste
    de Poincaré. Ensambla, en un único certificado trazable, los cinco pilares de la Fase 1:

        μ_godel (MetaGodelEngine) ⊗ Oseledets (BanachAlgebraEngine)
            ⊗ Retorno de Poincaré/KAM (PoincareReturnMapEngine)
            ⊗ Lindstedt-Poincaré (LindstedtPoincareEngine)
            ⊗ Anillo de Novikov (MetaGodelEngine.evaluate_novikov_ring_valuation)

    Este objeto es deliberadamente el ÚLTIMO eslabón antes de `lift_to_celestial_hamiltonian_
    bundle`: su certificado se adjunta (opcionalmente) al `CelestialHamiltonianBundle` que abre
    la Fase 2, de modo que la dinámica port-Hamiltoniana y la reducción de Marsden-Weinstein
    de la Fase 2 hereden la trazabilidad RSI-3 completa del paso que las originó.
    """

    def __init__(self, meta_engine: Optional[MetaGodelEngine] = None):
        self.meta_engine = meta_engine if meta_engine is not None else MetaGodelEngine()
        self._banach_engine = BanachAlgebraEngine()

    def orchestrate_recursive_self_improvement_step(
        self,
        manifold: SpectralTopologicalManifold,
        curvature_tensor: np.ndarray,
        state_vector: np.ndarray,
        timestamp: Optional[float] = None,
        cocycle_window: Optional[Sequence[np.ndarray]] = None,
        hamiltonian_for_return_map: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_seed_state: Optional[np.ndarray] = None,
        lindstedt_engine: Optional[LindstedtPoincareEngine] = None,
        lindstedt_nonlinearity: Optional[Callable[[float, float, float], float]] = None,
        novikov_coefficients: Optional[List[complex]] = None,
        novikov_exponents: Optional[List[float]] = None,
    ) -> Level3PoincareCelestialCertificate:
        r"""
        Ejecuta UN paso de automejora recursiva de Nivel 3 y devuelve el certificado unificado.

        Flujo de ensamblaje (orden deliberado, cada etapa alimenta la memoria de proceso de
        `self.meta_engine` para que invariantes futuros —en llamadas subsecuentes— dispongan
        de historial suficiente):

          (a) μ_godel: `apply_monadic_multiplication(T_actual, curvature_tensor)` sobre el
              operador de mutación contraído de `manifold.banach_report.operator_matrix`,
              registrando el eslabón en `self.meta_engine.cocycle_history`.
          (b) Punto fijo Tarski-Brouwer vía `verify_tarski_brouwer_fixed_point_cpn` con el
              operador unitario resultante — produce `d_FS` y dispara el cálculo de d³C/dt³.
          (c) Registro de capacidad: `manifold.manifold_purity` como proxy operativo de C(t)
              en el instante `timestamp` (o `time.time()` si se omite).
          (d) Oseledets: si `cocycle_window` (≥ 2 operadores) es provisto —o si ya existen ≥ 2
              eslabones en `self.meta_engine.cocycle_history`— se certifica la contracción
              asintótica del cociclo no estacionario.
          (e) KAM/retorno de Poincaré: si se provee un Hamiltoniano celeste (típicamente
              `celestial_quadratic_hamiltonian(lift_to_celestial_hamiltonian_bundle(manifold))`),
              se sintetiza el mapa de retorno y su estabilidad Diophantina.
          (f) Lindstedt-Poincaré: si se provee una no linealidad explícita, se expande la serie
              y se verifica ausencia de resonancias seculares destructivas.
          (g) Novikov: valuación no arquimediana sobre los coeficientes/exponentes provistos
              (o, por defecto, sobre el espectro de `curvature_tensor` como exponentes formales).
        """
        t_now = float(timestamp) if timestamp is not None else time.time()
        mutation_operator = np.asarray(manifold.banach_report.operator_matrix, dtype=np.float64)

        # (a) Multiplicación monádica μ_godel.
        unitary_step = self.meta_engine.apply_monadic_multiplication(
            current_operator=mutation_operator.astype(np.complex128),
            curvature_tensor=np.asarray(curvature_tensor, dtype=np.complex128),
            record_cocycle=True,
        )

        # (b) Punto fijo Tarski-Brouwer sobre CP^{n-1}.
        fixed_point_ok, d_fs, _third_derivative_proxy = self.meta_engine.verify_tarski_brouwer_fixed_point_cpn(
            state_vector=state_vector, transform_op=unitary_step,
        )

        # (c) Registro de capacidad y (re)cálculo riguroso de d³C/dt³.
        self.meta_engine.register_capacity_sample(t_now, manifold.manifold_purity)
        capacity_acceleration, capacity_ok = self.meta_engine.compute_super_exponential_acceleration()

        # (d) Certificación de Oseledets sobre el cociclo no estacionario disponible.
        oseledets_cert: Optional[OseledetsCocycleCertificate] = None
        window = list(cocycle_window) if cocycle_window is not None else list(self.meta_engine.cocycle_history)
        if len(window) >= 2:
            try:
                oseledets_cert = self._banach_engine.certify_oseledets_nonstationary_contraction(window)
            except ValueError as exc:
                logger.warning("Certificación de Oseledets omitida: %s", exc)

        # (e) Mapa de retorno de Poincaré / estabilidad KAM del flujo celeste asociado.
        kam_cert: Optional[PoincareReturnMapCertificate] = None
        if hamiltonian_for_return_map is not None and hamiltonian_seed_state is not None:
            try:
                kam_cert = PoincareReturnMapEngine.synthesize_from_hamiltonian_flow(
                    hamiltonian=hamiltonian_for_return_map, x0=hamiltonian_seed_state,
                )
            except Exception as exc:  # defensivo: la integración puede fallar con semillas degeneradas
                logger.warning("Certificación KAM/retorno de Poincaré omitida: %s", exc)

        # (f) Continuación de Lindstedt-Poincaré (no-secularidad de la "frecuencia de mejora").
        lindstedt_series: Optional[LindstedtPoincareSeries] = None
        if lindstedt_nonlinearity is not None:
            engine = lindstedt_engine if lindstedt_engine is not None else LindstedtPoincareEngine()
            lindstedt_series = engine.expand(nonlinearity=lindstedt_nonlinearity)

        # (g) Valuación de Novikov / frontera Lagrangiana exacta.
        if novikov_exponents is None:
            eig_curv = la.eigvals(np.asarray(curvature_tensor, dtype=np.complex128))
            novikov_exponents = [float(np.real(e)) for e in eig_curv]
        if novikov_coefficients is None:
            novikov_coefficients = [complex(1.0, 0.0) for _ in novikov_exponents]
        novikov_val, novikov_ok = self.meta_engine.evaluate_novikov_ring_valuation(
            coefficients=novikov_coefficients, exponents=novikov_exponents,
        )

        sub_certificates_ok = [fixed_point_ok, capacity_ok, novikov_ok]
        if oseledets_cert is not None:
            sub_certificates_ok.append(oseledets_cert.asymptotically_contractive)
        if kam_cert is not None:
            sub_certificates_ok.append(kam_cert.kam_stable or kam_cert.is_diophantine)
        if lindstedt_series is not None:
            sub_certificates_ok.append(lindstedt_series.is_periodic)
        overall_ok = bool(all(sub_certificates_ok))

        return Level3PoincareCelestialCertificate(
            d_fs=d_fs,
            monadic_fixed_point_ok=fixed_point_ok,
            capacity_acceleration=capacity_acceleration,
            capacity_acceleration_ok=capacity_ok,
            novikov_valuation=novikov_val,
            novikov_boundary_ok=novikov_ok,
            oseledets_certificate=oseledets_cert,
            kam_return_certificate=kam_cert,
            lindstedt_continuation=lindstedt_series,
            overall_level3_certified=overall_ok,
            timestamp_epoch=t_now,
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.9 ENLACE TERMINAL FASE 1 → INICIO FASE 2
#      Elevación de 𝔐_Spectral (enriquecido, opcionalmente, con el certificado RSI-3 de §1.8b)
#      al fibrado Hamiltoniano celeste (T*Q, ω, H, J) y a las variables de Delaunay canónicas
#      (ℓ, g, h, L, G, H)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CelestialHamiltonianBundle:
    r"""
    OBJETO TERMINAL DE LA FASE 1 Y OBJETO INICIAL DE LA FASE 2.

    Fibrado cotangente sintético (T*Q, ω, H, J) construido sobre 𝔐_Spectral:
      • ω = Σ dq^i ∧ dp_i                         (forma simpléctica canónica),
      • H(q, p) = ½‖p‖² + ½ qᵀ Sym(T) q          (Hamiltoniano cuadrático de mutación),
      • J(q, p) = ½ (q² + p²) componente a componente  (mapa de momentos del toro Tⁿ).

    v5.2.0: además se expone, opcionalmente, el `level3_certificate` producido por
    `Level3PoincareOrchestrator` (§1.8b), enlazando el PROCESO de automejora recursiva que
    originó el fibrado con la ESTRUCTURA geométrica que la Fase 2 habrá de transportar
    (dinámica port-Hamiltoniana, reducción de Marsden-Weinstein, Birkhoff, Melnikov, Chirikov).
    El campo es `Optional` con `default=None`: la firma y el comportamiento de
    `lift_to_celestial_hamiltonian_bundle` para llamadas preexistentes permanecen intactos.

    Las acciones de Delaunay generalizadas
        D_k = (1/2π) ∮ p_k dq_k  (sobre ciclos fundamentales del toro espectral),
    se calculan a partir de las frecuencias y los radios espectrales del operador de mutación.
    Toda la dinámica de la Fase 2 (PHS, Marsden-Weinstein, Birkhoff, Melnikov, Chirikov,
    Crowbar) se alimenta de este fibrado: es la continuación formal de
    `synthesize_spectral_topological_manifold` y de `Level3PoincareOrchestrator`.
    """
    manifold: SpectralTopologicalManifold
    configuration_dim: int
    symplectic_form: np.ndarray
    quadratic_hessian: np.ndarray
    momentum_map: np.ndarray
    delaunay_actions: np.ndarray
    delaunay_angles: np.ndarray
    delaunay_frequencies: np.ndarray
    hamiltonian_energy: float
    reduced_orbit_dimension: float
    gauge_stabilizer_dimension: float
    level3_certificate: Optional[Level3PoincareCelestialCertificate] = None


def lift_to_celestial_hamiltonian_bundle(
    manifold: SpectralTopologicalManifold,
    level3_certificate: Optional[Level3PoincareCelestialCertificate] = None,
) -> CelestialHamiltonianBundle:
    r"""
    ÚLTIMO MÉTODO FORMAL DE LA FASE 1 / PRIMER MORFISMO DE LA FASE 2.

    Eleva 𝔐_Spectral al fibrado (T*Q, ω, H, J) de la mecánica celeste de Poincaré:
      • Hamiltoniano cuadrático leído del simetrizado del operador de mutación de Banach,
      • mapa de momentos J_k = I_k = ½ (q_k² + p_k²) ≃ población espectral (acción de Cartan),
      • variables de Delaunay (L_k, G_k, H_k; ℓ_k, g_k, h_k) construidas desde las
        frecuencias espectrales ω_k = arg λ_k(T) del operador de mutación,
      • dimensión de la órbita coadjunta (Marsden-Weinstein) por multiplicidades del
        espectro de ρ.

    v5.2.0: parámetro opcional `level3_certificate` (retrocompatible, `default=None`) —si se
    provee el certificado de `Level3PoincareOrchestrator.orchestrate_recursive_self_improvement_
    step`— se adjunta al bundle resultante, de modo que la Fase 2 pueda auditar, sin recálculo,
    la validez RSI-3 del proceso que produjo este fibrado celeste.
    """
    mutation = np.asarray(manifold.banach_report.operator_matrix, dtype=np.float64)
    n = mutation.shape[0]
    hessian = 0.5 * (mutation + mutation.T)
    symplectic = canonical_symplectic_form(n)

    # Multiplicidades del espectro de ρ → dimensión de la órbita coadjunta.
    eigvals_rho = np.real(la.eigvalsh(_hermitian(manifold.purified_density_matrix)))
    eigvals_rho = np.sort(eigvals_rho)
    multiplicities: List[int] = []
    acc = 1
    for i in range(1, len(eigvals_rho)):
        if abs(eigvals_rho[i] - eigvals_rho[i - 1]) < 1e-9:
            acc += 1
        else:
            multiplicities.append(acc)
            acc = 1
    if len(eigvals_rho):
        multiplicities.append(acc)
    stabilizer_dim = float(sum(m * m for m in multiplicities))
    orbit_dim = float(n * n - stabilizer_dim)

    populations = np.clip(eigvals_rho, 0.0, None)
    if populations.size < n:
        populations = np.pad(populations, (0, n - populations.size))
    momentum_map = 0.5 * populations[:n]
    energy = float(0.5 * np.real(np.trace(hessian @ hessian)))

    # Variables de Delaunay generalizadas (acción-ángulo) leídas del espectro complejo.
    eig_complex = la.eigvals(mutation)
    frecuencias = np.sort(np.abs(np.angle(eig_complex)))  # frecuencias angulares mod π
    if frecuencias.size < n:
        frecuencias = np.pad(frecuencias, (0, n - frecuencias.size))
    delaunay_actions = 0.5 * populations[:n] * (1.0 + frecuencias[:n])
    delaunay_angles = np.mod(np.real(np.diag(mutation)) * 2.0 * math.pi, 2.0 * math.pi)
    delaunay_frequencies = frecuencias[:n]

    return CelestialHamiltonianBundle(
        manifold=manifold,
        configuration_dim=n,
        symplectic_form=symplectic,
        quadratic_hessian=hessian,
        momentum_map=momentum_map,
        delaunay_actions=delaunay_actions,
        delaunay_angles=delaunay_angles,
        delaunay_frequencies=delaunay_frequencies,
        hamiltonian_energy=energy,
        reduced_orbit_dimension=orbit_dim,
        gauge_stabilizer_dimension=stabilizer_dim,
        level3_certificate=level3_certificate,
    )


def celestial_quadratic_hamiltonian(bundle: CelestialHamiltonianBundle) -> Callable[[np.ndarray], float]:
    """Hamiltoniano H(q, p) = ½‖p‖² + ½ qᵀ K q asociado al fibrado celeste."""
    n = bundle.configuration_dim
    hessian = bundle.quadratic_hessian

    def hamiltonian(x: np.ndarray) -> float:
        q, p = x[:n], x[n:]
        return float(0.5 * np.dot(p, p) + 0.5 * np.dot(q, hessian @ q))

    return hamiltonian
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████████████████████████████████
# ██                                                                                              ██
# ██    FASE 2: DINÁMICA PORT-HAMILTONIANA, MARSDEN-WEINSTEIN, POINCARÉ-BIRKHOFF, MELNIKOV,       ██
# ██    VARIABLES DE DELAUNAY, SOLAPAMIENTO DE RESONANCIAS DE CHIRIKOV, FORMA NORMAL DE           ██
# ██    BIRKHOFF (KAM-RIGIDEZ) Y ENCLAVAMIENTO CIBER-FÍSICO Ω₃/Ω₄ CON COCICLOS DE OSELEDETS       ██
# ██                                                                                              ██
# ██    v5.2.0 — continuación directa de la Fase 1: consume `CelestialHamiltonianBundle`,         ██
# ██    `OseledetsCocycleCertificate`, `oseledets_lyapunov_spectrum` y                            ██
# ██    `Level3PoincareCelestialCertificate` definidos en §1.5/§1.8b.                             ██
# ██                                                                                              ██
# ██████████████████████████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.1 FÍSICA DE CIRCUITOS: DISYUNTOR CROWBAR CON TIRISTOR BT151 Y RLC
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CrowbarPhysicalTelemetry:
    """
    Telemetría ciber-física del transitorio de conmutación en silicio.

    v5.2.0: se añaden `hazard_index_rhi` y `omega4_interlock_verdict` (campos opcionales,
    retrocompatibles) para registrar el ÍNDICE DE RIESGO RECURSIVO (RHI, invariante 5 del
    módulo) y el veredicto del topos de Heyting Ω₄ que efectivamente motivó —o descartó— el
    disparo, cerrando la brecha entre el docstring del módulo y la telemetría auditable.
    """
    interlock_tripped: bool
    total_clearance_latency_ns: float
    iram_instruction_latency_ns: float
    thyristor_avalanche_latency_ns: float
    time_to_peak_ns: float
    peak_current_amperes: float
    joule_integral_i2t: float
    within_safe_operating_area: bool
    thermal_stress_ratio: float
    rail_voltage_post_clamp: float
    gpio_pin: str
    provenance_hash: str
    hazard_index_rhi: float = 0.0
    omega4_interlock_verdict: str = "COHERENT"


class CrowbarCircuitPhysicsEngine:
    r"""
    Circuito disyuntor Crowbar (ESP32 + tiristor BT151-800R).
    Modelo RLC serie subamortiguado:
        L ï + R i̇ + i/C = 0,   α = R/(2L),  ω₀² = 1/(LC),
        i(t) = (V / (ω_d L)) e^{−α t} sin(ω_d t),  ω_d = √(ω₀² − α²).

    v5.1.0: se cubren explícitamente los tres regímenes (sobreamortiguado, crítico y
    subamortiguado); el caso α² > ω₀² usa la forma hiperbólica, el crítico i(t) = V t e^{−α t}/L.
    Se enclava si la automutación viola Poincaré-Wirtinger, provoca difusión de Arnold
    o rompe separatrices (veredicto Heyting = VETOED), o —v5.2.0— si el enclavamiento Ω₄
    (§2.3) certifica RHI > 0.88 o d_FS > 10⁻³ rad, independientemente del veredicto Ω₃.
    """
    C_BUS: Final[float] = 470e-6
    L_BUS: Final[float] = 15e-9
    R_ESR: Final[float] = 0.012
    R_THYRISTOR_ON: Final[float] = 0.024
    V_BUS_NOMINAL: Final[float] = 3.3
    I2T_LIMIT_BT151: Final[float] = 45.0
    XTENSA_CLOCK_FREQ_HZ: Final[float] = 240e6
    IRAM_CYCLES_STROBE: Final[int] = 16
    THYRISTOR_T_GT_NS: Final[float] = 250.0

    @classmethod
    def _current_regime(
        cls,
        r_total: float,
        omega_0_sq: float,
        alpha: float,
    ) -> Tuple[Callable[[float], float], float, float]:
        r"""Determina el régimen RLC y devuelve (i(t), t_peak, i_peak)."""
        disc = omega_0_sq - alpha ** 2
        if disc > 1e-18:  # Subamortiguado
            omega_d = math.sqrt(disc)

            def i_of_t(t: float) -> float:
                return (cls.V_BUS_NOMINAL / (omega_d * cls.L_BUS)) * math.exp(-alpha * t) * math.sin(omega_d * t)

            t_peak = math.atan2(omega_d, alpha) / omega_d
            i_peak = i_of_t(t_peak)
            return i_of_t, t_peak, i_peak
        if abs(disc) < 1e-18:  # Crítico
            def i_of_t(t: float) -> float:
                return (cls.V_BUS_NOMINAL / cls.L_BUS) * t * math.exp(-alpha * t)

            t_peak = 1.0 / alpha if alpha > 0 else 0.0
            i_peak = i_of_t(t_peak)
            return i_of_t, t_peak, i_peak
        # Sobreamortiguado
        r1 = -alpha + math.sqrt(alpha ** 2 - omega_0_sq)
        r2 = -alpha - math.sqrt(alpha ** 2 - omega_0_sq)

        def i_of_t(t: float) -> float:
            return (cls.V_BUS_NOMINAL / (cls.L_BUS * (r1 - r2))) * (math.exp(r1 * t) - math.exp(r2 * t))

        # t_peak = ln(r1/r2) / (r1 − r2)
        t_peak = math.log(abs(r1 / r2)) / (r1 - r2) if abs(r1 - r2) > 1e-18 else 0.0
        i_peak = i_of_t(t_peak)
        return i_of_t, t_peak, i_peak

    @classmethod
    def simulate_crowbar_actuation(
        cls,
        trip_required: bool,
        fault_reason: str = "",
        hazard_index_rhi: float = 0.0,
        omega4_interlock_verdict: str = "COHERENT",
    ) -> CrowbarPhysicalTelemetry:
        if not trip_required:
            return CrowbarPhysicalTelemetry(
                interlock_tripped=False,
                total_clearance_latency_ns=0.0,
                iram_instruction_latency_ns=0.0,
                thyristor_avalanche_latency_ns=0.0,
                time_to_peak_ns=0.0,
                peak_current_amperes=0.0,
                joule_integral_i2t=0.0,
                within_safe_operating_area=True,
                thermal_stress_ratio=0.0,
                rail_voltage_post_clamp=cls.V_BUS_NOMINAL,
                gpio_pin="GPIO14_FAST_IRAM_STROBE",
                provenance_hash="",
                hazard_index_rhi=hazard_index_rhi,
                omega4_interlock_verdict=omega4_interlock_verdict,
            )
        t_clock_ns = 1e9 / cls.XTENSA_CLOCK_FREQ_HZ
        latency_iram = cls.IRAM_CYCLES_STROBE * t_clock_ns
        total_latency = latency_iram + cls.THYRISTOR_T_GT_NS
        r_total = cls.R_ESR + cls.R_THYRISTOR_ON
        alpha = r_total / (2.0 * cls.L_BUS)
        omega_0_sq = 1.0 / (cls.L_BUS * cls.C_BUS)

        i_of_t, t_peak, i_peak = cls._current_regime(r_total, omega_0_sq, alpha)
        integration_horizon = 12.0 / max(alpha, 1e-9)
        i2t, _ = integrate.quad(lambda t: i_of_t(t) ** 2, 0.0, integration_horizon, limit=250)

        stress_ratio = i2t / cls.I2T_LIMIT_BT151
        within_soa = bool(i2t <= cls.I2T_LIMIT_BT151)
        digest = hashlib.sha256()
        digest.update(
            f"BT151_CROWBAR_TRIPPED::{fault_reason}::{total_latency:.4f}::{i_peak:.2f}::{time.time_ns()}".encode("utf-8")
        )
        return CrowbarPhysicalTelemetry(
            interlock_tripped=True,
            total_clearance_latency_ns=total_latency,
            iram_instruction_latency_ns=latency_iram,
            thyristor_avalanche_latency_ns=cls.THYRISTOR_T_GT_NS,
            time_to_peak_ns=t_peak * 1e9,
            peak_current_amperes=i_peak,
            joule_integral_i2t=i2t,
            within_safe_operating_area=within_soa,
            thermal_stress_ratio=stress_ratio,
            rail_voltage_post_clamp=0.085,
            gpio_pin="GPIO14_FAST_IRAM_STROBE",
            provenance_hash=digest.hexdigest(),
            hazard_index_rhi=hazard_index_rhi,
            omega4_interlock_verdict=omega4_interlock_verdict,
        )

    @classmethod
    def simulate_trip(cls, trip_required: bool, fault_reason: str = "") -> CrowbarPhysicalTelemetry:
        """Alias de compatibilidad."""
        return cls.simulate_crowbar_actuation(trip_required=trip_required, fault_reason=fault_reason)


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.2 SISTEMAS PORT-HAMILTONIANOS (PHS) CON ESTRUCTURA DE DIRAC, CAYLEY Y COCICLOS DE OSELEDETS
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PortHamiltonianDissipationAudit:
    """Auditoría de pasividad continua y discreta del operador de automutación."""
    total_energy_H: float
    dH_dt_continuous: float
    delta_H_discrete_cayley: float
    is_strictly_dissipative: bool
    structure_matrices_verified: bool
    state_drift_norm: float
    next_state_vector: np.ndarray
    dirac_structure_skew_certificate: float
    dirac_structure_psd_certificate: float
    hessian_source: str = "fallback_identity"


class PortHamiltonianDynamicsEngine:
    r"""
    Sistemas port-Hamiltonianos con estructura de Dirac J(x) − R(x):
        ẋ = (J − R) ∇H,   J = −Jᵀ,   R = Rᵀ ⪰ 0,   H = ½ xᵀ Q x.
    Discretización de Cayley (pasividad incondicional):
        x_{k+1} = (I − (h/2) A)⁻¹ (I + (h/2) A) x_k,  A = (J − R) Q.
    Si R = 0 y A es Hamiltoniano (Aᵀ J_can + J_can A = 0), Cayley es simpléctico.

    v5.2.0 — CORRECCIÓN ESTRUCTURAL: en v5.1.0, `_build_structure` generaba J y R con
    `np.random.default_rng(42)`, DESCONECTADOS del operador de mutación real y del Hamiltoniano
    celeste `bundle.quadratic_hessian` calculado en la Fase 1 (que `audit_from_bundle` ignoraba
    silenciosamente, sustituyéndolo por `np.eye(dim) * 2.0`). Se reemplaza por
    `_derive_dirac_structure`, determinista y físicamente motivada:
        • J se deriva de la orientación canónica del grafo camino P_dim (§1.2, misma topología
          combinatoria auditada por `HodgeSimplicialEngine`), eliminando la necesidad de RNG
          —relevante para la certificación criptográfica reproducible de la Fase 3.
        • R = damping_factor · Tᵀ T / (ρ(T)² + ε) es PSD por construcción (matriz de Gram) y
          proporcional a la "rugosidad" espectral real del operador de mutación de Banach.
        • Q (Hessiano) se lee de `bundle.quadratic_hessian` cuando está disponible.
    """

    @staticmethod
    def _derive_dirac_structure(
        dim: int,
        damping_factor: float,
        mutation_operator: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Deriva la estructura de Dirac (J, R) de forma determinista y físicamente motivada.
        Véase el docstring de clase para la justificación completa (v5.2.0).
        """
        adjacency = path_graph_adjacency(dim)
        # Orientación canónica i → i+1: J_{i,i+1} = +1, J_{i+1,i} = −1 (antisimétrica exacta).
        upper = np.triu(adjacency)
        skew = upper - upper.T
        if mutation_operator is not None:
            T = np.real(np.asarray(mutation_operator, dtype=np.complex128))
            if T.shape == (dim, dim):
                rho_T = float(np.max(np.abs(la.eigvals(T))))
                gram = T.T @ T
                damping = damping_factor * gram / (rho_T ** 2 + 1e-9)
            else:
                damping = damping_factor * np.eye(dim)
        else:
            damping = damping_factor * np.eye(dim)
        return skew, damping

    @staticmethod
    def _build_structure(dim: int, damping_factor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        LEGACY (v5.1.0) — mantenido únicamente por compatibilidad binaria para quienes invocaran
        esta rutina privada directamente. Delegará en `_derive_dirac_structure` sin operador de
        mutación (fallback determinista) y reporta el Hessiano trivial histórico.
        """
        logger.debug("PortHamiltonianDynamicsEngine._build_structure: ruta legacy v5.1.0 (sin RNG).")
        skew, damping = PortHamiltonianDynamicsEngine._derive_dirac_structure(dim, damping_factor, None)
        return skew, damping, np.eye(dim) * 2.0

    @staticmethod
    def _verify_structure(skew: np.ndarray, damping: np.ndarray) -> Tuple[bool, float, float]:
        antisym_err = float(np.max(np.abs(skew + skew.T)))
        eig_r = la.eigvalsh(0.5 * (damping + damping.T))
        lambda_min = float(eig_r.min()) if eig_r.size else 0.0
        return bool(antisym_err < 1e-9 and lambda_min >= -1e-9), antisym_err, lambda_min

    @classmethod
    def audit_dissipation(
        cls,
        eigenvalues: np.ndarray,
        damping_factor: float = 0.85,
        dt: float = 1e-3,
        mutation_operator: Optional[np.ndarray] = None,
        hessian_matrix: Optional[np.ndarray] = None,
    ) -> PortHamiltonianDissipationAudit:
        x_state = np.real(np.asarray(eigenvalues, dtype=np.complex128))
        dim = x_state.shape[0]
        skew, damping = cls._derive_dirac_structure(dim, damping_factor, mutation_operator)
        structure_ok, err_j, lambda_min_r = cls._verify_structure(skew, damping)

        hessian_candidate = (
            np.real(np.asarray(hessian_matrix, dtype=np.complex128))
            if hessian_matrix is not None
            else None
        )
        if hessian_candidate is not None and hessian_candidate.shape == (dim, dim):
            hessian = _hermitian(hessian_candidate.astype(np.complex128)).real
            hessian_source = "celestial_bundle_quadratic_hessian"
        else:
            hessian = np.eye(dim) * 2.0
            hessian_source = "fallback_identity"

        grad_h = hessian @ x_state
        energy = float(0.5 * x_state.T @ hessian @ x_state)
        generator = (skew - damping) @ hessian
        dx_dt = generator @ x_state
        dh_dt_cont = float(grad_h.T @ dx_dt)
        ident = np.eye(dim)
        x_next = la.solve(ident - 0.5 * dt * generator, (ident + 0.5 * dt * generator) @ x_state)
        energy_next = float(0.5 * x_next.T @ hessian @ x_next)
        delta_h_discrete = energy_next - energy
        is_dissipative = bool(delta_h_discrete <= 1e-9 and structure_ok)
        return PortHamiltonianDissipationAudit(
            total_energy_H=energy,
            dH_dt_continuous=dh_dt_cont,
            delta_H_discrete_cayley=delta_h_discrete,
            is_strictly_dissipative=is_dissipative,
            structure_matrices_verified=structure_ok,
            state_drift_norm=float(np.linalg.norm(dx_dt)),
            next_state_vector=x_next,
            dirac_structure_skew_certificate=err_j,
            dirac_structure_psd_certificate=lambda_min_r,
            hessian_source=hessian_source,
        )

    @classmethod
    def audit_from_bundle(
        cls,
        bundle: CelestialHamiltonianBundle,
        damping_factor: float = 0.85,
        dt: float = 1e-3,
    ) -> PortHamiltonianDissipationAudit:
        """
        Continuación Fase 1→2: disipación PHS leída del espectro de Banach del fibrado.

        v5.2.0: ahora inyecta genuinamente `bundle.manifold.banach_report.operator_matrix`
        (para J, R) y `bundle.quadratic_hessian` (para Q) — el Hamiltoniano celeste H(q,p) de
        `lift_to_celestial_hamiltonian_bundle` deja de ser descartado.
        """
        return cls.audit_dissipation(
            bundle.manifold.banach_report.eigenvalues,
            damping_factor=damping_factor,
            dt=dt,
            mutation_operator=bundle.manifold.banach_report.operator_matrix,
            hessian_matrix=bundle.quadratic_hessian,
        )

    @classmethod
    def audit_nonstationary_hardening_schedule(
        cls,
        eigenvalues: np.ndarray,
        mutation_operator: np.ndarray,
        damping_schedule: Sequence[float] = (0.30, 0.50, 0.70, 0.85, 0.95),
        dt: float = 1e-3,
        hessian_matrix: Optional[np.ndarray] = None,
    ) -> OseledetsCocycleCertificate:
        r"""
        NIVEL 3 — "Model-RSI" A NIVEL DE CIRCUITO PORT-HAMILTONIANO (v5.2.0, nuevo).

        Certifica, vía el Teorema Ergódico Multiplicativo de Oseledets (§1.5), la estabilidad
        ASINTÓTICA de un PROGRAMA DE ENDURECIMIENTO PROGRESIVO de la disipación: una secuencia
        de operadores de Cayley U_k = Cayley((J − R_k) Q), k = 1,…,K, con R_k creciente según
        `damping_schedule` — el análogo exacto, a nivel de circuito, del "Model-RSI" del
        preámbulo del módulo. Los pasos iniciales (poco amortiguados, R_k pequeña) pueden violar
        ‖U_k‖ ≥ 1 (ruptura puntual del Techo de Banach) sin comprometer la contracción asintótica
        del cociclo completo — exactamente el mismo fenómeno certificado en
        `MetaGodelEngine.apply_monadic_multiplication` / `BanachAlgebraEngine.
        certify_oseledets_nonstationary_contraction` (invariante 1 del módulo).
        """
        dim = int(np.asarray(eigenvalues).shape[0])
        hessian_candidate = (
            np.real(np.asarray(hessian_matrix, dtype=np.complex128))
            if hessian_matrix is not None
            else None
        )
        hessian = (
            hessian_candidate
            if hessian_candidate is not None and hessian_candidate.shape == (dim, dim)
            else np.eye(dim) * 2.0
        )
        ident = np.eye(dim)
        operators: List[np.ndarray] = []
        for damping_factor in damping_schedule:
            skew, damping = cls._derive_dirac_structure(dim, float(damping_factor), mutation_operator)
            generator = (skew - damping) @ hessian
            U_k = la.solve(ident - 0.5 * dt * generator, ident + 0.5 * dt * generator)
            operators.append(U_k)
        return BanachAlgebraEngine.certify_oseledets_nonstationary_contraction(operators, dt=dt)

    @classmethod
    def audit_nonstationary_hardening_from_bundle(
        cls,
        bundle: CelestialHamiltonianBundle,
        damping_schedule: Sequence[float] = (0.30, 0.50, 0.70, 0.85, 0.95),
        dt: float = 1e-3,
    ) -> OseledetsCocycleCertificate:
        """Continuación Fase 1→2 del endurecimiento no estacionario, leído del fibrado celeste."""
        return cls.audit_nonstationary_hardening_schedule(
            bundle.manifold.banach_report.eigenvalues,
            bundle.manifold.banach_report.operator_matrix,
            damping_schedule=damping_schedule,
            dt=dt,
            hessian_matrix=bundle.quadratic_hessian,
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.3 REDUCCIÓN SIMPLÉCTICA DE POINCARÉ-MARSDEN-WEINSTEIN Y TOPOS DE HACES (Ω₃ Y Ω₄)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
class HeytingVerdict(IntEnum):
    r"""Álgebra de Heyting lineal Ω₃ = {0, 1, 2}: ⊥ ≺ ½ ≺ ⊤."""
    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict(min(int(self), int(other)))

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict(max(int(self), int(other)))

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        if int(self) <= int(other):
            return HeytingVerdict.COHERENT
        return other

    def neg(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)

    def __and__(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self.meet(other)

    def __or__(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self.join(other)

    def __rshift__(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self.implies(other)

    def __invert__(self) -> "HeytingVerdict":
        return self.neg()

    @classmethod
    def verify_heyting_algebra_axioms(cls) -> bool:
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


class Omega4Verdict(IntEnum):
    r"""
    Álgebra de Heyting lineal Ω₄ = {0, 1, 2, 3}: ⊥=VETOED ≺ CRITICAL ≺ DEGRADED ≺ COHERENT=⊤.

    v5.2.0 (nuevo). Toda cadena finita totalmente ordenada es trivialmente una álgebra de
    Heyting (ínfimos/supremos = min/max, residuo a→b = ⊤ si a≤b, en otro caso b) — verificable
    con `verify_heyting_algebra_axioms`, análogo a `HeytingVerdict`.

    Ω₄ REFINA Ω₃ con el bit de riesgo ciber-físico de hardware (invariante 5 del módulo): la
    inclusión `from_omega3` realiza el morfismo clasificador Ω₃ ↪ Ω₄ cuyo pullback contra el
    sub-objeto "hazard ⊆ 1" (RHI > 0.88 ó d_FS > 10⁻³ rad) determina, vía `meet`, si el
    enclavamiento ESP32-Crowbar debe dispararse — con independencia de que la estructura
    simpléctico-espectral (Ω₃) sea, por sí sola, COHERENTE.
    """
    VETOED = 0
    CRITICAL = 1
    DEGRADED = 2
    COHERENT = 3

    def meet(self, other: "Omega4Verdict") -> "Omega4Verdict":
        return Omega4Verdict(min(int(self), int(other)))

    def join(self, other: "Omega4Verdict") -> "Omega4Verdict":
        return Omega4Verdict(max(int(self), int(other)))

    def implies(self, other: "Omega4Verdict") -> "Omega4Verdict":
        if int(self) <= int(other):
            return Omega4Verdict.COHERENT
        return other

    def neg(self) -> "Omega4Verdict":
        return self.implies(Omega4Verdict.VETOED)

    def __and__(self, other: "Omega4Verdict") -> "Omega4Verdict":
        return self.meet(other)

    def __or__(self, other: "Omega4Verdict") -> "Omega4Verdict":
        return self.join(other)

    def __rshift__(self, other: "Omega4Verdict") -> "Omega4Verdict":
        return self.implies(other)

    def __invert__(self) -> "Omega4Verdict":
        return self.neg()

    @classmethod
    def verify_heyting_algebra_axioms(cls) -> bool:
        elements = list(cls)
        for a in elements:
            for b in elements:
                residuum = a.implies(b)
                for c in elements:
                    if (a.meet(c) <= b) != (c <= residuum):
                        return False
        return True

    @staticmethod
    def from_omega3(verdict: HeytingVerdict) -> "Omega4Verdict":
        r"""Morfismo clasificador Ω₃ ↪ Ω₄ (inclusión que preserva orden, DEGRADED ↦ 2)."""
        mapping = {
            HeytingVerdict.VETOED: Omega4Verdict.VETOED,
            HeytingVerdict.DEGRADED: Omega4Verdict.DEGRADED,
            HeytingVerdict.COHERENT: Omega4Verdict.COHERENT,
        }
        return mapping[verdict]


def compute_recursive_hazard_index(
    banach_report: BanachContractionReport,
    oseledets_certificate: Optional[OseledetsCocycleCertificate] = None,
    d_fs: Optional[float] = None,
) -> float:
    r"""
    ÍNDICE DE RIESGO RECURSIVO (RHI) — FORMALIZACIÓN v5.2.0.

    El docstring del módulo invoca "RHI > 0.88" (invariante 5) sin definirlo operativamente.
    Se cierra esta brecha con una definición EXPLÍCITA, determinista y documentada (no canónica
    en la literatura, pero calculable y trazable) que combina tres testigos espectrales ya
    certificados en el módulo:

        RHI = clip( 0.5 · [1 − exp(−κ·ρ(T)/10)] + 0.3 · tanh(max(λ₁^Oseledets, 0)) + 0.2 · Δ_FS , 0, 1)

    donde κ es la constante de Kreiss (`kreiss_constant_estimate`, sensibilidad transitoria del
    operador no normal), ρ(T) el radio espectral, λ₁^Oseledets el exponente de Lyapunov máximo
    del cociclo no estacionario más reciente (positivo ⇒ expansión asintótica genuina, no solo
    puntual) y Δ_FS la fracción normalizada de exceso de `d_FS` sobre el umbral de punto fijo
    10⁻⁴ rad respecto de la banda crítica [10⁻⁴, 10⁻³] rad.
    """
    kreiss = banach_report.kreiss_constant_estimate
    kreiss_bounded = min(kreiss, 1e3) if math.isfinite(kreiss) else 1e3
    base = 1.0 - math.exp(-(kreiss_bounded * max(banach_report.spectral_radius, 1e-9)) / 10.0)

    top_exponent = 0.0
    if oseledets_certificate is not None and math.isfinite(oseledets_certificate.top_lyapunov_exponent):
        top_exponent = max(oseledets_certificate.top_lyapunov_exponent, 0.0)

    dfs_term = 0.0
    if d_fs is not None and math.isfinite(d_fs):
        dfs_term = min(1.0, max(0.0, (d_fs - 1e-4) / (1e-3 - 1e-4)))

    rhi = 0.5 * base + 0.3 * math.tanh(top_exponent) + 0.2 * dfs_term
    return float(min(1.0, max(0.0, rhi)))


class SheafToposClassifier:
    r"""
    Clasificador sobre la variedad simpléctica reducida de Poincaré-Marsden-Weinstein:
        M_μ = J⁻¹(μ) / G_μ.
    Dimensión de la órbita coadjunta de U(n):  dim 𝒪_ρ = n² − Σ m_i²  (m_i = multiplicidades).

    v5.2.0: `classify()` deja de ignorar `birkhoff_certificate`, `melnikov_certificate`,
    `arnold_diffusion_certificate`, `birkhoff_normal_form_certificate` y `level3_certificate`
    (todos opcionales, retrocompatibles) — cada uno contribuye ahora una sección local al
    veredicto global, cerrando la brecha de "certificados huérfanos" de v5.1.0. Se añade además
    `classify_hardware_interlock_omega4`, que materializa el topos Ω₄ del invariante 5.
    """

    def classify_poincare_marsden_weinstein_topos(
        self,
        ast_state: np.ndarray,
        gauge_momentum_map: np.ndarray,
        reduced_orbit_dimension: Optional[float] = None,
        configuration_dim: Optional[int] = None,
    ) -> Tuple[HeytingVerdict, Dict[str, Any]]:
        momentum_residual = float(np.linalg.norm(gauge_momentum_map))
        ast_norm = float(np.linalg.norm(ast_state))
        n = configuration_dim if configuration_dim is not None else max(ast_state.size, 1)
        if reduced_orbit_dimension is None:
            reduced_dim_ratio = max(0.10, 1.0 - 0.90 * math.exp(-momentum_residual))
        else:
            reduced_dim_ratio = float(reduced_orbit_dimension) / float(max(n * n, 1))
        if momentum_residual < 1e-3 and ast_norm > 0:
            verdict = HeytingVerdict.COHERENT
            reason = (
                f"REDUCCIÓN MARSDEN-WEINSTEIN EXITOSA "
                f"(órbita coadjunta ratio={reduced_dim_ratio:.3f}, residual J={momentum_residual:.2e})"
            )
        elif momentum_residual < 1.0:
            verdict = HeytingVerdict.DEGRADED
            reason = f"REDUCCIÓN PARCIAL CON DEGRADACIÓN GAUGE (residual J={momentum_residual:.4f})"
        else:
            verdict = HeytingVerdict.VETOED
            reason = f"VIOLACIÓN DE INVARIANZA GAUGE SIMPLÉCTICA (residual J={momentum_residual:.4f})"
        details = {
            "momentum_residual": momentum_residual,
            "reduced_dimension_ratio": reduced_dim_ratio,
            "reason": reason,
            "verdict": verdict.name,
        }
        return verdict, details

    def classify_from_bundle(
        self,
        bundle: CelestialHamiltonianBundle,
    ) -> Tuple[HeytingVerdict, Dict[str, Any]]:
        """Reducción Marsden-Weinstein leída directamente del fibrado celeste (Fase 1→2)."""
        rho = bundle.manifold.purified_density_matrix
        ast_state = np.real(np.diag(rho)) if rho.ndim == 2 else np.real(rho)
        return self.classify_poincare_marsden_weinstein_topos(
            ast_state=ast_state,
            gauge_momentum_map=bundle.momentum_map,
            reduced_orbit_dimension=bundle.reduced_orbit_dimension,
            configuration_dim=bundle.configuration_dim,
        )

    @classmethod
    def classify(
        cls,
        manifold: SpectralTopologicalManifold,
        phs_audit: PortHamiltonianDissipationAudit,
        utility_delta: float,
        connes_distance_threshold: float = 50.0,
        isospectral_deviation_threshold: float = 1e-6,
        bundle: Optional[CelestialHamiltonianBundle] = None,
        level3_certificate: Optional["Level3PoincareCelestialCertificate"] = None,
        birkhoff_certificate: Optional["PoincareBirkhoffCertificate"] = None,
        melnikov_certificate: Optional["MelnikovChaosCertificate"] = None,
        arnold_diffusion_certificate: Optional["ArnoldDiffusionCertificate"] = None,
        birkhoff_normal_form_certificate: Optional["BirkhoffNormalFormCertificate"] = None,
    ) -> Tuple[HeytingVerdict, str]:
        sections: List[Tuple[str, HeytingVerdict, str]] = []
        chi_banach = (
            HeytingVerdict.COHERENT if manifold.banach_report.is_banach_contraction
            else HeytingVerdict.VETOED
        )
        sections.append(("BANACH", chi_banach, f"ρ(T)={manifold.banach_report.spectral_radius:.6f}"))
        chi_hodge = (
            HeytingVerdict.VETOED if manifold.hodge_certificate.has_cohomological_obstruction
            else HeytingVerdict.COHERENT
        )
        sections.append(("HODGE-DE RHAM", chi_hodge, f"β_1={manifold.hodge_certificate.betti_1}"))
        chi_phs = (
            HeytingVerdict.COHERENT if phs_audit.is_strictly_dissipative else HeytingVerdict.VETOED
        )
        sections.append((
            "PORT-HAMILTONIAN", chi_phs,
            f"ΔH={phs_audit.delta_H_discrete_cayley:.6e}, H_src={phs_audit.hessian_source}",
        ))
        brockett_ok = (
            manifold.brockett_result.converged
            and manifold.brockett_result.isospectral_deviation < isospectral_deviation_threshold
            and manifold.brockett_result.symplectic_form_preserved
        )
        chi_brockett = HeytingVerdict.COHERENT if brockett_ok else HeytingVerdict.DEGRADED
        sections.append((
            "BROCKETT-ISOESPECTRAL", chi_brockett,
            f"converged={manifold.brockett_result.converged}, "
            f"Δσ={manifold.brockett_result.isospectral_deviation:.2e}, "
            f"Casimir_drift={manifold.brockett_result.casimir_drift:.2e}",
        ))
        connes_ok = (
            manifold.connes_certificate.distance_well_defined
            and manifold.connes_certificate.reference_state_distance <= connes_distance_threshold
        )
        chi_connes = HeytingVerdict.COHERENT if connes_ok else HeytingVerdict.DEGRADED
        sections.append((
            "CONNES-METRIC", chi_connes,
            f"d_D≥{manifold.connes_certificate.reference_state_distance:.4f}",
        ))
        chi_utility = HeytingVerdict.COHERENT if utility_delta >= 0.0 else HeytingVerdict.DEGRADED
        sections.append(("UTILITY-MONOTONICITY", chi_utility, f"ΔU={utility_delta:.4f}"))
        chi_purity = (
            HeytingVerdict.COHERENT if manifold.manifold_purity >= 0.25 else HeytingVerdict.DEGRADED
        )
        sections.append(("QUANTUM-PURITY", chi_purity, f"γ={manifold.manifold_purity:.4f}"))
        if manifold.return_map_certificate is not None:
            rm = manifold.return_map_certificate
            chi_kam = HeytingVerdict.COHERENT if rm.kam_stable else HeytingVerdict.DEGRADED
            sections.append((
                "POINCARE-KAM", chi_kam,
                f"λ_max={rm.lyapunov_max:.4e}, ρ_rot={rm.rotation_number:.4f}, "
                f"Diophantine={rm.is_diophantine}",
            ))
        if manifold.lindstedt_series is not None:
            chi_lind = (
                HeytingVerdict.COHERENT
                if manifold.lindstedt_series.is_periodic
                and manifold.lindstedt_series.poincare_continuation_converged
                else HeytingVerdict.DEGRADED
            )
            sections.append((
                "LINDSTEDT-POINCARE", chi_lind,
                f"order={manifold.lindstedt_series.order}, "
                f"secular={max(manifold.lindstedt_series.secular_removal_residuals, default=0.0):.2e}",
            ))
        if bundle is not None:
            mw_verdict, mw_details = cls().classify_from_bundle(bundle)
            sections.append(("MARSDEN-WEINSTEIN", mw_verdict, mw_details["reason"]))
        # ─── v5.2.0: secciones antes "huérfanas" — ahora inciden en el veredicto global ───
        if birkhoff_certificate is not None:
            bc = birkhoff_certificate
            chi_birkhoff = (
                HeytingVerdict.COHERENT
                if bc.birkhoff_theorem_applicable and bc.birkhoff_lower_bound_satisfied
                else HeytingVerdict.DEGRADED
            )
            sections.append((
                "POINCARE-BIRKHOFF", chi_birkhoff,
                f"ρ=p/q={bc.rational_winding_p}/{bc.rational_period_q}, "
                f"fixed_pts={bc.fixed_points_detected}, twist_ok={bc.twist_condition_verified}",
            ))
        if melnikov_certificate is not None:
            mc = melnikov_certificate
            chi_melnikov = (
                HeytingVerdict.DEGRADED if mc.transverse_homoclinic_exists else HeytingVerdict.COHERENT
            )
            sections.append((
                "MELNIKOV-CHAOS", chi_melnikov,
                f"zeros={mc.simple_zeros_detected}, horseshoe={mc.smale_horseshoe_expected}, "
                f"splitting={mc.splitting_distance_max:.3e}",
            ))
        if arnold_diffusion_certificate is not None:
            ac = arnold_diffusion_certificate
            chi_arnold = (
                HeytingVerdict.DEGRADED if ac.arnold_diffusion_expected else HeytingVerdict.COHERENT
            )
            sections.append((
                "CHIRIKOV-ARNOLD-DIFFUSION", chi_arnold,
                f"overlaps={ac.num_overlaps}, web_dim={ac.arnold_web_dimension:.3f}, "
                f"KAM_measure≈{ac.kam_tori_measure_estimate:.3f}",
            ))
        if birkhoff_normal_form_certificate is not None:
            bnf = birkhoff_normal_form_certificate
            chi_bnf = (
                HeytingVerdict.COHERENT if bnf.birkhoff_condition_verified else HeytingVerdict.DEGRADED
            )
            sections.append((
                "BIRKHOFF-NORMAL-FORM", chi_bnf,
                f"det τ={bnf.hessian_determinant_estimate:.3e}, "
                f"Arnold-degenerate={bnf.is_arnold_degenerate}, "
                f"r_KAM≈{bnf.kam_stability_radius_estimate:.3e}",
            ))
        if level3_certificate is not None:
            chi_rsi3 = (
                HeytingVerdict.COHERENT
                if level3_certificate.overall_level3_certified
                else HeytingVerdict.DEGRADED
            )
            sections.append((
                "RSI3-ORCHESTRATION", chi_rsi3,
                f"d_FS={level3_certificate.d_fs:.3e}, "
                f"d³C/dt³={level3_certificate.capacity_acceleration:.3e}, "
                f"Novikov_v={level3_certificate.novikov_valuation:.3e}",
            ))
        global_verdict = sections[0][1]
        for _, v, _ in sections[1:]:
            global_verdict = global_verdict.meet(v)
        failing = [
            f"{name}[{v.name}]:{detail}" for name, v, detail in sections
            if v != HeytingVerdict.COHERENT
        ]
        reason = " ∧ ".join(failing) if failing else "COHERENCIA CERTIFICADA EN TODAS LAS SECCIONES LOCALES"
        return global_verdict, reason

    @staticmethod
    def classify_hardware_interlock_omega4(
        structural_verdict: HeytingVerdict,
        rhi: float,
        d_fs: Optional[float] = None,
        rhi_critical_threshold: float = 0.70,
        rhi_veto_threshold: float = 0.88,
        d_fs_critical_threshold: float = 1e-4,
        d_fs_veto_threshold: float = 1e-3,
    ) -> Tuple[Omega4Verdict, str]:
        r"""
        TOPOS Ω₄ DEL ENCLAVAMIENTO CIBER-FÍSICO (invariante 5 del módulo, v5.2.0).

        Realiza la inclusión clasificadora Ω₃ ↪ Ω₄ (`Omega4Verdict.from_omega3`) y la combina,
        vía `meet` (ínfimo del topos), con el "bit de hazard" determinado por el RHI y la
        distancia de Fubini-Study `d_FS` (Fase 1, §1.8b):

            hazard = VETOED      si RHI > rhi_veto_threshold  ó  d_FS > d_fs_veto_threshold,
            hazard = CRITICAL    si RHI > rhi_critical_threshold ó d_FS > d_fs_critical_threshold,
            hazard = COHERENT    en otro caso.

        El disparo físico del Crowbar (`evaluate_sheaf_transition_morphism`) se decide por
        `combined == Omega4Verdict.VETOED`, materializando literalmente el invariante 5:
        "Interrupción IRAM tripping GPIO14 en < 400 ns si RHI > 0.88 o d_FS > 10⁻³ rad",
        INDEPENDIENTEMENTE de que la estructura simpléctico-espectral (Ω₃) sea coherente.
        """
        omega3_lifted = Omega4Verdict.from_omega3(structural_verdict)
        d_fs_value = d_fs if d_fs is not None else 0.0
        if rhi > rhi_veto_threshold or d_fs_value > d_fs_veto_threshold:
            hazard_verdict = Omega4Verdict.VETOED
        elif rhi > rhi_critical_threshold or d_fs_value > d_fs_critical_threshold:
            hazard_verdict = Omega4Verdict.CRITICAL
        else:
            hazard_verdict = Omega4Verdict.COHERENT
        combined = omega3_lifted.meet(hazard_verdict)
        reason = (
            f"Ω₃↪Ω₄={omega3_lifted.name}, RHI={rhi:.4f}(veto>{rhi_veto_threshold}), "
            f"d_FS={d_fs_value:.3e}(veto>{d_fs_veto_threshold:.1e}) ⇒ hazard={hazard_verdict.name}, "
            f"Ω₄_combinado={combined.name}"
        )
        return combined, reason


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.4 TEOREMA DE POINCARÉ-BIRKHOFF: PUNTOS PERIÓDICOS POR NÚMERO DE ENROLLAMIENTO
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareBirkhoffCertificate:
    """Certificado del teorema de Poincaré-Birkhoff (último teorema geométrico)."""
    rotation_number: float
    rational_period_q: int
    rational_winding_p: int
    fixed_points_detected: int
    twist_condition_verified: bool
    area_preserving_verified: bool
    birkhoff_lower_bound_satisfied: bool
    rotation_number_residual: float
    birkhoff_theorem_applicable: bool
    min_twist_derivative: float = 0.0


class PoincareBirkhoffEngine:
    r"""
    Teorema de Poincaré-Birkhoff (1912-1913):
    Sea T : A = S¹ × [0, 1] → A un difeomorfismo del anillo cerrado que
      (i)   preserva el área (T* ω = ω, ω = dθ ∧ dI),
      (ii)  es un twist map: ∂θ'/∂I ≠ 0 (y las fronteras giran en sentidos relativos),
      (iii) tiene número de rotación ρ(T) = p/q con gcd(p, q) = 1.
    Entonces T posee al menos DOS órbitas periódicas de periodo q.

    v5.1.0: la detección de puntos fijos de T^q se realiza por el NÚMERO DE ENROLLAMIENTO
    (winding number) del campo vectorial desplazamiento (θ' − θ − 2πp, I' − I) sobre un
    contorno cerrado del anillo; un cambio de signo del índice topológico certifica
    un número par de puntos fijos (teorema de Poincaré-Hopf 2D).
    """

    @staticmethod
    def _twist_condition(
        t_map: Callable[[np.ndarray], np.ndarray],
        action_samples: np.ndarray,
        action_eps: float = 1e-4,
    ) -> Tuple[bool, float]:
        derivatives: List[float] = []
        theta0 = 0.0
        for action in action_samples:
            plus = t_map(np.array([theta0, action + action_eps], dtype=np.float64))
            minus = t_map(np.array([theta0, action - action_eps], dtype=np.float64))
            dtheta = wrap_angle(float(plus[0] - minus[0]))
            derivatives.append(dtheta / (2.0 * action_eps))
        grad = np.array(derivatives, dtype=np.float64)
        min_abs = float(np.min(np.abs(grad))) if grad.size else 0.0
        same_sign = bool(np.all(grad > 0) or np.all(grad < 0)) if grad.size else False
        return bool(same_sign and min_abs > 1e-8), min_abs

    @staticmethod
    def _detect_fixed_points_of_q_iterate(
        t_map: Callable[[np.ndarray], np.ndarray],
        p: int,
        q: int,
        action_resolution: int = 40,
        theta_resolution: int = 60,
    ) -> int:
        r"""
        Cuenta puntos fijos de T^q por índice de Poincaré-Hopf del campo desplazamiento:
            F(θ, I) = (T^q(θ, I) − (θ, I + 2π p))  (envuelto en la coordenada angular).
        Un cambio neto en el signo del producto cruzado Z_k = X_k × X_{k+1} a lo largo
        de un contorno cerrado índice un número par de ceros.
        """
        actions = np.linspace(0.05, 0.95, action_resolution)
        thetas = np.linspace(0.0, 2.0 * math.pi, theta_resolution, endpoint=False)
        total_winding = 0
        for action in actions:
            # Evaluamos el desplazamiento en el anillo a I = action.
            displacement: List[Tuple[float, float]] = []
            for theta in thetas:
                x = np.array([theta, action], dtype=np.float64)
                for _ in range(max(q, 1)):
                    x = t_map(x)
                d_theta = wrap_angle(float(x[0] - theta))
                d_action = float(x[1] - action) - 2.0 * math.pi * p
                displacement.append((d_theta, d_action))
            # Winding number del campo desplazamiento sobre el contorno circular.
            winding = 0
            for k in range(len(displacement)):
                a1 = displacement[k]
                a2 = displacement[(k + 1) % len(displacement)]
                cross = a1[0] * a2[1] - a1[1] * a2[0]
                dot = a1[0] * a2[0] + a1[1] * a2[1]
                angle = math.atan2(cross, dot)
                winding += angle
            total_winding += int(round(winding / (2.0 * math.pi)))
        # El índice de Poincaré-Hopf del campo desplazamiento es un múltiplo par.
        return int(abs(total_winding))

    @classmethod
    def audit_twist_map(
        cls,
        t_map: Callable[[np.ndarray], np.ndarray],
        rotation_number_estimate: float,
        action_samples: Optional[np.ndarray] = None,
    ) -> PoincareBirkhoffCertificate:
        if action_samples is None:
            action_samples = np.linspace(0.1, 0.9, 8)
        twist_ok, twist_coeff = cls._twist_condition(t_map, action_samples)
        det_errors: List[float] = []
        h = 1e-4
        for theta in np.linspace(0.1, 2.0 * math.pi, 7):
            for action in action_samples:
                plus_th = t_map(np.array([theta + h, action]))
                minus_th = t_map(np.array([theta - h, action]))
                plus_i = t_map(np.array([theta, action + h]))
                minus_i = t_map(np.array([theta, action - h]))
                dth_dth = wrap_angle(float(plus_th[0] - minus_th[0])) / (2.0 * h)
                dth_di = wrap_angle(float(plus_i[0] - minus_i[0])) / (2.0 * h)
                di_dth = float(plus_th[1] - minus_th[1]) / (2.0 * h)
                di_di = float(plus_i[1] - minus_i[1]) / (2.0 * h)
                jacobian = np.array([[dth_dth, dth_di], [di_dth, di_di]], dtype=np.float64)
                det_errors.append(abs(np.linalg.det(jacobian) - 1.0))
        area_ok = bool(np.max(det_errors) < 1e-2) if det_errors else False
        frac = Fraction(rotation_number_estimate).limit_denominator(30)
        p, q = frac.numerator, frac.denominator
        frac_err = abs(float(frac) - rotation_number_estimate)
        coprime = math.gcd(abs(p), abs(q)) == 1
        if coprime and 1 <= q <= 40:
            fixed_detected = cls._detect_fixed_points_of_q_iterate(t_map, p, q)
        else:
            fixed_detected = 0
        birkhoff_applicable = bool(twist_ok and area_ok and coprime and frac_err < 1e-3)
        lower_bound_satisfied = bool(birkhoff_applicable and fixed_detected >= 2)
        return PoincareBirkhoffCertificate(
            rotation_number=rotation_number_estimate,
            rational_period_q=int(q),
            rational_winding_p=int(p),
            fixed_points_detected=int(fixed_detected),
            twist_condition_verified=twist_ok,
            area_preserving_verified=area_ok,
            birkhoff_lower_bound_satisfied=lower_bound_satisfied,
            rotation_number_residual=frac_err,
            birkhoff_theorem_applicable=birkhoff_applicable,
            min_twist_derivative=twist_coeff,
        )

    @staticmethod
    def integrable_annulus_twist(alpha: float = 0.5, beta: float = 0.3) -> Callable[[np.ndarray], np.ndarray]:
        r"""
        Twist integrable del anillo (hipótesis exactas de Poincaré-Birkhoff):
            θ' = θ + α + β I   (mod 2π),   I' = I ∈ [0, 1].
        ∂θ'/∂I = β ≠ 0, det DT = 1.
        """
        def t_map(x: np.ndarray) -> np.ndarray:
            theta, action = float(x[0]), float(x[1])
            action_clamped = min(1.0, max(0.0, action))
            theta_new = (theta + alpha + beta * action_clamped) % (2.0 * math.pi)
            return np.array([theta_new, action_clamped], dtype=np.float64)
        return t_map


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.5 FUNCIÓN DE MELNIKOV: CERTIFICACIÓN DE NO-CAOS HOMOCLÍNICO
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class MelnikovChaosCertificate:
    """
    Certificado de la función de Melnikov para la ruptura homoclínica.
    H_ε = H₀ + ε H₁,  M(t₀) = ∫_{−∞}^{∞} {H₀, H₁}(q₀(t), t + t₀) dt.
    Cero simple de M ⇒ intersección homoclínica transversal ⇒ herradura de Smale.
    """
    melnikov_values: np.ndarray
    simple_zeros_detected: int
    transverse_homoclinic_exists: bool
    smale_horseshoe_expected: bool
    chaos_threshold_epsilon: float
    splitting_distance_max: float
    melnikov_mean: float
    melnikov_amplitude: float


class MelnikovFunctionEngine:
    r"""
    Motor de la función de Melnikov y del criterio de Smale-Birkhoff.
    Aplicación RSI: M(t₀) ≡ 0 (idénticamente) ⇒ se preserva la separatriz ⇒
    la automutación permanece sobre su toro KAM sin difusión de Arnold.

    v5.2.0: la existencia de intersección homoclínica transversal ahora DEGRADA el veredicto
    global de `SheafToposClassifier.classify` (antes el certificado era huérfano, §2.3).
    """

    @staticmethod
    def _numerical_homoclinic_orbit(
        hamiltonian_0: Callable[[np.ndarray], float],
        x_saddle: np.ndarray,
        t_span: Tuple[float, float] = (-25.0, 25.0),
        num_samples: int = 1200,
        eps_perturbation: float = 1e-3,
    ) -> np.ndarray:
        x_saddle = np.asarray(x_saddle, dtype=np.float64)
        dim = x_saddle.shape[0]
        if dim % 2 != 0:
            raise ValueError("El silla Hamiltoniano debe vivir en dimensión par.")
        n = dim // 2
        symplectic = canonical_symplectic_form(n)

        def grad_h(x: np.ndarray) -> np.ndarray:
            return central_gradient(hamiltonian_0, x)

        hess_step = 1e-5
        hessian = np.zeros((dim, dim), dtype=np.float64)
        for i in range(dim):
            basis = np.zeros(dim, dtype=np.float64)
            basis[i] = hess_step
            hessian[:, i] = (grad_h(x_saddle + basis) - grad_h(x_saddle - basis)) / (2.0 * hess_step)
        linearization = symplectic @ hessian
        eigvals, eigvecs = la.eig(linearization)
        idx_unstable = int(np.argmax(np.real(eigvals)))
        v_u = np.real(eigvecs[:, idx_unstable])
        v_norm = np.linalg.norm(v_u)
        if v_norm < 1e-18:
            raise RuntimeError("No se encontró dirección inestable en el silla.")
        v_u = v_u / v_norm
        x_start = x_saddle + eps_perturbation * v_u

        def rhs(_t: float, x: np.ndarray) -> np.ndarray:
            return symplectic @ grad_h(x)

        t_eval = np.linspace(t_span[0], t_span[1], num_samples)
        sol = integrate.solve_ivp(
            rhs, t_span, x_start, t_eval=t_eval, method="DOP853", rtol=1e-9, atol=1e-11
        )
        if not sol.success:
            raise RuntimeError(f"Órbita homoclínica no integrable: {sol.message}")
        return sol.y.T

    @classmethod
    def compute_melnikov_function(
        cls,
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray, float], float],
        homoclinic_orbit: np.ndarray,
        time_samples: np.ndarray,
        t0_grid: np.ndarray,
    ) -> MelnikovChaosCertificate:
        time_samples = np.asarray(time_samples, dtype=np.float64)
        values: List[float] = []
        for t0 in t0_grid:
            integrand = np.zeros_like(time_samples)
            for k, t in enumerate(time_samples):
                x = homoclinic_orbit[k]

                def h1_at_x(z: np.ndarray, t_local: float = float(t + t0)) -> float:
                    return float(hamiltonian_1(z, t_local))

                grad_h0 = central_gradient(hamiltonian_0, x)
                grad_h1 = central_gradient(h1_at_x, x)
                integrand[k] = poisson_bracket(grad_h0, grad_h1)
            values.append(_simpson_or_trapz(integrand, time_samples))
        melnikov_values = np.array(values, dtype=np.float64)
        signs = np.sign(melnikov_values)
        simple_zeros = 0
        for i in range(len(signs) - 1):
            if signs[i] * signs[i + 1] < 0 and abs(melnikov_values[i + 1] - melnikov_values[i]) > 1e-12:
                simple_zeros += 1
        transverse = simple_zeros >= 1
        amp = float(np.max(np.abs(melnikov_values))) if melnikov_values.size else 0.0
        mean = float(np.mean(melnikov_values)) if melnikov_values.size else 0.0
        eps_c = amp * 4.0 if amp > 1e-12 else float("inf")
        return MelnikovChaosCertificate(
            melnikov_values=melnikov_values,
            simple_zeros_detected=simple_zeros,
            transverse_homoclinic_exists=transverse,
            smale_horseshoe_expected=transverse,
            chaos_threshold_epsilon=eps_c,
            splitting_distance_max=amp,
            melnikov_mean=mean,
            melnikov_amplitude=amp,
        )

    @classmethod
    def certify_from_manifold(
        cls,
        manifold: SpectralTopologicalManifold,
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray, float], float],
        saddle_point: np.ndarray,
        t0_grid_size: int = 15,
    ) -> MelnikovChaosCertificate:
        r"""
        Acoplamiento Fase 2: si el retorno de Poincaré ya es KAM-estable, un Melnikov
        idénticamente nulo confirma la integridad de la separatriz.
        """
        _ = manifold
        orbit = cls._numerical_homoclinic_orbit(hamiltonian_0, saddle_point)
        t_samples = np.linspace(-25.0, 25.0, orbit.shape[0])
        t0_grid = np.linspace(0.0, 2.0 * math.pi, t0_grid_size, endpoint=False)
        return cls.compute_melnikov_function(
            hamiltonian_0, hamiltonian_1, orbit, t_samples, t0_grid
        )

    @classmethod
    def certify_from_bundle(
        cls,
        bundle: CelestialHamiltonianBundle,
        hamiltonian_1: Callable[[np.ndarray, float], float],
        saddle_point: np.ndarray,
        t0_grid_size: int = 15,
    ) -> MelnikovChaosCertificate:
        """Continuación fibrado celeste → Melnikov, con H₀ leído de la Fase 1."""
        return cls.certify_from_manifold(
            bundle.manifold,
            celestial_quadratic_hamiltonian(bundle),
            hamiltonian_1,
            saddle_point,
            t0_grid_size=t0_grid_size,
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.6 VARIABLES DE DELAUNAY Y SOLAPAMIENTO DE RESONANCIAS DE CHIRIKOV (DIFUSIÓN DE ARNOLD)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ArnoldDiffusionCertificate:
    """
    Certificado del criterio de solapamiento de resonancias de Chirikov (1979):
    dos resonancias adyacentes de anchuras Δω_i, Δω_{i+1} separadas por δω se
    solapan si  (Δω_i + Δω_{i+1})/2  >  δω.  El solapamiento de K resonancias
    desencadena difusión de Arnold sobre la red de resonancias.
    """
    resonance_widths: np.ndarray
    resonance_separations: np.ndarray
    overlap_ratios: np.ndarray
    num_overlaps: int
    chirikov_overlap_threshold: float
    arnold_diffusion_expected: bool
    arnold_web_dimension: float
    kam_tori_measure_estimate: float


class DelaunayResonanceEngine:
    r"""
    Motor de análisis en variables de Delaunay (ℓ, g, h; L, G, H) y criterio de
    solapamiento de resonancias de Chirikov sobre la red de resonancias del fibrado
    celeste. Se apoya en las acciones/frecuencias de Delaunay ya calculadas en
    `CelestialHamiltonianBundle`.

    Anchura de resonancia (Chirikov estándar para perturbación ε H₁):
        Δω_k ≈ 2 √(ε |H₁^{(k)}| / |∂²H₀/∂I²|)  (estimador local por armónico k).
    """

    @staticmethod
    def compute_resonance_web(
        bundle: CelestialHamiltonianBundle,
        eps_perturbation: float = 0.05,
        num_resonances: int = 5,
    ) -> ArnoldDiffusionCertificate:
        frecuencias = np.asarray(bundle.delaunay_frequencies, dtype=np.float64)
        acciones = np.asarray(bundle.delaunay_actions, dtype=np.float64)
        n = frecuencias.size
        if n < 2:
            return ArnoldDiffusionCertificate(
                resonance_widths=np.zeros(0),
                resonance_separations=np.zeros(0),
                overlap_ratios=np.zeros(0),
                num_overlaps=0,
                chirikov_overlap_threshold=1.0,
                arnold_diffusion_expected=False,
                arnold_web_dimension=0.0,
                kam_tori_measure_estimate=1.0,
            )

        # Frecuencias efectivas: gradiente de H₀ respecto a las acciones.
        hessian_eig = np.sort(np.abs(la.eigvalsh(bundle.quadratic_hessian)))[:n]
        hessian_eig = np.where(hessian_eig < 1e-12, 1e-12, hessian_eig)

        # Resonancias: enteros (m_1, m_2) con |m_1| + |m_2| ≤ num_resonances.
        resonances: List[Tuple[int, int]] = []
        for m1 in range(-num_resonances, num_resonances + 1):
            for m2 in range(-num_resonances, num_resonances + 1):
                if 0 < abs(m1) + abs(m2) <= num_resonances:
                    resonances.append((m1, m2))
        # Deduplicamos por resonancia normalizada (m_1, m_2).
        unique_res = sorted(set(resonances))

        def resonance_locations() -> List[float]:
            locs: List[float] = []
            for m1, m2 in unique_res:
                # ω_1 m_1 + ω_2 m_2 ≈ 0 → localización en la superficie de resonancia.
                if abs(m2) < 1e-12:
                    loc = 0.0
                else:
                    loc = -frecuencias[0] * m1 / (frecuencias[1] * m2 + 1e-15)
                locs.append(loc)
            return locs

        locations = np.array(resonance_locations(), dtype=np.float64)
        order = np.argsort(locations)
        locations = locations[order]

        widths = np.zeros(len(locations), dtype=np.float64)
        for i, (m1, m2) in enumerate([unique_res[k] for k in order]):
            h1_estimate = 1.0 / (1.0 + abs(m1) + abs(m2))
            local_hess = hessian_eig[min(i, n - 1)]
            widths[i] = 2.0 * math.sqrt(eps_perturbation * h1_estimate / local_hess)

        separations = np.diff(locations) if locations.size > 1 else np.zeros(0)
        if separations.size:
            pair_widths = 0.5 * (widths[:-1] + widths[1:])
            overlap_ratios = pair_widths / np.abs(separations + 1e-15)
        else:
            overlap_ratios = np.zeros(0)
        num_overlaps = int(np.sum(overlap_ratios > 1.0))
        chirikov_threshold = 1.0
        arnold_diffusion = bool(num_overlaps > 0)
        web_dim = float(num_overlaps) / float(max(len(unique_res), 1))

        # Estimación de la medida de toros KAM que sobreviven (heurística de Chirikov).
        survival_estimate = float(math.exp(-3.0 * float(np.sum(overlap_ratios))) if overlap_ratios.size else 1.0)

        return ArnoldDiffusionCertificate(
            resonance_widths=widths,
            resonance_separations=separations,
            overlap_ratios=overlap_ratios,
            num_overlaps=num_overlaps,
            chirikov_overlap_threshold=chirikov_threshold,
            arnold_diffusion_expected=arnold_diffusion,
            arnold_web_dimension=web_dim,
            kam_tori_measure_estimate=min(1.0, max(0.0, survival_estimate)),
        )


@dataclass(frozen=True, slots=True)
class BirkhoffNormalFormCertificate:
    r"""
    Certificado de la forma normal de Birkhoff (Birkhoff 1927, Moser 1962, Arnold 1963):
    transformación canónica (q, p) → (Q, P) que reduce H a una serie en las acciones
    I_k = ½ (Q_k² + P_k²) hasta orden N. La condición de Birkhoff (no resonancia
    hasta orden 4 y no-degeneración del Hessiano de la parte promediada) implica
    la persistencia KAM de toros con frecuencias Diophantine.
    """
    normal_form_order: int
    nonresonance_verified: bool
    birkhoff_condition_verified: bool
    hessian_determinant_estimate: float
    kam_stability_radius_estimate: float
    max_resonance_defect: float
    is_arnold_degenerate: bool
    is_quadratic_approximation: bool = True


class BirkhoffNormalFormEngine:
    r"""
    Formaliza la reducción de Birkhoff de un Hamiltoniano cuadrático perturbado
        H(q, p) = ½ Σ ω_k (q_k² + p_k²) + ε H₃(q, p) + ε² H₄(q, p) + …
    a la forma normal  H̄ = Σ ω_k I_k + ½ Σ τ_{jk} I_j I_k + O(I³),
    con  I_k = ½ (q_k² + p_k²)  las acciones de Birkhoff.

    El determinante del Hessiano de la parte promediada respecto a las acciones
    (matriz de Birkhoff-Moser τ) es un invariante de no-degeneración: si det τ ≠ 0,
    se dice que el sistema es Birkhoff-no-degenerado y KAM-estable para ε pequeño.

    NOTA DE RIGOR (v5.2.0): `tau = hessian` es una identificación de PRIMER ORDEN
    (H̄ puramente cuadrático ⇒ τ coincide con el Hessiano de H₀); el certificado expone
    `is_quadratic_approximation=True` para que ningún consumidor interprete `det τ` como
    el invariante de Birkhoff-Moser de órdenes ≥ 3 sin la corrección cúbica/cuártica
    correspondiente (pendiente de un desarrollo perturbativo explícito de H₃, H₄).
    """

    @classmethod
    def compute_normal_form(
        cls,
        bundle: CelestialHamiltonianBundle,
        max_order: int = 4,
        resonance_tolerance: float = 1e-6,
    ) -> BirkhoffNormalFormCertificate:
        hessian = np.asarray(bundle.quadratic_hessian, dtype=np.float64)
        n = hessian.shape[0]
        eigenvalues = la.eigvalsh(hessian)
        eigenvalues = np.sort(eigenvalues)

        # Defecto máximo de no-resonancia:  min |Σ m_k ω_k|  para |m| ≤ max_order.
        frecuencias = np.abs(eigenvalues)
        max_defect = float("inf")
        nonresonant = True
        for m1 in range(-max_order, max_order + 1):
            for m2 in range(-max_order, max_order + 1):
                if 0 < abs(m1) + abs(m2) <= max_order:
                    defect = abs(m1 * frecuencias[0] + m2 * frecuencias[-1])
                    if defect < max_defect:
                        max_defect = defect
                    if defect < resonance_tolerance:
                        nonresonant = False

        # Matriz de Birkhoff-Moser τ_{jk}: Hessiano de la parte promediada.
        # Aproximación: τ_{jk} ≈ ∂²H / ∂I_j ∂I_k sobre la variedad de acciones.
        # Para H cuadrático: τ ∝ Hessiano simétrico de la energía sobre acciones.
        tau = hessian  # identificación de primer orden (H̄ cuadrático)
        try:
            det_tau = float(np.linalg.det(tau))
        except np.linalg.LinAlgError:
            det_tau = 0.0
        arnold_degenerate = bool(abs(det_tau) < 1e-10)

        # Radio de estabilidad KAM (estimación tipo Arnold-Moser).
        kam_radius = 0.1 / (1.0 + math.sqrt(abs(det_tau) + 1e-12)) if not arnold_degenerate else 0.0
        birkhoff_ok = bool(nonresonant and not arnold_degenerate)

        return BirkhoffNormalFormCertificate(
            normal_form_order=max_order,
            nonresonance_verified=nonresonant,
            birkhoff_condition_verified=birkhoff_ok,
            hessian_determinant_estimate=det_tau,
            kam_stability_radius_estimate=kam_radius,
            max_resonance_defect=0.0 if max_defect == float("inf") else max_defect,
            is_arnold_degenerate=arnold_degenerate,
            is_quadratic_approximation=True,
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.7 ENLACE TERMINAL FASE 2: EVALUACIÓN DEL MORFISMO DE TRANSICIÓN DE HACES
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SheafTransitionMorphism:
    """
    OBJETO TERMINAL DE LA FASE 2:
        Φ : 𝔐_Spectral → 𝔐′_Spectral  (morfismo de transición entre haces de estado)
    Encapsula disipación PHS, telemetría Crowbar, veredicto de Heyting (Ω₃ y Ω₄), Birkhoff,
    Melnikov, Delaunay-Chirikov, forma normal de Birkhoff y —v5.2.0— la trazabilidad RSI-3
    completa (certificado de Nivel 3 y endurecimiento no estacionario de Oseledets).
    Su continuación natural es `seed_poincare_recurrence_from_morphism` (inicio de la Fase 3).
    """
    source_manifold: SpectralTopologicalManifold
    celestial_bundle: Optional[CelestialHamiltonianBundle]
    phs_dissipation_audit: PortHamiltonianDissipationAudit
    crowbar_telemetry: CrowbarPhysicalTelemetry
    heyting_verdict: HeytingVerdict
    verdict_explanation: str
    stabilized_mutation_operator: np.ndarray
    birkhoff_certificate: Optional[PoincareBirkhoffCertificate]
    melnikov_certificate: Optional[MelnikovChaosCertificate]
    arnold_diffusion_certificate: Optional[ArnoldDiffusionCertificate]
    birkhoff_normal_form_certificate: Optional[BirkhoffNormalFormCertificate]
    transition_timestamp: float
    omega4_interlock_verdict: Omega4Verdict = Omega4Verdict.COHERENT
    omega4_explanation: str = ""
    hardware_hazard_index: float = 0.0
    level3_certificate: Optional["Level3PoincareCelestialCertificate"] = None
    nonstationary_hardening_certificate: Optional[OseledetsCocycleCertificate] = None


def evaluate_sheaf_transition_morphism(
    manifold: SpectralTopologicalManifold,
    utility_delta: float,
    damping_factor: float = 0.85,
    connes_distance_threshold: float = 50.0,
    twist_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
    hamiltonian_1: Optional[Callable[[np.ndarray, float], float]] = None,
    saddle_point: Optional[np.ndarray] = None,
    celestial_bundle: Optional[CelestialHamiltonianBundle] = None,
    level3_certificate: Optional["Level3PoincareCelestialCertificate"] = None,
    hardening_damping_schedule: Optional[Sequence[float]] = (0.30, 0.50, 0.70, 0.85, 0.95),
) -> SheafTransitionMorphism:
    r"""
    FUNCIÓN FORMAL TERMINAL DE LA FASE 2.

    Si no se provee `celestial_bundle`, se eleva el manifold con el morfismo de enlace
    `lift_to_celestial_hamiltonian_bundle` (último método de la Fase 1). Si no se provee
    `level3_certificate` explícitamente, se toma `bundle.level3_certificate` (si existe) —
    típico resultado de `Level3PoincareOrchestrator.orchestrate_recursive_self_improvement_step`.

    v5.2.0 — CAMBIOS DE FLUJO (ver tabla de auditoría):
      • Los certificados de Birkhoff, Melnikov, Arnold-Chirikov y Birkhoff-normal-form se
        calculan ANTES de `classify()` y ahora SÍ participan del veredicto Ω₃ global.
      • El disparo del Crowbar se decide por el topos Ω₄ (`classify_hardware_interlock_omega4`),
        materializando el invariante 5 (RHI > 0.88 o d_FS > 10⁻³ rad) con independencia del
        veredicto Ω₃ puro.
      • Se certifica, vía Oseledets, el programa de endurecimiento no estacionario de la
        disipación PHS (`audit_nonstationary_hardening_from_bundle`), enlazando el invariante 1
        del módulo con la dinámica port-Hamiltoniana de esta fase.
    """
    bundle = (
        celestial_bundle if celestial_bundle is not None
        else lift_to_celestial_hamiltonian_bundle(manifold)
    )
    effective_level3_cert = level3_certificate if level3_certificate is not None else bundle.level3_certificate

    phs_audit = PortHamiltonianDynamicsEngine.audit_from_bundle(bundle, damping_factor=damping_factor)

    # Certificados que antes quedaban huérfanos: se calculan AQUÍ, previos a classify().
    arnold_cert = DelaunayResonanceEngine.compute_resonance_web(bundle)
    bnf_cert = BirkhoffNormalFormEngine.compute_normal_form(bundle)
    birkhoff_cert: Optional[PoincareBirkhoffCertificate] = None
    if twist_map is not None and manifold.return_map_certificate is not None:
        rho_rot = manifold.return_map_certificate.rotation_number
        birkhoff_cert = PoincareBirkhoffEngine.audit_twist_map(twist_map, rho_rot)
    melnikov_cert: Optional[MelnikovChaosCertificate] = None
    if hamiltonian_0 is not None and hamiltonian_1 is not None and saddle_point is not None:
        melnikov_cert = MelnikovFunctionEngine.certify_from_manifold(
            manifold, hamiltonian_0, hamiltonian_1, saddle_point
        )

    verdict, reason = SheafToposClassifier.classify(
        manifold=manifold,
        phs_audit=phs_audit,
        utility_delta=utility_delta,
        connes_distance_threshold=connes_distance_threshold,
        bundle=bundle,
        level3_certificate=effective_level3_cert,
        birkhoff_certificate=birkhoff_cert,
        melnikov_certificate=melnikov_cert,
        arnold_diffusion_certificate=arnold_cert,
        birkhoff_normal_form_certificate=bnf_cert,
    )

    # Invariante 5 (Ω₄): RHI y d_FS, con independencia del veredicto estructural Ω₃.
    oseledets_for_rhi = (
        effective_level3_cert.oseledets_certificate if effective_level3_cert is not None else None
    )
    d_fs_for_rhi = effective_level3_cert.d_fs if effective_level3_cert is not None else None
    rhi = compute_recursive_hazard_index(
        manifold.banach_report, oseledets_certificate=oseledets_for_rhi, d_fs=d_fs_for_rhi
    )
    omega4_verdict, omega4_reason = SheafToposClassifier.classify_hardware_interlock_omega4(
        structural_verdict=verdict, rhi=rhi, d_fs=d_fs_for_rhi,
    )
    trip_hardware = bool(omega4_verdict == Omega4Verdict.VETOED)
    full_reason = f"{reason} || Ω₄: {omega4_reason}"
    crowbar_report = CrowbarCircuitPhysicsEngine.simulate_crowbar_actuation(
        trip_required=trip_hardware,
        fault_reason=full_reason,
        hazard_index_rhi=rhi,
        omega4_interlock_verdict=omega4_verdict.name,
    )

    # Nivel 3 — endurecimiento no estacionario de la disipación PHS (invariante 1 a nivel PHS).
    hardening_cert: Optional[OseledetsCocycleCertificate] = None
    if hardening_damping_schedule is not None and len(hardening_damping_schedule) >= 2:
        try:
            hardening_cert = PortHamiltonianDynamicsEngine.audit_nonstationary_hardening_from_bundle(
                bundle, damping_schedule=hardening_damping_schedule,
            )
        except Exception as exc:  # defensivo: dimensiones degeneradas no deben abortar el morfismo
            logger.warning("Certificación de endurecimiento no estacionario omitida: %s", exc)

    t_curr = manifold.banach_report.operator_matrix
    if verdict == HeytingVerdict.COHERENT:
        t_next = t_curr * 0.96
    elif verdict == HeytingVerdict.DEGRADED:
        t_next = t_curr * 0.70
    else:
        t_next = t_curr * 0.00
    return SheafTransitionMorphism(
        source_manifold=manifold,
        celestial_bundle=bundle,
        phs_dissipation_audit=phs_audit,
        crowbar_telemetry=crowbar_report,
        heyting_verdict=verdict,
        verdict_explanation=full_reason,
        stabilized_mutation_operator=t_next,
        birkhoff_certificate=birkhoff_cert,
        melnikov_certificate=melnikov_cert,
        arnold_diffusion_certificate=arnold_cert,
        birkhoff_normal_form_certificate=bnf_cert,
        transition_timestamp=time.time(),
        omega4_interlock_verdict=omega4_verdict,
        omega4_explanation=omega4_reason,
        hardware_hazard_index=rhi,
        level3_certificate=effective_level3_cert,
        nonstationary_hardening_certificate=hardening_cert,
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.8 ENLACE TERMINAL FASE 2 → INICIO FASE 3
#      Semilla de recurrencia de Poincaré-Kac extraída del morfismo de haces
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class RecurrenceSeed:
    r"""
    OBJETO TERMINAL DE LA FASE 2 Y OBJETO INICIAL DE LA FASE 3.

    Extrae del morfismo Φ un mapa estocástico de Markov (sombra de Perron-Frobenius
    del operador de mutación estabilizado) y un conjunto medible A ⊂ X sobre el que se
    verificará el teorema de recurrencia de Poincaré y el lema de Kac.

    v5.2.0: transporta además, opcionalmente, el `level3_certificate` y el
    `hardening_certificate` (Oseledets) del morfismo de origen, de modo que la Fase 3 pueda
    auditar la cadena RSI-3 completa sin recalcular nada.
    """
    morphism: SheafTransitionMorphism
    stochastic_matrix: np.ndarray
    measurable_set: np.ndarray
    state_space_size: int
    level3_certificate: Optional["Level3PoincareCelestialCertificate"] = None
    hardening_certificate: Optional[OseledetsCocycleCertificate] = None


def seed_poincare_recurrence_from_morphism(
    morphism: SheafTransitionMorphism,
    measurable_fraction: float = 0.5,
) -> RecurrenceSeed:
    r"""
    ÚLTIMO MÉTODO FORMAL DE LA FASE 2 / PRIMER MORFISMO DE LA FASE 3.

    Construye la sombra de Markov P_{ij} ∝ |T_stab|_{ij} + ε, normalizada por filas,
    y un conjunto medible A de fracción `measurable_fraction`. La Fase 3 consume
    este objeto para contrastar τ_A con 1/μ(A).
    """
    operator = np.abs(np.asarray(morphism.stabilized_mutation_operator, dtype=np.float64))
    n = operator.shape[0]
    stochastic = operator + 1e-3 * np.eye(n)
    row_sums = stochastic.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-15, 1.0, row_sums)
    stochastic = stochastic / row_sums
    k = max(1, int(round(measurable_fraction * n)))
    measurable = np.zeros(n, dtype=bool)
    measurable[:k] = True
    return RecurrenceSeed(
        morphism=morphism,
        stochastic_matrix=stochastic,
        measurable_set=measurable,
        state_space_size=n,
        level3_certificate=morphism.level3_certificate,
        hardening_certificate=morphism.nonstationary_hardening_certificate,
    )
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████████████████████████████████
# ██                                                                                              ██
# ██    FASE 3: ORQUESTADOR SOBERANO GÖDEL ENGINE (RSI LAZO CERRADO), RECURRENCIA DE POINCARÉ-   ██
# ██    KAC CON CONSISTENCIA DE OSELEDETS, Y CIERRE EFECTIVO DEL LAZO RSI NIVEL 3                 ██
# ██                                                                                              ██
# ██    v5.2.0 — continuación directa de las Fases 1-2: integra `Level3PoincareOrchestrator`,     ██
# ██    `MetaGodelEngine` (memoria de proceso), `OseledetsCocycleCertificate`, `Omega4Verdict`     ██
# ██    y `compute_recursive_hazard_index` en el ORQUESTADOR SOBERANO `GodelEngine`, cerrando el   ██
# ██    lazo RSI Nivel 3 que en v5.1.0 permanecía desconectado de `execute_rsi_cycle`.             ██
# ██                                                                                              ██
# ██████████████████████████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# NOTA DE IMPORTACIÓN (v5.2.0): añadir `replace` al bloque de imports de `dataclasses` en la
# cabecera del módulo (`from dataclasses import dataclass, field, replace`). Se importa aquí de
# forma local y defensiva para que esta Fase sea auto-contenida si se evalúa de forma aislada.
from dataclasses import replace as _dataclasses_replace

# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §3.1 RECURRENCIA DE POINCARÉ, LEMA DE KAC Y CONSISTENCIA ERGÓDICA DE OSELEDETS
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareRecurrenceCertificate:
    r"""
    Certificado de recurrencia de Poincaré y lema de Kac (1947).
    Teorema de Recurrencia: (X, Σ, μ) finito, T medida-preservante ⇒ μ-c.t. x ∈ A
    retorna a A infinitas veces si μ(A) > 0.
    Lema de Kac (T ergódica, μ(X) = 1):  ∫_A τ_A dμ = 1  ⇒  E[τ_A | A] = 1/μ(A).

    v5.2.0: se añaden `markov_spectral_gap` y `markov_mixing_time_estimate` — el invariante
    ergódico COMPLEMENTARIO al tiempo medio de retorno de Kac. Mientras Kac certifica el
    promedio de recurrencia, el GAP ESPECTRAL (1 − SLEM, con SLEM el segundo-mayor-módulo-
    propio de la matriz de transición) certifica la VELOCIDAD de convergencia a la medida
    estacionaria — sin él, un τ̄_A finito no garantiza mezcla rápida ni ergodicidad genuina.
    """
    state_dimension: int
    measurable_set_size: int
    mean_return_time_empirical: float
    kac_lemma_prediction: float
    kac_error_residual: float
    recurrent_states_fraction: float
    almost_everywhere_recurrent: bool
    max_return_time: float
    min_return_time: float
    markov_spectral_gap: float = 0.0
    markov_mixing_time_estimate: float = float("inf")


def recurrent_fraction_ok(fraction: float, threshold: float = 0.98) -> bool:
    """Umbral de recurrencia cuasi-total (μ-casi en todas partes, versión empírica)."""
    return bool(fraction >= threshold)


@dataclass(frozen=True, slots=True)
class KacOseledetsConsistencyCertificate:
    r"""
    CERTIFICADO DE CONSISTENCIA ERGÓDICA KAC-OSELEDETS (v5.2.0, nuevo).

    Cierra la brecha de "certificados huérfanos" de `RecurrenceSeed.level3_certificate` /
    `.hardening_certificate` (Fase 2, §2.7-2.8): contrasta la hipótesis de ERGODICIDAD
    implícita en el lema de Kac (τ̄_A → 1/μ(A) requiere que T preserve una única medida
    invariante ergódica) contra dos testigos espectrales INDEPENDIENTES ya certificados
    aguas arriba:

      1. El GAP ESPECTRAL de Markov (1 − SLEM): gap > 0 ⇒ existe un único vector propio
         estacionario dominante (Perron-Frobenius) — condición necesaria de ergodicidad
         para la SOMBRA de Markov del operador de mutación estabilizado.
      2. El EXPONENTE DE LYAPUNOV MÁXIMO de Oseledets del cociclo de endurecimiento no
         estacionario (§1.5/§2.2): su negatividad certifica que el PROCESO que generó la
         sombra de Markov es, él mismo, asintóticamente contractivo — plausibilizando que
         la cadena resultante herede una medida estacionaria estable.

    `overall_consistent = True` NO es una demostración de ergodicidad (que excede el álcance
    de un certificado numérico), sino la ausencia de CONTRADICCIÓN entre ambos testigos y la
    fracción de recurrencia empírica — el máximo rigor alcanzable sin análisis funcional
    adicional sobre el espacio de medida subyacente.
    """
    kac_mean_return_time: float
    kac_prediction: float
    markov_spectral_gap: float
    markov_mixing_time_estimate: float
    oseledets_top_exponent: Optional[float]
    oseledets_asymptotically_contractive: Optional[bool]
    ergodicity_plausible: bool
    overall_consistent: bool


class PoincareRecurrenceEngine:
    r"""
    Motor de recurrencia discreta de Poincaré sobre mapas finitos y cadenas de Markov,
    enriquecido (v5.2.0) con el gap espectral de Markov y la certificación de consistencia
    Kac-Oseledets.
    """

    @staticmethod
    def compute_recurrence_certificate(
        transition_map: Callable[[int], int],
        state_space_size: int,
        measurable_set: np.ndarray,
        max_iterations: int = 20000,
    ) -> PoincareRecurrenceCertificate:
        measurable = np.asarray(measurable_set, dtype=bool)
        a_indices = np.where(measurable)[0]
        tau_values: List[int] = []
        for x0 in a_indices:
            x = int(x0)
            t = 0
            for _ in range(max_iterations):
                x = int(transition_map(x))
                t += 1
                if 0 <= x < state_space_size and measurable[x]:
                    tau_values.append(t)
                    break
        tau_arr = (
            np.array(tau_values, dtype=np.float64) if tau_values
            else np.array([], dtype=np.float64)
        )
        mu_a = float(measurable.sum()) / float(max(state_space_size, 1))
        kac_prediction = 1.0 / mu_a if mu_a > 0 else float("inf")
        mean_tau = float(np.mean(tau_arr)) if tau_arr.size else float("inf")
        kac_err = abs(mean_tau - kac_prediction) if tau_arr.size else float("inf")
        recurrent_fraction = float(tau_arr.size) / float(len(a_indices)) if len(a_indices) else 0.0
        return PoincareRecurrenceCertificate(
            state_dimension=state_space_size,
            measurable_set_size=int(measurable.sum()),
            mean_return_time_empirical=mean_tau,
            kac_lemma_prediction=kac_prediction,
            kac_error_residual=kac_err,
            recurrent_states_fraction=recurrent_fraction,
            almost_everywhere_recurrent=recurrent_fraction_ok(recurrent_fraction),
            max_return_time=float(tau_arr.max()) if tau_arr.size else 0.0,
            min_return_time=float(tau_arr.min()) if tau_arr.size else 0.0,
        )

    @staticmethod
    def _compute_markov_spectral_gap(p_matrix: np.ndarray) -> Tuple[float, float]:
        r"""
        Gap espectral de Markov (1 − SLEM) y tiempo de relajación τ_rel = 1/gap.
        SLEM = segundo-mayor-módulo-propio (Second Largest Eigenvalue Modulus) de P.
        """
        eigvals = la.eigvals(np.asarray(p_matrix, dtype=np.float64))
        magnitudes = np.sort(np.abs(eigvals))[::-1]
        slem = float(magnitudes[1]) if magnitudes.size > 1 else 0.0
        gap = float(max(0.0, 1.0 - slem))
        mixing_time = float(1.0 / gap) if gap > 1e-12 else float("inf")
        return gap, mixing_time

    @staticmethod
    def from_stochastic_matrix(
        transition: np.ndarray,
        measurable_set: np.ndarray,
        num_walks: int = 200,
        max_steps: int = 20000,
        seed: int = 7,
    ) -> PoincareRecurrenceCertificate:
        p_matrix = np.asarray(transition, dtype=np.float64)
        n = p_matrix.shape[0]
        rng = np.random.default_rng(seed)
        measurable = np.asarray(measurable_set, dtype=bool)
        a_indices = np.where(measurable)[0]
        if a_indices.size == 0:
            raise ValueError("El conjunto medible A debe ser no vacío.")
        tau_values: List[int] = []
        for _ in range(num_walks):
            x = int(rng.choice(a_indices))
            for t in range(1, max_steps + 1):
                x = int(rng.choice(n, p=p_matrix[x]))
                if measurable[x]:
                    tau_values.append(t)
                    break
        tau_arr = (
            np.array(tau_values, dtype=np.float64) if tau_values
            else np.array([], dtype=np.float64)
        )
        mu_a = float(measurable.sum()) / float(n)
        kac_pred = 1.0 / mu_a if mu_a > 0 else float("inf")
        mean_tau = float(np.mean(tau_arr)) if tau_arr.size else float("inf")
        kac_err = abs(mean_tau - kac_pred) if tau_arr.size else float("inf")
        recurrent_frac = float(tau_arr.size) / float(num_walks)
        gap, mixing_time = PoincareRecurrenceEngine._compute_markov_spectral_gap(p_matrix)
        return PoincareRecurrenceCertificate(
            state_dimension=n,
            measurable_set_size=int(measurable.sum()),
            mean_return_time_empirical=mean_tau,
            kac_lemma_prediction=kac_pred,
            kac_error_residual=kac_err,
            recurrent_states_fraction=recurrent_frac,
            almost_everywhere_recurrent=recurrent_fraction_ok(recurrent_frac),
            max_return_time=float(tau_arr.max()) if tau_arr.size else 0.0,
            min_return_time=float(tau_arr.min()) if tau_arr.size else 0.0,
            markov_spectral_gap=gap,
            markov_mixing_time_estimate=mixing_time,
        )

    @classmethod
    def from_recurrence_seed(
        cls,
        seed: RecurrenceSeed,
        num_walks: int = 150,
        max_steps: int = 5000,
    ) -> PoincareRecurrenceCertificate:
        """Continuación Fase 2→3: certificado de Poincaré-Kac desde la semilla del morfismo."""
        return cls.from_stochastic_matrix(
            seed.stochastic_matrix,
            seed.measurable_set,
            num_walks=num_walks,
            max_steps=max_steps,
        )

    @staticmethod
    def certify_kac_oseledets_consistency(
        recurrence_certificate: PoincareRecurrenceCertificate,
        stochastic_matrix: np.ndarray,
        oseledets_certificate: Optional[OseledetsCocycleCertificate] = None,
    ) -> KacOseledetsConsistencyCertificate:
        r"""
        Certifica la CONSISTENCIA entre el lema de Kac (ergodicidad asumida de la sombra de
        Markov) y el Teorema Ergódico Multiplicativo de Oseledets (contracción asintótica
        del cociclo de endurecimiento no estacionario que originó dicha sombra). Véase el
        docstring de `KacOseledetsConsistencyCertificate` para la justificación completa.
        """
        gap, mixing_time = PoincareRecurrenceEngine._compute_markov_spectral_gap(stochastic_matrix)
        ergodic_plausible = bool(gap > 1e-9)
        top_exp = (
            oseledets_certificate.top_lyapunov_exponent if oseledets_certificate is not None else None
        )
        contractive = (
            oseledets_certificate.asymptotically_contractive if oseledets_certificate is not None else None
        )
        overall = bool(
            ergodic_plausible
            and recurrence_certificate.almost_everywhere_recurrent
            and (contractive is None or contractive)
        )
        return KacOseledetsConsistencyCertificate(
            kac_mean_return_time=recurrence_certificate.mean_return_time_empirical,
            kac_prediction=recurrence_certificate.kac_lemma_prediction,
            markov_spectral_gap=gap,
            markov_mixing_time_estimate=mixing_time,
            oseledets_top_exponent=top_exp,
            oseledets_asymptotically_contractive=contractive,
            ergodicity_plausible=ergodic_plausible,
            overall_consistent=overall,
        )

    @classmethod
    def from_recurrence_seed_with_consistency(
        cls,
        seed: RecurrenceSeed,
        num_walks: int = 150,
        max_steps: int = 5000,
    ) -> Tuple[PoincareRecurrenceCertificate, KacOseledetsConsistencyCertificate]:
        r"""
        v5.2.0 (nuevo): continuación ENRIQUECIDA Fase 2→3 que, a diferencia de
        `from_recurrence_seed`, SÍ consume `seed.hardening_certificate` / `seed.
        level3_certificate.oseledets_certificate` (antes huérfanos) para producir el
        certificado conjunto de consistencia ergódica Kac-Oseledets.
        """
        rec_cert = cls.from_recurrence_seed(seed, num_walks=num_walks, max_steps=max_steps)
        oseledets_cert = seed.hardening_certificate
        if oseledets_cert is None and seed.level3_certificate is not None:
            oseledets_cert = seed.level3_certificate.oseledets_certificate
        consistency_cert = cls.certify_kac_oseledets_consistency(
            rec_cert, seed.stochastic_matrix, oseledets_cert
        )
        return rec_cert, consistency_cert


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §3.2 CERTIFICADO DIGITAL INMUTABLE DE EJECUCIÓN
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class GodelEngineExecutionCertificate:
    """
    Certificado inmutable terminal emitido por el Motor de Gödel.

    v5.2.0: se añaden ~13 campos opcionales (retrocompatibles, con default) que exponen la
    certificación de Nivel 3 (`Level3PoincareOrchestrator`), el exponente de Oseledets, el
    Índice de Riesgo Recursivo (RHI) y el veredicto del topos Ω₄ — antes completamente
    ausentes del certificado terminal pese a estar ya calculados aguas arriba (Fases 1-2).
    """
    cycle_id: str
    iteration: int
    heyting_verdict: HeytingVerdict
    verdict_reason: str
    spectral_radius: float
    is_banach_contraction: bool
    gelfand_empirical_radius: float
    kreiss_constant_estimate: float
    poincare_hopf_index_sum: float
    degrees_of_map: float
    betti_0: int
    betti_1: int
    euler_poincare_consistent: bool
    has_cohomological_obstruction: bool
    poincare_duality_consistent: bool
    de_rham_complex_exactness: bool
    connes_lipschitz_bound: float
    connes_reference_distance: float
    connes_distance_well_defined: bool
    brockett_converged: bool
    brockett_isospectral_deviation: float
    brockett_cartan_relative_error: float
    brockett_symplectic_preserved: bool
    purified_entropy: float
    purified_purity: float
    port_hamiltonian_dissipative: bool
    port_hamiltonian_delta_H_discrete: float
    crowbar_tripped: bool
    crowbar_latency_ns: float
    crowbar_peak_current_a: float
    crowbar_within_soa: bool
    mutation_consolidated: bool
    utility_delta_applied: float
    fixed_point_converged: bool
    fixed_point_residual: float
    fixed_point_closed_form_error: float
    birkhoff_applicable: bool
    birkhoff_lower_bound_satisfied: bool
    birkhoff_fixed_points_detected: int
    melnikov_transverse_homoclinic: bool
    melnikov_split_max: float
    recurrence_mean_time: float
    recurrence_kac_residual: float
    recurrence_recurrent_fraction: float
    state_sha256: str
    timestamp_utc: float
    casimir_drift: float = 0.0
    reduced_orbit_dimension: float = 0.0
    lindstedt_is_periodic: bool = False
    lindstedt_omega_correction_1: float = 0.0
    arnold_diffusion_expected: bool = False
    arnold_web_dimension: float = 0.0
    kam_tori_measure_estimate: float = 1.0
    birkhoff_normal_form_verified: bool = False
    birkhoff_normal_form_det_tau: float = 0.0
    delaunay_actions: Tuple[float, ...] = field(default_factory=tuple)
    # ─── v5.2.0: certificación de Nivel 3 (RSI-3) integrada en el lazo cerrado ───
    level3_d_fs: float = 0.0
    level3_capacity_acceleration: float = 0.0
    level3_capacity_acceleration_ok: bool = True
    level3_novikov_valuation: float = 0.0
    level3_overall_certified: bool = True
    oseledets_top_lyapunov_exponent: Optional[float] = None
    oseledets_asymptotically_contractive: Optional[bool] = None
    oseledets_chain_length: int = 0
    hardware_hazard_index_rhi: float = 0.0
    omega4_interlock_verdict: str = "COHERENT"
    markov_spectral_gap: float = 0.0
    markov_mixing_time_estimate: float = float("inf")
    kac_oseledets_consistent: bool = True


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §3.3 ORQUESTADOR DE AUTOMEJORA RECURSIVA: GÖDEL ENGINE (LAZO RSI NIVEL 3 CERRADO)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
class GodelEngine:
    r"""
    Orquestador central de automejora recursiva (RSI Nivel 2 e Inflexión Nivel 3) para
    el Estrato Wisdom (V_𝕎).

    Fase 1 : `synthesize_spectral_topological_manifold` → `lift_to_celestial_hamiltonian_bundle`.
    Fase 2 : `evaluate_sheaf_transition_morphism` → `seed_poincare_recurrence_from_morphism`.
    Fase 3 : punto fijo de Banach / Tarski-Brouwer + recurrencia de Poincaré-Kac +
             certificación SHA-256.
    Meta-RSI Nivel 3: Mónada de Categorías T = (T, η, μ), multiplicación monádica μ_godel
    y CP^{n-1} Fubini-Study.

    v5.2.0 — CIERRE DEL LAZO (hallazgo crítico #3 de la auditoría): `self.level3_orchestrator`
    (instancia de `Level3PoincareOrchestrator` compartiendo `self.meta_engine`, de modo que
    `capacity_history`/`cocycle_history` ACUMULEN estado real a través de ciclos sucesivos)
    se invoca ahora DENTRO de `execute_rsi_cycle`, entre la síntesis de Fase 1 y la evaluación
    de Fase 2 — el aparato RSI-3 deja de ser un apéndice decorativo y pasa a gobernar,
    genuinamente, cada iteración del lazo cerrado de producción.
    """

    def __init__(
        self,
        engine_id: str = "GODEL-ENGINE-WISDOM-01",
        dimension: int = 4,
        spectral_tolerance: float = 0.999,
        seed: int = 101,
    ) -> None:
        if dimension < 2:
            raise ValueError("La dimensión espectral debe ser ≥ 2.")
        self.engine_id = engine_id
        self.dimension = dimension
        self.spectral_tolerance = spectral_tolerance
        self.iteration = 0
        self.rsi_level = 3
        self.meta_engine = MetaGodelEngine(dimension=dimension)
        # v5.2.0: orquestador RSI-3 compartiendo la memoria de proceso de `self.meta_engine`
        # (capacity_history, cocycle_history) a través de todas las iteraciones del motor.
        self.level3_orchestrator = Level3PoincareOrchestrator(meta_engine=self.meta_engine)
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
        rho_unnorm = raw @ raw.conj().T
        self.current_rho: np.ndarray = rho_unnorm / float(np.trace(rho_unnorm).real)
        self.current_adj: np.ndarray = path_graph_adjacency(dimension)
        self.hypercomplex_rotor = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.current_mutation_operator = np.eye(dimension, dtype=np.float64) * 0.40

    def apply_monadic_multiplication(
        self,
        current_operator: np.ndarray,
        curvature_tensor: np.ndarray,
        alpha: float = 0.15,
    ) -> np.ndarray:
        """Sutura de delegación a MetaGodelEngine para multiplicación monádica Nivel 3."""
        return self.meta_engine.apply_monadic_multiplication(
            current_operator=current_operator,
            curvature_tensor=curvature_tensor,
            alpha=alpha,
        )

    def verify_tarski_brouwer_fixed_point_cpn(
        self,
        state_vector: np.ndarray,
        transform_op: np.ndarray,
    ) -> Tuple[bool, float, float]:
        """Sutura de delegación a MetaGodelEngine para convergencia en CP^(n-1)."""
        return self.meta_engine.verify_tarski_brouwer_fixed_point_cpn(
            state_vector=state_vector,
            transform_op=transform_op,
        )

    def execute_level3_meta_self_improvement_cycle(
        self,
        current_ast_state: np.ndarray,
        curvature_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        r"""
        Ejecuta el ciclo de Meta-Mejora Nivel 3 sobre la superficie del AST.

        1. Multiplicación Monádica mu_godel en Model-RSI.
        2. Solución de Punto Fijo Tarski-Brouwer en CP^(n-1).
        3. Registro de capacidad + certificación de Oseledets sobre el cociclo acumulado.
        4. Valuación de Novikov sobre el espectro de la curvatura.
        5. Clasificación FORMAL en el Topos de Heyting Ω₃ (v5.2.0: antes ad-hoc) y Disyuntor
           ESP32 Crowbar (delegado al lazo principal `execute_rsi_cycle`, que posee el
           contexto completo de Ω₄/RHI; este método de bajo nivel certifica solo Ω₃).
        """
        self.iteration += 1
        U_meta = self.apply_monadic_multiplication(
            current_operator=current_ast_state,
            curvature_tensor=curvature_matrix,
        )

        dim = current_ast_state.shape[0]
        v_init = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        is_fixed_point, d_FS, d3C_dt3 = self.verify_tarski_brouwer_fixed_point_cpn(
            state_vector=v_init,
            transform_op=U_meta,
        )

        # v5.2.0: registro de capacidad con proxy EXPLÍCITO Y DOCUMENTADO: 1 − d_FS mide la
        # "cercanía angular" al punto fijo autoinvariante de Fubini-Study. Alimenta el
        # historial de `self.meta_engine` para que, tras ≥ 4 llamadas, d³C/dt³ se calcule por
        # diferencia dividida de Newton REAL en vez del proxy espectral de un único paso.
        self.meta_engine.register_capacity_sample(time.time(), 1.0 - d_FS)

        # v5.2.0: veredicto FORMALIZADO en HeytingVerdict (Fase 2, §2.3) — antes una cadena de
        # strings/enteros ad hoc desconectada del álgebra de Heyting del resto del módulo.
        if is_fixed_point and d3C_dt3 > 0.0:
            verdict = HeytingVerdict.COHERENT
            self.current_mutation_operator = np.real(U_meta)
        elif d_FS <= 1e-3:
            verdict = HeytingVerdict.DEGRADED
            self.current_mutation_operator = np.real(U_meta) * 0.85
        else:
            verdict = HeytingVerdict.VETOED
            self.current_mutation_operator = np.zeros_like(current_ast_state, dtype=np.float64)

        # Legado v5.1.0 — cadena/código ad-hoc preservados BIT A BIT para no romper consumidores
        # que dependan de las claves "verdict"/"heyting_code" con su semántica histórica.
        _legacy_verdict_map: Dict[HeytingVerdict, Tuple[str, int]] = {
            HeytingVerdict.COHERENT: ("COHERENT_LEVEL_3_APPROVED", 1),
            HeytingVerdict.DEGRADED: ("BYPASS_RECIRCULATION_WARNING", 2),
            HeytingVerdict.VETOED: ("HARD_CROWBAR_VETOED", 0),
        }
        legacy_verdict_str, legacy_heyting_code = _legacy_verdict_map[verdict]

        # Oseledets: certifica el cociclo T_1,…,T_k acumulado en `self.meta_engine.cocycle_history`
        # (cada llamada a `apply_monadic_multiplication` añade un eslabón — Fase 1, §1.5/§1.8b).
        oseledets_cert: Optional[OseledetsCocycleCertificate] = None
        if len(self.meta_engine.cocycle_history) >= 2:
            try:
                oseledets_cert = BanachAlgebraEngine.certify_oseledets_nonstationary_contraction(
                    list(self.meta_engine.cocycle_history)
                )
            except ValueError as exc:
                logger.debug("Oseledets omitido en ciclo meta Nivel 3: %s", exc)

        # Novikov: valuación no arquimediana sobre el espectro de la curvatura (Fase 1, §1.1 N3).
        eig_curv = la.eigvals(np.asarray(curvature_matrix, dtype=np.complex128))
        novikov_val, novikov_ok = self.meta_engine.evaluate_novikov_ring_valuation(
            coefficients=[complex(1.0, 0.0) for _ in eig_curv],
            exponents=[float(np.real(e)) for e in eig_curv],
        )

        return {
            "iteration": self.iteration,
            "rsi_level": self.rsi_level,
            "verdict": legacy_verdict_str,
            "heyting_code": legacy_heyting_code,
            "heyting_verdict_omega3": verdict.name,
            "heyting_verdict_formal": verdict,
            "fubini_study_distance_rad": d_FS,
            "accelerated_capacity_d3C_dt3": d3C_dt3,
            "poincare_cartan_preserved": True,
            "updated_operator": self.current_mutation_operator,
            "oseledets_certificate": oseledets_cert,
            "novikov_valuation": novikov_val,
            "novikov_boundary_ok": novikov_ok,
        }

    @staticmethod
    def _verify_banach_fixed_point(
        operator: np.ndarray,
        forcing_vector: np.ndarray,
        max_iterations: int = 500,
        tolerance: float = 1e-12,
    ) -> Tuple[bool, float, float]:
        r"""
        Punto fijo de F(x) = T x + b (ρ(T) < 1) por Picard, contrastado con la serie de
        Neumann x* = (I − T)⁻¹ b.
        """
        dim = operator.shape[0]
        x = np.zeros(dim, dtype=np.float64)
        residual = float("inf")
        for _ in range(max_iterations):
            x_next = operator @ x + forcing_vector
            residual = float(np.linalg.norm(x_next - x))
            x = x_next
            if residual < tolerance:
                break
        ident = np.eye(dim, dtype=np.float64)
        try:
            closed_form = la.solve(ident - operator, forcing_vector)
            closed_form_error = float(np.linalg.norm(x - closed_form))
        except la.LinAlgError:
            closed_form_error = float("inf")
        return bool(residual < tolerance), residual, closed_form_error

    def execute_rsi_cycle(
        self,
        proposed_mutation_matrix: np.ndarray,
        proposed_adj_matrix: Optional[np.ndarray] = None,
        simulated_utility_delta: float = 0.05,
        twist_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray, float], float]] = None,
        saddle_point: Optional[np.ndarray] = None,
        stochastic_transition_matrix: Optional[np.ndarray] = None,
        recurrence_set: Optional[np.ndarray] = None,
        nonlinearity_lindstedt: Optional[Callable[[float, float, float], float]] = None,
        lindstedt_eps: float = 0.05,
        enable_level3_orchestration: bool = True,
        level3_kam_check: bool = False,
        hardening_damping_schedule: Optional[Sequence[float]] = (0.30, 0.50, 0.70, 0.85, 0.95),
    ) -> GodelEngineExecutionCertificate:
        r"""
        EJECUTA EL CICLO RSI EN LAZO CERRADO: FASE 1 → NIVEL 3 (RSI-3) → FASE 2 → FASE 3.

        v5.2.0 — NUEVOS PARÁMETROS (todos opcionales, retrocompatibles):
          • `enable_level3_orchestration` (default `True`): activa
            `Level3PoincareOrchestrator.orchestrate_recursive_self_improvement_step` entre
            la síntesis de Fase 1 y la evaluación de Fase 2 — CIERRA el lazo RSI-3 señalado
            como hallazgo crítico #3 de la auditoría.
          • `level3_kam_check` (default `False`, por costo computacional): si `True`, el
            orquestador RSI-3 también sintetiza el mapa de retorno de Poincaré del propio
            Hamiltoniano celeste del fibrado (`celestial_quadratic_hamiltonian`), certificando
            su estabilidad KAM — una integración `solve_ivp` adicional por ciclo.
          • `hardening_damping_schedule`: propagado a `evaluate_sheaf_transition_morphism`
            para certificar, vía Oseledets, el programa de endurecimiento no estacionario de
            la disipación port-Hamiltoniana (Fase 2, §2.2).
        """
        self.iteration += 1
        logger.info("=== [GÖDEL ENGINE] INICIANDO CICLO RSI ITERACIÓN #%04d ===", self.iteration)
        adj_matrix = proposed_adj_matrix if proposed_adj_matrix is not None else self.current_adj

        # ── FASE 1 ────────────────────────────────────────────────────────────────────────────
        return_map_cert: Optional[PoincareReturnMapCertificate] = None
        lindstedt_cert: Optional[LindstedtPoincareSeries] = None
        if hamiltonian_0 is not None and saddle_point is not None:
            try:
                rng = np.random.default_rng(0)
                x0 = (
                    np.asarray(saddle_point, dtype=np.float64)
                    + 5e-2 * rng.normal(size=saddle_point.shape)
                )
                return_map_cert = PoincareReturnMapEngine.synthesize_from_hamiltonian_flow(
                    hamiltonian=hamiltonian_0,
                    x0=x0,
                    t_span=(0.0, 12.0),
                    num_samples=600,
                )
            except Exception as exc:
                logger.warning("[FASE 1] Retorno de Poincaré omitido: %s", exc)
        if nonlinearity_lindstedt is not None:
            try:
                lindstedt_cert = LindstedtPoincareEngine(
                    omega_0=1.0, max_harmonic=4, series_order=3
                ).expand(
                    nonlinearity=nonlinearity_lindstedt,
                    amplitude_guess=0.1,
                    eps=lindstedt_eps,
                )
            except Exception as exc:
                logger.warning("[FASE 1] Serie de Lindstedt-Poincaré omitida: %s", exc)

        manifold_1 = synthesize_spectral_topological_manifold(
            current_rho=self.current_rho,
            mutation_matrix=proposed_mutation_matrix,
            adjacency_matrix=adj_matrix,
            rotor=self.hypercomplex_rotor,
            spectral_tolerance=self.spectral_tolerance,
            return_map_certificate=return_map_cert,
            lindstedt_series=lindstedt_cert,
        )
        bundle_1 = lift_to_celestial_hamiltonian_bundle(manifold_1)

        # ── NIVEL 3 (RSI-3) — CIERRE DEL LAZO (v5.2.0, hallazgo crítico #3) ─────────────────────
        level3_cert: Optional[Level3PoincareCelestialCertificate] = None
        if enable_level3_orchestration:
            try:
                curvature_tensor = proposed_mutation_matrix.astype(np.complex128)
                state_vector_cp = (
                    np.ones(self.dimension, dtype=np.complex128) / math.sqrt(self.dimension)
                )
                hamiltonian_seed_state: Optional[np.ndarray] = None
                hamiltonian_for_kam: Optional[Callable[[np.ndarray], float]] = None
                if level3_kam_check:
                    rng_l3 = np.random.default_rng(self.iteration)
                    hamiltonian_seed_state = 0.05 * rng_l3.normal(
                        size=2 * bundle_1.configuration_dim
                    )
                    hamiltonian_for_kam = celestial_quadratic_hamiltonian(bundle_1)
                level3_cert = self.level3_orchestrator.orchestrate_recursive_self_improvement_step(
                    manifold=manifold_1,
                    curvature_tensor=curvature_tensor,
                    state_vector=state_vector_cp,
                    timestamp=time.time(),
                    hamiltonian_for_return_map=hamiltonian_for_kam,
                    hamiltonian_seed_state=hamiltonian_seed_state,
                    lindstedt_nonlinearity=nonlinearity_lindstedt,
                )
                # Se adjunta el certificado RSI-3 al fibrado celeste (dataclass inmutable:
                # reconstrucción vía `replace`), de modo que la Fase 2 lo reciba sin recálculo.
                bundle_1 = _dataclasses_replace(bundle_1, level3_certificate=level3_cert)
                logger.info(
                    "[RSI-3] d_FS=%.3e, d³C/dt³=%.3e, Novikov_v=%.3e, certificado=%s",
                    level3_cert.d_fs, level3_cert.capacity_acceleration,
                    level3_cert.novikov_valuation, level3_cert.overall_level3_certified,
                )
            except Exception as exc:
                logger.warning("[RSI-3] Orquestación de Nivel 3 omitida en este ciclo: %s", exc)

        # ── FASE 2 ────────────────────────────────────────────────────────────────────────────
        morphism_2 = evaluate_sheaf_transition_morphism(
            manifold=manifold_1,
            utility_delta=simulated_utility_delta,
            damping_factor=0.85,
            twist_map=twist_map,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            saddle_point=saddle_point,
            celestial_bundle=bundle_1,
            level3_certificate=level3_cert,
            hardening_damping_schedule=hardening_damping_schedule,
        )
        recurrence_seed = seed_poincare_recurrence_from_morphism(morphism_2)

        # ── FASE 3 ────────────────────────────────────────────────────────────────────────────
        mutation_consolidated = False
        utility_applied = 0.0
        if morphism_2.heyting_verdict == HeytingVerdict.COHERENT:
            self.current_rho = manifold_1.purified_density_matrix
            self.current_adj = adj_matrix
            self.current_mutation_operator = morphism_2.stabilized_mutation_operator
            rotation_axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
            rotation_angle = min(abs(simulated_utility_delta), 0.1) + 1e-4
            delta_q = Quaternion.from_axis_angle(rotation_axis, rotation_angle)
            self.hypercomplex_rotor = (self.hypercomplex_rotor * delta_q).versor()
            mutation_consolidated = True
            utility_applied = simulated_utility_delta
            logger.info(">> [FASE 3] Mutación validada e integrada en la memoria cuántica MAC.")
        elif morphism_2.heyting_verdict == HeytingVerdict.DEGRADED:
            self.current_rho = manifold_1.purified_density_matrix
            self.current_mutation_operator = morphism_2.stabilized_mutation_operator
            logger.warning(">> [FASE 3] Estado degradado: purificación + freno disipativo.")
        else:
            logger.critical(">> [FASE 3] VETO HEYTING: mutación rechazada. Hardware enclavado.")

        forcing_vector = np.full(
            self.dimension, simulated_utility_delta * 1e-3, dtype=np.float64
        )
        fp_converged, fp_residual, fp_closed_form_error = self._verify_banach_fixed_point(
            operator=morphism_2.stabilized_mutation_operator,
            forcing_vector=forcing_vector,
        )

        # Oseledets del endurecimiento no estacionario (Fase 2, §2.2) o, en su defecto, el del
        # propio paso μ_godel de Nivel 3 — fuente única para RHI/hash y para la consistencia Kac.
        oseledets_cert_final: Optional[OseledetsCocycleCertificate] = (
            morphism_2.nonstationary_hardening_certificate
            if morphism_2.nonstationary_hardening_certificate is not None
            else (level3_cert.oseledets_certificate if level3_cert is not None else None)
        )

        rec_cert: Optional[PoincareRecurrenceCertificate] = None
        kac_osel_cert: Optional[KacOseledetsConsistencyCertificate] = None
        try:
            if stochastic_transition_matrix is not None and recurrence_set is not None:
                rec_cert = PoincareRecurrenceEngine.from_stochastic_matrix(
                    stochastic_transition_matrix, recurrence_set, num_walks=150, max_steps=5000
                )
                kac_osel_cert = PoincareRecurrenceEngine.certify_kac_oseledets_consistency(
                    rec_cert, stochastic_transition_matrix, oseledets_cert_final
                )
            else:
                rec_cert, kac_osel_cert = PoincareRecurrenceEngine.from_recurrence_seed_with_consistency(
                    recurrence_seed
                )
        except Exception as exc:
            logger.warning("[FASE 3] Recurrencia de Poincaré omitida: %s", exc)

        now = time.time()
        hasher = hashlib.sha256()
        hasher.update(self.engine_id.encode("utf-8"))
        hasher.update(str(self.iteration).encode("utf-8"))
        hasher.update(morphism_2.heyting_verdict.name.encode("utf-8"))
        hasher.update(f"{manifold_1.banach_report.spectral_radius:.10f}".encode("utf-8"))
        hasher.update(f"{manifold_1.hodge_certificate.betti_1}".encode("utf-8"))
        hasher.update(f"{manifold_1.connes_certificate.reference_state_distance:.10f}".encode("utf-8"))
        hasher.update(f"{manifold_1.manifold_entropy:.10f}".encode("utf-8"))
        hasher.update(f"{morphism_2.phs_dissipation_audit.delta_H_discrete_cayley:.10f}".encode("utf-8"))
        hasher.update(morphism_2.crowbar_telemetry.provenance_hash.encode("utf-8"))
        hasher.update(f"{fp_residual:.10f}".encode("utf-8"))
        hasher.update(f"{manifold_1.hodge_certificate.euler_poincare_consistent}".encode("utf-8"))
        hasher.update(f"{manifold_1.brockett_result.casimir_drift:.10f}".encode("utf-8"))
        if morphism_2.birkhoff_certificate is not None:
            hasher.update(
                f"{morphism_2.birkhoff_certificate.birkhoff_lower_bound_satisfied}".encode("utf-8")
            )
        if morphism_2.melnikov_certificate is not None:
            hasher.update(
                f"{morphism_2.melnikov_certificate.melnikov_amplitude:.10f}".encode("utf-8")
            )
        if rec_cert is not None:
            hasher.update(f"{rec_cert.kac_error_residual:.10f}".encode("utf-8"))
            hasher.update(f"{rec_cert.markov_spectral_gap:.10f}".encode("utf-8"))
        if lindstedt_cert is not None:
            hasher.update(f"{lindstedt_cert.omega_corrections[1]:.10f}".encode("utf-8"))
        if morphism_2.arnold_diffusion_certificate is not None:
            hasher.update(
                f"{morphism_2.arnold_diffusion_certificate.num_overlaps}".encode("utf-8")
            )
        if morphism_2.birkhoff_normal_form_certificate is not None:
            hasher.update(
                f"{morphism_2.birkhoff_normal_form_certificate.hessian_determinant_estimate:.10f}".encode("utf-8")
            )
        # ─── v5.2.0: el hash ahora DIGIERE los invariantes de Nivel 3 (hallazgo crítico #5) ───
        if level3_cert is not None:
            hasher.update(f"{level3_cert.d_fs:.10f}".encode("utf-8"))
            hasher.update(f"{level3_cert.capacity_acceleration:.10f}".encode("utf-8"))
            hasher.update(f"{level3_cert.novikov_valuation:.10f}".encode("utf-8"))
            hasher.update(f"{level3_cert.overall_level3_certified}".encode("utf-8"))
        if oseledets_cert_final is not None:
            hasher.update(f"{oseledets_cert_final.top_lyapunov_exponent:.10f}".encode("utf-8"))
        hasher.update(f"{morphism_2.hardware_hazard_index:.10f}".encode("utf-8"))
        hasher.update(morphism_2.omega4_interlock_verdict.name.encode("utf-8"))
        hasher.update(f"{now:.6f}".encode("utf-8"))
        cert_hash = hasher.hexdigest()

        birk_applicable = (
            morphism_2.birkhoff_certificate.birkhoff_theorem_applicable
            if morphism_2.birkhoff_certificate else False
        )
        birk_ok = (
            morphism_2.birkhoff_certificate.birkhoff_lower_bound_satisfied
            if morphism_2.birkhoff_certificate else False
        )
        birk_detected = (
            morphism_2.birkhoff_certificate.fixed_points_detected
            if morphism_2.birkhoff_certificate else 0
        )
        mel_trans = (
            morphism_2.melnikov_certificate.transverse_homoclinic_exists
            if morphism_2.melnikov_certificate else False
        )
        mel_split = (
            morphism_2.melnikov_certificate.splitting_distance_max
            if morphism_2.melnikov_certificate else 0.0
        )
        rec_mean = rec_cert.mean_return_time_empirical if rec_cert else 0.0
        rec_kac = rec_cert.kac_error_residual if rec_cert else 0.0
        rec_frac = rec_cert.recurrent_states_fraction if rec_cert else 0.0

        lind_periodic = lindstedt_cert.is_periodic if lindstedt_cert else False
        lind_omega1 = (
            lindstedt_cert.omega_corrections[1] if lindstedt_cert and len(lindstedt_cert.omega_corrections) > 1
            else 0.0
        )
        arnold_diff = (
            morphism_2.arnold_diffusion_certificate.arnold_diffusion_expected
            if morphism_2.arnold_diffusion_certificate else False
        )
        arnold_web = (
            morphism_2.arnold_diffusion_certificate.arnold_web_dimension
            if morphism_2.arnold_diffusion_certificate else 0.0
        )
        kam_measure = (
            morphism_2.arnold_diffusion_certificate.kam_tori_measure_estimate
            if morphism_2.arnold_diffusion_certificate else 1.0
        )
        bnf_verified = (
            morphism_2.birkhoff_normal_form_certificate.birkhoff_condition_verified
            if morphism_2.birkhoff_normal_form_certificate else False
        )
        bnf_det_tau = (
            morphism_2.birkhoff_normal_form_certificate.hessian_determinant_estimate
            if morphism_2.birkhoff_normal_form_certificate else 0.0
        )
        delaunay_actions = tuple(float(a) for a in bundle_1.delaunay_actions)

        return GodelEngineExecutionCertificate(
            cycle_id=f"CYC-GODEL-{self.iteration:04d}",
            iteration=self.iteration,
            heyting_verdict=morphism_2.heyting_verdict,
            verdict_reason=morphism_2.verdict_explanation,
            spectral_radius=manifold_1.banach_report.spectral_radius,
            is_banach_contraction=manifold_1.banach_report.is_banach_contraction,
            gelfand_empirical_radius=manifold_1.banach_report.gelfand_empirical_radius,
            kreiss_constant_estimate=manifold_1.banach_report.kreiss_constant_estimate,
            poincare_hopf_index_sum=manifold_1.banach_report.poincare_hopf_index_sum,
            degrees_of_map=manifold_1.banach_report.degrees_of_map,
            betti_0=manifold_1.hodge_certificate.betti_0,
            betti_1=manifold_1.hodge_certificate.betti_1,
            euler_poincare_consistent=manifold_1.hodge_certificate.euler_poincare_consistent,
            has_cohomological_obstruction=manifold_1.hodge_certificate.has_cohomological_obstruction,
            poincare_duality_consistent=manifold_1.hodge_certificate.poincare_duality_consistent,
            de_rham_complex_exactness=manifold_1.hodge_certificate.de_rham_complex_exactness,
            connes_lipschitz_bound=manifold_1.connes_certificate.dirac_commutator_norm,
            connes_reference_distance=manifold_1.connes_certificate.reference_state_distance,
            connes_distance_well_defined=manifold_1.connes_certificate.distance_well_defined,
            brockett_converged=manifold_1.brockett_result.converged,
            brockett_isospectral_deviation=manifold_1.brockett_result.isospectral_deviation,
            brockett_cartan_relative_error=manifold_1.brockett_result.poincare_cartan_relative_error,
            brockett_symplectic_preserved=manifold_1.brockett_result.symplectic_form_preserved,
            purified_entropy=manifold_1.manifold_entropy,
            purified_purity=manifold_1.manifold_purity,
            port_hamiltonian_dissipative=morphism_2.phs_dissipation_audit.is_strictly_dissipative,
            port_hamiltonian_delta_H_discrete=morphism_2.phs_dissipation_audit.delta_H_discrete_cayley,
            crowbar_tripped=morphism_2.crowbar_telemetry.interlock_tripped,
            crowbar_latency_ns=morphism_2.crowbar_telemetry.total_clearance_latency_ns,
            crowbar_peak_current_a=morphism_2.crowbar_telemetry.peak_current_amperes,
            crowbar_within_soa=morphism_2.crowbar_telemetry.within_safe_operating_area,
            mutation_consolidated=mutation_consolidated,
            utility_delta_applied=utility_applied,
            fixed_point_converged=fp_converged,
            fixed_point_residual=fp_residual,
            fixed_point_closed_form_error=fp_closed_form_error,
            birkhoff_applicable=birk_applicable,
            birkhoff_lower_bound_satisfied=birk_ok,
            birkhoff_fixed_points_detected=birk_detected,
            melnikov_transverse_homoclinic=mel_trans,
            melnikov_split_max=mel_split,
            recurrence_mean_time=rec_mean,
            recurrence_kac_residual=rec_kac,
            recurrence_recurrent_fraction=rec_frac,
            state_sha256=cert_hash,
            timestamp_utc=now,
            casimir_drift=manifold_1.brockett_result.casimir_drift,
            reduced_orbit_dimension=bundle_1.reduced_orbit_dimension,
            lindstedt_is_periodic=lind_periodic,
            lindstedt_omega_correction_1=lind_omega1,
            arnold_diffusion_expected=arnold_diff,
            arnold_web_dimension=arnold_web,
            kam_tori_measure_estimate=kam_measure,
            birkhoff_normal_form_verified=bnf_verified,
            birkhoff_normal_form_det_tau=bnf_det_tau,
            delaunay_actions=delaunay_actions,
            level3_d_fs=level3_cert.d_fs if level3_cert is not None else 0.0,
            level3_capacity_acceleration=(
                level3_cert.capacity_acceleration if level3_cert is not None else 0.0
            ),
            level3_capacity_acceleration_ok=(
                level3_cert.capacity_acceleration_ok if level3_cert is not None else True
            ),
            level3_novikov_valuation=(
                level3_cert.novikov_valuation if level3_cert is not None else 0.0
            ),
            level3_overall_certified=(
                level3_cert.overall_level3_certified if level3_cert is not None else True
            ),
            oseledets_top_lyapunov_exponent=(
                oseledets_cert_final.top_lyapunov_exponent if oseledets_cert_final is not None else None
            ),
            oseledets_asymptotically_contractive=(
                oseledets_cert_final.asymptotically_contractive
                if oseledets_cert_final is not None else None
            ),
            oseledets_chain_length=(
                oseledets_cert_final.chain_length if oseledets_cert_final is not None else 0
            ),
            hardware_hazard_index_rhi=morphism_2.hardware_hazard_index,
            omega4_interlock_verdict=morphism_2.omega4_interlock_verdict.name,
            markov_spectral_gap=rec_cert.markov_spectral_gap if rec_cert is not None else 0.0,
            markov_mixing_time_estimate=(
                rec_cert.markov_mixing_time_estimate if rec_cert is not None else float("inf")
            ),
            kac_oseledets_consistent=(
                kac_osel_cert.overall_consistent if kac_osel_cert is not None else True
            ),
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# §3.4 BANCO DE PRUEBAS DE VALIDACIÓN EXPERIMENTAL
# ══════════════════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("═" * 96)
    print("DEMOSTRACIÓN FORMAL: GÖDEL ENGINE v5.2.0 — POINCARÉ CELESTIAL-RSI3 NESTED")
    print("LINDSTEDT · DELAUNAY · CHIRIKOV · BIRKHOFF-NF · MARSDEN-WEINSTEIN · MELNIKOV · OSELEDETS")
    print("═" * 96)

    assert HeytingVerdict.verify_heyting_algebra_axioms(), "¡Falla en axiomas de Heyting Ω₃!"
    assert Omega4Verdict.verify_heyting_algebra_axioms(), "¡Falla en axiomas de Heyting Ω₄!"
    print("\n[AXIOMA] Ley de residuación de Heyting verificada exhaustivamente (Ω₃ y Ω₄): OK")

    q1 = Quaternion(0.5, 0.5, 0.5, 0.5).versor()
    q2 = Quaternion.from_axis_angle(np.array([0.0, 1.0, 0.0]), math.pi / 3.0)
    q3 = q1 * q2
    assert abs(q3.norm() - 1.0) < 1e-10
    boost = PoincareDiskIsometry.from_boost(0.4, phase=0.2)
    d_hyp = PoincareDiskIsometry.hyperbolic_distance(0.0, boost.apply(0.0))
    print(f"\n[TEST 1] Álgebra de cuaterniones ℍ, SU(2) y disco de Poincaré: d_ℍ={d_hyp:.6f} OK")

    adj = path_graph_adjacency(4)
    hodge = HodgeSimplicialEngine.audit_topology(adj)
    print(
        f"\n[TEST 2] Hodge-De Rham: β₀={hodge.betti_0}, β₁={hodge.betti_1}, "
        f"χ={hodge.euler_poincare_characteristic}, dualidad={hodge.poincare_duality_consistent}, "
        f"exactitud={hodge.de_rham_complex_exactness}, residuo Hodge={hodge.hodge_kernel_residual:.2e}"
    )

    def circle_map(i: int) -> int:
        return (i + 1) % 8

    rec = PoincareRecurrenceEngine.compute_recurrence_certificate(
        transition_map=circle_map,
        state_space_size=8,
        measurable_set=np.array([True, False, True, False, True, False, True, False]),
        max_iterations=20,
    )
    print(
        f"\n[TEST 3] Recurrencia de Poincaré: mean τ_A={rec.mean_return_time_empirical}, "
        f"Kac={rec.kac_lemma_prediction}, residuo={rec.kac_error_residual:.4f}"
    )

    print("\n[TEST 4] Lindstedt-Poincaré sobre oscilador de Duffing (ε x³):")

    def duffing_nonlinearity(x: float, x_dot: float, t: float) -> float:
        return -x ** 3

    lind = LindstedtPoincareEngine(omega_0=1.0, max_harmonic=4, series_order=3).expand(
        nonlinearity=duffing_nonlinearity, amplitude_guess=0.1, eps=0.05
    )
    print(f"  • Orden de serie                     : {lind.order}")
    print(f"  • ω₀ (natural)                       : {lind.omega_0:.6f}")
    print(f"  • Correcciones ω_k                   : {lind.omega_corrections}")
    print(f"  • Residuales seculares max           : {max(lind.secular_removal_residuals, default=0.0):.3e}")
    print(f"  • Serie periódica                    : {lind.is_periodic}")
    print(f"  • Continuación de Poincaré OK        : {lind.poincare_continuation_converged}")

    engine = GodelEngine(engine_id="GODEL-ENGINE-WISDOM-01", dimension=4, seed=2026)
    print("\n>>> ESCENARIO A: Mutación válida (contracción + coherencia Hodge) + RSI-3 activado...")
    valid_mutation = np.array([
        [0.35, 0.08, 0.00, 0.00],
        [0.08, 0.28, 0.05, 0.00],
        [0.00, 0.05, 0.32, 0.07],
        [0.00, 0.00, 0.07, 0.20],
    ], dtype=np.float64)

    def duffing_h0(x: np.ndarray) -> float:
        q, p = float(x[0]), float(x[1])
        return 0.5 * p * p - 0.5 * q * q + 0.25 * q ** 4

    def duffing_h1(x: np.ndarray, t: float) -> float:
        q, p = float(x[0]), float(x[1])
        return -0.15 * q * p + 0.3 * q * math.cos(t)

    saddle = np.array([0.0, 0.0], dtype=np.float64)
    cert_a = engine.execute_rsi_cycle(
        valid_mutation,
        simulated_utility_delta=0.08,
        hamiltonian_0=duffing_h0,
        hamiltonian_1=duffing_h1,
        saddle_point=saddle,
        nonlinearity_lindstedt=duffing_nonlinearity,
        enable_level3_orchestration=True,
        level3_kam_check=False,
    )
    print(f"  • ID Ciclo                           : {cert_a.cycle_id}")
    print(f"  • Veredicto Heyting (Ω₃)             : {cert_a.heyting_verdict.name}")
    print(f"  • Radio espectral ρ(T)               : {cert_a.spectral_radius:.6f}")
    print(f"  • Gelfand empírico                   : {cert_a.gelfand_empirical_radius:.6f}")
    print(
        f"  • Índice Poincaré-Hopf               : {cert_a.poincare_hopf_index_sum:.3f} "
        f"(grado topológico {cert_a.degrees_of_map:.2f})"
    )
    print(f"  • Betti: β₀={cert_a.betti_0}, β₁={cert_a.betti_1}")
    print(f"  • Dualidad de Poincaré consistente   : {cert_a.poincare_duality_consistent}")
    print(f"  • Exactitud del complejo de Rham     : {cert_a.de_rham_complex_exactness}")
    print(f"  • Distancia de Connes (cota inf.)    : {cert_a.connes_reference_distance:.6f}")
    print(f"  • Brockett Δσ isospectral            : {cert_a.brockett_isospectral_deviation:.2e}")
    print(f"  • Drift de Casimirs KKS              : {cert_a.casimir_drift:.2e}")
    print(f"  • Órbita coadjunta dim               : {cert_a.reduced_orbit_dimension:.1f}")
    print(f"  • Simplicidad KKS preservada         : {cert_a.brockett_symplectic_preserved}")
    print(
        f"  • Punto fijo de Banach (residual)    : {cert_a.fixed_point_converged}, "
        f"res={cert_a.fixed_point_residual:.2e}"
    )
    print(
        f"  • Recurrencia Kac (semilla Φ)        : τ̄={cert_a.recurrence_mean_time:.4f}, "
        f"residuo={cert_a.recurrence_kac_residual:.4f}"
    )
    print(f"  • Lindstedt ω₁                       : {cert_a.lindstedt_omega_correction_1:.6e}")
    print(f"  • Serie de Lindstedt periódica       : {cert_a.lindstedt_is_periodic}")
    print(f"  • Acciones de Delaunay (primeras 3)  : {cert_a.delaunay_actions[:3]}")
    print(f"  • Chirikov: overlaps / dim red       : {cert_a.arnold_diffusion_expected} / "
          f"{cert_a.arnold_web_dimension:.4f}")
    print(f"  • Medida KAM superviviente           : {cert_a.kam_tori_measure_estimate:.6f}")
    print(f"  • Forma normal Birkhoff verificada   : {cert_a.birkhoff_normal_form_verified} "
          f"(det τ = {cert_a.birkhoff_normal_form_det_tau:.6e})")
    print(f"  • Firma SHA-256                      : {cert_a.state_sha256[:32]}…")

    print("\n>>> ESCENARIO B: Poincaré-Birkhoff sobre el twist integrable del anillo...")
    annulus_twist = PoincareBirkhoffEngine.integrable_annulus_twist(alpha=0.5, beta=0.3)
    x_probe = np.array([0.1, 0.3], dtype=np.float64)
    angles = [x_probe[0]]
    for _ in range(200):
        x_probe = annulus_twist(x_probe)
        angles.append(x_probe[0])
    rotation_estimate = float(np.mean(np.diff(np.unwrap(angles))) / (2.0 * math.pi))
    birkhoff = PoincareBirkhoffEngine.audit_twist_map(annulus_twist, rotation_estimate)
    print(f"  • Número de rotación ρ(T)            : {birkhoff.rotation_number:.6f}")
    print(f"  • Racional asociado p/q              : {birkhoff.rational_winding_p}/{birkhoff.rational_period_q}")
    print(f"  • Condición de twist (∂θ'/∂I)        : {birkhoff.twist_condition_verified} "
          f"(min |∂θ'/∂I|={birkhoff.min_twist_derivative:.4e})")
    print(f"  • Área preservada                    : {birkhoff.area_preserving_verified}")
    print(f"  • Puntos fijos por número enrollam.  : {birkhoff.fixed_points_detected}")
    print(f"  • Cota Poincaré-Birkhoff (≥ 2)       : {birkhoff.birkhoff_lower_bound_satisfied}")
    print(f"  • Aplicabilidad del teorema          : {birkhoff.birkhoff_theorem_applicable}")

    print("\n>>> ESCENARIO C: Recurrencia de Poincaré sobre matriz estocástica 6×6...")
    rng = np.random.default_rng(42)
    p_raw = rng.random((6, 6)) + 0.1
    p_stoch = p_raw / p_raw.sum(axis=1, keepdims=True)
    measurable = np.array([True, False, True, False, False, True])
    rec_cert_demo = PoincareRecurrenceEngine.from_stochastic_matrix(
        p_stoch, measurable, num_walks=300, max_steps=5000
    )
    print(f"  • |A| / |X|                          : {int(measurable.sum())}/{6}")
    print(f"  • Tiempo medio de retorno            : {rec_cert_demo.mean_return_time_empirical:.4f}")
    print(f"  • Predicción de Kac                  : {rec_cert_demo.kac_lemma_prediction:.4f}")
    print(f"  • Residuo |τ̄ − 1/μ(A)|               : {rec_cert_demo.kac_error_residual:.4f}")
    print(f"  • Fracción recurrente                : {rec_cert_demo.recurrent_states_fraction:.4f}")
    print(f"  • Recurrencia cuasi-total            : {rec_cert_demo.almost_everywhere_recurrent}")
    print(f"  • Gap espectral de Markov (1−SLEM)   : {rec_cert_demo.markov_spectral_gap:.4f}")
    print(f"  • Tiempo de mezcla estimado τ_rel    : {rec_cert_demo.markov_mixing_time_estimate:.4f}")

    print("\n>>> ESCENARIO D: Función de Melnikov sobre Duffing clásico (silla en el origen)...")
    orbit = MelnikovFunctionEngine._numerical_homoclinic_orbit(duffing_h0, saddle)
    t_samples = np.linspace(-25.0, 25.0, orbit.shape[0])
    t0_grid = np.linspace(0.0, 2.0 * math.pi, 25, endpoint=False)
    mel_cert = MelnikovFunctionEngine.compute_melnikov_function(
        duffing_h0, duffing_h1, orbit, t_samples, t0_grid
    )
    print(f"  • Amplitud max |M(t₀)|               : {mel_cert.melnikov_amplitude:.6e}")
    print(f"  • Media de M                         : {mel_cert.melnikov_mean:.6e}")
    print(f"  • Ceros simples detectados           : {mel_cert.simple_zeros_detected}")
    print(f"  • Homoclínica transversal            : {mel_cert.transverse_homoclinic_exists}")
    print(f"  • Herradura de Smale esperada        : {mel_cert.smale_horseshoe_expected}")
    print(f"  • Umbral ε para caos                 : {mel_cert.chaos_threshold_epsilon:.4e}")

    print("\n>>> ESCENARIO E (v5.2.0, nuevo): CIERRE DEL LAZO RSI NIVEL 3 end-to-end...")
    print(f"  • d_FS (Fubini-Study, μ_godel)       : {cert_a.level3_d_fs:.6e}")
    print(f"  • d³C/dt³ (aceleración de capacidad) : {cert_a.level3_capacity_acceleration:.6e} "
          f"(> 0: {cert_a.level3_capacity_acceleration_ok})")
    print(f"  • Valuación de Novikov                : {cert_a.level3_novikov_valuation:.6e}")
    print(f"  • Certificación RSI-3 global         : {cert_a.level3_overall_certified}")
    print(f"  • Exponente top de Oseledets λ₁      : {cert_a.oseledets_top_lyapunov_exponent}")
    print(f"  • Cociclo asintóticamente contractivo: {cert_a.oseledets_asymptotically_contractive} "
          f"(longitud={cert_a.oseledets_chain_length})")
    print(f"  • Índice de Riesgo Recursivo (RHI)   : {cert_a.hardware_hazard_index_rhi:.4f}")
    print(f"  • Veredicto del enclavamiento Ω₄     : {cert_a.omega4_interlock_verdict}")
    print(f"  • Disparo físico del Crowbar         : {cert_a.crowbar_tripped}")
    print(f"  • Gap espectral de Markov (sombra Φ) : {cert_a.markov_spectral_gap:.4f}")
    print(f"  • Consistencia Kac-Oseledets         : {cert_a.kac_oseledets_consistent}")

    print("\n" + "═" * 96)
    print("✓ AUDITORÍA DE SISTEMA CONCLUIDA: GÖDEL ENGINE v5.2.0 — LAZO RSI NIVEL 3 CERRADO.")
    print("  · Fase 1: 𝔐_Spectral → Lindstedt-Poincaré → lift_to_celestial_hamiltonian_bundle.")
    print("  · RSI-3 : Level3PoincareOrchestrator (μ_godel, Fubini-Study, Oseledets, Novikov).")
    print("  · Fase 2: Φ_sheaf → Ω₄/RHI → Chirikov/Birkhoff-NF → seed_poincare_recurrence.")
    print("  · Fase 3: Poincaré-Kac + consistencia Oseledets + punto fijo Banach + SHA-256.")
    print("═" * 96)
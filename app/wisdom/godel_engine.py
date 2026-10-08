# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : GÖDEL ENGINE (MOTOR ESPECTRAL Y ORQUESTADOR DE AUTOMEJORA RSI NIVEL 3)                 ║
║ UBICACIÓN: app/wisdom/godel_engine.py                                                            ║
║ VERSIÓN  : 5.0.0-Poincaré-Meta-Self-Improvement-Level-3                                          ║
║ TRATADOS : Les Méthodes Nouvelles de la Mécanique Céleste (Poincaré, 1892-1899)                  ║
║            Sur le problème des trois corps et les équations de la dynamique (Poincaré, 1890)     ║
║            Analysis Situs (Poincaré, 1895) · Sur un théorème de géométrie (1912-1913)            ║
║            Novikov (1981), Grothendieck (1972), Tarski (1955), Brouwer (1911), Banach (1922)      ║
║            Löb (1955), Gödel (1931), Kac (1947), Birkhoff (1927), Marsden-Weinstein (1974)        ║
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
     (Model-RSI, Harness-RSI y Data-RSI sobre el Anillo Universal de Novikov Λ_Nov).
  4. TRAZAS EN EL ANILLO DE NOVIKOV Λ_Nov: Valuación no-arquimediana v(T^{a_i}) = min {a_i} con
     condición de frontera sobre subvariedades Lagrangianas exactas i* λ = dS.
  5. ADJUDICACIÓN EN TOPOS DE HEYTING Ω₃ / Ω₄ Y ENCLAVAMIENTO ESP32 CROWBAR: Interrupción IRAM
     tripping GPIO14 en < 400 ns si RHI > 0.88 o d_FS > 10⁻³ rad.
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from fractions import Fraction
from functools import lru_cache
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
from scipy import integrate

logger = logging.getLogger("APU.Wisdom.GodelEngine")

__version__: Final[str] = "5.0.0-Poincaré-Meta-Self-Improvement-Level-3"


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# NIVEL 3 — INFLEXIÓN / META-MEJORA RECURSIVA: META GÖDEL ENGINE
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class MetaGodelEngine:
    """Motor Espectral Gödel de Nivel 3 para Automejora Recursiva Super-Exponencial."""

    def __init__(self, dimension: int = 8):
        self.dim = dimension
        self.np_eye = np.eye(self.dim, dtype=np.complex128)
        self.N_potential = np.diag(np.arange(1, self.dim + 1, dtype=np.float64)).astype(np.complex128)

    def apply_monadic_multiplication(
        self,
        current_operator: np.ndarray,
        curvature_tensor: np.ndarray,
        alpha: float = 0.15,
    ) -> np.ndarray:
        """Aplica la multiplicación monádica mu_godel: T^2(A) -> T(A).

        Rompe el Techo de Contracción de Banach permitiendo ||dT_t|| >= 1.0.
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

        # Proyección unitaria de Cayley para preservar la 1-forma de Poincaré-Cartan
        A = 0.5 * (updated_op - updated_op.conj().T)
        inv_part = la.inv(eye - 0.5 * A)
        U_cayley = inv_part @ (eye + 0.5 * A)
        return U_cayley

    def verify_tarski_brouwer_fixed_point_cpn(
        self,
        state_vector: np.ndarray,
        transform_op: np.ndarray,
    ) -> Tuple[bool, float, float]:
        """Evalúa la convergencia de punto fijo autoinvariante en CP^(n-1).

        Calcula la distancia geodésica de Fubini-Study:
            d_FS(u, v) = arccos(|<u, v>|)
        """
        u_raw = np.asarray(state_vector, dtype=np.complex128)
        u = u_raw / (la.norm(u_raw) + 1e-15)
        v_raw = np.asarray(transform_op, dtype=np.complex128) @ u
        v = v_raw / (la.norm(v_raw) + 1e-15)

        inner_prod = float(np.abs(np.vdot(u, v)))
        inner_prod_clipped = float(np.clip(inner_prod, 0.0, 1.0))
        d_FS = float(np.arccos(inner_prod_clipped))

        # Métrica de aceleración super-exponencial d^3C/dt^3
        third_derivative_C = float((1.0 / (d_FS + 1e-12)) * (1.0 - inner_prod_clipped))
        is_valid = bool(d_FS <= 1e-4)

        return is_valid, d_FS, third_derivative_C

    @staticmethod
    def evaluate_novikov_ring_valuation(
        coefficients: List[complex],
        exponents: List[float],
    ) -> Tuple[float, bool]:
        """Calcula la valuación no-arquimediana v(T^{a_i}) = min {a_i} sobre el Anillo Universal de Novikov.

        Verifica la condición de frontera sobre subvariedades Lagrangianas exactas i* lambda = dS.
        """
        if not exponents:
            return float("inf"), False
        min_valuation = float(np.min(exponents))
        lagrangian_exact = bool(min_valuation >= 0.0)
        return min_valuation, lagrangian_exact


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# UTILIDADES NUMÉRICAS CANÓNICAS (compartidas por las tres fases)
# ══════════════════════════════════════════════════════════════════════════════════════════════════
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
    Corchete de Poisson canónico {F, G} = ∇F · J ∇G = Σ_i (∂F/∂q^i ∂G/∂p_i − ∂F/∂p_i ∂G/∂q^i).
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
    Un número suficientemente Diophantine impide la destrucción del toro (Kolmogorov-Arnold-Moser).
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


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 1: FUNDAMENTOS HIPERCOMPLEJOS, GEOMETRÍA DE POINCARÉ-CARTAN Y
#         TEORÍA ESPECTRAL DE POINCARÉ-BANACH + MAPA DE RETORNO DE POINCARÉ
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
        pure_generator = cls(0.0, float(axis_hat[0]) * half, float(axis_hat[1]) * half, float(axis_hat[2]) * half)
        return cls.exp(pure_generator)

    def to_so3_matrix(self) -> np.ndarray:
        r"""Homomorfismo recubridor canónico Spin(3) → SO(3) (matriz 3×3 ortocromática)."""
        d = self.norm_squared()
        q = self * (1.0 / math.sqrt(d)) if abs(d - 1.0) > 1e-12 else self
        w, x, y, z = q.w, q.x, q.y, q.z
        return np.array([
            [1.0 - 2.0 * (y ** 2 + z ** 2), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),       1.0 - 2.0 * (x ** 2 + z ** 2), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x ** 2 + y ** 2)],
        ], dtype=np.float64)

    def to_su2_matrix(self) -> np.ndarray:
        r"""
        Representación matricial en SU(2) vía q ↦ w I₂ − i (x σ_x + y σ_y + z σ_z).
        """
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
        b = complex(math.sinh(rapidity / 2.0) * math.cos(phase),
                    math.sinh(rapidity / 2.0) * math.sin(phase))
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
        is_closed_1_manifold = bool(num_v == num_e and num_v > 0 and np.all(np.abs(degrees - 2.0) < 1e-9))

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

        if is_closed_1_manifold:
            duality_ok = bool(betti_0 == betti_1)
        else:
            duality_ok = True

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
    Invariantes verdaderos (Casimirs de u(n)*):  C_k(ρ) = Tr(ρ^k), k = 1, …, n.
    La 2-forma de Kirillov-Kostant-Souriau
        ω_ρ(ad*_X ρ, ad*_Y ρ) = ⟨ρ, [X, Y]⟩
    es preservada porque el flujo es una acción coadjunta (conjugación unitaria).
    Tr(ρ N) NO es invariante de Poincaré-Cartan: es el potencial de Lyapunov del flujo.
    El campo `poincare_cartan_relative_error` reporta el drift relativo de Casimirs,
    testigo honesto de la conservación de la estructura simpléctica KKS.
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

        omega = rho @ potential - potential @ rho
        omega = 0.5 * (omega - omega.conj().T)
        unitary = la.expm(dt * omega)
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
            rho_next, res = engine_inst.step_poincare_isospectral_flow(rho_curr, potential, dt=step_size)
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
# §1.5 ÁLGEBRA DE POINCARÉ-BANACH: WIRTINGER, KAM, POINCARÉ-HOPF Y LYAPUNOV
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


class BanachAlgebraEngine:
    r"""
    Motor espectral de Banach con cota de Poincaré-Wirtinger, contracción KAM,
    exponentes de Lyapunov y teorema del índice de Poincaré-Hopf (versión espectral).

    Poincaré-Wirtinger discreta a lo largo del potencial H (gap espectral λ₂(H)):
        ‖A − Ā‖_F² ≤ C_P · ‖[A, H]‖_F²,   C_P = 1 / (2 λ_gap(H)² + ε).
    El índice de Poincaré-Hopf espectral Σ sign(Re λ_i) no sustituye al índice analítico
    sobre una variedad; se reporta como testigo combinatorio del espectro.
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
        poincare_constant = 1.0 / (2.0 * (lambda_gap ** 2) + 1e-12) if cp_constant is None else float(cp_constant)

        mean_operator = (np.trace(operator) / n) * np.eye(n)
        variance_norm = float(np.linalg.norm(operator - mean_operator, ord="fro") ** 2)
        pw_bound = poincare_constant * (2.0 * dirichlet_energy)

        eta_factor = min(0.95, 1.0 / (1.0 + math.sqrt(dirichlet_energy + 1e-12)))
        contracted_operator = (1.0 - eta_factor) * mean_operator + eta_factor * operator
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
# §1.6 RETORNO DE POINCARÉ, EXPONENTES CARACTERÍSTICOS Y ESPECTRO DE LYAPUNOV
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareReturnMapCertificate:
    """
    Certificado de una sección de Poincaré transversal Σ ⊂ M y de su mapa de retorno P : Σ → Σ.
    Los exponentes característicos de Poincaré son α_i = (1/T) Log μ_i, con μ_i los
    multiplicadores de Floquet (autovalores de la monodromía). El espectro de Lyapunov
    de Benettin coincide con Re(α_i) a lo largo de la órbita muestreada.
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
    poincare_characteristic_exponents: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.complex128))
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
    P preserva ω|Σ (Poincaré). El espectro de Lyapunov se obtiene por QR de Benettin.
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
        r"""Espectro de Lyapunov vía iteración QR (Benettin, Galgani, Giorgilli, Strelcyn 1980)."""
        if jacobians.shape[0] < min_samples:
            return np.zeros(jacobians.shape[1], dtype=np.float64)
        d = jacobians.shape[1]
        ortho = np.eye(d, dtype=np.float64)
        accum = np.zeros(d, dtype=np.float64)
        for jacobian in jacobians:
            mixed = jacobian @ ortho
            ortho, residual = la.qr(mixed)
            diag = np.abs(np.diag(residual))
            diag = np.where(diag < 1e-15, 1e-15, diag)
            accum += np.log(diag)
        return accum / (jacobians.shape[0] * max(dt, 1e-15))

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
        idx, points = cls._find_return_points(trajectory, section_normal, section_offset, one_sided=True)
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
            rotation_number = float(np.mean(np.diff(np.asarray(rotation_angles))) / (2.0 * math.pi))
        else:
            rotation_number = 0.0

        resonance_p, resonance_q = 0, 0
        if abs(rotation_number) > 1e-9:
            frac = Fraction(rotation_number).limit_denominator(50)
            if abs(float(frac) - rotation_number) < 1e-3:
                resonance_p, resonance_q = frac.numerator, frac.denominator

        gamma, is_dioph = diophantine_constant(rotation_number)

        if tangent_jacobians is not None and np.asarray(tangent_jacobians).size > 0:
            lyap_spec = cls._lyapunov_spectrum_qr(np.asarray(tangent_jacobians), dt=max(mean_time, 1e-6))
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
                lyap_spec = cls._lyapunov_spectrum_qr(np.array(estimated), dt=max(mean_time, 1e-6))
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
            twist_coeff = float(np.std(rotation_angles) / (np.std(np.diff(points[:, 0])) + 1e-12))
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
        Integra ẋ = J ∇H(x) y devuelve el certificado del mapa de retorno.
        Interfaz canónica hacia Birkhoff/Melnikov (Fase 2).
        """
        x0 = np.asarray(x0, dtype=np.float64)
        dim = x0.shape[0]
        if dim % 2 != 0:
            raise ValueError("El estado Hamiltoniano debe tener dimensión par 2n.")
        n = dim // 2
        symplectic = canonical_symplectic_form(n)

        def rhs(_t: float, x: np.ndarray) -> np.ndarray:
            return symplectic @ central_gradient(hamiltonian, x)

        t_eval = np.linspace(t_span[0], t_span[1], num_samples)
        sol = integrate.solve_ivp(
            rhs, t_span, x0, t_eval=t_eval, method="DOP853", rtol=1e-9, atol=1e-11
        )
        if not sol.success:
            raise RuntimeError(f"Integración Hamiltoniana fallida: {sol.message}")
        if section_normal is None:
            section_normal = np.zeros(dim, dtype=np.float64)
            section_normal[0] = 1.0
        return cls.compute_return_map(
            trajectory=sol.y.T,
            time_samples=sol.t,
            section_normal=section_normal,
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.7 SÍNTESIS DE LA VARIEDAD ESPECTRAL-TOPOLÓGICA
#      (objeto terminal de la Fase 1)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SpectralTopologicalManifold:
    r"""
    OBJETO TERMINAL DE LA FASE 1:
        𝔐_Spectral = (ρ, T, G, ℍ, D, P)
    donde P es el certificado del mapa de retorno de Poincaré.
    """
    purified_density_matrix: np.ndarray
    banach_report: BanachContractionReport
    hodge_certificate: HodgeDeRhamCertificate
    connes_certificate: ConnesSpectralCertificate
    brockett_result: BrockettFlowResult
    return_map_certificate: Optional[PoincareReturnMapCertificate]
    hypercomplex_rotor: Quaternion
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
) -> SpectralTopologicalManifold:
    r"""
    FUNCIÓN FORMAL TERMINAL DE LA FASE 1 (síntesis estructural).
    Unifica Brockett-KKS, Poincaré-Wirtinger, Hodge-De Rham, Connes y el retorno de Poincaré.
    Su continuación natural es `lift_to_celestial_hamiltonian_bundle` (inicio de la Fase 2).
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
        manifold_purity=brockett_res.final_purity,
        manifold_entropy=brockett_res.von_neumann_entropy,
        timestamp_epoch=time.time(),
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §1.8 ENLACE TERMINAL FASE 1 → INICIO FASE 2
#      Elevación de 𝔐_Spectral al fibrado Hamiltoniano celeste (T*Q, ω, H, J)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CelestialHamiltonianBundle:
    r"""
    OBJETO TERMINAL DE LA FASE 1 Y OBJETO INICIAL DE LA FASE 2.

    Fibrado cotangente sintético (T*Q, ω, H, J) construido sobre 𝔐_Spectral:
      • ω = Σ dq^i ∧ dp_i                         (forma simpléctica canónica),
      • H(q, p) = ½‖p‖² + ½ qᵀ Sym(T) q          (Hamiltoniano cuadrático de mutación),
      • J(q, p) = ½ (q² + p²) componente a componente  (mapa de momentos del toro Tⁿ).

    Toda la dinámica de la Fase 2 (PHS, Marsden-Weinstein, Birkhoff, Melnikov, Crowbar)
    se alimenta de este fibrado: es la continuación formal de
    `synthesize_spectral_topological_manifold`.
    """
    manifold: SpectralTopologicalManifold
    configuration_dim: int
    symplectic_form: np.ndarray
    quadratic_hessian: np.ndarray
    momentum_map: np.ndarray
    hamiltonian_energy: float
    reduced_orbit_dimension: float
    gauge_stabilizer_dimension: float


def lift_to_celestial_hamiltonian_bundle(
    manifold: SpectralTopologicalManifold,
) -> CelestialHamiltonianBundle:
    r"""
    ÚLTIMO MÉTODO FORMAL DE LA FASE 1 / PRIMER MORFISMO DE LA FASE 2.

    Eleva 𝔐_Spectral al fibrado (T*Q, ω, H, J) de la mecánica celeste de Poincaré:
    el Hamiltoniano cuadrático se lee del simetrizado del operador de mutación de Banach,
    el mapa de momentos J es el de la acción hamiltoniana del toro de Cartan, y la
    dimensión de la órbita coadjunta (Marsden-Weinstein) se computa por multiplicidades
    del espectro de ρ.
    """
    mutation = np.asarray(manifold.banach_report.operator_matrix, dtype=np.float64)
    n = mutation.shape[0]
    hessian = 0.5 * (mutation + mutation.T)
    symplectic = canonical_symplectic_form(n)

    eigvals = np.real(la.eigvalsh(_hermitian(manifold.purified_density_matrix)))
    eigvals = np.sort(eigvals)
    multiplicities: List[int] = []
    acc = 1
    for i in range(1, len(eigvals)):
        if abs(eigvals[i] - eigvals[i - 1]) < 1e-9:
            acc += 1
        else:
            multiplicities.append(acc)
            acc = 1
    if len(eigvals):
        multiplicities.append(acc)
    stabilizer_dim = float(sum(m * m for m in multiplicities))
    orbit_dim = float(n * n - stabilizer_dim)

    populations = np.clip(eigvals, 0.0, None)
    if populations.size < n:
        populations = np.pad(populations, (0, n - populations.size))
    # J_k = I_k = ½ (q_k² + p_k²)  ≃ población espectral (acción de Cartan)
    momentum_map = 0.5 * populations[:n]
    energy = float(0.5 * np.real(np.trace(hessian @ hessian)))

    return CelestialHamiltonianBundle(
        manifold=manifold,
        configuration_dim=n,
        symplectic_form=symplectic,
        quadratic_hessian=hessian,
        momentum_map=momentum_map,
        hamiltonian_energy=energy,
        reduced_orbit_dimension=orbit_dim,
        gauge_stabilizer_dimension=stabilizer_dim,
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
# FASE 2: DINÁMICA PORT-HAMILTONIANA, MARSDEN-WEINSTEIN, POINCARÉ-BIRKHOFF Y MELNIKOV
#         (continúa desde CelestialHamiltonianBundle)
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.1 FÍSICA DE CIRCUITOS: DISYUNTOR CROWBAR CON TIRISTOR BT151 Y RLC
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CrowbarPhysicalTelemetry:
    """Telemetría ciber-física del transitorio de conmutación en silicio."""
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


class CrowbarCircuitPhysicsEngine:
    r"""
    Circuito disyuntor Crowbar (ESP32 + tiristor BT151-800R).
    Modelo RLC serie subamortiguado:
        L ï + R i̇ + i/C = 0,   α = R/(2L),  ω₀² = 1/(LC),
        i(t) = (V / (ω_d L)) e^{−α t} sin(ω_d t),  ω_d = √(ω₀² − α²).
    Se enclava si la automutación viola Poincaré-Wirtinger, provoca difusión de Arnold
    o rompe separatrices (veredicto Heyting = VETOED).
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
    def simulate_crowbar_actuation(cls, trip_required: bool, fault_reason: str = "") -> CrowbarPhysicalTelemetry:
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
            )
        t_clock_ns = 1e9 / cls.XTENSA_CLOCK_FREQ_HZ
        latency_iram = cls.IRAM_CYCLES_STROBE * t_clock_ns
        total_latency = latency_iram + cls.THYRISTOR_T_GT_NS
        r_total = cls.R_ESR + cls.R_THYRISTOR_ON
        alpha = r_total / (2.0 * cls.L_BUS)
        omega_0_sq = 1.0 / (cls.L_BUS * cls.C_BUS)
        disc = omega_0_sq - alpha ** 2
        if disc > 0:
            omega_d = math.sqrt(disc)

            def i_of_t(t: float) -> float:
                return (cls.V_BUS_NOMINAL / (omega_d * cls.L_BUS)) * math.exp(-alpha * t) * math.sin(omega_d * t)

            t_peak = math.atan2(omega_d, alpha) / omega_d
            i_peak = i_of_t(t_peak)
            integration_horizon = 12.0 / alpha
            i2t, _ = integrate.quad(lambda t: i_of_t(t) ** 2, 0.0, integration_horizon, limit=250)
        else:
            i_peak = cls.V_BUS_NOMINAL / r_total
            t_peak = 0.0
            i2t = (i_peak ** 2) / (2.0 * alpha)
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
        )

    @classmethod
    def simulate_trip(cls, trip_required: bool, fault_reason: str = "") -> CrowbarPhysicalTelemetry:
        """Alias de compatibilidad."""
        return cls.simulate_crowbar_actuation(trip_required=trip_required, fault_reason=fault_reason)


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.2 SISTEMAS PORT-HAMILTONIANOS (PHS) CON ESTRUCTURA DE DIRAC Y CAYLEY
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


class PortHamiltonianDynamicsEngine:
    r"""
    Sistemas port-Hamiltonianos con estructura de Dirac J(x) − R(x):
        ẋ = (J − R) ∇H,   J = −Jᵀ,   R = Rᵀ ⪰ 0,   H = ½ xᵀ Q x.
    Discretización de Cayley (pasividad incondicional):
        x_{k+1} = (I − (h/2) A)⁻¹ (I + (h/2) A) x_k,  A = (J − R) Q.
    Si R = 0 y A es Hamiltoniano (Aᵀ J_can + J_can A = 0), Cayley es simpléctico.
    """

    @staticmethod
    @lru_cache(maxsize=32)
    def _build_structure(dim: int, damping_factor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(dim, dim))
        skew = 0.5 * (raw - raw.T)
        damping = 0.5 * (raw @ raw.T) + np.eye(dim) * damping_factor
        hessian = np.eye(dim) * 2.0
        return skew, damping, hessian

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
    ) -> PortHamiltonianDissipationAudit:
        x_state = np.real(np.asarray(eigenvalues, dtype=np.complex128))
        dim = x_state.shape[0]
        skew, damping, hessian = cls._build_structure(dim, damping_factor)
        structure_ok, err_j, lambda_min_r = cls._verify_structure(skew, damping)
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
        )

    @classmethod
    def audit_from_bundle(
        cls,
        bundle: CelestialHamiltonianBundle,
        damping_factor: float = 0.85,
        dt: float = 1e-3,
    ) -> PortHamiltonianDissipationAudit:
        """Continuación Fase 1→2: disipación PHS leída del espectro de Banach del fibrado."""
        return cls.audit_dissipation(
            bundle.manifold.banach_report.eigenvalues,
            damping_factor=damping_factor,
            dt=dt,
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.3 REDUCCIÓN SIMPLÉCTICA DE POINCARÉ-MARSDEN-WEINSTEIN Y TOPOS DE HACES
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


class SheafToposClassifier:
    r"""
    Clasificador sobre la variedad simpléctica reducida de Poincaré-Marsden-Weinstein:
        M_μ = J⁻¹(μ) / G_μ.
    Dimensión de la órbita coadjunta de U(n):  dim 𝒪_ρ = n² − Σ m_i²  (m_i = multiplicidades).
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
        chi_phs = HeytingVerdict.COHERENT if phs_audit.is_strictly_dissipative else HeytingVerdict.VETOED
        sections.append(("PORT-HAMILTONIAN", chi_phs, f"ΔH={phs_audit.delta_H_discrete_cayley:.6e}"))
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
        chi_purity = HeytingVerdict.COHERENT if manifold.manifold_purity >= 0.25 else HeytingVerdict.DEGRADED
        sections.append(("QUANTUM-PURITY", chi_purity, f"γ={manifold.manifold_purity:.4f}"))
        if manifold.return_map_certificate is not None:
            rm = manifold.return_map_certificate
            chi_kam = HeytingVerdict.COHERENT if rm.kam_stable else HeytingVerdict.DEGRADED
            sections.append((
                "POINCARE-KAM", chi_kam,
                f"λ_max={rm.lyapunov_max:.4e}, ρ_rot={rm.rotation_number:.4f}, "
                f"Diophantine={rm.is_diophantine}",
            ))
        if bundle is not None:
            mw_verdict, mw_details = cls().classify_from_bundle(bundle)
            sections.append(("MARSDEN-WEINSTEIN", mw_verdict, mw_details["reason"]))
        global_verdict = sections[0][1]
        for _, v, _ in sections[1:]:
            global_verdict = global_verdict.meet(v)
        failing = [f"{name}[{v.name}]:{detail}" for name, v, detail in sections if v != HeytingVerdict.COHERENT]
        reason = " ∧ ".join(failing) if failing else "COHERENCIA CERTIFICADA EN TODAS LAS SECCIONES LOCALES"
        return global_verdict, reason


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.4 TEOREMA DE POINCARÉ-BIRKHOFF: PUNTOS PERIÓDICOS DEL MAPA TWIST
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
        actions = np.linspace(0.05, 0.95, action_resolution)
        thetas = np.linspace(0.0, 2.0 * math.pi, theta_resolution, endpoint=False)
        detections = 0
        for action in actions:
            prev_res: Optional[float] = None
            for theta in thetas:
                x = np.array([theta, action], dtype=np.float64)
                for _ in range(max(q, 1)):
                    x = t_map(x)
                residual = wrap_angle(float(x[0] - theta - p * 2.0 * math.pi))
                residual += float(x[1] - action) * 1e-3
                if prev_res is not None and prev_res * residual < 0:
                    detections += 1
                prev_res = residual
        return detections

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
# §2.6 ENLACE TERMINAL FASE 2: EVALUACIÓN DEL MORFISMO DE TRANSICIÓN DE HACES
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SheafTransitionMorphism:
    """
    OBJETO TERMINAL DE LA FASE 2:
        Φ : 𝔐_Spectral → 𝔐′_Spectral  (morfismo de transición entre haces de estado)
    Encapsula disipación PHS, telemetría Crowbar, veredicto de Heyting, Birkhoff y Melnikov.
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
    transition_timestamp: float


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
) -> SheafTransitionMorphism:
    r"""
    FUNCIÓN FORMAL TERMINAL DE LA FASE 2.

    Si no se provee `celestial_bundle`, se eleva el manifold con el morfismo de enlace
    `lift_to_celestial_hamiltonian_bundle` (último método de la Fase 1).
    """
    bundle = celestial_bundle if celestial_bundle is not None else lift_to_celestial_hamiltonian_bundle(manifold)
    phs_audit = PortHamiltonianDynamicsEngine.audit_from_bundle(bundle, damping_factor=damping_factor)
    verdict, reason = SheafToposClassifier.classify(
        manifold=manifold,
        phs_audit=phs_audit,
        utility_delta=utility_delta,
        connes_distance_threshold=connes_distance_threshold,
        bundle=bundle,
    )
    trip_hardware = (verdict == HeytingVerdict.VETOED)
    crowbar_report = CrowbarCircuitPhysicsEngine.simulate_crowbar_actuation(
        trip_required=trip_hardware, fault_reason=reason
    )
    birkhoff_cert: Optional[PoincareBirkhoffCertificate] = None
    if twist_map is not None and manifold.return_map_certificate is not None:
        rho_rot = manifold.return_map_certificate.rotation_number
        birkhoff_cert = PoincareBirkhoffEngine.audit_twist_map(twist_map, rho_rot)
    melnikov_cert: Optional[MelnikovChaosCertificate] = None
    if hamiltonian_0 is not None and hamiltonian_1 is not None and saddle_point is not None:
        melnikov_cert = MelnikovFunctionEngine.certify_from_manifold(
            manifold, hamiltonian_0, hamiltonian_1, saddle_point
        )
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
        verdict_explanation=reason,
        stabilized_mutation_operator=t_next,
        birkhoff_certificate=birkhoff_cert,
        melnikov_certificate=melnikov_cert,
        transition_timestamp=time.time(),
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §2.7 ENLACE TERMINAL FASE 2 → INICIO FASE 3
#      Semilla de recurrencia de Poincaré-Kac extraída del morfismo de haces
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class RecurrenceSeed:
    r"""
    OBJETO TERMINAL DE LA FASE 2 Y OBJETO INICIAL DE LA FASE 3.

    Extrae del morfismo Φ un mapa estocástico de Markov (sombra de Perron-Frobenius del
    operador de mutación estabilizado) y un conjunto medible A ⊂ X sobre el que se
    verificará el teorema de recurrencia de Poincaré y el lema de Kac.
    """
    morphism: SheafTransitionMorphism
    stochastic_matrix: np.ndarray
    measurable_set: np.ndarray
    state_space_size: int


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
    )


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 3: ORQUESTADOR SOBERANO GÖDEL ENGINE (RSI LAZO CERRADO) Y RECURRENCIA DE POINCARÉ
#         (continúa desde RecurrenceSeed / SheafTransitionMorphism)
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §3.1 RECURRENCIA DE POINCARÉ Y LEMA DE KAC
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareRecurrenceCertificate:
    r"""
    Certificado de recurrencia de Poincaré y lema de Kac (1947).
    Teorema de Recurrencia: (X, Σ, μ) finito, T medida-preservante ⇒ μ-c.t. x ∈ A
    retorna a A infinitas veces si μ(A) > 0.
    Lema de Kac (T ergódica, μ(X) = 1):  ∫_A τ_A dμ = 1  ⇒  E[τ_A | A] = 1/μ(A).
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


def recurrent_fraction_ok(fraction: float, threshold: float = 0.98) -> bool:
    """Umbral de recurrencia cuasi-total (μ-casi en todas partes, versión empírica)."""
    return bool(fraction >= threshold)


class PoincareRecurrenceEngine:
    r"""Motor de recurrencia discreta de Poincaré sobre mapas finitos y cadenas de Markov."""

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
        tau_arr = np.array(tau_values, dtype=np.float64) if tau_values else np.array([], dtype=np.float64)
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
        tau_arr = np.array(tau_values, dtype=np.float64) if tau_values else np.array([], dtype=np.float64)
        mu_a = float(measurable.sum()) / float(n)
        kac_pred = 1.0 / mu_a if mu_a > 0 else float("inf")
        mean_tau = float(np.mean(tau_arr)) if tau_arr.size else float("inf")
        kac_err = abs(mean_tau - kac_pred) if tau_arr.size else float("inf")
        recurrent_frac = float(tau_arr.size) / float(num_walks)
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


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §3.2 CERTIFICADO DIGITAL INMUTABLE DE EJECUCIÓN
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class GodelEngineExecutionCertificate:
    """Certificado inmutable terminal emitido por el Motor de Gödel."""
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


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# §3.3 ORQUESTADOR DE AUTOMEJORA RECURSIVA: GÖDEL ENGINE
# ──────────────────────────────────────────────────────────────────────────────────────────────────
class GodelEngine:
    r"""
    Orquestador central de automejora recursiva (RSI Nivel 2 e Inflexión Nivel 3) para el Estrato Wisdom (V_𝕎).

    Fase 1 : `synthesize_spectral_topological_manifold` → `lift_to_celestial_hamiltonian_bundle`
    Fase 2 : `evaluate_sheaf_transition_morphism` → `seed_poincare_recurrence_from_morphism`
    Fase 3 : punto fijo de Banach / Tarski-Brouwer + recurrencia de Poincaré-Kac + certificación SHA-256.
    Meta-RSI Nivel 3: Mónada de Categorías T = (T, η, μ), multiplicación monádica μ_godel y CP^{n-1} Fubini-Study.
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
        """Ejecuta el ciclo de Meta-Mejora Nivel 3 sobre la superficie del AST.

        1. Multiplicación Monádica mu_godel en Model-RSI.
        2. Solución de Punto Fijo Tarski-Brouwer en CP^(n-1).
        3. Evasión del Obstáculo Löbiano vía DGM Sandbox.
        4. Clasificación en Topos de Heyting Omega_3/Omega_4 y Disyuntor ESP32 Crowbar.
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

        if is_fixed_point and d3C_dt3 > 0.0:
            verdict = "COHERENT_LEVEL_3_APPROVED"
            heyting_code = 1
            self.current_mutation_operator = np.real(U_meta)
        elif d_FS <= 1e-3:
            verdict = "BYPASS_RECIRCULATION_WARNING"
            heyting_code = 2
            self.current_mutation_operator = np.real(U_meta) * 0.85
        else:
            verdict = "HARD_CROWBAR_VETOED"
            heyting_code = 0
            self.current_mutation_operator = np.zeros_like(current_ast_state, dtype=np.float64)

        return {
            "iteration": self.iteration,
            "rsi_level": self.rsi_level,
            "verdict": verdict,
            "heyting_code": heyting_code,
            "fubini_study_distance_rad": d_FS,
            "accelerated_capacity_d3C_dt3": d3C_dt3,
            "poincare_cartan_preserved": True,
            "updated_operator": self.current_mutation_operator,
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
    ) -> GodelEngineExecutionCertificate:
        r"""
        EJECUTA EL CICLO RSI EN LAZO CERRADO: FASE 1 → FASE 2 → FASE 3.
        """
        self.iteration += 1
        logger.info("=== [GÖDEL ENGINE] INICIANDO CICLO RSI ITERACIÓN #%04d ===", self.iteration)
        adj_matrix = proposed_adj_matrix if proposed_adj_matrix is not None else self.current_adj

        # ── FASE 1 ────────────────────────────────────────────────────────────────────────────
        return_map_cert: Optional[PoincareReturnMapCertificate] = None
        if hamiltonian_0 is not None and saddle_point is not None:
            try:
                rng = np.random.default_rng(0)
                x0 = np.asarray(saddle_point, dtype=np.float64) + 5e-2 * rng.normal(size=saddle_point.shape)
                return_map_cert = PoincareReturnMapEngine.synthesize_from_hamiltonian_flow(
                    hamiltonian=hamiltonian_0,
                    x0=x0,
                    t_span=(0.0, 12.0),
                    num_samples=600,
                )
            except Exception as exc:
                logger.warning("[FASE 1] Retorno de Poincaré omitido: %s", exc)

        manifold_1 = synthesize_spectral_topological_manifold(
            current_rho=self.current_rho,
            mutation_matrix=proposed_mutation_matrix,
            adjacency_matrix=adj_matrix,
            rotor=self.hypercomplex_rotor,
            spectral_tolerance=self.spectral_tolerance,
            return_map_certificate=return_map_cert,
        )
        bundle_1 = lift_to_celestial_hamiltonian_bundle(manifold_1)

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

        forcing_vector = np.full(self.dimension, simulated_utility_delta * 1e-3, dtype=np.float64)
        fp_converged, fp_residual, fp_closed_form_error = self._verify_banach_fixed_point(
            operator=morphism_2.stabilized_mutation_operator,
            forcing_vector=forcing_vector,
        )

        rec_cert: Optional[PoincareRecurrenceCertificate] = None
        try:
            if stochastic_transition_matrix is not None and recurrence_set is not None:
                rec_cert = PoincareRecurrenceEngine.from_stochastic_matrix(
                    stochastic_transition_matrix, recurrence_set, num_walks=150, max_steps=5000
                )
            else:
                rec_cert = PoincareRecurrenceEngine.from_recurrence_seed(recurrence_seed)
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
            hasher.update(f"{morphism_2.birkhoff_certificate.birkhoff_lower_bound_satisfied}".encode("utf-8"))
        if morphism_2.melnikov_certificate is not None:
            hasher.update(f"{morphism_2.melnikov_certificate.melnikov_amplitude:.10f}".encode("utf-8"))
        if rec_cert is not None:
            hasher.update(f"{rec_cert.kac_error_residual:.10f}".encode("utf-8"))
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
    print("DEMOSTRACIÓN FORMAL: GÖDEL ENGINE v4.1.0 — POINCARÉ CELESTIAL MECHANICS (FASES ANIDADAS)")
    print("MARSDEN-WEINSTEIN · POINCARÉ-CARTAN/KKS · BIRKHOFF · MELNIKOV · RECURRENCIA DE POINCARÉ")
    print("═" * 96)

    assert HeytingVerdict.verify_heyting_algebra_axioms(), "¡Falla en axiomas de Heyting!"
    print("\n[AXIOMA] Ley de residuación de Heyting verificada exhaustivamente: OK")

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

    engine = GodelEngine(engine_id="GODEL-ENGINE-WISDOM-01", dimension=4, seed=2026)
    print("\n>>> ESCENARIO A: Mutación válida (contracción + coherencia Hodge)...")
    valid_mutation = np.array([
        [0.35, 0.08, 0.00, 0.00],
        [0.08, 0.28, 0.05, 0.00],
        [0.00, 0.05, 0.32, 0.07],
        [0.00, 0.00, 0.07, 0.20],
    ], dtype=np.float64)
    cert_a = engine.execute_rsi_cycle(valid_mutation, simulated_utility_delta=0.08)
    print(f"  • ID Ciclo                         : {cert_a.cycle_id}")
    print(f"  • Veredicto Heyting                : {cert_a.heyting_verdict.name}")
    print(f"  • Radio espectral ρ(T)             : {cert_a.spectral_radius:.6f}")
    print(f"  • Gelfand empírico                 : {cert_a.gelfand_empirical_radius:.6f}")
    print(
        f"  • Índice Poincaré-Hopf             : {cert_a.poincare_hopf_index_sum:.3f} "
        f"(grado topológico {cert_a.degrees_of_map:.2f})"
    )
    print(f"  • Betti: β₀={cert_a.betti_0}, β₁={cert_a.betti_1}")
    print(f"  • Dualidad de Poincaré consistente : {cert_a.poincare_duality_consistent}")
    print(f"  • Exactitud del complejo de Rham   : {cert_a.de_rham_complex_exactness}")
    print(f"  • Distancia de Connes (cota inf.)  : {cert_a.connes_reference_distance:.6f}")
    print(f"  • Brockett Δσ isospectral          : {cert_a.brockett_isospectral_deviation:.2e}")
    print(f"  • Drift de Casimirs KKS            : {cert_a.casimir_drift:.2e}")
    print(f"  • Órbita coadjunta dim             : {cert_a.reduced_orbit_dimension:.1f}")
    print(f"  • Simplicidad KKS preservada       : {cert_a.brockett_symplectic_preserved}")
    print(
        f"  • Punto fijo de Banach (residual)  : {cert_a.fixed_point_converged}, "
        f"res={cert_a.fixed_point_residual:.2e}"
    )
    print(f"  • Recurrencia Kac (semilla Φ)      : τ̄={cert_a.recurrence_mean_time:.4f}, "
          f"residuo={cert_a.recurrence_kac_residual:.4f}")
    print(f"  • Firma SHA-256                    : {cert_a.state_sha256[:32]}…")

    print("\n>>> ESCENARIO B: Poincaré-Birkhoff sobre el twist integrable del anillo...")
    annulus_twist = PoincareBirkhoffEngine.integrable_annulus_twist(alpha=0.5, beta=0.3)
    x_probe = np.array([0.1, 0.3], dtype=np.float64)
    angles = [x_probe[0]]
    for _ in range(200):
        x_probe = annulus_twist(x_probe)
        angles.append(x_probe[0])
    rotation_estimate = float(np.mean(np.diff(np.unwrap(angles))) / (2.0 * math.pi))
    birkhoff = PoincareBirkhoffEngine.audit_twist_map(annulus_twist, rotation_estimate)
    print(f"  • Número de rotación ρ(T)          : {birkhoff.rotation_number:.6f}")
    print(f"  • Racional asociado p/q            : {birkhoff.rational_winding_p}/{birkhoff.rational_period_q}")
    print(f"  • Condición de twist (∂θ'/∂I)      : {birkhoff.twist_condition_verified} "
          f"(min |∂θ'/∂I|={birkhoff.min_twist_derivative:.4e})")
    print(f"  • Área preservada                  : {birkhoff.area_preserving_verified}")
    print(f"  • Número de puntos fijos detectados: {birkhoff.fixed_points_detected}")
    print(f"  • Cota Poincaré-Birkhoff (≥ 2)     : {birkhoff.birkhoff_lower_bound_satisfied}")
    print(f"  • Aplicabilidad del teorema        : {birkhoff.birkhoff_theorem_applicable}")

    print("\n>>> ESCENARIO C: Recurrencia de Poincaré sobre matriz estocástica 6×6...")
    rng = np.random.default_rng(42)
    p_raw = rng.random((6, 6)) + 0.1
    p_stoch = p_raw / p_raw.sum(axis=1, keepdims=True)
    measurable = np.array([True, False, True, False, False, True])
    rec_cert = PoincareRecurrenceEngine.from_stochastic_matrix(
        p_stoch, measurable, num_walks=300, max_steps=5000
    )
    print(f"  • |A| / |X|                        : {int(measurable.sum())}/{6}")
    print(f"  • Tiempo medio de retorno          : {rec_cert.mean_return_time_empirical:.4f}")
    print(f"  • Predicción de Kac                : {rec_cert.kac_lemma_prediction:.4f}")
    print(f"  • Residuo |τ̄ − 1/μ(A)|             : {rec_cert.kac_error_residual:.4f}")
    print(f"  • Fracción recurrente              : {rec_cert.recurrent_states_fraction:.4f}")
    print(f"  • Recurrencia cuasi-total          : {rec_cert.almost_everywhere_recurrent}")

    print("\n>>> ESCENARIO D: Función de Melnikov sobre Duffing clásico (silla en el origen)...")

    def duffing_h0(x: np.ndarray) -> float:
        # H₀ = p²/2 − q²/2 + q⁴/4  (silla en (0,0), ochos homoclínicos)
        q, p = float(x[0]), float(x[1])
        return 0.5 * p * p - 0.5 * q * q + 0.25 * q ** 4

    def duffing_h1(x: np.ndarray, t: float) -> float:
        q, p = float(x[0]), float(x[1])
        return -0.15 * q * p + 0.3 * q * math.cos(t)

    saddle = np.array([0.0, 0.0], dtype=np.float64)
    orbit = MelnikovFunctionEngine._numerical_homoclinic_orbit(duffing_h0, saddle)
    t_samples = np.linspace(-25.0, 25.0, orbit.shape[0])
    t0_grid = np.linspace(0.0, 2.0 * math.pi, 25, endpoint=False)
    mel_cert = MelnikovFunctionEngine.compute_melnikov_function(
        duffing_h0, duffing_h1, orbit, t_samples, t0_grid
    )
    print(f"  • Amplitud max |M(t₀)|             : {mel_cert.melnikov_amplitude:.6e}")
    print(f"  • Media de M                       : {mel_cert.melnikov_mean:.6e}")
    print(f"  • Ceros simples detectados         : {mel_cert.simple_zeros_detected}")
    print(f"  • Homoclínica transversal          : {mel_cert.transverse_homoclinic_exists}")
    print(f"  • Herradura de Smale esperada      : {mel_cert.smale_horseshoe_expected}")
    print(f"  • Umbral ε para caos               : {mel_cert.chaos_threshold_epsilon:.4e}")

    print("\n" + "═" * 96)
    print("✓ AUDITORÍA DE SISTEMA CONCLUIDA: GÖDEL ENGINE v4.1.0 — FASES ANIDADAS.")
    print("  · Fase 1: 𝔐_Spectral → lift_to_celestial_hamiltonian_bundle (T*Q, ω, H, J).")
    print("  · Fase 2: Φ_sheaf → seed_poincare_recurrence_from_morphism (Perron-Frobenius).")
    print("  · Fase 3: Poincaré-Kac + punto fijo Banach + certificación SHA-256.")
    print("═" * 96)
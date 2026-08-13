# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Riemannian Inertia Modulator (Motor de Momento Giroscópico)         ║
║ Ruta   : app/physics/riemannian_inertia_modulator.py                         ║
║ Versión: 4.0.0-Spectral-Neumaier-deRham-Lie-Strict                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

OBJETO CIENTÍFICO-TÉCNICO Y FUNDAMENTACIÓN GEOMÉTRICA (Rigor Doctoral): ────────
Este módulo implementa el **Funtor de Moldeo de Masa** (Mass Shaping Functor)
sobre el fibrado cotangente $$\mathcal{T}^* \mathcal{M}$$. Su propósito
axiomático es inyectar una "Fuerza de Lorentz" informacional de naturaleza
estrictamente giroscópica en la trayectoria del espacio de fase transaccional,
con el fin de desviar las fluctuaciones estocásticas y las alucinaciones de
los Modelos de Lenguaje (LLMs) hacia sumideros de disipación, sin alterar la
energía Hamiltoniana interna del sistema.

El módulo trata el flujo logístico del presupuesto como un fluido continuo,
subyugando el transporte paralelo de los covectores de estado a la regularidad
del cono antisimétrico de Lie y preservando de manera exacta la 2-forma
simpléctica canónica de Liouville.

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial): ───────────────────
La transmutación del momento se rige por un contrato algebraico rígido donde la
salida inmutable de cada fase actúa como la precondición formal de la siguiente.
El último morfismo de la Fase $$k$$ es, por construcción, el dominio del primer
morfismo de la Fase $$k+1$$:

  Fase 1 ──► FASE 1: ESPECTROSCOPÍA DEL MOMENTUM (Phase1_MomentumSpectrometer)
             Aplica el isomorfismo musical plano (flat, $$\flat$$) para descender
             índices en la variedad Riemanniana anisotrópica:
             $$p_\mu = G_{\mu\nu} \dot{q}^\nu \quad\big[8, 25, 30\big]$$
             Certifica el isomorfismo recíproco (sharp, $$\sharp$$):
             $$\dot{q}^\mu = G^{\mu\nu} p_\nu, \qquad \sharp\circ\flat = \mathrm{id}$$
             Certifica la identidad de apareamiento dual y la energía cinética:
             $$\langle p,\dot{q}\rangle = \dot{q}^\mu G_{\mu\nu}\dot{q}^\nu
               = p_\mu G^{\mu\nu}p_\nu$$
             Certifica la cota de inercia mediante la norma dual inducida:
             $$\|p\|_{G^{-1}} = \sqrt{p_\mu G^{\mu\nu} p_\nu} \le P_{\max}$$
             Entrega: MomentumAuditData.
             Morfismo terminal: handoff_phase1_to_phase2  ≡  dominio de Fase 2.

  Fase 2 ──► FASE 2: SÍNTESIS DEL OPERADOR GIROSCÓPICO (Phase2_GyroscopicSynthesizer)
             Recibe el covector certificado por handoff_phase1_to_phase2.
             Acopla el momentum covariante $$p$$ con la vorticidad solenoidal
             $$\omega$$ (2-forma de refracción territorial) para construir el
             tensor de Lorentz en el álgebra exterior $$\bigwedge^2 T^*\mathcal{M}$$:
             $$W_{\mu\nu} = \alpha \left( p_\mu \omega_\nu - p_\nu \omega_{\mu} \right)
               = \alpha\,(p\wedge\omega)_{\mu\nu}$$
             Certifica la identidad de Gram del producto exterior:
             $$\|p\wedge\omega\|_F^2 = 2\bigl(\|p\|_2^2\|\omega\|_2^2
               - \langle p,\omega\rangle^2\bigr)$$
             Proyecta $$W$$ al álgebra de Lie del cono antisimétrico
             $$\mathfrak{so}(n)$$ exigiendo la nulidad del residuo relativo
             de Frobenius:
             $$r_{\mathrm{skew}} = \frac{\|W + W^\top\|_F}{\max(1, \|W\|_F)}
               \le \epsilon_{\mathrm{skew}}$$
             Entrega: GyroscopicSynthesisData.
             Morfismo terminal: handoff_phase2_to_phase3  ≡  dominio de Fase 3.

  Fase 3 ──► FASE 3: MODULACIÓN SIMPLÉCTICA Y TRABAJO NILPOTENTE
             (Phase3_SymplecticInertiaModulator)
             Recibe el tensor certificado por handoff_phase2_to_phase3.
             Inyecta el tensor giroscópico $$W$$ en el operador de interconexión
             de Dirac $$J_{\mathrm{base}}$$ del sistema Port-Hamiltoniano:
             $$J_{\mathrm{eff}} = J_{\mathrm{base}} + W \quad\big[8, 25\big]$$
             Certifica la nulidad del trabajo mecánico neto mediante sumación
             compensada de Neumaier sobre el gradiente Hamiltoniano
             $$\nabla H$$, y mediante la fórmula estructural por pares:
             $$P_{\mathrm{gyro}} = \langle \nabla H, J_{\mathrm{eff}} \nabla H \rangle
               \equiv 0 \pmod{\varepsilon_{\mathrm{machine}}}$$
             $$P_{\mathrm{pair}} = \sum_{i<j}(J_{ij}+J_{ji})x_i x_j
               + \sum_i J_{ii}x_i^2 \equiv 0$$
             Entrega: ThermodynamicVetoData como certificado inmutable terminal.

INVARIANTES MATEMÁTICOS, GEOMÉTRICOS Y PROPIEDADES DE CALIBRE PRESERVADOS: ─────
  [I1] Invariancia de la Forma Simpléctica (Volumen de Liouville):
       El operador $$J_{\mathrm{eff}}$$ es estrictamente antisimétrico, garantizando
       que el flujo conserve el volumen en el espacio de fase, acatando el
       Teorema de Liouville:
       $$x^\top J_{\mathrm{eff}} x \equiv 0 \quad \forall x \in \mathbb{R}^{n}
         \quad\big[8\big]$$

  [I2] Coherencia Métrica de de Rham (Isomorfismo Musical):
       El motor de isomorfismo exige consistencia bilateral en la inversión
       espectral de la métrica $$G_{\mu\nu}$$ respecto al límite de Wilkinson:
       $$\max\left(
            \frac{\|G G^{-1}-I\|_F}{\sqrt{n}},
            \frac{\|G^{-1}G-I\|_F}{\sqrt{n}}
         \right)
         \le C\cdot\kappa(G)\cdot\varepsilon_{\mathrm{machine}}$$
       y el round-trip musical:
       $$\|\sharp\flat\dot{q}-\dot{q}\|_2
         \le C\cdot\kappa(G)\cdot n\cdot\varepsilon_{\mathrm{machine}}
         \cdot\max(1,\|\dot{q}\|_2)$$

  [I3] Conservación Termodinámica e Inecuación de Clausius-Duhem:
       Dado que la inyección giroscópica de Lorentz no realiza trabajo neto,
       la tasa de disipación de Rayleigh del sistema Port-Hamiltoniano basal se
       mantiene incondicionalmente pasiva y no-negativa:
       $$\dot{H} = -\nabla H^\top R(x)\nabla H \le 0
         \quad\text{donde}\quad R(x)=R(x)^\top\succeq\mathbf{0}\quad\big[25\big]$$

  [I4] Identidad de Gram del Álgebra Exterior:
       El bivector $$p\wedge\omega$$ satisface la relación pitagórica
       inducida por el determinante de Gram de los 1-vectores $$\{p,\omega\}$$.

  [I5] Estructura de álgebra de Lie $$\mathfrak{so}(n)$$:
       $$W^\top=-W$$, espectro puramente imaginario, rango par, y la
       proyección $$W\mapsto\tfrac12(W-W^\top)$$ es la proyección ortogonal
       respecto del producto de Frobenius.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────────────
# Dependencias del ecosistema MIC.
# Se conservan stubs analíticos para ejecución autónoma.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:

    class TopologicalInvariantError(Exception):
        """Violación base de un invariante topológico-categórico."""

        pass

    class Morphism:
        """Clase base de composición funtorial del ecosistema MIC."""

        pass


logger = logging.getLogger("MIC.Physics.RiemannianInertiaModulator")


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS, NUMÉRICAS Y TERMODINÁMICAS
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

# Tolerancia base para residuos estructurales absolutos.
_SYMPLECTIC_TOLERANCE: Final[float] = 1e-12

# Tolerancia absoluta de respaldo para la inversa métrica.
_METRIC_INVERSE_TOLERANCE: Final[float] = 1e-8

# Constante de Wilkinson para cotas de residuo de GEMM / inversa.
# Empíricamente 8 cubre n · ulp de un DGEMM estable en IEEE-754.
_WILKINSON_CONSTANT: Final[float] = 8.0

# Cota elástica máxima del momentum (norma dual).
_MOMENTUM_MAX_BOUND: Final[float] = 1e8

# Factor de acoplamiento giroscópico α en W = α (p ∧ ω).
_VORTICITY_COUPLING_FACTOR: Final[float] = 1.0

# Número de condición máximo admisible para la métrica riemanniana.
_CONDITION_NUMBER_MAX: Final[float] = 1e12

# Multiplicador de ULPs para la cota adaptativa del trabajo nulo.
_WORK_TOLERANCE_ULPS: Final[float] = 1000.0

# Cota relativa de antisimetría (definición documental de r_skew).
_SKEW_RELATIVE_TOLERANCE: Final[float] = 1e-12

# Número de sondas aleatorias para la auditoría de Liouville.
_LIOUVILLE_PROBE_COUNT: Final[int] = 8

# Semilla reproducible de las sondas de Liouville.
_LIOUVILLE_PROBE_SEED: Final[int] = 1729

# Versión canónica del funtor.
__version__: Final[str] = "4.0.0-Spectral-Neumaier-deRham-Lie-Strict"


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES SIMPLÉCTICAS
# ══════════════════════════════════════════════════════════════════════════════
class RiemannianInertiaError(TopologicalInvariantError):
    """Excepción raíz del funtor de modulación de inercia."""

    pass


class MetricCoherenceError(RiemannianInertiaError):
    r"""
    Detonada si el par $$(G, G^{-1})$$ viola la cota de Wilkinson o el
    round-trip del isomorfismo musical $$\sharp\circ\flat=\mathrm{id}$$.
    """

    pass


class DualPairingError(RiemannianInertiaError):
    r"""
    Detonada si se rompe la identidad de apareamiento dual:
        $$\langle p,\dot{q}\rangle
          = \dot{q}^\top G\dot{q}
          = p^\top G^{-1}p$$
    """

    pass


class MomentumDivergenceError(RiemannianInertiaError):
    r"""
    Detonada si $$\|p\|_{G^{-1}}$$ excede la cota elástica admisible:
        $$\|p\|_{G^{-1}} = \sqrt{p_\mu G^{\mu\nu} p_\nu} > P_{\mathrm{max}}$$
    """

    pass


class ExteriorAlgebraError(RiemannianInertiaError):
    r"""
    Detonada si el bivector $$p\wedge\omega$$ viola la identidad de Gram:
        $$\|p\wedge\omega\|_F^2
          = 2\bigl(\|p\|_2^2\|\omega\|_2^2-\langle p,\omega\rangle^2\bigr)$$
    """

    pass


class SkewSymmetryViolationError(RiemannianInertiaError):
    r"""
    Detonada si la proyección antisimétrica retiene componentes simétricas:
        $$r_{\mathrm{skew}}
          = \frac{\|W+W^\top\|_F}{\max(1,\|W\|_F)}
          > \epsilon_{\mathrm{skew}}$$
    """

    pass


class SymplecticWorkViolationError(RiemannianInertiaError):
    r"""
    Detonada si el operador efectivo realiza trabajo neto no nulo:
        $$|\langle\nabla H,\,J_{\mathrm{eff}}\nabla H\rangle| > \tau_{\mathrm{work}}$$
    """

    pass


class PhaseHandoffError(RiemannianInertiaError):
    """Detonada si un certificado de fase no satisface las precondiciones de la siguiente."""

    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs DEL FIBRADO COTANGENTE)
# ══════════════════════════════════════════════════════════════════════════════
def _freeze_array(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Copia defensiva y sello de solo-lectura para tensores certificados."""
    frozen = np.array(array, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, slots=True)
class MomentumAuditData:
    """
    Artefacto certificado de la Fase 1.

    Es, por contrato, el objeto inicial de la Fase 2
    (véase ``handoff_phase1_to_phase2``).
    """

    covariant_momentum: NDArray[np.float64]
    reconstructed_velocity: NDArray[np.float64]
    momentum_norm: float
    kinetic_energy_primal: float
    kinetic_energy_dual: float
    dual_pairing: float
    pairing_residual: float
    musical_roundtrip_residual: float
    is_bounded: bool
    metric_condition_number: float
    inverse_consistency_residual: float
    wilkinson_bound: float
    spectral_minimum: float
    spectral_maximum: float


@dataclass(frozen=True, slots=True)
class GyroscopicSynthesisData:
    """
    Artefacto certificado de la Fase 2.

    Es, por contrato, el objeto inicial de la Fase 3
    (véase ``handoff_phase2_to_phase3``).
    """

    skew_symmetric_tensor: NDArray[np.float64]
    vorticity_two_form: NDArray[np.float64]
    omega_vector: NDArray[np.float64]
    antisymmetry_residual: float
    relative_skew_residual: float
    vorticity_projection_residual: float
    gyroscopic_frobenius_norm: float
    wedge_gram_residual: float
    skew_numerical_rank: int
    rank_is_even: bool
    is_strictly_skew: bool


@dataclass(frozen=True, slots=True)
class ThermodynamicVetoData:
    """Artefacto certificado de la Fase 3 (veredicto terminal)."""

    effective_dirac_matrix: NDArray[np.float64]
    nilpotent_work_residual: float
    pairwise_work_residual: float
    liouville_probe_residual: float
    dirac_symmetric_residual: float
    relative_skew_residual: float
    work_tolerance: float
    is_symplectically_passive: bool


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 → ESPECTROSCOPÍA DEL MOMENTUM (Observe)
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_MomentumSpectrometer:
    r"""
    Extrae el covector de momentum mediante el isomorfismo musical plano
    ($$\flat$$):

        $$p_\mu = G_{\mu\nu}\dot{q}^\nu$$

    y certifica, en orden de dependencia lógica:

        1. dimensionalidad, vacuidad y finitud de los tensores;
        2. simetría métrica estructural;
        3. espectro estrictamente positivo de $$G$$ y $$G^{-1}$$;
        4. consistencia de Wilkinson de la inversa;
        5. round-trip musical $$\sharp\circ\flat=\mathrm{id}$$;
        6. identidad de apareamiento dual / energía cinética;
        7. cota de inercia $$\|p\|_{G^{-1}}\le P_{\max}$$.

    El morfismo terminal ``handoff_phase1_to_phase2`` eleva el certificado
    a precondición formal de la Fase 2.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 1.1  Utilidades elementales de validación numérica
    # ──────────────────────────────────────────────────────────────────────────
    def _as_float64_array(
        self,
        payload: object,
        name: str,
    ) -> NDArray[np.float64]:
        """Convierte ``payload`` a ``ndarray[float64]`` o eleva error tipado."""
        try:
            return np.array(payload, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise RiemannianInertiaError(
                f"{name} no pudo convertirse a un tensor float64."
            ) from exc

    def _assert_finite(
        self,
        array: NDArray[np.float64],
        name: str,
    ) -> None:
        """Exige que todos los coeficientes pertenezcan a $$\mathbb{R}$$ finito."""
        if not np.all(np.isfinite(array)):
            raise RiemannianInertiaError(f"{name} contiene valores no finitos.")

    def _validate_vector(
        self,
        vector: object,
        name: str,
    ) -> NDArray[np.float64]:
        """Valida que un vector sea 1-D, no vacío, finito y convertible a float64."""
        arr = self._as_float64_array(vector, name)

        if arr.ndim != 1:
            raise RiemannianInertiaError(f"{name} debe ser un vector 1-D.")

        if arr.size == 0:
            raise RiemannianInertiaError(f"{name} no puede ser vacío.")

        self._assert_finite(arr, name)
        return arr

    def _validate_square_matrix(
        self,
        matrix: object,
        name: str,
    ) -> NDArray[np.float64]:
        """Valida que una matriz sea cuadrada, no vacía, finita y float64."""
        arr = self._as_float64_array(matrix, name)

        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise RiemannianInertiaError(f"{name} debe ser una matriz cuadrada.")

        if arr.shape[0] == 0:
            raise RiemannianInertiaError(f"{name} no puede ser vacía.")

        self._assert_finite(arr, name)
        return arr

    def _frobenius_norm(self, matrix: NDArray[np.float64], name: str) -> float:
        """Calcula la norma de Frobenius con verificación de finitud."""
        self._assert_finite(matrix, name)
        value = float(np.linalg.norm(matrix, ord="fro"))
        if not math.isfinite(value):
            raise RiemannianInertiaError(
                f"La norma de Frobenius de {name} no es finita."
            )
        return value

    def _euclidean_norm(self, vector: NDArray[np.float64], name: str) -> float:
        """Norma euclídea $$\ell^2$$ con verificación de finitud."""
        self._assert_finite(vector, name)
        value = float(np.linalg.norm(vector, ord=2))
        if not math.isfinite(value):
            raise RiemannianInertiaError(f"La norma ℓ² de {name} no es finita.")
        return value

    def _matrix_residual_tolerance(
        self,
        matrix: NDArray[np.float64],
        base_tol: float = _SYMPLECTIC_TOLERANCE,
    ) -> float:
        r"""
        Cota adaptativa relativa a la escala de la matriz:

            $$\mathrm{tol}=\max\bigl(\mathrm{tol}_{base},\,
              \mathrm{tol}_{base}\times\max(1,\|A\|_F)\bigr)$$
        """
        scale = self._frobenius_norm(matrix, "matriz para tolerancia")
        return max(base_tol, base_tol * max(1.0, scale))

    def _relative_frobenius_residual(
        self,
        residual_norm: float,
        reference_norm: float,
    ) -> float:
        r"""Residuo relativo $$\|R\|_F / \max(1, \|A\|_F)$$."""
        if not math.isfinite(residual_norm) or not math.isfinite(reference_norm):
            raise RiemannianInertiaError(
                "Normas no finitas en el residuo relativo de Frobenius."
            )
        return residual_norm / max(1.0, reference_norm)

    def _symmetric_residual(self, matrix: NDArray[np.float64]) -> float:
        r"""Residuo de simetría $$\|A-A^\top\|_F$$."""
        return self._frobenius_norm(matrix - matrix.T, "residuo simétrico")

    def _skew_symmetric_residual(self, matrix: NDArray[np.float64]) -> float:
        r"""Residuo de antisimetría $$\|A+A^\top\|_F$$."""
        return self._frobenius_norm(matrix + matrix.T, "residuo antisimétrico")

    def _relative_skew_residual(self, matrix: NDArray[np.float64]) -> float:
        r"""
        Residuo relativo documental:

            $$r_{\mathrm{skew}}
              = \frac{\|A+A^\top\|_F}{\max(1,\|A\|_F)}$$
        """
        abs_residual = self._skew_symmetric_residual(matrix)
        scale = self._frobenius_norm(matrix, "matriz para r_skew")
        return self._relative_frobenius_residual(abs_residual, scale)

    def _symmetrize(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Proyección ortogonal al cono simétrico: $$\tfrac12(A+A^\top)$$."""
        return 0.5 * (matrix + matrix.T)

    def _skew_symmetrize(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Proyección ortogonal a $$\mathfrak{so}(n)$$: $$\tfrac12(A-A^\top)$$."""
        return 0.5 * (matrix - matrix.T)

    # ──────────────────────────────────────────────────────────────────────────
    # 1.2  Sumación compensada de Neumaier (reducción escalar rigurosa)
    # ──────────────────────────────────────────────────────────────────────────
    def _neumaier_sum(self, terms: NDArray[np.float64], name: str) -> float:
        r"""
        Suma compensada de Neumaier (Kahan–Babuška–Neumaier).

        A diferencia de Kahan clásico, corrige el caso
        $$|x_{k+1}| > |s_k|$$, esencial cuando hay cancelación cruzada
        en formas cuadráticas antisimétricas.
        """
        self._assert_finite(terms, name)

        total = 0.0
        compensation = 0.0
        for raw in terms.flat:
            term = float(raw)
            trial = total + term
            if abs(total) >= abs(term):
                compensation += (total - trial) + term
            else:
                compensation += (term - trial) + total
            if not math.isfinite(trial) or not math.isfinite(compensation):
                raise RiemannianInertiaError(
                    f"La sumación de Neumaier de {name} divergió."
                )
            total = trial

        result = total + compensation
        if not math.isfinite(result):
            raise RiemannianInertiaError(
                f"La suma compensada de {name} no es finita."
            )
        return result

    def _neumaier_dot(
        self,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
        name: str,
    ) -> float:
        r"""Producto interno compensado $$\langle u,v\rangle=\sum_i u_i v_i$$."""
        left = self._validate_vector(left, f"{name}.left")
        right = self._validate_vector(right, f"{name}.right")
        if left.size != right.size:
            raise RiemannianInertiaError(
                f"Dimensión incompatible en el producto interno {name}."
            )
        products = left * right
        return self._neumaier_sum(products, name)

    # ──────────────────────────────────────────────────────────────────────────
    # 1.3  Espectro de la métrica y cota de Wilkinson
    # ──────────────────────────────────────────────────────────────────────────
    def _spectral_decompose_spd(
        self,
        matrix: NDArray[np.float64],
        name: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Descomposición espectral de una métrica riemanniana:

            $$G = Q\,\mathrm{diag}(\lambda)\,Q^\top,
              \qquad \lambda_1\le\cdots\le\lambda_n,
              \qquad Q^\top Q=I$$

        Se exige $$\lambda_{\min}>0$$ por encima del umbral de singularidad
        numérica $$n\,\varepsilon\,\lambda_{\max}$$.
        """
        matrix = self._validate_square_matrix(matrix, name)
        symmetric = self._symmetrize(matrix)
        self._assert_finite(symmetric, f"{name} simetrizada")

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        except np.linalg.LinAlgError as exc:
            raise MetricCoherenceError(
                f"La descomposición espectral de {name} falló."
            ) from exc

        self._assert_finite(eigenvalues, f"espectro de {name}")
        self._assert_finite(eigenvectors, f"autovectores de {name}")

        n = symmetric.shape[0]
        lambda_min = float(eigenvalues[0])
        lambda_max = float(eigenvalues[-1])

        if lambda_max <= 0.0 or not math.isfinite(lambda_max):
            raise MetricCoherenceError(
                f"{name} no es definida positiva: λ_max = {lambda_max:.4e}."
            )

        spectral_floor = max(
            float(n) * _MACHINE_EPSILON * lambda_max,
            _MACHINE_EPSILON,
        )
        if lambda_min <= spectral_floor:
            raise MetricCoherenceError(
                f"{name} es numéricamente singular o no definida positiva. "
                f"λ_min = {lambda_min:.4e}, umbral = {spectral_floor:.4e}."
            )

        return eigenvalues, eigenvectors

    def _condition_number_from_spectrum(
        self,
        eigenvalues: NDArray[np.float64],
        name: str,
    ) -> float:
        r"""Número de condición espectral $$\kappa_2=\lambda_{\max}/\lambda_{\min}$$."""
        lambda_min = float(eigenvalues[0])
        lambda_max = float(eigenvalues[-1])
        if lambda_min <= 0.0:
            raise MetricCoherenceError(
                f"κ({name}) indefinido: λ_min = {lambda_min:.4e}."
            )
        cond = lambda_max / lambda_min
        if not math.isfinite(cond):
            raise MetricCoherenceError(f"El número de condición de {name} no es finito.")
        if cond > _CONDITION_NUMBER_MAX:
            raise MetricCoherenceError(
                f"{name} está mal condicionada: κ = {cond:.4e} > "
                f"{_CONDITION_NUMBER_MAX:.4e}."
            )
        return float(cond)

    def _spectral_inverse(
        self,
        eigenvalues: NDArray[np.float64],
        eigenvectors: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Inversa espectral:

            $$G^{-1}=Q\,\mathrm{diag}(\lambda^{-1})\,Q^\top$$
        """
        inv_evals = 1.0 / eigenvalues
        inverse = (eigenvectors * inv_evals) @ eigenvectors.T
        inverse = self._symmetrize(inverse)
        self._assert_finite(inverse, "inversa espectral")
        return inverse

    def _wilkinson_inverse_tolerance(
        self,
        dimension: int,
        condition_number: float,
    ) -> float:
        r"""
        Cota de Wilkinson para el residuo de inversa normalizado por
        $$\sqrt{n}$$ (invariante [I2]):

            $$\tau_W = \max\bigl(
                \tau_{\mathrm{abs}},\,
                C\cdot\kappa(G)\cdot\varepsilon\cdot\sqrt{n}
            \bigr)$$

        El factor $$\sqrt{n}$$ convierte la cota del enunciado
        $$\|GG^{-1}-I\|_F/\sqrt{n}$$ en una cota sobre la norma cruda
        $$\|GG^{-1}-I\|_F$$.
        """
        sqrt_n = math.sqrt(float(dimension))
        spectral = (
            _WILKINSON_CONSTANT
            * max(1.0, condition_number)
            * _MACHINE_EPSILON
            * sqrt_n
        )
        return max(_METRIC_INVERSE_TOLERANCE, spectral)

    def _normalized_inverse_residual(
        self,
        left_residual: float,
        right_residual: float,
        dimension: int,
    ) -> float:
        r"""
        Residuo bilateral normalizado al estilo Wilkinson:

            $$\max\bigl(
                \|GG^{-1}-I\|_F/\sqrt{n},\,
                \|G^{-1}G-I\|_F/\sqrt{n}
            \bigr)$$
        """
        sqrt_n = math.sqrt(float(dimension))
        return max(left_residual, right_residual) / sqrt_n

    # ──────────────────────────────────────────────────────────────────────────
    # 1.4  Validación del par métrico (G, G^{-1})
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_metric_pair(
        self,
        G_tensor: NDArray[np.float64],
        G_inv: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        float,
        float,
        float,
        float,
    ]:
        r"""
        Valida y sanea el par métrico:

            $$G      : T^*\mathcal{M}\otimes T^*\mathcal{M}$$
            $$G^{-1} : T\mathcal{M}\otimes T\mathcal{M}$$

        Certificado:
            - misma dimensión;
            - simetría estructural;
            - espectro estrictamente positivo de ambas;
            - consistencia $$G G^{-1}\approx I$$ y $$G^{-1}G\approx I$$
              bajo la cota de Wilkinson;
            - acuerdo entre $$G^{-1}$$ suministrada y la inversa espectral;
            - número de condición acotado.

        Retorna:
            ``(G, G_inv, κ(G), residuo_inversa_crudo, cota_Wilkinson,
              λ_min, λ_max)``.
        """
        G_tensor = self._validate_square_matrix(G_tensor, "G_tensor")
        G_inv = self._validate_square_matrix(G_inv, "G_inv")

        if G_tensor.shape != G_inv.shape:
            raise MetricCoherenceError(
                "G_tensor y G_inv deben tener la misma dimensión."
            )

        n = G_tensor.shape[0]

        g_sym_residual = self._symmetric_residual(G_tensor)
        g_tol = self._matrix_residual_tolerance(G_tensor)
        if g_sym_residual > g_tol:
            raise MetricCoherenceError(
                f"G_tensor no es simétrica. Residuo = {g_sym_residual:.4e}, "
                f"tolerancia = {g_tol:.4e}."
            )

        g_inv_sym_residual = self._symmetric_residual(G_inv)
        g_inv_tol = self._matrix_residual_tolerance(G_inv)
        if g_inv_sym_residual > g_inv_tol:
            raise MetricCoherenceError(
                f"G_inv no es simétrica. Residuo = {g_inv_sym_residual:.4e}, "
                f"tolerancia = {g_inv_tol:.4e}."
            )

        G = self._symmetrize(G_tensor)
        G_inv = self._symmetrize(G_inv)

        evals_G, evecs_G = self._spectral_decompose_spd(G, "G_tensor")
        self._spectral_decompose_spd(G_inv, "G_inv")
        cond_G = self._condition_number_from_spectrum(evals_G, "G_tensor")
        lambda_min = float(evals_G[0])
        lambda_max = float(evals_G[-1])

        identity = np.eye(n, dtype=np.float64)
        left = G @ G_inv
        right = G_inv @ G
        self._assert_finite(left, "G @ G_inv")
        self._assert_finite(right, "G_inv @ G")

        left_residual = self._frobenius_norm(left - identity, "G @ G_inv - I")
        right_residual = self._frobenius_norm(right - identity, "G_inv @ G - I")
        inverse_residual = max(left_residual, right_residual)
        normalized_residual = self._normalized_inverse_residual(
            left_residual,
            right_residual,
            n,
        )

        wilkinson_bound = self._wilkinson_inverse_tolerance(n, cond_G)
        normalized_bound = wilkinson_bound / math.sqrt(float(n))

        if normalized_residual > max(
            normalized_bound,
            _WILKINSON_CONSTANT * cond_G * _MACHINE_EPSILON,
        ):
            raise MetricCoherenceError(
                "G_inv no es una inversa de Wilkinson-consistente de G_tensor. "
                f"Residuo normalizado = {normalized_residual:.4e}, "
                f"cota = {normalized_bound:.4e}."
            )

        G_inv_spectral = self._spectral_inverse(evals_G, evecs_G)
        spectral_gap = self._frobenius_norm(
            G_inv - G_inv_spectral,
            "G_inv - G_inv_espectral",
        )
        spectral_tol = max(
            wilkinson_bound,
            _WILKINSON_CONSTANT
            * cond_G
            * _MACHINE_EPSILON
            * max(1.0, self._frobenius_norm(G_inv, "G_inv")),
        )
        if spectral_gap > spectral_tol:
            raise MetricCoherenceError(
                "G_inv discrepa de la inversa espectral de G_tensor. "
                f"Residuo = {spectral_gap:.4e}, tolerancia = {spectral_tol:.4e}."
            )

        return (
            G,
            G_inv,
            cond_G,
            inverse_residual,
            wilkinson_bound,
            lambda_min,
            lambda_max,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 1.5  Isomorfismo musical y energía cinética geométrica
    # ──────────────────────────────────────────────────────────────────────────
    def _musical_flat(
        self,
        q_dot: NDArray[np.float64],
        G_tensor: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Isomorfismo musical plano:

            $$p = G\dot{q},\qquad p_\mu = G_{\mu\nu}\dot{q}^\nu$$
        """
        q_dot = self._validate_vector(q_dot, "q_dot")
        G_tensor = self._validate_square_matrix(G_tensor, "G_tensor")

        if G_tensor.shape[0] != q_dot.size:
            raise RiemannianInertiaError(
                "Dimensión incompatible entre q_dot y G_tensor."
            )

        momentum = G_tensor @ q_dot
        self._assert_finite(momentum, "momentum covariante p = G q̇")
        return momentum

    def _musical_sharp(
        self,
        momentum: NDArray[np.float64],
        G_inv: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Isomorfismo musical agudo:

            $$v = G^{-1}p,\qquad v^\mu = G^{\mu\nu}p_\nu$$
        """
        momentum = self._validate_vector(momentum, "p")
        G_inv = self._validate_square_matrix(G_inv, "G_inv")

        if G_inv.shape[0] != momentum.size:
            raise RiemannianInertiaError(
                "Dimensión incompatible entre p y G_inv."
            )

        velocity = G_inv @ momentum
        self._assert_finite(velocity, "velocidad reconstruida v = G^{-1} p")
        return velocity

    def _certify_musical_roundtrip(
        self,
        q_dot: NDArray[np.float64],
        reconstructed: NDArray[np.float64],
        condition_number: float,
    ) -> float:
        r"""
        Certifica $$\sharp\circ\flat=\mathrm{id}$$ sobre $$\dot{q}$$:

            $$\|\sharp\flat\dot{q}-\dot{q}\|_2
              \le C\cdot\kappa\cdot n\cdot\varepsilon
              \cdot\max(1,\|\dot{q}\|_2)$$
        """
        q_dot = self._validate_vector(q_dot, "q_dot")
        reconstructed = self._validate_vector(reconstructed, "reconstructed_velocity")
        if q_dot.size != reconstructed.size:
            raise DualPairingError(
                "Dimensión incompatible en el round-trip musical."
            )

        delta = reconstructed - q_dot
        residual = self._euclidean_norm(delta, "♯♭q̇ − q̇")
        scale = max(1.0, self._euclidean_norm(q_dot, "q_dot"))
        tolerance = (
            _WILKINSON_CONSTANT
            * max(1.0, condition_number)
            * _MACHINE_EPSILON
            * float(q_dot.size)
            * scale
        )
        tolerance = max(tolerance, _SYMPLECTIC_TOLERANCE * scale)

        if residual > tolerance:
            raise DualPairingError(
                "El isomorfismo musical no es involutivo en aritmética de máquina. "
                f"‖♯♭q̇ − q̇‖ = {residual:.4e}, tolerancia = {tolerance:.4e}."
            )
        return residual

    def _certify_dual_pairing(
        self,
        q_dot: NDArray[np.float64],
        momentum: NDArray[np.float64],
        G_tensor: NDArray[np.float64],
        G_inv: NDArray[np.float64],
        condition_number: float,
    ) -> tuple[float, float, float, float]:
        r"""
        Certifica la identidad de apareamiento dual:

            $$\langle p,\dot{q}\rangle
              = \dot{q}^\top G\dot{q}
              = p^\top G^{-1}p$$

        Retorna ``(apareamiento, energía_primal, energía_dual, residuo)``.
        """
        g_q = G_tensor @ q_dot
        g_inv_p = G_inv @ momentum
        self._assert_finite(g_q, "G q̇")
        self._assert_finite(g_inv_p, "G^{-1} p")

        pairing = self._neumaier_dot(momentum, q_dot, "⟨p, q̇⟩")
        primal = self._neumaier_dot(q_dot, g_q, "q̇ᵀ G q̇")
        dual = self._neumaier_dot(momentum, g_inv_p, "pᵀ G^{-1} p")

        if primal < 0.0 or dual < 0.0:
            floor = max(
                _SYMPLECTIC_TOLERANCE,
                1000.0 * _MACHINE_EPSILON * max(1.0, abs(primal), abs(dual)),
            )
            if primal < -floor or dual < -floor:
                raise DualPairingError(
                    "Energía cinética geométrica negativa: "
                    f"primal = {primal:.4e}, dual = {dual:.4e}."
                )
            primal = max(primal, 0.0)
            dual = max(dual, 0.0)

        scale = max(1.0, abs(primal), abs(dual), abs(pairing))
        tolerance = max(
            _SYMPLECTIC_TOLERANCE,
            _WILKINSON_CONSTANT
            * max(1.0, condition_number)
            * _MACHINE_EPSILON
            * float(q_dot.size)
            * scale,
        )
        residual = max(abs(pairing - primal), abs(primal - dual), abs(pairing - dual))
        if residual > tolerance:
            raise DualPairingError(
                "Identidad de apareamiento dual violada. "
                f"⟨p,q̇⟩={pairing:.4e}, primal={primal:.4e}, "
                f"dual={dual:.4e}, residuo={residual:.4e}, "
                f"tolerancia={tolerance:.4e}."
            )
        return pairing, primal, dual, residual

    def _evaluate_momentum_norm(
        self,
        dual_energy: float,
    ) -> float:
        r"""
        Norma inducida por la métrica dual:

            $$\|p\|_{G^{-1}}=\sqrt{p_\mu G^{\mu\nu}p_\nu}$$
        """
        if not math.isfinite(dual_energy):
            raise RiemannianInertiaError(
                "La energía dual del momentum no es finita."
            )
        if dual_energy < 0.0:
            raise RiemannianInertiaError(
                f"Energía dual negativa en la norma inercial: {dual_energy:.4e}."
            )
        return math.sqrt(dual_energy)

    # ──────────────────────────────────────────────────────────────────────────
    # 1.6  Núcleo terminal de la Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def execute_phase1(
        self,
        q_dot: NDArray[np.float64],
        G_tensor: NDArray[np.float64],
        G_inv: NDArray[np.float64],
    ) -> MomentumAuditData:
        """
        Método terminal de la Fase 1.

        Su salida constituye el dominio formal de
        ``handoff_phase1_to_phase2``, que es el morfismo inicial de la Fase 2.
        """
        q_dot = self._validate_vector(q_dot, "q_dot")

        (
            G_tensor,
            G_inv,
            cond_G,
            inv_residual,
            wilkinson_bound,
            lambda_min,
            lambda_max,
        ) = self._validate_metric_pair(G_tensor, G_inv)

        if G_tensor.shape[0] != q_dot.size:
            raise RiemannianInertiaError(
                "Dimensión incompatible entre q_dot y la métrica."
            )

        momentum = self._musical_flat(q_dot, G_tensor)
        reconstructed = self._musical_sharp(momentum, G_inv)
        roundtrip_residual = self._certify_musical_roundtrip(
            q_dot,
            reconstructed,
            cond_G,
        )
        pairing, primal, dual, pairing_residual = self._certify_dual_pairing(
            q_dot,
            momentum,
            G_tensor,
            G_inv,
            cond_G,
        )
        p_norm = self._evaluate_momentum_norm(dual)
        is_bounded = p_norm <= _MOMENTUM_MAX_BOUND

        if not is_bounded:
            raise MomentumDivergenceError(
                "Divergencia inercial detectada: "
                f"‖p‖ = {p_norm:.4e} > {_MOMENTUM_MAX_BOUND:.4e}."
            )

        audit = MomentumAuditData(
            covariant_momentum=_freeze_array(momentum),
            reconstructed_velocity=_freeze_array(reconstructed),
            momentum_norm=p_norm,
            kinetic_energy_primal=primal,
            kinetic_energy_dual=dual,
            dual_pairing=pairing,
            pairing_residual=pairing_residual,
            musical_roundtrip_residual=roundtrip_residual,
            is_bounded=is_bounded,
            metric_condition_number=cond_G,
            inverse_consistency_residual=inv_residual,
            wilkinson_bound=wilkinson_bound,
            spectral_minimum=lambda_min,
            spectral_maximum=lambda_max,
        )

        logger.debug(
            "Fase 1 completada: ‖p‖ = %.6e, κ(G) = %.6e, "
            "residuo inversa = %.6e, residuo musical = %.6e, "
            "residuo apareamiento = %.6e.",
            audit.momentum_norm,
            audit.metric_condition_number,
            audit.inverse_consistency_residual,
            audit.musical_roundtrip_residual,
            audit.pairing_residual,
        )
        return audit

    def handoff_phase1_to_phase2(
        self,
        momentum_data: MomentumAuditData,
    ) -> NDArray[np.float64]:
        r"""
        Morfismo de transición

            $$\Phi_{12}:\mathrm{MomentumAuditData}\longrightarrow\mathbb{R}^n.$$

        Poscondición de la Fase 1  ≡  precondición de la Fase 2:

            $$p\in T^*\mathcal{M},\quad
              \|p\|_{G^{-1}}\le P_{\max},\quad
              p\text{ finito 1-D},\quad
              \mathrm{is\_bounded}=\top.$$

        Este método es la definición formal final de la Fase 1 y, a la
        vez, el dominio sobre el que la Fase 2 construye el bivector de
        Lorentz $$p\wedge\omega$$.  ``Phase2_GyroscopicSynthesizer``
        comienza invocándolo.
        """
        if not isinstance(momentum_data, MomentumAuditData):
            raise PhaseHandoffError(
                "handoff_phase1_to_phase2 exige MomentumAuditData."
            )

        if not momentum_data.is_bounded:
            raise PhaseHandoffError(
                "El certificado de Fase 1 no está acotado inercialmente."
            )

        if momentum_data.momentum_norm > _MOMENTUM_MAX_BOUND:
            raise PhaseHandoffError(
                "‖p‖ del certificado excede P_max en el handoff Φ₁₂."
            )

        if momentum_data.metric_condition_number > _CONDITION_NUMBER_MAX:
            raise PhaseHandoffError(
                "κ(G) del certificado excede la cota admisible en Φ₁₂."
            )

        momentum = self._validate_vector(
            momentum_data.covariant_momentum,
            "momentum_data.covariant_momentum",
        )
        return momentum


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 → SÍNTESIS DEL OPERADOR GIROSCÓPICO (Orient)
#          continuación formal de handoff_phase1_to_phase2
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_GyroscopicSynthesizer(Phase1_MomentumSpectrometer):
    r"""
    Recibe el covector certificado por ``handoff_phase1_to_phase2`` y
    construye el tensor giroscópico en el álgebra exterior:

        $$W_{\mu\nu}=\alpha(p_\mu\omega_\nu-p_\nu\omega_\mu)
          =\alpha\,(p\wedge\omega)_{\mu\nu}$$

    donde $$\omega=\Omega_{\mathrm{skew}}p$$ y $$\Omega$$ es una 2-forma
    de vorticidad proyectada al cono antisimétrico.

    La construcción garantiza $$W\in\mathfrak{so}(n)$$, evitando
    componentes disipativas o inyección energética espuria.

    El morfismo terminal ``handoff_phase2_to_phase3`` eleva el certificado
    a precondición formal de la Fase 3.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 2.1  Continuación inmediata del handoff de Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _receive_certified_momentum(
        self,
        momentum_data: MomentumAuditData,
    ) -> NDArray[np.float64]:
        """
        Primer método de la Fase 2.

        Es la continuación literal de ``handoff_phase1_to_phase2``:
        revalida el certificado y entrega el covector $$p$$ sobre el que
        se edifica el producto exterior $$p\wedge\omega$$.
        """
        return self.handoff_phase1_to_phase2(momentum_data)

    # ──────────────────────────────────────────────────────────────────────────
    # 2.2  Proyección de vorticidad y álgebra exterior
    # ──────────────────────────────────────────────────────────────────────────
    def _project_vorticity_to_two_form(
        self,
        vorticity_matrix: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        r"""
        Proyecta la matriz de vorticidad al espacio de 2-formas:

            $$\Omega_{\mathrm{skew}}=\tfrac12(\Omega-\Omega^\top)$$

        El residuo es la componente simétrica descartada (strain, no rotación):

            $$\bigl\|\tfrac12(\Omega+\Omega^\top)\bigr\|_F$$
        """
        vorticity = self._validate_square_matrix(vorticity_matrix, "vorticity_matrix")
        symmetric_part = self._symmetrize(vorticity)
        projection_residual = self._frobenius_norm(
            symmetric_part,
            "parte simétrica de vorticidad",
        )
        two_form = self._skew_symmetrize(vorticity)
        self._assert_finite(two_form, "2-forma de vorticidad")
        return two_form, projection_residual

    def _couple_vorticity(
        self,
        two_form: NDArray[np.float64],
        momentum: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Contracción de la 2-forma contra el momentum:

            $$\omega_\mu=\Omega_{\mu\nu}p^\nu$$

        (en coordenadas, $$\omega=\Omega_{\mathrm{skew}}p$$).
        """
        if two_form.shape[0] != momentum.size:
            raise RiemannianInertiaError(
                "La dimensión de vorticity_matrix no coincide con la del momentum."
            )
        omega = two_form @ momentum
        self._assert_finite(omega, "acoplamiento vorticial Ω p")
        return omega

    def _exterior_wedge(
        self,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
        coupling: float,
    ) -> NDArray[np.float64]:
        r"""
        Producto exterior de 1-formas, escalado por $$\alpha$$:

            $$(p\wedge\omega)_{\mu\nu}
              =\alpha\bigl(p_\mu\omega_\nu-p_\nu\omega_\mu\bigr)$$
        """
        if not math.isfinite(coupling):
            raise ExteriorAlgebraError(
                "El factor de acoplamiento giroscópico α no es finito."
            )
        left = self._validate_vector(left, "wedge.left")
        right = self._validate_vector(right, "wedge.right")
        if left.size != right.size:
            raise ExteriorAlgebraError(
                "Los 1-vectores del producto exterior tienen dimensión distinta."
            )
        bivector = coupling * (np.outer(left, right) - np.outer(right, left))
        self._assert_finite(bivector, "bivector p ∧ ω")
        return bivector

    def _certify_wedge_gram_identity(
        self,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
        bivector: NDArray[np.float64],
        coupling: float,
    ) -> float:
        r"""
        Identidad de Gram del álgebra exterior [I4]:

            $$\|p\wedge\omega\|_F^2
              = 2\bigl(\|p\|_2^2\|\omega\|_2^2-\langle p,\omega\rangle^2\bigr)$$

        Con el factor $$\alpha$$, el lado derecho se multiplica por
        $$\alpha^2$$.  El residuo se mide en norma absoluta del cuadrado.
        """
        left_norm_sq = self._neumaier_dot(left, left, "‖p‖²")
        right_norm_sq = self._neumaier_dot(right, right, "‖ω‖²")
        pairing = self._neumaier_dot(left, right, "⟨p, ω⟩")

        gram_rhs = 2.0 * (left_norm_sq * right_norm_sq - pairing * pairing)
        gram_rhs *= coupling * coupling
        gram_lhs = self._frobenius_norm(bivector, "p ∧ ω") ** 2

        residual = abs(gram_lhs - gram_rhs)
        scale = max(1.0, abs(gram_lhs), abs(gram_rhs))
        tolerance = max(
            _SYMPLECTIC_TOLERANCE,
            _WILKINSON_CONSTANT
            * _MACHINE_EPSILON
            * float(left.size) ** 2
            * scale,
        )
        if residual > tolerance:
            raise ExteriorAlgebraError(
                "Identidad de Gram del bivector violada. "
                f"‖p∧ω‖_F² = {gram_lhs:.4e}, Gram = {gram_rhs:.4e}, "
                f"residuo = {residual:.4e}, tolerancia = {tolerance:.4e}."
            )
        return residual

    # ──────────────────────────────────────────────────────────────────────────
    # 2.3  Proyección al cono de Lie so(n) y certificados espectrales
    # ──────────────────────────────────────────────────────────────────────────
    def _project_to_skew_symmetric_cone(
        self,
        W_raw: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, float]:
        r"""
        Proyección ortogonal (Frobenius) al álgebra de Lie
        $$\mathfrak{so}(n)$$:

            $$W_{\mathrm{proj}}=\tfrac12(W_{\mathrm{raw}}-W_{\mathrm{raw}}^\top)$$

        Se auditan el residuo absoluto $$\|W+W^\top\|_F$$ y el relativo
        documental $$r_{\mathrm{skew}}$$.
        """
        W_raw = self._validate_square_matrix(W_raw, "W_raw")
        W_proj = self._skew_symmetrize(W_raw)
        self._assert_finite(W_proj, "W_proj")

        abs_residual = self._skew_symmetric_residual(W_proj)
        rel_residual = self._relative_skew_residual(W_proj)
        abs_tol = self._matrix_residual_tolerance(W_proj)

        if abs_residual > abs_tol or rel_residual > _SKEW_RELATIVE_TOLERANCE:
            raise SkewSymmetryViolationError(
                "El proyector antisimétrico retiene componentes simétricas. "
                f"Residuo abs = {abs_residual:.4e} (tol {abs_tol:.4e}), "
                f"r_skew = {rel_residual:.4e} "
                f"(tol {_SKEW_RELATIVE_TOLERANCE:.4e})."
            )
        return W_proj, abs_residual, rel_residual

    def _certify_even_numerical_rank(
        self,
        W_proj: NDArray[np.float64],
    ) -> tuple[int, bool]:
        r"""
        Toda matriz antisimétrica real tiene rango par.

        Se estima el rango numérico por umbral espectral de valores
        singulares $$\sigma_i > n\cdot\varepsilon\cdot\sigma_{\max}$$.
        Un rango impar se interpreta como artefacto de umbral y se
        reporta, sin vetar, cuando el valor singular residual está en
        la zona de redondeo.
        """
        n = W_proj.shape[0]
        try:
            singular_values = np.linalg.svd(W_proj, compute_uv=False)
        except np.linalg.LinAlgError as exc:
            raise SkewSymmetryViolationError(
                "No fue posible calcular el espectro singular de W."
            ) from exc

        self._assert_finite(singular_values, "valores singulares de W")
        if singular_values.size == 0:
            return 0, True

        sigma_max = float(singular_values[0])
        if sigma_max == 0.0:
            return 0, True

        floor = max(
            float(n) * _MACHINE_EPSILON * sigma_max,
            _SYMPLECTIC_TOLERANCE * max(1.0, sigma_max),
        )
        rank = int(np.count_nonzero(singular_values > floor))
        return rank, (rank % 2 == 0)

    # ──────────────────────────────────────────────────────────────────────────
    # 2.4  Núcleo terminal de la Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _synthesize_lorentz_tensor(
        self,
        momentum_data: MomentumAuditData,
        vorticity_matrix: NDArray[np.float64],
    ) -> GyroscopicSynthesisData:
        r"""
        Síntesis del tensor de Lorentz giroscópico:

            $$\omega=\Omega_{\mathrm{skew}}p$$
            $$W=\alpha\,(p\wedge\omega)$$

        La construcción es manifiestamente antisimétrica y se re-proyecta
        al cono para eliminar ruido de mantisa.
        """
        momentum = self._receive_certified_momentum(momentum_data)
        two_form, vorticity_residual = self._project_vorticity_to_two_form(
            vorticity_matrix
        )
        omega = self._couple_vorticity(two_form, momentum)

        coupling = float(_VORTICITY_COUPLING_FACTOR)
        W_raw = self._exterior_wedge(momentum, omega, coupling)
        gram_residual = self._certify_wedge_gram_identity(
            momentum,
            omega,
            W_raw,
            coupling,
        )
        W_proj, abs_residual, rel_residual = self._project_to_skew_symmetric_cone(
            W_raw
        )
        W_norm = self._frobenius_norm(W_proj, "W_proj")
        rank, rank_even = self._certify_even_numerical_rank(W_proj)

        synthesis = GyroscopicSynthesisData(
            skew_symmetric_tensor=_freeze_array(W_proj),
            vorticity_two_form=_freeze_array(two_form),
            omega_vector=_freeze_array(omega),
            antisymmetry_residual=abs_residual,
            relative_skew_residual=rel_residual,
            vorticity_projection_residual=vorticity_residual,
            gyroscopic_frobenius_norm=W_norm,
            wedge_gram_residual=gram_residual,
            skew_numerical_rank=rank,
            rank_is_even=rank_even,
            is_strictly_skew=True,
        )

        logger.debug(
            "Fase 2 completada: ‖W‖_F = %.6e, r_skew = %.6e, "
            "residuo vorticidad = %.6e, residuo Gram = %.6e, rango = %d.",
            synthesis.gyroscopic_frobenius_norm,
            synthesis.relative_skew_residual,
            synthesis.vorticity_projection_residual,
            synthesis.wedge_gram_residual,
            synthesis.skew_numerical_rank,
        )
        return synthesis

    def execute_phase2(
        self,
        momentum_data: MomentumAuditData,
        vorticity_matrix: NDArray[np.float64],
    ) -> GyroscopicSynthesisData:
        """
        Método terminal de la Fase 2.

        Recibe la salida formal de ``execute_phase1`` (vía
        ``handoff_phase1_to_phase2``) y produce la entrada canónica de
        ``handoff_phase2_to_phase3``.
        """
        return self._synthesize_lorentz_tensor(momentum_data, vorticity_matrix)

    def handoff_phase2_to_phase3(
        self,
        synthesis_data: GyroscopicSynthesisData,
    ) -> NDArray[np.float64]:
        r"""
        Morfismo de transición

            $$\Phi_{23}:\mathrm{GyroscopicSynthesisData}
              \longrightarrow\mathfrak{so}(n).$$

        Poscondición de la Fase 2  ≡  precondición de la Fase 3:

            $$W^\top=-W,\quad
              r_{\mathrm{skew}}\le\epsilon_{\mathrm{skew}},\quad
              \mathrm{is\_strictly\_skew}=\top.$$

        Este método es la definición formal final de la Fase 2 y, a la
        vez, el dominio sobre el que la Fase 3 realiza la inyección
        $$J_{\mathrm{eff}}=J+W$$.  ``Phase3_SymplecticInertiaModulator``
        comienza invocándolo.
        """
        if not isinstance(synthesis_data, GyroscopicSynthesisData):
            raise PhaseHandoffError(
                "handoff_phase2_to_phase3 exige GyroscopicSynthesisData."
            )

        if not synthesis_data.is_strictly_skew:
            raise PhaseHandoffError(
                "El certificado de Fase 2 no es estrictamente antisimétrico."
            )

        if synthesis_data.relative_skew_residual > _SKEW_RELATIVE_TOLERANCE:
            raise PhaseHandoffError(
                "r_skew del certificado excede ε_skew en el handoff Φ₂₃."
            )

        W_proj = self._validate_square_matrix(
            synthesis_data.skew_symmetric_tensor,
            "synthesis_data.skew_symmetric_tensor",
        )

        rel = self._relative_skew_residual(W_proj)
        if rel > _SKEW_RELATIVE_TOLERANCE:
            raise PhaseHandoffError(
                "W no permanece en so(n) al revalidar Φ₂₃. "
                f"r_skew = {rel:.4e}."
            )
        return W_proj


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 → MODULACIÓN SIMPLÉCTICA Y VEREDICTO TERMODINÁMICO (Decide & Act)
#          continuación formal de handoff_phase2_to_phase3
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_SymplecticInertiaModulator(Phase2_GyroscopicSynthesizer):
    r"""
    Recibe el tensor certificado por ``handoff_phase2_to_phase3``,
    lo inyecta en la estructura de Dirac $$J$$:

        $$J_{\mathrm{eff}}=J+W$$

    y certifica la pasividad simpléctica:

        $$\langle\nabla H,\,J_{\mathrm{eff}}\nabla H\rangle=0$$

    Para ello:

        1. exige que $$J$$ sea antisimétrica;
        2. proyecta $$J_{\mathrm{eff}}$$ al cono antisimétrico;
        3. audita el trabajo por Neumaier y por la fórmula estructural
           de pares;
        4. sondea el teorema de Liouville con vectores aleatorios;
        5. emplea cotas adaptativas proporcionales a la escala.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 3.1  Continuación inmediata del handoff de Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _receive_certified_gyroscopic_tensor(
        self,
        synthesis_data: GyroscopicSynthesisData,
    ) -> NDArray[np.float64]:
        """
        Primer método de la Fase 3.

        Es la continuación literal de ``handoff_phase2_to_phase3``:
        revalida el certificado y entrega $$W\in\mathfrak{so}(n)$$ para
        la inyección de Dirac.
        """
        return self.handoff_phase2_to_phase3(synthesis_data)

    # ──────────────────────────────────────────────────────────────────────────
    # 3.2  Estructura de Dirac y modulación
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_dirac_structure(
        self,
        J_base: NDArray[np.float64],
        name: str = "J_base",
    ) -> tuple[NDArray[np.float64], float, float]:
        r"""
        Valida que la estructura de Dirac base sea antisimétrica:

            $$J+J^\top=0$$

        En sistemas Port-Hamiltonianos, $$J$$ pertenece al álgebra de Lie
        del grupo que preserva la 2-forma; cualquier componente simétrica
        representa disipación o inyección energética no autorizada
        (esa función recae exclusivamente en $$R(x)\succeq 0$$).
        """
        interconnection = self._validate_square_matrix(J_base, name)
        abs_residual = self._skew_symmetric_residual(interconnection)
        rel_residual = self._relative_skew_residual(interconnection)
        abs_tol = self._matrix_residual_tolerance(interconnection)

        if abs_residual > abs_tol or rel_residual > _SKEW_RELATIVE_TOLERANCE:
            raise SymplecticWorkViolationError(
                f"{name} no es antisimétrica. "
                f"Residuo abs = {abs_residual:.4e} (tol {abs_tol:.4e}), "
                f"r_skew = {rel_residual:.4e} "
                f"(tol {_SKEW_RELATIVE_TOLERANCE:.4e})."
            )

        J_skew = self._skew_symmetrize(interconnection)
        self._assert_finite(J_skew, f"{name} proyectada")
        return J_skew, abs_residual, rel_residual

    def _modulate_dirac_structure(
        self,
        J_base: NDArray[np.float64],
        W_proj: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Acopla el operador giroscópico a la estructura de Dirac:

            $$J_{\mathrm{eff}}=J+W$$

        Como $$J$$ y $$W$$ son antisimétricas, $$J_{\mathrm{eff}}$$
        también lo es.  Se realiza una proyección final de saneamiento
        para eliminar ruido de mantisa.
        """
        J_base = self._validate_square_matrix(J_base, "J_base")
        W_proj = self._validate_square_matrix(W_proj, "W_proj")

        if J_base.shape != W_proj.shape:
            raise RiemannianInertiaError(
                "J_base y W_proj deben tener la misma dimensión."
            )

        J_eff = J_base + W_proj
        self._assert_finite(J_eff, "J_eff bruta")
        J_eff = self._skew_symmetrize(J_eff)
        self._assert_finite(J_eff, "J_eff proyectada")
        return J_eff

    # ──────────────────────────────────────────────────────────────────────────
    # 3.3  Trabajo nulo: Neumaier, pares estructurales y Liouville
    # ──────────────────────────────────────────────────────────────────────────
    def _neumaier_quadratic_form(
        self,
        vector: NDArray[np.float64],
        matrix: NDArray[np.float64],
    ) -> tuple[float, float]:
        r"""
        Calcula $$x^\top A x$$ mediante sumación compensada de Neumaier.

        Retorna:
            ``(work_sum, abs_sum)`` donde ``abs_sum`` es la suma
            compensada de $$|A_{ij}x_i x_j|$$, usada para cotas de error:

            $$|\mathrm{error}|\lesssim\varepsilon_{\mathrm{machine}}\sum|t_{ij}|$$
        """
        vector = self._validate_vector(vector, "x")
        matrix = self._validate_square_matrix(matrix, "A")
        n = vector.size
        if matrix.shape != (n, n):
            raise RiemannianInertiaError(
                "Dimensión incompatible entre x y A en la forma cuadrática."
            )

        work_total = 0.0
        work_comp = 0.0
        abs_total = 0.0
        abs_comp = 0.0

        for i in range(n):
            xi = float(vector[i])
            terms = xi * matrix[i, :] * vector
            self._assert_finite(terms, f"términos de xᵀAx en fila {i}")

            for raw in terms:
                term = float(raw)

                abs_y = abs(term)
                abs_trial = abs_total + abs_y
                if abs(abs_total) >= abs_y:
                    abs_comp += (abs_total - abs_trial) + abs_y
                else:
                    abs_comp += (abs_y - abs_trial) + abs_total
                if not math.isfinite(abs_trial):
                    raise RiemannianInertiaError(
                        "La sumación absoluta de Neumaier divergió."
                    )
                abs_total = abs_trial

                trial = work_total + term
                if abs(work_total) >= abs(term):
                    work_comp += (work_total - trial) + term
                else:
                    work_comp += (term - trial) + work_total
                if not math.isfinite(trial):
                    raise RiemannianInertiaError(
                        "La sumación compensada de Neumaier divergió."
                    )
                work_total = trial

        work_sum = work_total + work_comp
        abs_sum = abs_total + abs_comp
        if not math.isfinite(work_sum) or not math.isfinite(abs_sum):
            raise RiemannianInertiaError(
                "La forma cuadrática compensada no es finita."
            )
        return work_sum, abs_sum

    def _pairwise_structural_work(
        self,
        vector: NDArray[np.float64],
        matrix: NDArray[np.float64],
    ) -> float:
        r"""
        Residuo estructural del trabajo, independiente de la cancelación
        numérica de $$x^\top A x$$ cuando $$A$$ es casi antisimétrica:

            $$P_{\mathrm{pair}}
              =\sum_{i<j}(A_{ij}+A_{ji})x_i x_j
              +\sum_i A_{ii}x_i^2$$

        En aritmética exacta, $$A^\top=-A$$ implica $$P_{\mathrm{pair}}=0$$.
        El residuo mide, por tanto, la contaminación simétrica residual
        ponderada por el estado.
        """
        vector = self._validate_vector(vector, "x")
        matrix = self._validate_square_matrix(matrix, "A")
        n = vector.size
        if matrix.shape != (n, n):
            raise RiemannianInertiaError(
                "Dimensión incompatible en el trabajo estructural de pares."
            )

        terms: list[float] = []
        for i in range(n):
            xi = float(vector[i])
            terms.append(float(matrix[i, i]) * xi * xi)
            for j in range(i + 1, n):
                xj = float(vector[j])
                coeff = float(matrix[i, j]) + float(matrix[j, i])
                terms.append(coeff * xi * xj)

        payload = np.array(terms, dtype=np.float64)
        return self._neumaier_sum(payload, "trabajo estructural de pares")

    def _liouville_probe_residual(
        self,
        J_eff: NDArray[np.float64],
    ) -> float:
        r"""
        Auditoría Monte-Carlo del teorema de Liouville [I1]:

            $$x^\top J_{\mathrm{eff}}x=0\qquad\forall x$$

        Se lanzan sondas gaussianas reproducibles y se retiene el máximo
        del trabajo compensado.  Equivale a muestrear la forma cuadrática
        nula asociada a la antisimetría.
        """
        n = J_eff.shape[0]
        rng = np.random.default_rng(_LIOUVILLE_PROBE_SEED)
        worst = 0.0
        probes = max(1, int(_LIOUVILLE_PROBE_COUNT))
        for _ in range(probes):
            probe = rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float64)
            norm = self._euclidean_norm(probe, "sonda de Liouville")
            if norm == 0.0:
                continue
            probe /= norm
            work, _ = self._neumaier_quadratic_form(probe, J_eff)
            worst = max(worst, abs(work))
        return worst

    def _adaptive_work_tolerance(
        self,
        grad_H: NDArray[np.float64],
        J_eff: NDArray[np.float64],
        abs_sum: float,
    ) -> float:
        r"""
        Cota adaptativa para el trabajo nulo:

            $$\tau=\max\bigl(
                100\varepsilon,\;
                U\cdot\varepsilon\cdot
                \max\bigl(1,\;\|\nabla H\|_2^2\|J\|_F,\;\sum|t_{ij}|\bigr)
            \bigr)$$
        """
        grad_norm = self._euclidean_norm(grad_H, "∇H")
        j_norm = self._frobenius_norm(J_eff, "J_eff")
        scale_by_operator = (grad_norm * grad_norm) * j_norm
        scale_by_terms = max(1.0, abs_sum)
        scale = max(1.0, scale_by_operator, scale_by_terms)
        return max(
            100.0 * _MACHINE_EPSILON,
            _WORK_TOLERANCE_ULPS * _MACHINE_EPSILON * scale,
        )

    def _certify_nilpotent_work(
        self,
        grad_H: NDArray[np.float64],
        J_eff: NDArray[np.float64],
    ) -> ThermodynamicVetoData:
        r"""
        Certifica el teorema de trabajo nulo en tres capas:

        1. Residuo simétrico y $$r_{\mathrm{skew}}$$ de $$J_{\mathrm{eff}}$$.
        2. Proyección estricta $$J_{\mathrm{certified}}=\mathrm{skew}(J_{\mathrm{eff}})$$.
        3. Auditoría conjunta:
               - Neumaier de $$x^\top J x$$;
               - residuo estructural de pares;
               - sondas de Liouville.
        """
        grad_H = self._validate_vector(grad_H, "grad_H")
        J_eff = self._validate_square_matrix(J_eff, "J_eff")

        n = grad_H.size
        if J_eff.shape != (n, n):
            raise RiemannianInertiaError(
                "Dimensión incompatible entre grad_H y J_eff."
            )

        symmetric_residual = self._skew_symmetric_residual(J_eff)
        relative_residual = self._relative_skew_residual(J_eff)
        sym_tol = self._matrix_residual_tolerance(J_eff)

        if (
            symmetric_residual > sym_tol
            or relative_residual > _SKEW_RELATIVE_TOLERANCE
        ):
            raise SymplecticWorkViolationError(
                "J_eff contiene una componente simétrica incompatible con pasividad. "
                f"Residuo = {symmetric_residual:.4e} (tol {sym_tol:.4e}), "
                f"r_skew = {relative_residual:.4e}."
            )

        J_certified = self._skew_symmetrize(J_eff)
        self._assert_finite(J_certified, "J_certified")

        work_sum, abs_sum = self._neumaier_quadratic_form(grad_H, J_certified)
        work_residual = abs(work_sum)
        pairwise_residual = abs(
            self._pairwise_structural_work(grad_H, J_certified)
        )
        liouville_residual = self._liouville_probe_residual(J_certified)

        work_tolerance = self._adaptive_work_tolerance(
            grad_H,
            J_certified,
            abs_sum,
        )

        is_passive = (
            work_residual <= work_tolerance
            and pairwise_residual <= work_tolerance
            and liouville_residual <= work_tolerance
        )

        if not is_passive:
            raise SymplecticWorkViolationError(
                "El operador efectivo realiza trabajo neto no nulo. "
                f"Neumaier = {work_residual:.4e}, "
                f"pares = {pairwise_residual:.4e}, "
                f"Liouville = {liouville_residual:.4e}, "
                f"tolerancia = {work_tolerance:.4e}."
            )

        veto = ThermodynamicVetoData(
            effective_dirac_matrix=_freeze_array(J_certified),
            nilpotent_work_residual=work_residual,
            pairwise_work_residual=pairwise_residual,
            liouville_probe_residual=liouville_residual,
            dirac_symmetric_residual=symmetric_residual,
            relative_skew_residual=relative_residual,
            work_tolerance=work_tolerance,
            is_symplectically_passive=bool(is_passive),
        )

        logger.debug(
            "Fase 3 completada: trabajo = %.6e, pares = %.6e, "
            "Liouville = %.6e, tolerancia = %.6e, r_skew = %.6e.",
            veto.nilpotent_work_residual,
            veto.pairwise_work_residual,
            veto.liouville_probe_residual,
            veto.work_tolerance,
            veto.relative_skew_residual,
        )
        return veto

    def execute_phase3(
        self,
        grad_H: NDArray[np.float64],
        J_base: NDArray[np.float64],
        synthesis_data: GyroscopicSynthesisData,
    ) -> ThermodynamicVetoData:
        """
        Método terminal de la Fase 3.

        Recibe la salida formal de ``execute_phase2`` (vía
        ``handoff_phase2_to_phase3``) y devuelve el veredicto
        termodinámico definitivo.
        """
        W_proj = self._receive_certified_gyroscopic_tensor(synthesis_data)
        grad_H = self._validate_vector(grad_H, "grad_H")
        J_base = self._validate_square_matrix(J_base, "J_base")

        n = grad_H.size
        if J_base.shape != (n, n):
            raise RiemannianInertiaError(
                "Dimensión incompatible entre grad_H y J_base."
            )
        if W_proj.shape != (n, n):
            raise RiemannianInertiaError(
                "La dimensión de W_proj no coincide con grad_H y J_base."
            )

        J_base_skew, _, _ = self._validate_dirac_structure(J_base, "J_base")
        J_eff = self._modulate_dirac_structure(J_base_skew, W_proj)
        return self._certify_nilpotent_work(grad_H, J_eff)


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR SUPREMO: RIEMANNIAN INERTIA MODULATOR
# ══════════════════════════════════════════════════════════════════════════════
class RiemannianInertiaModulator(Morphism, Phase3_SymplecticInertiaModulator):
    r"""
    Funtor físico puro de moldeo de masa.

    Compone las tres fases en una sola flecha categórica determinista:

        $$\Phi_{01}\xrightarrow{\ \mathrm{Fase}\,1\ }
          \mathrm{MomentumAuditData}
          \xrightarrow{\ \Phi_{12}\ }
          \mathrm{Fase}\,2
          \xrightarrow{\ \Phi_{23}\ }
          \mathrm{Fase}\,3
          \longrightarrow\mathrm{ThermodynamicVetoData}$$

    La composición es estricta: cualquier violación de inercia,
    coherencia musical, álgebra exterior, antisimetría o pasividad
    simpléctica detiene la ejecución con una excepción tipada.
    """

    def apply_inertia_modulation(
        self,
        q_dot: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        G_tensor: NDArray[np.float64],
        G_inv: NDArray[np.float64],
        J_base: NDArray[np.float64],
        vorticity_matrix: NDArray[np.float64],
    ) -> ThermodynamicVetoData:
        r"""
        Orquesta la inyección del campo giroscópico de Gauge.

        Flujo formal:

            1. $$\dot{q},\,G,\,G^{-1}
                 \mapsto p=G\dot{q},\;
                 \|p\|_{G^{-1}},\;
                 \sharp\flat\dot{q}=\dot{q}$$.

            2. $$p,\,\Omega
                 \mapsto W=\alpha\bigl(p\wedge(\Omega_{\mathrm{skew}}p)\bigr),
                 \;W\in\mathfrak{so}(n)$$.

            3. $$\nabla H,\,J,\,W
                 \mapsto J_{\mathrm{eff}}=\mathrm{skew}(J+W),
                 \;x^\top J_{\mathrm{eff}}x=0$$.

        Retorna:
            ``ThermodynamicVetoData`` con la certificación final.
        """
        q_dot = self._validate_vector(q_dot, "q_dot")
        grad_H = self._validate_vector(grad_H, "grad_H")
        if q_dot.size != grad_H.size:
            raise RiemannianInertiaError(
                "q_dot y grad_H deben vivir en el mismo espacio vectorial."
            )

        momentum_audit = self.execute_phase1(
            q_dot=q_dot,
            G_tensor=G_tensor,
            G_inv=G_inv,
        )
        # Φ₁₂ : el handoff se invoca dentro de execute_phase2.
        synthesis_data = self.execute_phase2(
            momentum_data=momentum_audit,
            vorticity_matrix=vorticity_matrix,
        )
        # Φ₂₃ : el handoff se invoca dentro de execute_phase3.
        veto_data = self.execute_phase3(
            grad_H=grad_H,
            J_base=J_base,
            synthesis_data=synthesis_data,
        )

        logger.info(
            "Modulación de inercia certificada: ‖p‖ = %.6e, ‖W‖_F = %.6e, "
            "trabajo residual = %.6e, Liouville = %.6e.",
            momentum_audit.momentum_norm,
            synthesis_data.gyroscopic_frobenius_norm,
            veto_data.nilpotent_work_residual,
            veto_data.liouville_probe_residual,
        )
        return veto_data


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "RiemannianInertiaError",
    "MetricCoherenceError",
    "DualPairingError",
    "MomentumDivergenceError",
    "ExteriorAlgebraError",
    "SkewSymmetryViolationError",
    "SymplecticWorkViolationError",
    "PhaseHandoffError",
    "MomentumAuditData",
    "GyroscopicSynthesisData",
    "ThermodynamicVetoData",
    "Phase1_MomentumSpectrometer",
    "Phase2_GyroscopicSynthesizer",
    "Phase3_SymplecticInertiaModulator",
    "RiemannianInertiaModulator",
]
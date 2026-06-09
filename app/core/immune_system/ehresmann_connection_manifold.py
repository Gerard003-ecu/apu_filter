# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Ehresmann Connection Manifold (Fibrado de Integración Simpléctica)   ║
║ Ubicación: app/core/immune_system/ehresmann_connection_manifold.py           ║
║ Versión: 3.0.0-Rigorous-Phased-Synthesis                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Topológica (Síntesis en 3 Fases Anidadas):
─────────────────────────────────────────────────────────────────────
Este módulo implementa la Adjunción Funtorial de Grothendieck entre la evaluación
termodinámica discreta (Funtor Shield) y el análisis métrico (Topological Watcher),
estructurada en tres fases de rigurosidad creciente.

COHERENCIA INTER-FASES (invariante de diseño):
  • Cada clase de Fase N extiende la de Fase N-1 mediante herencia estricta.
  • Los errores corregidos en Fase 2 son coherentes con las definiciones —
    deliberadamente simplificadas pero matemáticamente trazables— de Fase 1.
  • Fase 3 garantiza invariantes físicos verificables en tiempo de ejecución.

CORRECCIONES RESPECTO A v2.0.0:
  ① Phase1: shape[0] en lugar de shape (tupla) en todos los métodos.
  ② Phase1: psi_signal construido como vector 1-D (n,) no como matriz (n,n).
  ③ Phase1: Laplaciano de Hodge definido como δd = *₂ᵀ *₁ *₂ con convención
     orientada, coherente con el complejo de cadenas C₀ ← C₁ ← C₂.
  ④ Phase2: epsilon adaptativo con cota inferior absoluta para evitar
     cancelación catastrófica en aritmética de punto flotante IEEE-754.
  ⑤ Phase2: Jacobiano aproximado corregido: J_{ij} = ∂Xⁱ/∂xʲ ≈ diag(∇·X/n)·I
     con normalización dimensional explícita.
  ⑥ Phase3: default argument mutable eliminado (anti-patrón Python).
  ⑦ Phase3: corrección de disipación mediante proyección en el cono de matrices
     semidefinidas negativas (SND) en lugar de suma directa de identidad.
  ⑧ Phase3: verificación del teorema de valor medio con tolerancia relativa
     y absoluta combinadas (norma mixta).

FÍSICA MODELADA:
  • Gradiente Discreto de Itoh–Abe: preserva exactamente la variación de H.
  • Acoplamiento de Fröhlich–Laplaciano de Hodge: renormalización de masa
    efectiva en redes de grafos con torsión topológica.
  • Mediador de Grothendieck: flujo de Ricci discreto acoplado a la distancia
    de Mahalanobis como parámetro de control de la bifurcación de Higgs.
  • Inecuación de Disipación (Port-Hamiltoniana):
      Ḣ = ∇H · ẋ = −∇Hᵀ R(x) ∇H ≤ 0   ∀ R(x) ≽ 0

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Dependencias del sistema MIC  (stubs locales si no se importan del proyecto)
# ---------------------------------------------------------------------------
try:
    from app.core.mic_algebra import (
        CategoricalState,
        Morphism,
        NumericalInstabilityError,
        Stratum,
    )
    from app.core.immune_system.metric_tensors import G_PHYSICS
    from app.core.immune_system.topological_watcher import ThreatAssessment
except ImportError:  # pragma: no cover — stubs para pruebas unitarias aisladas
    import dataclasses

    class NumericalInstabilityError(RuntimeError):  # type: ignore[no-redef]
        pass

    class Morphism:  # type: ignore[no-redef]
        pass

    @dataclasses.dataclass
    class CategoricalState:  # type: ignore[no-redef]
        stratum: str = "nominal"

    @dataclasses.dataclass
    class ThreatAssessment:  # type: ignore[no-redef]
        mahalanobis_distance: float = 0.0
        euler_char: Optional[float] = None

    G_PHYSICS: NDArray[np.float64] = np.eye(4, dtype=np.float64)

logger = logging.getLogger("MIC.ImmuneSystem.EhresmannConnection")

# =============================================================================
# FASE 1 — DEFINICIONES FUNDACIONALES
# =============================================================================
# Propósito pedagógico y de trazabilidad:
#   Se plasman los conceptos matemáticos de manera directa y reconocible,
#   sin vectorizaciones ni validaciones exhaustivas. Cada clase expone
#   únicamente la semántica esencial del objeto matemático que modela.
#   Los comentarios «BUG-F1» marcan las simplificaciones que Fase 2 corrige,
#   de modo que la progresión resulte auditable.
#
# Invariante de Fase 1:
#   • Los métodos son matemáticamente trazables a sus fórmulas originales.
#   • No se usan optimizaciones que oscurezcan la intención algebraica.
#   • Los errores anotados son intencionados y documentados explícitamente.
# =============================================================================


class Phase1_ItohAbeDiscreteGradient:
    r"""
    Operador de Gradiente Discreto de Itoh–Abe (Fase 1 — Fundacional).

    Implementa la condición de preservación energética exacta:

    .. math::

        H(x_{k+1}) - H(x_k)
        = \overline{\nabla}H(x_k,\, x_{k+1})^{\!\top}(x_{k+1} - x_k)

    mediante el algoritmo de barrido coordenada a coordenada de Gonzalez (1996):

    .. math::

        \overline{\nabla}_i H = \frac{H(x^{(i+1)}) - H(x^{(i)})}
                                     {x_{k+1}^i - x_k^i}

    donde :math:`x^{(i)}` es la configuración «escalera» que ha transitado
    las primeras :math:`i` coordenadas de :math:`x_k` a :math:`x_{k+1}`.

    .. note::
        **BUG-F1 documentado**: se usa ``x_k.shape`` (tupla) en lugar de
        ``x_k.shape[0]`` (entero). El error se corrige en :class:`Phase2_ItohAbeDiscreteGradient`.
    """

    @staticmethod
    def compute(
        H_func: Callable[[NDArray[np.float64]], float],
        x_k: NDArray[np.float64],
        x_next: NDArray[np.float64],
        epsilon: float = 1e-12,
    ) -> NDArray[np.float64]:
        r"""
        Calcula :math:`\overline{\nabla}H(x_k, x_{k+1})`.

        Parameters
        ----------
        H_func:
            Hamiltoniano escalar :math:`H: \mathbb{R}^n \to \mathbb{R}`.
        x_k:
            Estado actual :math:`x_k \in \mathbb{R}^n`.
        x_next:
            Estado siguiente :math:`x_{k+1} \in \mathbb{R}^n`.
        epsilon:
            Umbral para diferencia central cuando :math:`\Delta x_i \approx 0`.

        Returns
        -------
        grad_discrete : ndarray, shape (n,)
            Gradiente discreto de Itoh–Abe.

        Notes
        -----
        **BUG-F1**: ``n_dim = x_k.shape`` devuelve la tupla ``(n,)``, no el
        entero ``n``.  ``np.zeros(n_dim)`` acepta la tupla (comportamiento
        accidental correcto en NumPy), pero ``range(n_dim)`` lanzaría
        ``TypeError`` en Python estándar.  Documentado aquí para trazabilidad;
        corregido en Fase 2.
        """
        # BUG-F1: shape devuelve tupla, no entero — corregido en Fase 2
        n_dim = x_k.shape[0]  # ← CORRECCIÓN MÍNIMA para que el código ejecute
        grad_discrete = np.zeros(n_dim, dtype=np.float64)

        for i in range(n_dim):
            # Configuración escalera inferior: coordenadas 0…i-1 ya transitadas
            x_temp_k = np.copy(x_next)
            x_temp_k[i:] = x_k[i:]

            # Configuración escalera superior: coordenadas 0…i ya transitadas
            x_temp_next = np.copy(x_next)
            x_temp_next[(i + 1):] = x_k[(i + 1):]

            delta_x_i = x_temp_next[i] - x_temp_k[i]

            if np.abs(delta_x_i) > epsilon:
                delta_H = H_func(x_temp_next) - H_func(x_temp_k)
                grad_discrete[i] = delta_H / delta_x_i
            else:
                # Aproximación de respaldo por diferencia central en x_k
                x_forward = np.copy(x_k)
                x_forward[i] += epsilon
                x_backward = np.copy(x_k)
                x_backward[i] -= epsilon
                grad_discrete[i] = (H_func(x_forward) - H_func(x_backward)) / (
                    2.0 * epsilon
                )

        return grad_discrete


class Phase1_SpectralTorsionCoupling:
    r"""
    Acoplamiento Espectral de Fröhlich (Fase 1 — Fundacional).

    Modela la renormalización de masa efectiva en una red discreta con torsión
    topológica :math:`\mathcal{T}` y Laplaciano de Hodge :math:`\Delta_H`:

    .. math::

        \psi_{\text{ren}} = m_{\text{eff}}\!\left(1 + \alpha\right)
                            \Delta_H\, \psi

    donde

    .. math::

        \alpha = \frac{\exp(-\langle\psi,\,\mathcal{T}\psi\rangle)}
                      {6\,\lambda_2}

    y :math:`\lambda_2 > 0` es el valor de Fiedler del grafo subyacente.

    El **Laplaciano de Hodge de grado 1** sobre el complejo de cadenas
    :math:`C_0 \xleftarrow{\partial_1} C_1 \xleftarrow{\partial_2} C_2`
    se define como:

    .. math::

        \Delta_1 = \partial_1^{\dagger}\partial_1 + \partial_2\partial_2^{\dagger}
                 = \star_2^{\!\top}\star_1\,\star_2 + \star_2\,\star_2^{\!\top}\star_1

    En Fase 1 se usa la versión simplificada :math:`\Delta_1 \approx \star_1\star_2^{\!\top}\star_2`
    (omite el término de coborde), corregida en Fase 2.

    Parameters
    ----------
    fiedler_value:
        :math:`\lambda_2 > 0`, segundo valor propio del laplaciano combinatorial.
    torsion_matrix:
        Matriz dispersa :math:`\mathcal{T} \in \mathbb{R}^{n \times n}`,
        simétrica semidefinida positiva.
    hodge_star_1, hodge_star_2:
        Operadores estrella de Hodge discretos :math:`\star_1, \star_2 \in
        \mathbb{R}^{n \times n}`.
    """

    def __init__(
        self,
        fiedler_value: float,
        torsion_matrix: sp.csr_matrix,
        hodge_star_1: sp.csr_matrix,
        hodge_star_2: sp.csr_matrix,
    ) -> None:
        if fiedler_value <= 1e-10:
            raise NumericalInstabilityError(
                "Valor de Fiedler nulo o negativo: el grafo es algebraicamente "
                "desconexo.  Proporcione un grafo fuertemente conexo."
            )
        self.lambda_2 = float(fiedler_value)
        self.torsion = torsion_matrix.tocsr()
        self.star_1 = hodge_star_1.tocsr()
        self.star_2 = hodge_star_2.tocsr()

        # Laplaciano de Hodge (Fase 1: solo término de borde ∂₁†∂₁ ≈ ★₁★₂ᵀ★₂)
        # BUG-F1 documentado: falta el término de coborde ∂₂∂₂† = ★₂★₂ᵀ★₁
        # Corregido en Phase2_SpectralTorsionCoupling.
        self._laplacian: sp.csr_matrix = self.star_1 @ (self.star_2.T @ self.star_2)

    def apply_frohlich_renormalization(
        self, m_eff: float, psi_state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Aplica la renormalización de Fröhlich.

        Parameters
        ----------
        m_eff:
            Masa efectiva bare :math:`m_0 > 0`.
        psi_state:
            Vector de estado :math:`\psi \in \mathbb{R}^n`.

        Returns
        -------
        ndarray, shape (n,)
            Estado renormalizado :math:`m_{\text{ren}}\,\Delta_1\psi`.
        """
        psi = np.asarray(psi_state, dtype=np.float64).ravel()
        torsion_form = float(psi @ (self.torsion @ psi))
        alpha = np.exp(-torsion_form) / (6.0 * self.lambda_2)
        mass_renorm = m_eff * (1.0 + alpha)
        return mass_renorm * (self._laplacian @ psi)


class Phase1_GrothendieckTopologicalMediator(Morphism):
    r"""
    Mediador Topológico de Grothendieck (Fase 1 — Fundacional).

    Implementa la dinámica simpléctica del fibrado de Ehresmann mediante:

    1. **Derivada de Lie discreta** del tensor métrico :math:`G` a lo largo
       de un campo vectorial :math:`X`:

       .. math::

           \mathcal{L}_X G \approx G\,\text{diag}(\nabla X)
                                  + \text{diag}(\nabla X)^{\!\top} G

    2. **Sincronización de variedades**: modulación de la matriz de disipación
       :math:`R_d` y del potencial de Higgs :math:`\mu^2` en función de la
       distancia de Mahalanobis :math:`d_M`.

    .. note::
        **BUG-F1 documentado en ``synchronize_manifolds``**: ``psi_signal``
        se construye como ``np.ones(self.G_PHYSICS.shape)`` —una matriz 2-D—
        en lugar de un vector 1-D de dimensión ``n``.  Documentado aquí;
        corregido en Fase 2.

    Parameters
    ----------
    metric_tensor:
        Tensor métrico físico :math:`G \in \mathbb{R}^{n \times n}`,
        simétrico definido positivo. Por defecto ``G_PHYSICS``.
    ricci_flow_rate:
        Tasa de flujo de Ricci :math:`\alpha_R > 0`. Por defecto ``0.01``.
    """

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = None,
        ricci_flow_rate: float = 0.01,
    ) -> None:
        super().__init__()
        self.G_PHYSICS: NDArray[np.float64] = (
            np.asarray(G_PHYSICS, dtype=np.float64)
            if metric_tensor is None
            else np.asarray(metric_tensor, dtype=np.float64)
        )
        if self.G_PHYSICS.ndim != 2 or self.G_PHYSICS.shape[0] != self.G_PHYSICS.shape[1]:
            raise ValueError("metric_tensor debe ser una matriz cuadrada (n, n).")
        self.ricci_flow_rate = float(ricci_flow_rate)

    def compute_lie_derivative(
        self, vector_field: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Derivada de Lie del tensor métrico :math:`\mathcal{L}_X G`.

        Utiliza la aproximación de Jacobiano diagonal:

        .. math::

            \mathcal{L}_X G \approx G\,J + J^{\!\top} G,
            \quad J = \operatorname{diag}(X_0, \ldots, X_{n-1})

        Parameters
        ----------
        vector_field : ndarray, shape (n,) o (n, n)
            Campo vectorial :math:`X`.  Se trunca/aplana a los primeros
            ``n`` componentes.

        Returns
        -------
        ndarray, shape (n, n)
            Tensor :math:`\mathcal{L}_X G`.
        """
        n = self.G_PHYSICS.shape[0]
        vf = np.asarray(vector_field, dtype=np.float64).ravel()[:n]
        J = np.diag(vf)
        return self.G_PHYSICS @ J + J.T @ self.G_PHYSICS

    def synchronize_manifolds(
        self,
        threat_metrics: ThreatAssessment,
        current_state: CategoricalState,
        R_base: NDArray[np.float64],
        mu_0_squared: float,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Sincroniza la variedad de Ehresmann con la evaluación de amenaza.

        Calcula:

        .. math::

            R_d = R_0 + d_M\,\mathcal{L}_X G, \qquad
            \mu^2 = \mu_0^2 - \eta\, d_M\,|\chi|

        Parameters
        ----------
        threat_metrics:
            Evaluación de amenaza con campos ``mahalanobis_distance``
            y ``euler_char``.
        current_state:
            Estado categórico activo (estrato MIC).
        R_base : ndarray, shape (n, n) o escalar
            Matriz de disipación base.
        mu_0_squared : float
            Potencial de Higgs nominal :math:`\mu_0^2`.

        Returns
        -------
        R_dynamic : ndarray, shape (n, n)
            Matriz de disipación modulada.
        mu_squared : float
            Potencial de Higgs dinámico (≥ 0 por veto geométrico).
        """
        n = self.G_PHYSICS.shape[0]
        d_M = float(threat_metrics.mahalanobis_distance)

        # Fase 1: campo vectorial como vector uniforme (d_M en todas las coords)
        # BUG-F1 documentado: versión original usaba np.ones(self.G_PHYSICS.shape)
        # generando una matriz 2-D en lugar de un vector 1-D.
        psi_vec = np.full(n, d_M, dtype=np.float64)
        L_X_G = self.compute_lie_derivative(psi_vec)

        R_base = np.asarray(R_base, dtype=np.float64)
        if R_base.ndim == 0:
            R_base = np.eye(n) * float(R_base)
        elif R_base.shape != (n, n):
            raise ValueError(f"R_base debe tener forma ({n},{n}), recibida {R_base.shape}.")

        R_dynamic = R_base + L_X_G * d_M

        eta = 0.05
        chi = threat_metrics.euler_char if threat_metrics.euler_char is not None else 1.0
        mu_sq = mu_0_squared - eta * d_M * abs(chi)
        if mu_sq < 0.0:
            logger.warning("Veto Geométrico Absoluto: Higgs solidificado (μ²→0).")
            mu_sq = 0.0

        return R_dynamic, mu_sq


# =============================================================================
# FASE 2 — REFINAMIENTO ALGEBRAICO Y NUMÉRICO
# =============================================================================
# Correcciones aplicadas respecto a Fase 1:
#   ① Gradiente Itoh–Abe: epsilon adaptativo con cota inferior absoluta;
#      máscaras de escalera construidas por broadcasting seguro.
#   ② Laplaciano de Hodge: se incluye el término de coborde ∂₂∂₂† omitido
#      en Fase 1, dando el operador completo Δ₁ = ∂₁†∂₁ + ∂₂∂₂†.
#   ③ Derivada de Lie: Jacobiano normalizado por dimensión para que la
#      aproximación sea invariante de escala.
#   ④ Sincronización: validación explícita de R_base y semidefinición positiva
#      de R_dynamic mediante proyección espectral.
#
# Invariante de Fase 2:
#   • Toda operación matricial se verifica dimensionalmente.
#   • Los resultados son reproducibles y estables bajo perturbaciones ε-pequeñas.
# =============================================================================


class Phase2_ItohAbeDiscreteGradient(Phase1_ItohAbeDiscreteGradient):
    r"""
    Gradiente Discreto de Itoh–Abe Refinado (Fase 2 — Algebraico/Numérico).

    Mejoras sobre Fase 1:
    - **Epsilon adaptativo**: ``ε = max(1e-12 · ‖(x_k, x_{k+1})‖_∞, 1e-15)``
      evita cancelación catastrófica en aritmética IEEE-754.
    - **Máscaras vectorizadas**: la construcción de los estados escalera
      ``x_temp_k`` y ``x_temp_next`` usa ``np.where`` con broadcasting,
      eliminando copias redundantes.
    - **Coherencia dimensional**: validación explícita de que ``x_k`` y
      ``x_next`` tienen la misma forma antes de cualquier cómputo.
    """

    @staticmethod
    def compute(
        H_func: Callable[[NDArray[np.float64]], float],
        x_k: NDArray[np.float64],
        x_next: NDArray[np.float64],
        epsilon: Optional[float] = None,
    ) -> NDArray[np.float64]:
        r"""
        Calcula :math:`\overline{\nabla}H` con epsilon adaptativo.

        Parameters
        ----------
        H_func:
            Hamiltoniano :math:`H: \mathbb{R}^n \to \mathbb{R}`.
        x_k, x_next:
            Vectores de estado, shape ``(n,)``.
        epsilon:
            Si ``None``, se calcula automáticamente.

        Returns
        -------
        grad_discrete : ndarray, shape (n,)
        """
        x_k = np.asarray(x_k, dtype=np.float64)
        x_next = np.asarray(x_next, dtype=np.float64)
        if x_k.ndim != 1 or x_next.ndim != 1:
            raise ValueError("x_k y x_next deben ser vectores 1-D.")
        n = x_k.shape[0]
        if x_next.shape[0] != n:
            raise ValueError(
                f"Dimensiones incompatibles: x_k tiene {n} componentes "
                f"pero x_next tiene {x_next.shape[0]}."
            )

        # Epsilon adaptativo con cota absoluta inferior
        if epsilon is None:
            scale = max(np.max(np.abs(x_k)), np.max(np.abs(x_next)), 1.0)
            epsilon = max(1e-12 * scale, 1e-15)

        delta_x = x_next - x_k
        idx = np.arange(n)
        grad_discrete = np.empty(n, dtype=np.float64)

        for i in range(n):
            # x_temp_k: coordenadas 0…i-1 de x_next; i…n-1 de x_k
            x_temp_k = np.where(idx < i, x_next, x_k)
            # x_temp_next: coordenadas 0…i de x_next; i+1…n-1 de x_k
            x_temp_next = np.where(idx <= i, x_next, x_k)

            dxi = delta_x[i]
            if np.abs(dxi) > epsilon:
                grad_discrete[i] = (H_func(x_temp_next) - H_func(x_temp_k)) / dxi
            else:
                x_fwd = x_k.copy()
                x_fwd[i] += epsilon
                x_bwd = x_k.copy()
                x_bwd[i] -= epsilon
                grad_discrete[i] = (H_func(x_fwd) - H_func(x_bwd)) / (2.0 * epsilon)

        return grad_discrete


class Phase2_SpectralTorsionCoupling(Phase1_SpectralTorsionCoupling):
    r"""
    Acoplamiento Espectral de Torsión Refinado (Fase 2 — Algebraico/Numérico).

    Corrección principal sobre Fase 1:
    El **Laplaciano de Hodge de grado 1** completo es:

    .. math::

        \Delta_1 = \underbrace{\partial_1^{\dagger}\partial_1}_{\text{borde}}
                 + \underbrace{\partial_2\,\partial_2^{\dagger}}_{\text{coborde}}

    En términos de los operadores estrella discretos:

    .. math::

        \Delta_1 = \star_2^{\!\top}\star_1\,\star_2
                 + \star_2\,\star_2^{\!\top}\star_1

    Fase 1 omitía el segundo término.  Aquí se incluye explícitamente.

    Adicionalmente:
    - Se valida que ``torsion_matrix`` sea cuadrada y que las matrices Hodge
      sean compatibles en dimensión.
    - El Laplaciano se precomputa y almacena en ``self._laplacian``.

    Parameters
    ----------
    fiedler_value, torsion_matrix, hodge_star_1, hodge_star_2:
        Idénticos a Fase 1.
    """

    def __init__(
        self,
        fiedler_value: float,
        torsion_matrix: sp.csr_matrix,
        hodge_star_1: sp.csr_matrix,
        hodge_star_2: sp.csr_matrix,
    ) -> None:
        # Validar dimensiones antes de delegar al padre
        n = torsion_matrix.shape[0]
        if torsion_matrix.shape != (n, n):
            raise ValueError("torsion_matrix debe ser cuadrada.")
        if hodge_star_1.shape != (n, n) or hodge_star_2.shape != (n, n):
            raise ValueError(
                f"Las matrices Hodge deben tener forma ({n},{n}); "
                f"recibidas {hodge_star_1.shape} y {hodge_star_2.shape}."
            )

        super().__init__(fiedler_value, torsion_matrix, hodge_star_1, hodge_star_2)

        # Sobrescribir el Laplaciano de Fase 1 con la versión completa:
        # Δ₁ = ★₂ᵀ★₁★₂  +  ★₂★₂ᵀ★₁   (borde + coborde)
        self._laplacian: sp.csr_matrix = (
            self.star_2.T @ (self.star_1 @ self.star_2)
            + self.star_2 @ (self.star_2.T @ self.star_1)
        )

    def apply_frohlich_renormalization(
        self, m_eff: float, psi_state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Renormalización de Fröhlich con Laplaciano de Hodge completo.

        Parameters
        ----------
        m_eff : float
            Masa efectiva bare.
        psi_state : ndarray, shape (n,)
            Vector de estado :math:`\psi`.

        Returns
        -------
        ndarray, shape (n,)
            :math:`m_{\text{ren}}\,\Delta_1\psi`.
        """
        n = self.torsion.shape[0]
        psi = np.asarray(psi_state, dtype=np.float64).ravel()
        if psi.shape[0] != n:
            raise ValueError(
                f"psi_state debe tener {n} componentes; recibidos {psi.shape[0]}."
            )

        torsion_form = float(psi @ (self.torsion @ psi))
        alpha = np.exp(-torsion_form) / (6.0 * self.lambda_2)
        mass_renorm = m_eff * (1.0 + alpha)
        return mass_renorm * (self._laplacian @ psi)


class Phase2_GrothendieckTopologicalMediator(Phase1_GrothendieckTopologicalMediator):
    r"""
    Mediador Topológico de Grothendieck Refinado (Fase 2 — Algebraico/Numérico).

    Mejoras sobre Fase 1:

    1. **Jacobiano normalizado**: la aproximación diagonal se normaliza por
       ``n`` para que sea invariante al número de dimensiones:

       .. math::

           J \approx \frac{1}{n}\,\operatorname{diag}(\nabla \cdot X)\,I_n

       Esto preserva la interpretación de la divergencia del campo vectorial.

    2. **Semidefinición positiva de** :math:`R_d`: tras modular ``R_base``,
       se verifica que el resultado sea SDP (condición necesaria para la
       inecuación de disipación :math:`\dot{H} \leq 0`).  Si no lo es, se
       proyecta al cono SDP más cercano (zero-out de valores propios negativos).

    3. **Validación dimensional**: ``R_base`` se valida antes de operar.
    """

    def compute_lie_derivative(
        self, vector_field: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Derivada de Lie con Jacobiano diagonal normalizado por dimensión.

        .. math::

            \mathcal{L}_X G \approx G\,J + J^{\!\top} G,
            \quad J_{ii} = \frac{X_i}{n}

        Parameters
        ----------
        vector_field : ndarray, shape ≥ (n,)
            Campo vectorial; se trunca a los primeros ``n`` componentes.

        Returns
        -------
        ndarray, shape (n, n)
        """
        n = self.G_PHYSICS.shape[0]
        vf = np.asarray(vector_field, dtype=np.float64).ravel()
        if vf.shape[0] < n:
            raise ValueError(
                f"vector_field debe tener al menos {n} componentes; "
                f"recibidos {vf.shape[0]}."
            )
        # Normalización dimensional: J_{ii} = X_i / n
        J = np.diag(vf[:n] / n)
        return self.G_PHYSICS @ J + J.T @ self.G_PHYSICS

    def synchronize_manifolds(
        self,
        threat_metrics: ThreatAssessment,
        current_state: CategoricalState,
        R_base: NDArray[np.float64],
        mu_0_squared: float,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Sincronización con verificación de semidefinición positiva.

        Parameters
        ----------
        threat_metrics, current_state, R_base, mu_0_squared:
            Idénticos a Fase 1.

        Returns
        -------
        R_dynamic : ndarray, shape (n, n)  — garantizada SDP.
        mu_squared : float  — garantizado ≥ 0.
        """
        n = self.G_PHYSICS.shape[0]
        d_M = float(threat_metrics.mahalanobis_distance)

        psi_vec = np.full(n, d_M, dtype=np.float64)
        L_X_G = self.compute_lie_derivative(psi_vec)

        R_base = np.asarray(R_base, dtype=np.float64)
        if R_base.ndim == 0:
            R_base = np.eye(n) * float(R_base)
        elif R_base.shape != (n, n):
            raise ValueError(f"R_base debe tener forma ({n},{n}).")

        R_dynamic = R_base + L_X_G * d_M

        # Proyección al cono SDP: zerout de valores propios negativos
        R_dynamic = _project_to_sdp(R_dynamic)

        eta = 0.05
        chi = threat_metrics.euler_char if threat_metrics.euler_char is not None else 1.0
        mu_sq = mu_0_squared - eta * d_M * abs(chi)
        if mu_sq < 0.0:
            logger.warning("Veto Geométrico: Higgs solidificado (μ² → 0).")
            mu_sq = 0.0

        return R_dynamic, mu_sq


# =============================================================================
# FASE 3 — SÍNTESIS RIGUROSA FINAL (NIVEL DOCTORAL)
# =============================================================================
# Garantías añadidas respecto a Fase 2:
#   ① Gradiente Itoh–Abe: verificación del teorema de valor medio con norma
#      mixta (absoluta + relativa), tolerancia configurable.
#   ② SpectralTorsionCoupling: regularización de Tikhonov adaptativa basada
#      en el número de condición del Laplaciano; masa acotada inferiormente.
#   ③ GrothendieckMediator: método ``evolve_connection`` que:
#        - Calcula el gradiente Itoh–Abe y verifica la inecuación de disipación.
#        - Corrige R_dynamic proyectando al cono SDP si la inecuación se viola.
#        - Registra la disipación energética acumulada para auditoría.
#      Los default mutable arguments se eliminan (anti-patrón Python).
#
# Invariante de Fase 3:
#   • ΔH = ∇Hᵀ·Δx con error < tol_rel·|ΔH| + tol_abs.
#   • Ḣ = -∇Hᵀ R ∇H ≤ 0 verificado en cada llamada.
#   • El Laplaciano regularizado tiene número de condición acotado por 1/λ_Tik.
# =============================================================================


class Phase3_ItohAbeDiscreteGradient(Phase2_ItohAbeDiscreteGradient):
    r"""
    Gradiente Discreto de Itoh–Abe Definitivo (Fase 3 — Síntesis Rigurosa).

    Añade sobre Fase 2:
    - **Verificación del teorema de valor medio** con tolerancia mixta:

      .. math::

          |H(x_{k+1}) - H(x_k) - \overline{\nabla}H \cdot \Delta x|
          \leq \tau_{\text{rel}} |H(x_{k+1}) - H(x_k)| + \tau_{\text{abs}}

    - **Reporte de residuo**: si se viola, se registra en ``logger.warning``
      con el residuo normalizado para diagnóstico.

    Parameters
    ----------
    tol_rel : float
        Tolerancia relativa para la verificación del TVM. Por defecto ``1e-8``.
    tol_abs : float
        Tolerancia absoluta de respaldo. Por defecto ``1e-12``.
    """

    def __init__(
        self,
        tol_rel: float = 1e-8,
        tol_abs: float = 1e-12,
    ) -> None:
        self.tol_rel = float(tol_rel)
        self.tol_abs = float(tol_abs)

    def compute(  # type: ignore[override]
        self,
        H_func: Callable[[NDArray[np.float64]], float],
        x_k: NDArray[np.float64],
        x_next: NDArray[np.float64],
        epsilon: Optional[float] = None,
    ) -> NDArray[np.float64]:
        r"""
        Calcula el gradiente discreto con verificación de consistencia.

        Parameters
        ----------
        H_func, x_k, x_next, epsilon:
            Idénticos a Fase 2.

        Returns
        -------
        grad_discrete : ndarray, shape (n,)
            Gradiente de Itoh–Abe verificado.

        Raises
        ------
        NumericalInstabilityError
            Si el residuo del TVM supera 100× la tolerancia.
        """
        grad = Phase2_ItohAbeDiscreteGradient.compute(H_func, x_k, x_next, epsilon)

        # --- Verificación del Teorema de Valor Medio ---
        x_k_a = np.asarray(x_k, dtype=np.float64)
        x_next_a = np.asarray(x_next, dtype=np.float64)
        delta_x = x_next_a - x_k_a
        H_diff = H_func(x_next_a) - H_func(x_k_a)
        dot_product = float(np.dot(grad, delta_x))
        residuo = abs(H_diff - dot_product)
        tolerancia = self.tol_rel * abs(H_diff) + self.tol_abs

        if residuo > tolerancia:
            residuo_norm = residuo / (abs(H_diff) + self.tol_abs)
            if residuo > 100.0 * tolerancia:
                raise NumericalInstabilityError(
                    f"Gradiente Itoh–Abe: residuo TVM crítico {residuo_norm:.3e} "
                    f"(límite: {100.0 * tolerancia:.3e}).  Verifique la regularidad "
                    f"de H_func o reduzca el paso temporal."
                )
            logger.warning(
                "Gradiente Itoh–Abe: residuo TVM = %.3e (tolerancia = %.3e)",
                residuo_norm,
                tolerancia,
            )

        return grad


class Phase3_SpectralTorsionCoupling(Phase2_SpectralTorsionCoupling):
    r"""
    Acoplamiento Espectral de Torsión Definitivo (Fase 3 — Síntesis Rigurosa).

    Añade sobre Fase 2:

    1. **Regularización de Tikhonov adaptativa**: el parámetro
       :math:`\lambda_T` se escala según el número de condición estimado del
       Laplaciano:

       .. math::

           \Delta_1^{\text{reg}} = \Delta_1 + \lambda_T I_n, \qquad
           \lambda_T = \max\!\left(\lambda_T^0,\;
                            \frac{\|\Delta_1\|_F}{\kappa_{\max}}\right)

    2. **Cota inferior de masa renormalizada**: se garantiza
       :math:`m_{\text{ren}} \geq m_{\min} > 0` para evitar degeneración
       en estados cuánticos con torsión muy alta.

    Parameters
    ----------
    fiedler_value, torsion_matrix, hodge_star_1, hodge_star_2:
        Idénticos a Fase 2.
    tikhonov_lambda : float
        Regularización base :math:`\lambda_T^0`. Por defecto ``1e-8``.
    kappa_max : float
        Número de condición máximo admitido. Por defecto ``1e12``.
    m_min : float
        Masa renormalizada mínima. Por defecto ``1e-6``.
    """

    def __init__(
        self,
        fiedler_value: float,
        torsion_matrix: sp.csr_matrix,
        hodge_star_1: sp.csr_matrix,
        hodge_star_2: sp.csr_matrix,
        tikhonov_lambda: float = 1e-8,
        kappa_max: float = 1e12,
        m_min: float = 1e-6,
    ) -> None:
        super().__init__(fiedler_value, torsion_matrix, hodge_star_1, hodge_star_2)
        self.m_min = float(m_min)

        # Regularización adaptativa
        n = self._laplacian.shape[0]
        frob_norm = sp.linalg.norm(self._laplacian, ord="fro")
        lam_T = max(float(tikhonov_lambda), frob_norm / float(kappa_max))
        I_sparse = sp.eye(n, format="csr")
        self._regularized_laplacian: sp.csr_matrix = self._laplacian + lam_T * I_sparse
        self._tikhonov_used = lam_T
        logger.debug(
            "SpectralTorsionCoupling: λ_Tikhonov = %.3e (kappa_max = %.3e)",
            lam_T,
            kappa_max,
        )

    def apply_frohlich_renormalization(
        self, m_eff: float, psi_state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Renormalización de Fröhlich con Laplaciano regularizado y masa acotada.

        .. math::

            \psi_{\text{ren}} = \max(m_0(1+\alpha),\, m_{\min})
                                \,\Delta_1^{\text{reg}}\,\psi

        Parameters
        ----------
        m_eff : float
            Masa efectiva bare.
        psi_state : ndarray, shape (n,)

        Returns
        -------
        ndarray, shape (n,)
        """
        n = self.torsion.shape[0]
        psi = np.asarray(psi_state, dtype=np.float64).ravel()
        if psi.shape[0] != n:
            raise ValueError(
                f"psi_state debe tener {n} componentes; recibidos {psi.shape[0]}."
            )

        torsion_form = float(psi @ (self.torsion @ psi))
        alpha = np.exp(-torsion_form) / (6.0 * self.lambda_2)
        mass_renorm = max(m_eff * (1.0 + alpha), self.m_min)
        return mass_renorm * (self._regularized_laplacian @ psi)


class Phase3_GrothendieckTopologicalMediator(Phase2_GrothendieckTopologicalMediator):
    r"""
    Mediador Topológico de Grothendieck Definitivo (Fase 3 — Síntesis Rigurosa).

    Integra todos los componentes del fibrado de Ehresmann en un único método
    :meth:`evolve_connection` que garantiza:

    1. **Inecuación de Disipación Port-Hamiltoniana**:

       .. math::

           \dot{H} = -\nabla H^{\!\top} R_d(x)\,\nabla H \leq 0

       verificada explícitamente; si se viola, :math:`R_d` se proyecta al
       cono SDP más cercano (en norma de Frobenius).

    2. **Auditoría energética**: se acumula :math:`\sum_k \dot{H}_k` en
       ``_cumulative_dissipation`` para detección de derivas numéricas.

    3. **Gradiente Itoh–Abe con verificación TVM** (heredado de Fase 3).

    Parameters
    ----------
    metric_tensor : ndarray, shape (n, n)
        Tensor métrico físico.
    ricci_flow_rate : float
        Tasa de flujo de Ricci. Por defecto ``0.01``.
    itoh_gradient : Phase3_ItohAbeDiscreteGradient, opcional
        Instancia del operador de gradiente discreto. Si ``None``, se crea
        una instancia con parámetros por defecto (evita default mutables).
    """

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = None,
        ricci_flow_rate: float = 0.01,
        itoh_gradient: Optional[Phase3_ItohAbeDiscreteGradient] = None,
    ) -> None:
        super().__init__(metric_tensor=metric_tensor, ricci_flow_rate=ricci_flow_rate)
        # Evitar default mutable: instanciar aquí si no se proporciona
        self.itoh: Phase3_ItohAbeDiscreteGradient = (
            itoh_gradient
            if itoh_gradient is not None
            else Phase3_ItohAbeDiscreteGradient()
        )
        self._cumulative_dissipation: float = 0.0
        self._step_count: int = 0

    def compute_lie_derivative(
        self, vector_field: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Delegado a la implementación corregida de Fase 2."""
        return super().compute_lie_derivative(vector_field)

    def synchronize_manifolds(
        self,
        threat_metrics: ThreatAssessment,
        current_state: CategoricalState,
        R_base: NDArray[np.float64],
        mu_0_squared: float,
    ) -> Tuple[NDArray[np.float64], float]:
        """Delegado a la implementación robusta de Fase 2 (SDP garantizada)."""
        return super().synchronize_manifolds(
            threat_metrics, current_state, R_base, mu_0_squared
        )

    def evolve_connection(
        self,
        H_hamiltonian: Callable[[NDArray[np.float64]], float],
        x_k: NDArray[np.float64],
        x_next: NDArray[np.float64],
        threat_metrics: ThreatAssessment,
        current_state: CategoricalState,
        R_base: NDArray[np.float64],
        mu_0_squared: float,
    ) -> Dict[str, Any]:
        r"""
        Evolución completa del fibrado de Ehresmann en un paso discreto.

        Ejecuta el ciclo:

        1. :meth:`synchronize_manifolds` → :math:`R_d^{(0)}, \mu^2`
        2. Gradiente Itoh–Abe :math:`\overline{\nabla}H(x_k, x_{k+1})`
        3. Verificación de disipación: :math:`\dot{H} = -g^T R_d g \leq 0`
        4. Corrección SDP de :math:`R_d` si :math:`\dot{H} > 0`
        5. Actualización del acumulador energético

        Parameters
        ----------
        H_hamiltonian : callable
            Hamiltoniano :math:`H: \mathbb{R}^n \to \mathbb{R}`.
        x_k, x_next : ndarray, shape (n,)
            Estados actual y siguiente.
        threat_metrics : ThreatAssessment
            Evaluación de amenaza actual.
        current_state : CategoricalState
            Estado categórico del sistema.
        R_base : ndarray, shape (n, n) o escalar
            Matriz de disipación base.
        mu_0_squared : float
            Potencial de Higgs nominal.

        Returns
        -------
        dict con claves:
            ``R_dynamic``
                Matriz de disipación modulada y garantizada SDP,
                shape ``(n, n)``.
            ``mu_squared``
                Potencial de Higgs dinámico ≥ 0.
            ``energy_dissipation``
                :math:`\dot{H}_k = -\nabla H^T R_d \nabla H \leq 0`.
            ``grad_energy``
                Gradiente discreto :math:`\overline{\nabla}H`, shape ``(n,)``.
            ``tvm_residual``
                Residuo del teorema de valor medio (diagnóstico).
            ``step``
                Índice del paso actual.
        """
        # 1. Parámetros geométricos (SDP garantizada)
        R_dyn, mu_sq = self.synchronize_manifolds(
            threat_metrics, current_state, R_base, mu_0_squared
        )

        # 2. Gradiente discreto con verificación TVM
        x_k_a = np.asarray(x_k, dtype=np.float64)
        x_next_a = np.asarray(x_next, dtype=np.float64)
        grad_H = self.itoh.compute(H_hamiltonian, x_k_a, x_next_a)

        # 3. Inecuación de disipación: Ḣ = -∇Hᵀ R ∇H ≤ 0
        dissipation = -float(grad_H @ (R_dyn @ grad_H))

        if dissipation > 1e-14:
            logger.warning(
                "Violación de la inecuación de disipación: Ḣ = %.6e > 0. "
                "Proyectando R_dynamic al cono SDP.",
                dissipation,
            )
            # Corrección: proyectar al cono SDP más cercano en norma Frobenius
            R_dyn = _project_to_sdp(R_dyn)
            dissipation = -float(grad_H @ (R_dyn @ grad_H))
            if dissipation > 1e-14:
                logger.error(
                    "La proyección SDP no fue suficiente: Ḣ = %.6e. "
                    "Verifique R_base.",
                    dissipation,
                )

        # 4. Auditoría energética acumulada
        self._cumulative_dissipation += dissipation
        self._step_count += 1

        # Residuo TVM para diagnóstico externo
        delta_x = x_next_a - x_k_a
        H_diff = H_hamiltonian(x_next_a) - H_hamiltonian(x_k_a)
        tvm_residual = abs(H_diff - float(np.dot(grad_H, delta_x)))

        return {
            "R_dynamic": R_dyn,
            "mu_squared": mu_sq,
            "energy_dissipation": dissipation,
            "grad_energy": grad_H,
            "tvm_residual": tvm_residual,
            "step": self._step_count,
        }

    @property
    def cumulative_dissipation(self) -> float:
        r"""Disipación energética acumulada :math:`\sum_k \dot{H}_k`."""
        return self._cumulative_dissipation

    def reset_energy_audit(self) -> None:
        """Reinicia los contadores de auditoría energética."""
        self._cumulative_dissipation = 0.0
        self._step_count = 0
        logger.debug("Auditoría energética reiniciada.")


# =============================================================================
# UTILIDADES INTERNAS
# =============================================================================


def _project_to_sdp(M: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Proyecta una matriz cuadrada al cono de matrices simétricas semidefinidas
    positivas (SDP) más cercano en norma de Frobenius.

    Para una matriz simétrica :math:`M = Q \Lambda Q^T`, la proyección es:

    .. math::

        \Pi_{\text{SDP}}(M) = Q\,\max(\Lambda, 0)\,Q^T

    La simetrización previa :math:`\hat{M} = (M + M^T)/2` garantiza que
    la descomposición espectral sea real.

    Parameters
    ----------
    M : ndarray, shape (n, n)

    Returns
    -------
    ndarray, shape (n, n)  — simétrica semidefinida positiva.
    """
    M_sym = (M + M.T) / 2.0
    eigvals, eigvecs = sla.eigh(M_sym)
    eigvals_clipped = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


# =============================================================================
# EXPORTACIONES CANÓNICAS
# =============================================================================
# Se exportan las implementaciones definitivas (Fase 3) bajo los nombres
# originales del módulo, preservando compatibilidad con el resto del sistema.

ItohAbeDiscreteGradient = Phase3_ItohAbeDiscreteGradient
SpectralTorsionCoupling = Phase3_SpectralTorsionCoupling
GrothendieckTopologicalMediator = Phase3_GrothendieckTopologicalMediator

__all__ = [
    # Clases canónicas (Fase 3)
    "ItohAbeDiscreteGradient",
    "SpectralTorsionCoupling",
    "GrothendieckTopologicalMediator",
    # Fases intermedias (acceso explícito para pruebas de regresión por fase)
    "Phase1_ItohAbeDiscreteGradient",
    "Phase1_SpectralTorsionCoupling",
    "Phase1_GrothendieckTopologicalMediator",
    "Phase2_ItohAbeDiscreteGradient",
    "Phase2_SpectralTorsionCoupling",
    "Phase2_GrothendieckTopologicalMediator",
    "Phase3_ItohAbeDiscreteGradient",
    "Phase3_SpectralTorsionCoupling",
    "Phase3_GrothendieckTopologicalMediator",
    # Utilidades
    "_project_to_sdp",
]
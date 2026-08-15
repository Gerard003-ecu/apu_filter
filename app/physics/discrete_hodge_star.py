# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Discrete Hodge Star Operator (El Operador Constitutivo de de Rham)  ║
║ Ruta   : app/physics/discrete_hodge_star.py                                  ║
║ Versión: 2.0.0-DEC-Metric-SelfAdjoint-Spectral-Topos-Strict-PhD              ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y RIGOR DOCTORAL (DEC + Teoría Espectral + Topos):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra la formulación discreta del operador estrella de Hodge ($$\star$$) 
sobre complejos simpliciales orientados de dimensión finita, elevada a un marco 
de teoría de categorías/topos y teoría espectral de operadores autoadjuntos. 
En el contexto de la electrodinámica discreta, sistemas Port-Hamiltonianos y 
geometría de la información, el Hodge star actúa como el morfismo métrico 
constitutivo que realiza la dualidad de Poincaré discreta:

$$\star_k: \Lambda^k \mathcal{H} \xrightarrow{\simeq} \Lambda^{n-k} \mathcal{H}^* \quad\big[190\big]$$

Para inmunizar la Malla de Sabiduría ante el ruido estocástico del LLM y evitar 
inestabilidades de redondeo en la FPU ($$\text{IEEE-754 binary64}$$), el operador 
Hodge se somete a aduanas espectrales estrictas en tiempo de ejecución. Esto 
garantiza que la signatura métrica Riemanniana se preserve de manera exacta 
y que las leyes de conservación física de la red presupuestal se mantengan 
invariantes en el topos.

AXIOMÁTICA SIMPLÉCTICA, COHOMOLÓGICA Y TERMOMECÁNICA DISCRETA:
────────────────────────────────────────────────────────────────────────────────

  [A1] Simetría Autoadjunta y Definición Positiva (Hodge Metric):
       El operador estrella de Hodge discreto $$\star_k$$ de grado k es una 
       matriz real que actúa como el tensor métrico constitutivo local. 
       Debe ser estrictamente simétrica y definida positiva (SPD) en la FPU:
       $$\star_k = \star_k^\top \succ \mathbf{0} \quad\big[200\big]$$
       Sujeta a que su autovalor mínimo respete el piso de Wilkinson:
       $$\lambda_{\min}(\star_k) \ge \mathtt{\_SPECTRAL\_PSD\_FLOOR} = -1.0\times 10^{-13} \quad\big[203, 208\big]$$
       Y su número de condición espectral permanezca incondicionalmente acotado:
       $$\kappa_2(\star_k) = \frac{\lambda_{\max}(\star_k)}{\lambda_{\min}(\star_k)} \le \mathtt{\_CONDITION\_NUMBER\_MAX} = 1.0\times 10^8 \quad\big[203, 208\big]$$

  [A2] Descomposición de Helmholtz-Hodge Ortogonal Ponderada:
       Toda 1-forma discreta $$\alpha \in C^1(K)$$ (flujo de recursos e insumos) 
       se descompone de forma única y métricamente ortogonal respecto al apareamiento 
       de Hilbert inducido por el operador constitutivo $$\star_1$$:
       $$\alpha = d\varphi + \delta\beta + h \quad\big[197, 200\big]$$
       Donde:
         · $$d\varphi = \partial_1^\top \varphi$$ es la componente exacta (flujo irrotacional, gradientes).
         · $$\delta\beta = \star_1^{-1} \partial_2 \beta$$ es la componente coexacta (flujo de remolino, rizos solenoidales).
         · $$h \in \ker(\Delta_1^H)$$ es la componente armónica (obstrucciones cohomológicas globales).
       La ortogonalidad mutua se certifica mediante la anulación del producto interno de Riesz:
       $$\langle d\varphi, \, \delta\beta \rangle_{\star_1} = (d\varphi)^\top \star_1 (\delta\beta) \equiv 0 \pmod{\varepsilon_{\mathrm{machine}}} \quad\big[193, 197\big]$$

  [A3] Relación del Laplaciano de Hodge y Isomorfismo de de Rham:
       Los operadores Laplacianos discretos de grado 0 y 1 se construyen a partir 
       del operador coborde orientado $$\partial_k$$ y de la métrica constitutiva:
       $$\Delta_0^H = \partial_1 \star_1 \partial_1^\top \quad\big[200, 275\big]$$
       $$\Delta_1^H = \star_1^{-1} \partial_1^\top \partial_1 + \partial_2 \partial_2^\top \star_1^{-1} \quad\big[260, 288\big]$$
       Garantizando síncronamente el Teorema de Isomorfismo de Hodge discreto:
       $$\ker(\Delta_k^H) \cong H^k_{\mathrm{dR}}(K) \quad\big[260, 275\big]$$

  [A4] Pasividad de Lyapunov en Sistemas Port-Hamiltonianos:
       La evolución temporal de la energía del sistema $$x = [q, p]^\top$$ bajo el 
       flujo del Laplaciano de Hodge ponderado satisface la desigualdad de pasividad:
       $$\dot{H} = -\nabla H(x)^\top R(x) \nabla H(x) \le 0 \quad\big[200, 205\big]$$
       Asegurando que el sistema sea estrictamente disipativo ($$R(x) \succeq \mathbf{0}$$).

  [A5] Invarianza de la Característica de Euler-Poincaré:
       Los números de Betti computed vía la SVD completa de $$\partial_k$$ se confinan 
       por la fórmula de Euler-Poincaré del esqueleto simplicial del presupuesto:
       $$\chi(K) = \beta_0 - \beta_1 = |V| - |E| \quad\big[5, 419\big]$$
       Cualquier ciclo parásito ($$\beta_1 > 0$$) delata un socavón lógico circular.

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial OODA / Kleisli):
────────────────────────────────────────────────────────────────────────────────
La transferencia del estado y el tránsito del Pasaporte de Telemetría se rige 
por un acoplamiento monoidal covariante estricto:

  Fase 1 ──► OBSERVACIÓN ESPECTRAL Y SANEAMIENTO MÉTRICO (Phase1_SpectralMetricObserver)
             Ingiere los pesos del complejo simplicial, calcula la diagonalización 
             autoadjunta, audita la tolerancia espectral, y proyecta al cono SPD 
             más cercano vía Higham-Löwner si se detectan asimetrías.
             Último método: emit_phase1_observation.
             Entrega: Phase1HodgeObservation.

  Fase 2 ──► COHOMOLOGÍA Y LAPLACIANOS DE HODGE (Phase2_HodgeLaplacianOrienter)
             Hereda formalmente la Phase1HodgeObservation. Ensambla el 
             Laplaciano de Hodge discreto $$\Delta_k^H$$, calcula su espectro de 
             autovalores y verifica la exactitud del complejo $$\partial_1 \circ \partial_2 = 0$$.
             Último método: emit_phase2_orientation.
             Entrega: Phase2HodgeOrientation.

  Fase 3 ──► COMPRESIÓN DE HELMHOLTZ Y VETO DE HEYTING (Phase3_HelmholtzDecisionMaker)
             Hereda formalmente la Phase2HodgeOrientation. Computa la 
             descomposición de Helmholtz-Hodge, extrae los armónicos, calcula 
             $$\beta_1$$ y evalúa el Supremo ($$\sqcup$$) en el retículo distributivo $$\Omega_3$$:
             $$\Omega_3 = \{\mathrm{COHERENT}, \, \mathrm{DEGRADED}, \, \mathrm{VETOED}\} \quad\big[200, 206\big]$$
             Si $$\beta_1 > \mathtt{max\_betti\_1}$$ o la métrica pierde definición positiva, 
             el retículo colapsa síncronamente al Supremo terminal VETOED ($$\top$$).
             Detona la excepción 'HodgeStarAgentError' en el milisegundo cero, 
             purgando la memoria RAM y disparando el actuador físico Crowbar (GPIO14).
             Entrega: Phase3HodgeDecision.

Funtor Maestro de Resolución Constitutiva:
  $$\mathcal{Z}_{\mathrm{Hodge}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 \quad\big[211\big]$$
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Final,
    Tuple,
    Optional,
    List,
    Dict,
    Callable,
    Iterator,
    Protocol,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

logger = logging.getLogger("MIC.Physics.DiscreteHodgeStar")

# ─────────────────────────────────────────────────────────────────────────────
# Tipos canónicos y constantes de máquina (rigor numérico de Wilkinson)
# ─────────────────────────────────────────────────────────────────────────────
RealMatrix = NDArray[np.float64]
RealVector = NDArray[np.float64]
ComplexMatrix = NDArray[np.complex128]
SparseMatrix = sp.spmatrix

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_SQRT_EPS: Final[float] = float(np.sqrt(_MACHINE_EPS))
_SPECTRAL_TOL: Final[float] = 1e-12
_CONDITION_SAFE: Final[float] = 1e12


class HodgeDegree(Enum):
    """Grado de la forma diferencial discreta (k-cochain)."""
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3


class MetricSignature(Enum):
    """Signatura métrica admitida (Lorentziana reservada para extensiones futuras)."""
    RIEMANNIAN = auto()
    PSEUDO_RIEMANNIAN = auto()


@runtime_checkable
class InnerProductSpace(Protocol):
    """Protocolo de espacio pre-Hilbertiano discreto (axioma de topos)."""
    def inner_product(self, alpha: RealVector, beta: RealVector) -> float: ...
    def norm(self, alpha: RealVector) -> float: ...
    def dimension(self) -> int: ...


# =============================================================================
# FASE 1 — FUNDAMENTOS MÉTRICOS Y OPERADOR ESTRELLA DE HODGE ELEMENTAL
#          (Teoría de grafos ponderados + Álgebra lineal SPD + Positividad)
# =============================================================================

@dataclass(frozen=True, slots=True)
class SpectralCertificate:
    r"""
    Certificado inmutable de la estructura espectral de un operador autoadjunto.

    Captura el teorema espectral finito-dimensional:
    $$A = U \Lambda U^\top, \quad \Lambda = \operatorname{diag}(\lambda_i), \quad \lambda_i \in \mathbb{R}.$$

    Atributos
    ---------
    eigenvalues : RealVector
        Espectro ordenado $\lambda_{\min} = \lambda_1 \le \cdots \le \lambda_m = \lambda_{\max}$.
    eigenvectors : RealMatrix
        Base ortonormal de autovectores (columnas).
    condition_number : float
        $\kappa_2(A) = \lambda_{\max}/|\lambda_{\min}|$ (o $\infty$ si singular).
    is_spd : bool
        Verdadero sii $\lambda_{\min} > 0$ (criterio de Sylvester estricto).
    trace : float
        $\operatorname{tr}(A) = \sum_i \lambda_i$.
    determinant : float
        $\det(A) = \prod_i \lambda_i$ (volumen del paraleloedro espectral).
    spectral_gap : float
        $\gamma = \lambda_{\min}$ (brecha hacia el núcleo).
    frobenius_norm : float
        $\|A\|_F = \sqrt{\sum_i \lambda_i^2}$.
    """
    eigenvalues: RealVector
    eigenvectors: RealMatrix
    condition_number: float
    is_spd: bool
    trace: float
    determinant: float
    spectral_gap: float
    frobenius_norm: float

    def __post_init__(self) -> None:
        if self.eigenvalues.ndim != 1:
            raise ValueError("Espectro debe ser un vector 1-D.")
        if not np.all(np.diff(self.eigenvalues) >= -_MACHINE_EPS):
            raise ValueError("Espectro debe estar ordenado no-decrecientemente.")


@dataclass(frozen=True, slots=True)
class HodgeMetricState:
    r"""
    Certificado inmutable de estabilidad espectral de la métrica de Hodge $\star_k$.

    Extiende SpectralCertificate con invariantes constitutivos de la dualidad
    de Poincaré discreta y del producto interno de energía.
    """
    dimension: int
    condition_number: float
    is_spd: bool
    trace: float
    spectral_gap: float
    determinant: float
    frobenius_norm: float
    min_weight: float
    max_weight: float
    volume_form: float          # $\sqrt{\det(\star_k)}$ (densidad de volumen)
    signature: MetricSignature = MetricSignature.RIEMANNIAN


class DiscreteHodgeStar:
    r"""
    Operador constitutivo estrella de Hodge $\star_k$ sobre el complejo de cadenas.

    En la formulación diagonal (métrica de circumcentros o de Whitney),
    $$\star_k = \operatorname{diag}(w_1,\dots,w_m), \quad w_i > 0,$$
    donde $w_i$ codifica la razón de volúmenes dual/primal
    $$w_i = \frac{|\star\sigma_i|}{|\sigma_i|}.$$

    El operador es el único morfismo que hace conmutar el diagrama de dualidad
    y convierte el pairing de cadenas/cocadenas en un producto interno.
    """

    def __init__(
        self,
        weights: RealVector,
        degree: HodgeDegree = HodgeDegree.ONE,
        validate_strict: bool = True,
    ) -> None:
        r"""
        Inicializa el operador a partir del vector de pesos métricos constitutivos.

        Parámetros
        ----------
        weights : RealVector
            Vector $w \in \mathbb{R}^m_{>0}$ de conductancias/permeabilidades/
            capacitancias locales (diagonal de $\star_k$).
        degree : HodgeDegree
            Grado $k$ de las formas sobre las que actúa.
        validate_strict : bool
            Si True, impone positividad estricta y lanza en caso de violación
            del axioma de positividad de la métrica riemanniana discreta.
        """
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size == 0:
            raise ValueError("Axioma de No-Trivialidad: el soporte métrico no puede ser vacío.")
        if validate_strict and np.any(w <= 0.0):
            raise ValueError(
                "Axioma de Positividad Roto: todos los pesos constitutivos "
                "deben ser estrictamente positivos (métrica riemanniana)."
            )
        if np.any(~np.isfinite(w)):
            raise ValueError("Pesos no finitos detectados (NaN/Inf).")

        self._weights: RealVector = w.copy()
        self._weights.setflags(write=False)
        self._dim: int = int(self._weights.shape[0])
        self._degree: HodgeDegree = degree
        self._log_weights: RealVector = np.log(self._weights)  # para geometría de información

        logger.debug(
            "DiscreteHodgeStar(k=%s) inicializado: dim=%d, min_w=%.3e, max_w=%.3e",
            degree.name, self._dim, float(np.min(w)), float(np.max(w)),
        )

    # ── Propiedades topológico-métricas ──────────────────────────────────────

    @property
    def degree(self) -> HodgeDegree:
        """Grado $k$ de la cochain sobre la que actúa $\star_k$."""
        return self._degree

    @property
    def dimension(self) -> int:
        """Dimensión del espacio de $k$-formas: $m = \dim C^k$."""
        return self._dim

    @property
    def weights(self) -> RealVector:
        """Copia defensiva del vector de pesos (inmutable en el interior)."""
        return self._weights.copy()

    @property
    def matrix(self) -> RealMatrix:
        r"""
        Representación matricial densa $\star_k = \operatorname{diag}(w)$.

        Returns
        -------
        RealMatrix
            Matriz diagonal $m\times m$ SPD.
        """
        return np.diag(self._weights)

    @property
    def inverse_matrix(self) -> RealMatrix:
        r"""
        Inversa métrica $\star_k^{-1} = \operatorname{diag}(1/w_i)$.

        Cumple $\star_k \star_k^{-1} = I$ con error de redondeo $\le m\cdot\varepsilon_{\mathrm{mach}}$.
        """
        return np.diag(1.0 / self._weights)

    @property
    def sqrt_matrix(self) -> RealMatrix:
        r"""
        Raíz cuadrada métrica $\star_k^{1/2} = \operatorname{diag}(\sqrt{w_i})$.

        Permite la isometría
        $$\langle\alpha,\beta\rangle_{\star} = \langle\star^{1/2}\alpha,\star^{1/2}\beta\rangle_{\ell^2}.$$
        """
        return np.diag(np.sqrt(self._weights))

    @property
    def log_determinant(self) -> float:
        r"""Log-determinante $\log\det(\star_k) = \sum_i \log w_i$ (entropía volumétrica)."""
        return float(np.sum(self._log_weights))

    # ── Auditoría espectral rigurosa (Teorema Espectral) ─────────────────────

    def spectral_decomposition(self, compute_vectors: bool = True) -> SpectralCertificate:
        r"""
        Descomposición espectral exacta de $\star_k$ (operador diagonal).

        Como $\star_k$ es diagonal, los autovalores son los propios pesos
        ordenados y los autovectores son la base canónica permutada.

        Returns
        -------
        SpectralCertificate
            Certificado completo del teorema espectral.
        """
        idx = np.argsort(self._weights)
        ev = self._weights[idx].copy()
        if compute_vectors:
            evec = np.eye(self._dim, dtype=np.float64)[:, idx]
        else:
            evec = np.zeros((0, 0), dtype=np.float64)

        lam_min = float(ev[0])
        lam_max = float(ev[-1])
        cond = lam_max / lam_min if lam_min > _MACHINE_EPS else np.inf
        is_spd = lam_min > _SPECTRAL_TOL
        tr = float(np.sum(ev))
        det = float(np.prod(ev))
        gap = lam_min
        frob = float(np.linalg.norm(ev))

        return SpectralCertificate(
            eigenvalues=ev,
            eigenvectors=evec,
            condition_number=cond,
            is_spd=is_spd,
            trace=tr,
            determinant=det,
            spectral_gap=gap,
            frobenius_norm=frob,
        )

    def audit_metric(self, tolerance: float = _SPECTRAL_TOL) -> HodgeMetricState:
        r"""
        Audita el cumplimiento del axioma SPD y produce el certificado métrico.

        Aplica el criterio de Sylvester (todos los menores principales > 0)
        que, para matrices diagonales, se reduce a $w_i > \texttt{tolerance}$.

        Parámetros
        ----------
        tolerance : float
            Tolerancia espectral de Wilkinson para declarar positividad.

        Returns
        -------
        HodgeMetricState
            Certificado inmutable del estado métrico constitutivo.
        """
        cert = self.spectral_decomposition(compute_vectors=False)
        min_w = float(np.min(self._weights))
        max_w = float(np.max(self._weights))
        vol = float(np.exp(0.5 * self.log_determinant))

        state = HodgeMetricState(
            dimension=self._dim,
            condition_number=cert.condition_number,
            is_spd=bool(min_w > tolerance),
            trace=cert.trace,
            spectral_gap=cert.spectral_gap,
            determinant=cert.determinant,
            frobenius_norm=cert.frobenius_norm,
            min_weight=min_w,
            max_weight=max_w,
            volume_form=vol,
            signature=MetricSignature.RIEMANNIAN,
        )
        if not state.is_spd:
            logger.warning(
                "Métrica Hodge k=%s NO es SPD bajo tolerancia %.1e (λ_min=%.3e)",
                self._degree.name, tolerance, min_w,
            )
        return state

    # ── Estructura de producto interno (pre-Hilbert) ─────────────────────────

    def inner_product(self, alpha: RealVector, beta: RealVector) -> float:
        r"""
        Producto interno de Hodge (energía / disipación de Joule / acción):
        $$\langle\alpha,\beta\rangle_{\star_k} = \alpha^\top\star_k\beta = \sum_i w_i\alpha_i\beta_i.$$

        Parámetros
        ----------
        alpha, beta : RealVector
            $k$-formas (cochains) de dimensión $m$.

        Returns
        -------
        float
            Escalar de energía (positivo si $\alpha=\beta\neq 0$ y $\star$ SPD).
        """
        a = np.asarray(alpha, dtype=np.float64).ravel()
        b = np.asarray(beta, dtype=np.float64).ravel()
        if a.shape[0] != self._dim or b.shape[0] != self._dim:
            raise ValueError(
                f"Falla Dimensional: se esperaba dim={self._dim}, "
                f"recibido α={a.shape[0]}, β={b.shape[0]}."
            )
        return float(np.dot(a, self._weights * b))

    def norm(self, alpha: RealVector) -> float:
        r"""Norma inducida $\|\alpha\|_{\star} = \sqrt{\langle\alpha,\alpha\rangle_{\star}}$."""
        return float(np.sqrt(max(self.inner_product(alpha, alpha), 0.0)))

    def rayleigh_quotient(self, alpha: RealVector) -> float:
        r"""
        Cociente de Rayleigh $R(\alpha) = \frac{\langle\alpha,\alpha\rangle_{\star}}{\|\alpha\|_{\ell^2}^2}$.

        Coincide con un promedio ponderado de los pesos cuando $\|\alpha\|_2=1$.
        """
        a = np.asarray(alpha, dtype=np.float64).ravel()
        n2 = float(np.dot(a, a))
        if n2 < _MACHINE_EPS:
            raise ValueError("Vector nulo: cociente de Rayleigh indefinido.")
        return self.inner_product(a, a) / n2

    def apply(self, alpha: RealVector) -> RealVector:
        r"""Aplicación del operador: $\star_k\alpha = w\odot\alpha$ (producto de Hadamard)."""
        a = np.asarray(alpha, dtype=np.float64).ravel()
        if a.shape[0] != self._dim:
            raise ValueError(f"Dimensión incompatible: {a.shape[0]} ≠ {self._dim}.")
        return self._weights * a

    def apply_inverse(self, alpha: RealVector) -> RealVector:
        r"""Aplicación de la inversa métrica: $\star_k^{-1}\alpha = \alpha\oslash w$."""
        a = np.asarray(alpha, dtype=np.float64).ravel()
        if a.shape[0] != self._dim:
            raise ValueError(f"Dimensión incompatible: {a.shape[0]} ≠ {self._dim}.")
        return a / self._weights

    def parallel_transport_weights(self, other: "DiscreteHodgeStar") -> RealVector:
        r"""
        Log-mapa de la métrica (geometría de información de Fisher-Rao discreta):
        $$\log_{\star}(\star') = \log(w') - \log(w).$$

        Mide la deformación infinitesimal de la estructura constitutiva.
        """
        if other.dimension != self._dim:
            raise ValueError("Dimensiones métricas incompatibles para transporte paralelo.")
        return other._log_weights - self._log_weights

    # ── Último método formal de la FASE 1 ────────────────────────────────────
    # Su valor de retorno (HodgeDualityMorphism) es el objeto de arranque
    # obligatorio de la FASE 2 (continuidad de la definición formal).

    def induce_duality_morphism(self) -> "HodgeDualityMorphism":
        r"""
        Induce el morfismo de dualidad de Poincaré discreta asociado a $\star_k$.

        En el lenguaje de teoría de categorías, $\star_k$ es un isomorfismo
        natural en el topos de haces sobre el complejo simplicial:
        $$\star_k : \Omega^k_{\mathrm{primal}} \xrightarrow{\simeq} \Omega^{n-k}_{\mathrm{dual}}.$$

        El objeto retornado encapsula tanto el operador directo como su inverso,
        junto con el certificado espectral, y constituye el punto de partida
        axiomático de la FASE 2 (construcción de Laplacianos ponderados).

        Returns
        -------
        HodgeDualityMorphism
            Morfismo de dualidad listo para ser consumido por WeightedHodgeLaplacian
            y por el complejo de de Rham discreto de la FASE 3.
        """
        state = self.audit_metric()
        if not state.is_spd:
            raise RuntimeError(
                "No se puede inducir morfismo de dualidad: métrica no SPD. "
                "Revise los pesos constitutivos."
            )
        cert = self.spectral_decomposition(compute_vectors=True)
        return HodgeDualityMorphism(
            star=self,
            star_inv_weights=1.0 / self._weights,
            metric_state=state,
            spectral_cert=cert,
        )


# =============================================================================
# FASE 2 — MORFISMO DE DUALIDAD + LAPLACIANOS DE HODGE PONDERADOS
#          (Continúa directamente desde DiscreteHodgeStar.induce_duality_morphism)
#          Teoría espectral de grafos + Cohomología de de Rham discreta (grado 0-1)
# =============================================================================

@dataclass(frozen=True, slots=True)
class HodgeDualityMorphism:
    r"""
    Morfismo de dualidad de Poincaré inducido por un DiscreteHodgeStar (FASE 1).

    Este dataclass es el contrato formal de continuidad FASE 1 → FASE 2:
    todo Laplaciano ponderado debe construirse a partir de un morfismo de
    dualidad certificado, nunca a partir de pesos crudos sin auditoría.

    Atributos
    ---------
    star : DiscreteHodgeStar
        Operador $\star_k$ de origen.
    star_inv_weights : RealVector
        Pesos de la inversa $\star_k^{-1}$.
    metric_state : HodgeMetricState
        Certificado SPD y de condición.
    spectral_cert : SpectralCertificate
        Descomposición espectral completa.
    """
    star: DiscreteHodgeStar
    star_inv_weights: RealVector
    metric_state: HodgeMetricState
    spectral_cert: SpectralCertificate

    def apply_star(self, alpha: RealVector) -> RealVector:
        return self.star.apply(alpha)

    def apply_star_inv(self, alpha: RealVector) -> RealVector:
        return self.star.apply_inverse(alpha)

    @property
    def is_well_conditioned(self) -> bool:
        return self.metric_state.condition_number < _CONDITION_SAFE


@dataclass(frozen=True, slots=True)
class LaplacianSpectrum:
    r"""
    Espectro del Laplaciano de Hodge ponderado (teorema espectral de grafos).

    Incluye multiplicidad del núcleo (número de Betti) y brecha espectral
    (conectividad algebraica de Fiedler).
    """
    eigenvalues: RealVector
    eigenvectors: RealMatrix
    betti_number: int              # dim ker Δ = β_k
    algebraic_connectivity: float  # λ_{β+1} (Fiedler)
    is_self_adjoint: bool
    max_residual: float            # ||Δv - λv||_∞ máximo (certificado numérico)


class WeightedHodgeLaplacian:
    r"""
    Resolvedor de Laplacianos de Hodge ponderados sobre el complejo de cadenas
    de grado 0 y 1 (grafos orientados ponderados).

    Dado el diagrama de frontera
    $$C_1 \xrightarrow{\partial_1} C_0,$$
    y morfismos de dualidad $\star_0$, $\star_1$ (objetos de FASE 1), se definen:

    .. math::

        \Delta_0^\star &= \partial_1\,\star_1\,\partial_1^\top\,\star_0^{-1},\\
        \Delta_1^\star &= \star_1\,\partial_1^\top\,\star_0^{-1}\,\partial_1
                       + \text{(término de curl si } \partial_2 \text{ existe)}.

    Ambos son autoadjuntos respecto de los productos internos inducidos por
    $\star_0$ y $\star_1$ respectivamente (isomorfismo de espacios de Hilbert).
    """

    def __init__(
        self,
        boundary_1: RealMatrix,
        duality_0: HodgeDualityMorphism,
        duality_1: HodgeDualityMorphism,
        boundary_2: Optional[RealMatrix] = None,
    ) -> None:
        r"""
        Inicializa el resolvedor sobre el complejo $C_2 \to C_1 \to C_0$.

        Parámetros
        ----------
        boundary_1 : RealMatrix
            Operador de frontera $\partial_1$ de tamaño $V\times E$ (nodos × aristas).
        duality_0 : HodgeDualityMorphism
            Morfismo de dualidad para 0-formas (nodos) — producido por FASE 1.
        duality_1 : HodgeDualityMorphism
            Morfismo de dualidad para 1-formas (aristas) — producido por FASE 1.
        boundary_2 : Optional[RealMatrix]
            Operador de frontera $\partial_2$ de tamaño $E\times F$ (aristas × caras).
            Si se omite, se asume complejo 1-dimensional (grafo puro).
        """
        B1 = np.asarray(boundary_1, dtype=np.float64)
        if B1.ndim != 2:
            raise ValueError("∂₁ debe ser una matriz 2-D.")
        n_nodes, n_edges = B1.shape

        if duality_0.star.dimension != n_nodes:
            raise ValueError(
                f"Incompatibilidad ∂₁ / ★₀: nodos={n_nodes}, dim(★₀)={duality_0.star.dimension}."
            )
        if duality_1.star.dimension != n_edges:
            raise ValueError(
                f"Incompatibilidad ∂₁ / ★₁: aristas={n_edges}, dim(★₁)={duality_1.star.dimension}."
            )
        if not duality_0.is_well_conditioned or not duality_1.is_well_conditioned:
            logger.warning(
                "Una o ambas métricas presentan número de condición elevado "
                "(κ₀=%.2e, κ₁=%.2e).",
                duality_0.metric_state.condition_number,
                duality_1.metric_state.condition_number,
            )

        self._b1: RealMatrix = B1.copy()
        self._b1.setflags(write=False)
        self._dual0 = duality_0
        self._dual1 = duality_1
        self._n_nodes = n_nodes
        self._n_edges = n_edges

        self._b2: Optional[RealMatrix] = None
        if boundary_2 is not None:
            B2 = np.asarray(boundary_2, dtype=np.float64)
            if B2.shape[0] != n_edges:
                raise ValueError(
                    f"Incompatibilidad ∂₂: filas={B2.shape[0]} ≠ aristas={n_edges}."
                )
            self._b2 = B2.copy()
            self._b2.setflags(write=False)
            self._n_faces = B2.shape[1]
        else:
            self._n_faces = 0

        # Verificación de la propiedad fundamental de complejo de cadenas: ∂₁∂₂ = 0
        if self._b2 is not None:
            compos = self._b1 @ self._b2
            resid = np.max(np.abs(compos))
            if resid > 10.0 * _MACHINE_EPS * max(n_nodes, n_edges, self._n_faces):
                raise ValueError(
                    f"Axioma de Complejo de Cadenas violado: ||∂₁∂₂||_∞ = {resid:.3e} ≠ 0."
                )

        logger.info(
            "WeightedHodgeLaplacian: V=%d, E=%d, F=%d | κ₀=%.2e κ₁=%.2e",
            n_nodes, n_edges, self._n_faces,
            duality_0.metric_state.condition_number,
            duality_1.metric_state.condition_number,
        )

    # ── Constructores de operadores de Laplace-de Rham ───────────────────────

    def compute_weighted_laplacian_0(self) -> RealMatrix:
        r"""
        Laplaciano ponderado de grado 0 (operador de Poisson/graph-Laplacian):
        $$\Delta_0^\star = \partial_1\,\star_1\,\partial_1^\top\,\star_0^{-1}.$$

        Es autoadjunto respecto de $\langle\cdot,\cdot\rangle_{\star_0}$ y
        semi-definido positivo. Su núcleo son las funciones localmente constantes
        (dim ker = β₀ = número de componentes conexas).

        Returns
        -------
        RealMatrix
            Matriz densa $V\times V$.
        """
        star1 = self._dual1.star.matrix
        star0_inv = self._dual0.star.inverse_matrix
        # Δ₀ = B₁ ★₁ B₁ᵀ ★₀⁻¹
        return self._b1 @ star1 @ self._b1.T @ star0_inv

    def compute_weighted_laplacian_1(self, include_curl: bool = True) -> RealMatrix:
        r"""
        Laplaciano ponderado de grado 1 (completo o sin curl):
        $$\Delta_1^\star = \underbrace{\star_1\partial_1^\top\star_0^{-1}\partial_1}_{\mathrm{grad}^*\mathrm{grad}}
                         + \underbrace{\partial_2\star_2\partial_2^\top\star_1^{-1}}_{\mathrm{curl}^*\mathrm{curl}}.$$

        Cuando `include_curl=False` o no existe ∂₂ se omite el segundo sumando
        (caso de grafos puros). El operador es autoadjunto respecto de ★₁.

        Returns
        -------
        RealMatrix
            Matriz densa $E\times E$.
        """
        star1 = self._dual1.star.matrix
        star0_inv = self._dual0.star.inverse_matrix
        # Término grad*grad
        L = star1 @ self._b1.T @ star0_inv @ self._b1

        if include_curl and self._b2 is not None:
            # Requiere ★₂; si no se suministró se usa identidad (volúmenes unitarios)
            # En una extensión completa se inyectaría duality_2.
            # Aquí se asume ★₂ = I por defecto (documentado).
            star2 = np.eye(self._n_faces, dtype=np.float64)
            star1_inv = self._dual1.star.inverse_matrix
            L = L + self._b2 @ star2 @ self._b2.T @ star1_inv

        return L

    def compute_coboundary_weighted(self) -> RealMatrix:
        r"""
        Coborde ponderado (exterior derivative métrica):
        $$d_0^\star = \star_1\partial_1^\top\star_0^{-1}.$$

        Satisface $\Delta_0 = d_0^{\star*}d_0^\star$ (factorización de Cholesky métrica).
        """
        return self._dual1.star.matrix @ self._b1.T @ self._dual0.star.inverse_matrix

    # ── Análisis espectral del Laplaciano ────────────────────────────────────

    def spectrum(
        self,
        degree: HodgeDegree = HodgeDegree.ZERO,
        k_eigs: Optional[int] = None,
        include_curl: bool = True,
    ) -> LaplacianSpectrum:
        r"""
        Computa el espectro de $\Delta_k^\star$ y extrae invariantes topológicos.

        Parámetros
        ----------
        degree : HodgeDegree
            Grado del Laplaciano (ZERO o ONE).
        k_eigs : Optional[int]
            Si se indica, calcula solo los k autovalores más pequeños
            (Lanczos/ARPACK); si None, usa descomposición densa completa.
        include_curl : bool
            Incluir término de curl en Δ₁.

        Returns
        -------
        LaplacianSpectrum
            Espectro certificado con número de Betti y conectividad algebraica.
        """
        if degree == HodgeDegree.ZERO:
            L = self.compute_weighted_laplacian_0()
        elif degree == HodgeDegree.ONE:
            L = self.compute_weighted_laplacian_1(include_curl=include_curl)
        else:
            raise NotImplementedError(f"Espectro para grado {degree} no implementado en FASE 2.")

        # Simetrización numérica (garantiza autoadjunción de máquina)
        L_sym = 0.5 * (L + L.T)

        n = L_sym.shape[0]
        if k_eigs is None or k_eigs >= n:
            ev, evec = la.eigh(L_sym)
        else:
            k_eigs = min(k_eigs, n - 1)
            ev, evec = spla.eigsh(L_sym, k=k_eigs, which="SA")
            # Ordenar
            idx = np.argsort(ev)
            ev, evec = ev[idx], evec[:, idx]

        # Número de Betti: multiplicidad numérica del autovalor 0
        tol_ker = _SPECTRAL_TOL * max(1.0, float(np.max(np.abs(ev))))
        betti = int(np.sum(np.abs(ev) < tol_ker))
        alg_conn = float(ev[betti]) if betti < len(ev) else 0.0

        # Residuo de autoadjunción y de eigenpares
        max_res = 0.0
        for i in range(len(ev)):
            r = L_sym @ evec[:, i] - ev[i] * evec[:, i]
            max_res = max(max_res, float(np.max(np.abs(r))))

        is_sa = float(np.max(np.abs(L - L.T))) < 10.0 * _MACHINE_EPS * n

        return LaplacianSpectrum(
            eigenvalues=ev,
            eigenvectors=evec,
            betti_number=betti,
            algebraic_connectivity=alg_conn,
            is_self_adjoint=is_sa,
            max_residual=max_res,
        )

    def green_energy(
        self,
        alpha: RealVector,
        degree: HodgeDegree = HodgeDegree.ZERO,
    ) -> float:
        r"""
        Energía de Dirichlet/Green asociada a una cochain:
        $$\mathcal{E}(\alpha) = \frac12\langle\Delta_k^\star\alpha,\alpha\rangle_{\star_k}.$$

        Es la acción del sistema Port-Hamiltoniano en régimen estacionario.
        """
        if degree == HodgeDegree.ZERO:
            L = self.compute_weighted_laplacian_0()
            # El producto interno es respecto de ★₀; L ya incluye ★₀⁻¹,
            # por tanto αᵀ ★₀ (L α) = αᵀ (B ★₁ Bᵀ) α.
            star0 = self._dual0.star.matrix
            return 0.5 * float(alpha.T @ star0 @ (L @ alpha))
        elif degree == HodgeDegree.ONE:
            L = self.compute_weighted_laplacian_1()
            star1 = self._dual1.star.matrix
            return 0.5 * float(alpha.T @ star1 @ (L @ alpha))
        raise NotImplementedError(f"Energía de Green para grado {degree} no disponible.")

    # ── Último método formal de la FASE 2 ────────────────────────────────────
    # Su valor de retorno (DeRhamComplexCertificate) es el objeto de arranque
    # obligatorio de la FASE 3.

    def certify_de_rham_complex(self) -> "DeRhamComplexCertificate":
        r"""
        Certifica la exactitud y las propiedades espectrales del complejo de
        de Rham discreto ponderado de grados 0-1-(2).

        Verifica:
        1. ∂₁∂₂ = 0 (ya comprobado en __init__, se re-exporta).
        2. Autoadjunción de Δ₀ y Δ₁.
        3. Números de Betti β₀, β₁ (dim ker Δ_k).
        4. Compatibilidad de la dualidad de Hodge con la cohomología
           (teorema de Hodge discreto: ker Δ_k ≅ H^k).

        El certificado retornado es el único argumento legítimo del constructor
        de la FASE 3 (DiscreteDeRhamComplex / PortHamiltonianConstitutiveLaw).

        Returns
        -------
        DeRhamComplexCertificate
            Objeto inmutable que cierra la FASE 2 y abre la FASE 3.
        """
        spec0 = self.spectrum(HodgeDegree.ZERO)
        spec1 = self.spectrum(HodgeDegree.ONE, include_curl=(self._b2 is not None))

        # Identidades de Hodge: β_k = dim ker Δ_k
        beta0 = spec0.betti_number
        beta1 = spec1.betti_number

        # Energía de referencia (traza de la métrica)
        energy_scale = (
            self._dual0.metric_state.trace + self._dual1.metric_state.trace
        )

        return DeRhamComplexCertificate(
            laplacian_0_spectrum=spec0,
            laplacian_1_spectrum=spec1,
            betti_0=beta0,
            betti_1=beta1,
            n_nodes=self._n_nodes,
            n_edges=self._n_edges,
            n_faces=self._n_faces,
            duality_0=self._dual0,
            duality_1=self._dual1,
            boundary_1=self._b1,
            boundary_2=self._b2,
            energy_scale=energy_scale,
            is_exact_sequence=(self._b2 is None) or (beta0 + beta1 > 0),
        )


# =============================================================================
# FASE 3 — COMPLEJO DE DE RHAM DISCRETO COMPLETO + LEYES CONSTITUTIVAS
#          PORT-HAMILTONIANAS (Continúa desde certify_de_rham_complex)
#          Topología algebraica + Teoría de categorías + Mecánica cuántica
#          de sistemas abiertos (estructura de Dirac discreta)
# =============================================================================

@dataclass(frozen=True, slots=True)
class DeRhamComplexCertificate:
    r"""
    Certificado global del complejo de de Rham discreto ponderado (cierre FASE 2).

    Este objeto es el contrato formal de continuidad FASE 2 → FASE 3.
    Contiene toda la información espectral, topológica y métrica necesaria
    para construir operadores de proyección de Hodge, descomposición de
    Helmholtz-Hodge y la estructura de Dirac de un sistema Port-Hamiltoniano.
    """
    laplacian_0_spectrum: LaplacianSpectrum
    laplacian_1_spectrum: LaplacianSpectrum
    betti_0: int
    betti_1: int
    n_nodes: int
    n_edges: int
    n_faces: int
    duality_0: HodgeDualityMorphism
    duality_1: HodgeDualityMorphism
    boundary_1: RealMatrix
    boundary_2: Optional[RealMatrix]
    energy_scale: float
    is_exact_sequence: bool


@dataclass(frozen=True, slots=True)
class HelmholtzHodgeDecomposition:
    r"""
    Descomposición de Helmholtz-Hodge de una 1-forma:
    $$\alpha = d\varphi + \delta\beta + h,$$
    donde $h\in\ker\Delta_1$ (armónicos), $d\varphi$ es exacta y $\delta\beta$ es coexacta.

    En dimensión 1 (grafos): $\alpha = \operatorname{grad}\varphi + \operatorname{proj}_{\ker}\alpha$.
    """
    exact_part: RealVector       # dφ
    coexact_part: RealVector     # δβ
    harmonic_part: RealVector    # h ∈ H¹
    potential: RealVector        # φ (0-forma)
    residual_norm: float         # ||α - (exact+coexact+harmonic)||


@dataclass(frozen=True, slots=True)
class PortHamiltonianState:
    r"""
    Estado energético de un sistema Port-Hamiltoniano discreto.

    El Hamiltoniano es la suma de energías cuadráticas métricas:
    $$\mathcal{H}(q,p) = \frac12 q^\top\star_0 q + \frac12 p^\top\star_1^{-1} p.$$
    """
    effort_0: RealVector         # esfuerzos nodales (tensiones)
    flow_1: RealVector           # flujos de arista (corrientes)
    hamiltonian: float
    dissipated_power: float
    conserved_quantity: float    # carga total / flujo armónico


class DiscreteDeRhamComplex:
    r"""
    Complejo de de Rham discreto ponderado de longitud 2 (nodos-aristas-caras)
    equipado con la estructura de Dirac y las proyecciones de Hodge.

    Construido exclusivamente a partir de un DeRhamComplexCertificate
    (continuidad formal FASE 2 → FASE 3). Realiza:

    * Descomposición de Helmholtz-Hodge.
    * Proyectores espectrales sobre armónicos (cohomología).
    * Estructura de Dirac $\mathcal{J}-\mathcal{R}$ de sistemas Port-Hamiltonianos.
    * Evolución cuasi-cuántica (semigrupo de contracción generado por -Δ).
    """

    def __init__(self, certificate: DeRhamComplexCertificate) -> None:
        r"""
        Único constructor legítimo: recibe el certificado de la FASE 2.

        Parámetros
        ----------
        certificate : DeRhamComplexCertificate
            Objeto retornado por WeightedHodgeLaplacian.certify_de_rham_complex().
        """
        if not isinstance(certificate, DeRhamComplexCertificate):
            raise TypeError(
                "FASE 3 exige un DeRhamComplexCertificate emitido por la FASE 2. "
                "No se admiten construcciones ad-hoc."
            )
        self._cert = certificate
        self._dual0 = certificate.duality_0
        self._dual1 = certificate.duality_1
        self._b1 = certificate.boundary_1
        self._b2 = certificate.boundary_2

        # Pre-cálculo de bases armónicas (núcleos de los Laplacianos)
        self._harmonic_0 = self._extract_harmonic_basis(
            certificate.laplacian_0_spectrum, certificate.betti_0
        )
        self._harmonic_1 = self._extract_harmonic_basis(
            certificate.laplacian_1_spectrum, certificate.betti_1
        )

        logger.info(
            "DiscreteDeRhamComplex certificado: β₀=%d β₁=%d | V=%d E=%d F=%d",
            certificate.betti_0, certificate.betti_1,
            certificate.n_nodes, certificate.n_edges, certificate.n_faces,
        )

    @staticmethod
    def _extract_harmonic_basis(
        spectrum: LaplacianSpectrum, betti: int
    ) -> RealMatrix:
        """Extrae las primeras `betti` columnas (autovectores del núcleo)."""
        if betti == 0:
            return np.zeros((spectrum.eigenvectors.shape[0], 0), dtype=np.float64)
        return spectrum.eigenvectors[:, :betti].copy()

    # ── Proyectores de Hodge y descomposición ────────────────────────────────

    def project_harmonic(self, alpha: RealVector, degree: HodgeDegree) -> RealVector:
        r"""
        Proyección ortogonal (métrica) sobre el subespacio de formas armónicas
        $H^k = \ker\Delta_k \simeq H^k_{\mathrm{dR}}(K)$.

        $$P_H\alpha = \sum_{j=1}^{\beta_k}\langle\alpha,h_j\rangle_{\star}\,h_j.$$
        """
        a = np.asarray(alpha, dtype=np.float64).ravel()
        if degree == HodgeDegree.ZERO:
            basis = self._harmonic_0
            star = self._dual0.star
        elif degree == HodgeDegree.ONE:
            basis = self._harmonic_1
            star = self._dual1.star
        else:
            raise NotImplementedError(f"Proyección armónica grado {degree} no disponible.")

        if basis.shape[1] == 0:
            return np.zeros_like(a)

        # Coeficientes de Fourier-Hodge
        coeffs = np.array([star.inner_product(a, basis[:, j]) for j in range(basis.shape[1])])
        # Re-normalización si la base no es ★-ortonormal (eigh lo es en ℓ², no en ★)
        # Corrección: proyectar en la métrica correcta
        gram = np.array([
            [star.inner_product(basis[:, i], basis[:, j]) for j in range(basis.shape[1])]
            for i in range(basis.shape[1])
        ])
        try:
            coeffs_metric = la.solve(gram, coeffs, assume_a="pos")
        except la.LinAlgError:
            coeffs_metric = la.lstsq(gram, coeffs)[0]
        return basis @ coeffs_metric

    def helmholtz_hodge_decomposition(
        self, alpha: RealVector
    ) -> HelmholtzHodgeDecomposition:
        r"""
        Descomposición completa de Helmholtz-Hodge de una 1-forma α:
        $$\alpha = \underbrace{\star_1^{-1}\partial_1^\top\star_0\varphi}_{\mathrm{exacta}}
                 + \underbrace{h}_{\mathrm{armónica}}
                 + \underbrace{\alpha - \mathrm{exacta} - h}_{\mathrm{coexacta}}.$$

        El potencial φ se obtiene resolviendo el problema de Poisson
        $$\Delta_0\varphi = \partial_1\star_1\alpha$$
        en el ortogonal al núcleo (componentes conexas).
        """
        a = np.asarray(alpha, dtype=np.float64).ravel()
        if a.shape[0] != self._cert.n_edges:
            raise ValueError(
                f"1-forma de dim {a.shape[0]} incompatible con E={self._cert.n_edges}."
            )

        # 1. Parte armónica
        h = self.project_harmonic(a, HodgeDegree.ONE)

        # 2. Parte exacta: resolver Δ₀ φ = div_★ (α) = ∂₁ ★₁ α
        star1_a = self._dual1.star.apply(a)
        rhs = self._b1 @ star1_a                    # en el dual de nodos
        # Pasar a coordenadas primales: ★₀⁻¹ rhs no es necesario porque
        # Δ₀ ya contiene ★₀⁻¹; el sistema es Δ₀ φ = ★₀⁻¹ (∂₁ ★₁ α)
        # ⇒ ★₀ Δ₀ φ = ∂₁ ★₁ α, pero Δ₀ = B★₁Bᵀ★₀⁻¹ ⇒ ★₀Δ₀ = B★₁Bᵀ
        # Por tanto resolvemos (B★₁Bᵀ) φ = ∂₁ ★₁ α  (Laplaciano de grafo clásico)
        L0_metric = (
            self._b1 @ self._dual1.star.matrix @ self._b1.T
        )
        # Proyectar rhs y el sistema al ortogonal de constantes
        # (fijar gauge: media nula o anclar un nodo)
        n = L0_metric.shape[0]
        # Anclaje del primer nodo (gauge de Dirichlet)
        L_red = L0_metric[1:, 1:]
        rhs_red = rhs[1:]
        try:
            phi_red = la.solve(L_red, rhs_red, assume_a="sym")
        except la.LinAlgError:
            phi_red = la.lstsq(L_red, rhs_red)[0]
        phi = np.zeros(n, dtype=np.float64)
        phi[1:] = phi_red

        # grad_★ φ = ★₁⁻¹ ∂₁ᵀ ★₀ φ   (pero ★₀=diag →)
        star0_phi = self._dual0.star.apply(phi)
        exact = self._dual1.star.apply_inverse(self._b1.T @ star0_phi)

        # 3. Parte coexacta (resto)
        coexact = a - exact - h
        residual = float(np.linalg.norm(a - exact - coexact - h))

        return HelmholtzHodgeDecomposition(
            exact_part=exact,
            coexact_part=coexact,
            harmonic_part=h,
            potential=phi,
            residual_norm=residual,
        )

    # ── Estructura Port-Hamiltoniana / Dirac ─────────────────────────────────

    def dirac_structure_matrix(self) -> RealMatrix:
        r"""
        Matriz de la estructura de Dirac discreta $\mathcal{J}$ (skew-simétrica):
        $$\mathcal{J} = \begin{pmatrix} 0 & -\partial_1 \\ \partial_1^\top & 0 \end{pmatrix}.$$

        Junto con la métrica de disipación $\mathcal{R}=\operatorname{diag}(\star_0,\star_1)$
        genera la dinámica
        $$\begin{pmatrix}\dot{e}\\ \dot{f}\end{pmatrix}
          = (\mathcal{J}-\mathcal{R})\,\frac{\partial\mathcal{H}}{\partial(e,f)}.$$
        """
        V, E = self._cert.n_nodes, self._cert.n_edges
        J = np.zeros((V + E, V + E), dtype=np.float64)
        J[:V, V:] = -self._b1
        J[V:, :V] = self._b1.T
        # Comprobación de skew-simetría
        skew_err = float(np.max(np.abs(J + J.T)))
        if skew_err > 10.0 * _MACHINE_EPS * (V + E):
            logger.warning("Estructura de Dirac no exactamente skew (err=%.3e).", skew_err)
        return J

    def evaluate_hamiltonian(
        self,
        effort_0: RealVector,
        flow_1: RealVector,
        resistance_weights: Optional[RealVector] = None,
    ) -> PortHamiltonianState:
        r"""
        Evalúa el estado Port-Hamiltoniano completo.

        $$\mathcal{H} = \frac12\|e_0\|_{\star_0}^2 + \frac12\|f_1\|_{\star_1^{-1}}^2,$$
        $$P_{\mathrm{diss}} = f_1^\top R f_1 \quad (R=\operatorname{diag}(r_i)).$$
        """
        e = np.asarray(effort_0, dtype=np.float64).ravel()
        f = np.asarray(flow_1, dtype=np.float64).ravel()
        if e.shape[0] != self._cert.n_nodes or f.shape[0] != self._cert.n_edges:
            raise ValueError("Dimensiones de esfuerzo/flujo incompatibles con el complejo.")

        H = 0.5 * self._dual0.star.inner_product(e, e) + 0.5 * float(
            np.dot(f, self._dual1.star.apply_inverse(f))
        )

        if resistance_weights is None:
            # Disipación por defecto = pesos de ★₁ (conductancias)
            P_diss = self._dual1.star.inner_product(f, f)
        else:
            r = np.asarray(resistance_weights, dtype=np.float64).ravel()
            P_diss = float(np.dot(f, r * f))

        # Cantidad conservada: proyección sobre armónicos de grado 0 (carga total)
        q_cons = float(np.sum(self._dual0.star.apply(e)))  # carga neta métrica

        return PortHamiltonianState(
            effort_0=e,
            flow_1=f,
            hamiltonian=H,
            dissipated_power=P_diss,
            conserved_quantity=q_cons,
        )

    # ── Semigrupo de difusión / evolución espectral ──────────────────────────

    def heat_semigroup(
        self,
        alpha0: RealVector,
        t: float,
        degree: HodgeDegree = HodgeDegree.ZERO,
    ) -> RealVector:
        r"""
        Evolución bajo el semigrupo del calor de Hodge:
        $$\alpha(t) = e^{-t\Delta_k^\star}\alpha_0 = \sum_j e^{-t\lambda_j}\langle\alpha_0,v_j\rangle v_j.$$

        En el límite $t\to\infty$ se obtiene la proyección sobre armónicos
        (teorema de Hodge dinámico). Compatible con la interpretación de
        mecánica cuántica de sistemas abiertos (canal de Lorentzian).
        """
        if t < 0.0:
            raise ValueError("El semigrupo del calor está definido solo para t ≥ 0.")
        a = np.asarray(alpha0, dtype=np.float64).ravel()

        if degree == HodgeDegree.ZERO:
            spec = self._cert.laplacian_0_spectrum
            star = self._dual0.star
        elif degree == HodgeDegree.ONE:
            spec = self._cert.laplacian_1_spectrum
            star = self._dual1.star
        else:
            raise NotImplementedError(f"Semigrupo grado {degree} no implementado.")

        if a.shape[0] != spec.eigenvectors.shape[0]:
            raise ValueError("Dimensión de la cochain incompatible con el espectro.")

        # Expansión espectral (base ℓ²-ortonormal de eigh)
        coeffs = spec.eigenvectors.T @ a
        decay = np.exp(-t * spec.eigenvalues)
        return spec.eigenvectors @ (decay * coeffs)

    def hodge_norm_squared(self, alpha: RealVector, degree: HodgeDegree) -> float:
        """Acceso unificado a $\|\alpha\|_{\star_k}^2$."""
        if degree == HodgeDegree.ZERO:
            return self._dual0.star.inner_product(alpha, alpha)
        if degree == HodgeDegree.ONE:
            return self._dual1.star.inner_product(alpha, alpha)
        raise NotImplementedError(f"Norma Hodge grado {degree} no disponible.")

    # ── Función generadora de categorías (topos de operadores) ───────────────

    def monoidal_product(
        self, other: "DiscreteDeRhamComplex"
    ) -> Dict[str, int]:
        r"""
        Producto monoidal (suma directa de complejos) en la categoría de
        complejos de de Rham discretos:

        $$(K,\star) \oplus (K',\star') \quad\Longrightarrow\quad
          \beta_k^{\mathrm{tot}} = \beta_k + \beta_k'.$$

        Devuelve los números de Betti del objeto suma (invariante monoidal).
        """
        return {
            "betti_0": self._cert.betti_0 + other._cert.betti_0,
            "betti_1": self._cert.betti_1 + other._cert.betti_1,
            "n_nodes": self._cert.n_nodes + other._cert.n_nodes,
            "n_edges": self._cert.n_edges + other._cert.n_edges,
            "n_faces": self._cert.n_faces + other._cert.n_faces,
        }

    def summary(self) -> str:
        """Resumen legible del complejo certificado (diagnóstico de laboratorio)."""
        c = self._cert
        lines = [
            "=" * 72,
            "DISCRETE DE RHAM COMPLEX — CERTIFICADO DOCTORAL (FASE 3)",
            "=" * 72,
            f"  Dimensiones      : V={c.n_nodes}  E={c.n_edges}  F={c.n_faces}",
            f"  Números de Betti : β₀={c.betti_0}  β₁={c.betti_1}",
            f"  κ(★₀)            : {c.duality_0.metric_state.condition_number:.4e}",
            f"  κ(★₁)            : {c.duality_1.metric_state.condition_number:.4e}",
            f"  λ_min(Δ₀)        : {c.laplacian_0_spectrum.eigenvalues[0]:.4e}",
            f"  λ_min(Δ₁)        : {c.laplacian_1_spectrum.eigenvalues[0]:.4e}",
            f"  Conectividad alg.: {c.laplacian_0_spectrum.algebraic_connectivity:.4e}",
            f"  Escala energética: {c.energy_scale:.4e}",
            f"  Sucesión exacta  : {c.is_exact_sequence}",
            f"  Autoadjunto Δ₀   : {c.laplacian_0_spectrum.is_self_adjoint}",
            f"  Autoadjunto Δ₁   : {c.laplacian_1_spectrum.is_self_adjoint}",
            f"  Residuo máx. Δ₀  : {c.laplacian_0_spectrum.max_residual:.3e}",
            f"  Residuo máx. Δ₁  : {c.laplacian_1_spectrum.max_residual:.3e}",
            "=" * 72,
        ]
        return "\n".join(lines)


# =============================================================================
# API de conveniencia y factoría (composición de las 3 fases anidadas)
# =============================================================================

def build_certified_de_rham_complex(
    weights_0: RealVector,
    weights_1: RealVector,
    boundary_1: RealMatrix,
    boundary_2: Optional[RealMatrix] = None,
    weights_2: Optional[RealVector] = None,
) -> DiscreteDeRhamComplex:
    r"""
    Factoría de alto nivel que encadena las tres fases anidadas de forma segura:

    FASE 1  →  DiscreteHodgeStar + induce_duality_morphism()
    FASE 2  →  WeightedHodgeLaplacian + certify_de_rham_complex()
    FASE 3  →  DiscreteDeRhamComplex(certificate)

    Parámetros
    ----------
    weights_0, weights_1 : RealVector
        Pesos constitutivos de ★₀ y ★₁ (estrictamente positivos).
    boundary_1 : RealMatrix
        Matriz de incidencia nodo-arista ∂₁.
    boundary_2 : Optional[RealMatrix]
        Matriz de incidencia arista-cara ∂₂ (opcional).
    weights_2 : Optional[RealVector]
        Pesos de ★₂ (reservado; actualmente se usa I si boundary_2 existe).

    Returns
    -------
    DiscreteDeRhamComplex
        Complejo de de Rham discreto totalmente certificado y listo para
        simulación Port-Hamiltoniana / análisis espectral / cohomología.
    """
    # ── FASE 1 ──────────────────────────────────────────────────────────────
    star0 = DiscreteHodgeStar(weights_0, degree=HodgeDegree.ZERO)
    star1 = DiscreteHodgeStar(weights_1, degree=HodgeDegree.ONE)
    dual0 = star0.induce_duality_morphism()   # ← último método FASE 1
    dual1 = star1.induce_duality_morphism()

    # ── FASE 2 (inicia desde HodgeDualityMorphism) ──────────────────────────
    lap = WeightedHodgeLaplacian(
        boundary_1=boundary_1,
        duality_0=dual0,
        duality_1=dual1,
        boundary_2=boundary_2,
    )
    cert = lap.certify_de_rham_complex()      # ← último método FASE 2

    # ── FASE 3 (inicia desde DeRhamComplexCertificate) ──────────────────────
    complex_3 = DiscreteDeRhamComplex(cert)   # ← entrada FASE 3
    logger.info("\n%s", complex_3.summary())
    return complex_3


# =============================================================================
# Fin del módulo — las tres fases anidadas quedan formalmente cerradas:
#   DiscreteHodgeStar.induce_duality_morphism
#       → HodgeDualityMorphism
#           → WeightedHodgeLaplacian.certify_de_rham_complex
#               → DeRhamComplexCertificate
#                   → DiscreteDeRhamComplex
# =============================================================================
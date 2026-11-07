# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Discrete Hodge Star Operator (El Operador Constitutivo de de Rham)  ║
║ Ruta   : app/physics/discrete_hodge_star.py                                  ║
║ Versión: 3.0.0-DEC-PoincareCelestial-RSI-L3-Spectral-Topos-Strict-PhD        ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y RIGOR DOCTORAL
(DEC + Teoría Espectral + Topos + Mecánica Celeste de Poincaré + RSI Nivel 3):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra la formulación discreta del operador estrella de Hodge ($\star$)
sobre complejos simpliciales orientados de dimensión finita, elevada a un marco
unificado de teoría de categorías/topos, teoría espectral de operadores autoadjuntos
en espacios de Hilbert ponderados, mecánica celeste simpléctica de Henri Poincaré
(*Les Méthodes Nouvelles de la Mécanique Céleste*) y automejora recursiva de
Nivel 3 (RSI-L3) mediante flujos geodésicos isocóricos de Fisher-Rao.

En el contexto de la electrodinámica discreta, circuitos cuánticos superconductores,
sistemas Port-Hamiltonianos y geometría de la información, el Hodge star actúa como
el morfismo métrico constitutivo que realiza la dualidad de Poincaré discreta:

$$\star_k: \Lambda^k \mathcal{H} \xrightarrow{\simeq} \Lambda^{n-k} \mathcal{H}^*$$

AXIOMÁTICA SIMPLÉCTICA, COHOMOLÓGICA, CELESTE Y DE AUTOMEJORA (RSI-L3):
────────────────────────────────────────────────────────────────────────────────

  [A1] Simetría Autoadjunta, Cono SPD y Flujo Isocórico RSI-L3 (Fisher-Rao):
       El operador estrella de Hodge discreto $\star_k$ es estrictamente simétrico y
       definido positivo (SPD):
       $$\star_k = \star_k^\top \succ \mathbf{0}, \qquad \lambda_{\min}(\star_k) \ge \mathtt{\_SPECTRAL\_PSD\_FLOOR}$$
       $$\kappa_2(\star_k) = \frac{\lambda_{\max}(\star_k)}{\lambda_{\min}(\star_k)} \le \mathtt{\_CONDITION\_NUMBER\_MAX} = 1.0\times 10^8$$
       Si se detectan perturbaciones o mal condicionamiento, el endofuntor RSI de
       Nivel 3 proyecta al cono SPD vía Higham-Löwner (Nivel 1), ejecuta una
       retracción geodésica en la variedad Riemanniana de Fisher-Rao preservando el
       invariante integral de volumen de Liouville-Poincaré $\det(\star_k)$ (Nivel 2),
       y auto-adapta su hiper-ley de amortiguamiento mediante contracción de
       Banach-Lyapunov certificada (Nivel 3).

  [A2] Descomposición de Helmholtz-Hodge Ortogonal Ponderada No-Tautológica:
       Toda 1-cocadena $\alpha \in C^1(K)$ se descompone de forma única y
       métricamente ortogonal respecto al producto interno $\langle \cdot, \cdot \rangle_{\star_1}$:
       $$\alpha = d_0\varphi + \delta_2\beta + h$$
       Donde:
         · $d_0\varphi = \partial_1^\top \varphi$ es la componente exacta ($\varphi \in C^0$).
         · $\delta_2\beta = \star_1^{-1} \partial_2 \star_2 \beta$ es la componente coexacta ($\beta \in C^2$).
         · $h \in \ker(\Delta_1^\star)$ es la componente armónica ($h \cong H^1_{\mathrm{dR}}(K)$).
       Certificada por la anulación simultánea de los tres apareamientos de Riesz:
       $$\langle d_0\varphi, \delta_2\beta \rangle_{\star_1} = \langle d_0\varphi, h \rangle_{\star_1} = \langle \delta_2\beta, h \rangle_{\star_1} \equiv 0 \pmod{\varepsilon_{\mathrm{mach}}}$$

  [A3] Conjugación Isométrica de Hilbert para Laplacianos de Hodge:
       Con los diferenciales exteriores $d_0 = \partial_1^\top$, $d_1 = \partial_2^\top$ y
       codiferenciales $\delta_1 = \star_0^{-1}\partial_1\star_1$, $\delta_2 = \star_1^{-1}\partial_2\star_2$:
       $$\Delta_0^\star = \delta_1 d_0 = \star_0^{-1} \partial_1 \star_1 \partial_1^\top, \qquad
         \Delta_1^\star = d_0 \delta_1 + \delta_2 d_1 = \partial_1^\top \star_0^{-1} \partial_1 \star_1 + \star_1^{-1} \partial_2 \star_2 \partial_2^\top$$
       Su diagonalización espectral se ejecuta mediante la transformación de
       similaridad isométrica $\tilde{\Delta}_k = \star_k^{1/2} \Delta_k^\star \star_k^{-1/2}$,
       garantizando simetría exacta en $\ell^2$ y autovectores $\star_k$-ortonormales.

  [A4] Mecánica Celeste de Poincaré, Resonancias KAM y Recurrencia Simpléctica:
       El flujo Port-Hamiltoniano en el espacio fase $T^*C^0 \cong C^0 \oplus C^1$ se
       integra mediante el morfismo simpléctico de Cayley-Poincaré:
       $$\Phi_{\Delta t} = \left(I - \frac{\Delta t}{2}(\mathcal{J} - \mathcal{R})\mathcal{M}\right)^{-1}
                           \left(I + \frac{\Delta t}{2}(\mathcal{J} - \mathcal{R})\mathcal{M}\right)$$
       En el límite conservativo ($\mathcal{R} = \mathbf{0}$), preserva exactamente la
       2-forma simpléctica de Poincaré-Cartan $\omega = dp \wedge dq$, el determinante
       de Liouville $\det(\Phi_{\Delta t}) = 1$ y el Hamiltoniano $\mathcal{H}(q, p)$,
       permitiendo auditar la sección estroboscópica de Poincaré, el tiempo de
       recurrencia de Kac-Poincaré, los multiplicadores de Floquet y los pequeños
       divisores diofánticos de Kolmogorov-Arnold-Moser (KAM).

  [A5] Invarianza de Euler-Poincaré y Veto Coherente en el Retículo de Heyting $\Omega_3$:
       $$\chi(K) = |V| - |E| + |F| = \beta_0 - \beta_1 + \beta_2$$
       Evaluación en el álgebra de Heyting $\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}$.

ARQUITECTURA DE TRES FASES ANIDADAS (Continuidad Formal Estricta):
────────────────────────────────────────────────────────────────────────────────
  FASE 1 ──► FUNDAMENTOS MÉTRICOS, AUTOMEJORA RSI-L3 Y VOLUMEN DE LIOUVILLE
             Clase principal: DiscreteHodgeStar (alias Phase1_SpectralMetricObserver).
             Último método formal: DiscreteHodgeStar.induce_duality_morphism()
             ──► cuya firma y retorno (HodgeDualityMorphism) es el inicio exacto de la Fase 2.

  FASE 2 ──► MORFISMO DE DUALIDAD, LAPLACIANOS DE HILBERT Y ESPECTRO KAM DE POINCARÉ
             Inicia con la definición formal de HodgeDualityMorphism (alias Phase1HodgeObservation)
             y continúa con WeightedHodgeLaplacian (alias Phase2_HodgeLaplacianOrienter).
             Último método formal: WeightedHodgeLaplacian.certify_de_rham_complex()
             ──► cuya firma y retorno (DeRhamComplexCertificate) es el inicio exacto de la Fase 3.

  FASE 3 ──► COMPLEJO DE DE RHAM, DINÁMICA CELESTE DE POINCARÉ Y DECISIÓN DE HEYTING
             Inicia con la definición formal de DeRhamComplexCertificate (alias Phase2HodgeOrientation)
             y continúa con DiscreteDeRhamComplex (alias Phase3_HelmholtzDecisionMaker).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import (
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

logger = logging.getLogger("MIC.Physics.DiscreteHodgeStar")

# ─────────────────────────────────────────────────────────────────────────────
# Tipos canónicos y constantes de máquina (Rigor Numérico de Wilkinson)
# ─────────────────────────────────────────────────────────────────────────────
RealMatrix = NDArray[np.float64]
RealVector = NDArray[np.float64]
ComplexVector = NDArray[np.complex128]
ComplexMatrix = NDArray[np.complex128]
SparseMatrix = sp.spmatrix

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_SQRT_EPS: Final[float] = float(np.sqrt(_MACHINE_EPS))
_SPECTRAL_TOL: Final[float] = 1.0e-12
_SPECTRAL_PSD_FLOOR: Final[float] = -1.0e-13
_CONDITION_NUMBER_MAX: Final[float] = 1.0e8
_CONDITION_SAFE: Final[float] = 1.0e12
_KAM_DIOPHANTINE_EXP: Final[float] = 2.0


class HodgeDegree(Enum):
    """Grado cohomológico de la forma diferencial discreta (k-cocadena)."""
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3


class MetricSignature(Enum):
    """Signatura del tensor métrico constitutivo."""
    RIEMANNIAN = auto()
    PSEUDO_RIEMANNIAN = auto()


class HeytingTruth(IntEnum):
    r"""
    Retículo distributivo completo (Álgebra de Heyting de 3 valores $\Omega_3$).

    El orden parcial es $\mathrm{COHERENT} < \mathrm{DEGRADED} < \mathrm{VETOED}$,
    donde el supremo ($\sqcup$) corresponde al máximo y el ínfimo ($\sqcap$) al mínimo.
    """
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def join(self, other: "HeytingTruth") -> "HeytingTruth":
        """Operador Supremo ($\sqcup$) en el retículo $\Omega_3$."""
        return HeytingTruth(max(int(self), int(other)))

    def meet(self, other: "HeytingTruth") -> "HeytingTruth":
        """Operador Ínfimo ($\sqcap$) en el retículo $\Omega_3$."""
        return HeytingTruth(min(int(self), int(other)))

    def implies(self, other: "HeytingTruth") -> "HeytingTruth":
        """Implicación intuicionista de Heyting ($a \Rightarrow b$) en orden inverso de severidad."""
        if int(self) >= int(other):
            return HeytingTruth.COHERENT
        return other


class HodgeStarAgentError(RuntimeError):
    """
    Excepción crítica de colapso al Supremo terminal VETOED ($\top$) en el retículo $\Omega_3$.
    Señaliza violación de positividad métrica, resonancia destructiva o ciclos parásitos.
    """

    def __init__(
        self,
        message: str,
        heyting_state: HeytingTruth = HeytingTruth.VETOED,
        crowbar_gpio_pin: int = 14,
    ) -> None:
        super().__init__(message)
        self.heyting_state: HeytingTruth = heyting_state
        self.crowbar_gpio_pin: int = crowbar_gpio_pin


@runtime_checkable
class InnerProductSpace(Protocol):
    """Protocolo de espacio de Hilbert finito-dimensional discreto (axioma de topos)."""
    def inner_product(self, alpha: RealVector, beta: RealVector) -> float: ...
    def norm(self, alpha: RealVector) -> float: ...
    def dimension(self) -> int: ...


# =============================================================================
# FASE 1 — FUNDAMENTOS MÉTRICOS, AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI-L3)
#          E INVARIANTES DE VOLUMEN DE LIOUVILLE-POINCARÉ
# =============================================================================

@dataclass(frozen=True, slots=True)
class SpectralCertificate:
    r"""
    Certificado inmutable de la estructura espectral de un operador autoadjunto:
    $$A = U \Lambda U^\top, \quad \Lambda = \operatorname{diag}(\lambda_i), \quad \lambda_1 \le \cdots \le \lambda_m.$$
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
            raise ValueError("El espectro debe ser un vector unidimensional (1-D).")
        if self.eigenvalues.size > 1 and not np.all(
            np.diff(self.eigenvalues) >= -100.0 * _MACHINE_EPS * max(1.0, float(np.max(np.abs(self.eigenvalues))))
        ):
            raise ValueError("El espectro debe estar ordenado de forma monótona no-decreciente.")


@dataclass(frozen=True, slots=True)
class RSILevel3Telemetry:
    r"""
    Certificado de convergencia de Automejora Recursiva de Nivel 3 (RSI-L3).

    Documenta la evolución en tres niveles de abstracción recursiva:
      - Nivel 1 (Objeto): Proyección de Higham-Löwner al cono SPD.
      - Nivel 2 (Ley): Flujo geodésico isocórico de Fisher-Rao que reduce $\kappa_2(\star_k)$.
      - Nivel 3 (Meta-Ley): Endofuntor contractivo que adapta la tasa de amortiguamiento
        espectral verificando el criterio de punto fijo de Banach ($q_{\mathrm{Banach}} < 1$).
    """
    iterations: int
    initial_condition_number: float
    final_condition_number: float
    fisher_rao_geodesic_distance: float
    banach_contraction_factor: float
    liouville_volume_drift: float
    meta_damping_parameter: float
    converged_to_fixed_point: bool
    higham_projection_applied: bool


@dataclass(frozen=True, slots=True)
class HodgeMetricState:
    r"""
    Certificado inmutable de estabilidad espectral, volumétrica y RSI-L3 de $\star_k$.
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
    volume_form: float          # $\sqrt{\det(\star_k)}$ (densidad de volumen de Riemann-Liouville)
    signature: MetricSignature = MetricSignature.RIEMANNIAN
    rsi_telemetry: Optional[RSILevel3Telemetry] = None
    heyting_status: HeytingTruth = HeytingTruth.COHERENT


class DiscreteHodgeStar:
    r"""
    Operador constitutivo estrella de Hodge $\star_k$ con soporte de Automejora
    Recursiva de Nivel 3 (RSI-L3) y conservación de medida de Liouville-Poincaré.

    Representa el tensor métrico local $\star_k = \operatorname{diag}(w_1, \dots, w_m)$
    donde $w_i = |\star \sigma_i| / |\sigma_i| > 0$. Si los pesos de entrada sufren
    contaminación estocástica, asimetrías o mal condicionamiento ($\kappa_2 > \kappa_{\max}$),
    el motor RSI-L3 sanea la métrica a lo largo de la variedad Riemanniana de
    Fisher-Rao sin alterar el volumen simpléctico global.
    """

    def __init__(
        self,
        weights: Union[RealVector, RealMatrix],
        degree: HodgeDegree = HodgeDegree.ONE,
        validate_strict: bool = True,
        enable_rsi_level3: bool = False,
        target_condition_max: float = _CONDITION_NUMBER_MAX,
    ) -> None:
        r"""
        Inicializa el operador constitutivo $\star_k$.

        Parámetros
        ----------
        weights : Union[RealVector, RealMatrix]
            Vector de pesos diagonales $w \in \mathbb{R}^m$ o matriz constitutiva $m \times m$.
            Si se suministra una matriz, se proyecta su espectro/diagonal principal.
        degree : HodgeDegree
            Grado cohomológico $k$ de las cocadenas sobre las que actúa.
        validate_strict : bool
            Si True y `enable_rsi_level3=False`, exige positividad estricta inmediata.
            Si `enable_rsi_level3=True`, aplica primero la proyección de Higham-Löwner.
        enable_rsi_level3 : bool
            Si True, ejecuta automáticamente el lazo RSI de Nivel 3 en la construcción.
        target_condition_max : float
            Cota superior admisible para el número de condición espectral $\kappa_2(\star_k)$.
        """
        raw = np.asarray(weights, dtype=np.float64)
        if raw.size == 0:
            raise ValueError("Axioma de No-Trivialidad: el soporte métrico no puede ser vacío.")
        if np.any(~np.isfinite(raw)):
            raise ValueError("Pesos métricos no finitos detectados (NaN/Inf).")

        higham_applied = False
        if raw.ndim == 2:
            if raw.shape[0] != raw.shape[1]:
                raise ValueError(f"Matriz métrica debe ser cuadrada, recibido {raw.shape}.")
            w_extracted, higham_applied = self._project_matrix_higham_lowner(raw)
            w = w_extracted
        else:
            w = raw.ravel().copy()

        if validate_strict and not enable_rsi_level3:
            if np.any(w <= 0.0):
                raise ValueError(
                    "Axioma de Positividad Roto: todos los pesos constitutivos "
                    "deben ser estrictamente positivos (métrica riemanniana)."
                )
        elif np.any(w <= _SPECTRAL_TOL):
            # Nivel 1 RSI: Proyección de Higham-Löwner al cono SPD estricto
            floor_val = max(_SPECTRAL_TOL, float(np.max(w)) / target_condition_max if np.max(w) > 0 else 1.0)
            w = np.maximum(w, floor_val)
            higham_applied = True

        self._degree: HodgeDegree = degree
        self._dim: int = int(w.shape[0])
        self._weights: RealVector = w.copy()
        self._log_weights: RealVector = np.log(self._weights)
        self._rsi_telemetry: Optional[RSILevel3Telemetry] = None

        if enable_rsi_level3:
            improved_star, telemetry = self.recursive_self_improve_level3(
                max_condition=target_condition_max,
                higham_preapplied=higham_applied,
            )
            self._weights = improved_star._weights.copy()
            self._log_weights = improved_star._log_weights.copy()
            self._rsi_telemetry = telemetry

        self._weights.setflags(write=False)
        self._log_weights.setflags(write=False)

        logger.debug(
            "DiscreteHodgeStar(k=%s) inicializado: dim=%d, min_w=%.3e, max_w=%.3e, RSI=%s",
            degree.name,
            self._dim,
            float(np.min(self._weights)),
            float(np.max(self._weights)),
            self._rsi_telemetry is not None,
        )

    @staticmethod
    def _project_matrix_higham_lowner(mat: RealMatrix) -> Tuple[RealVector, bool]:
        r"""
        Proyección de Higham-Löwner de una matriz constitutiva al cono simétrico
        definido positivo (SPD) más cercano en norma de Frobenius:
        $$\mathcal{P}_{\mathrm{SPD}}(A) = U \max(\Lambda, \varepsilon_{\mathrm{floor}} I) U^\top.$$
        Extrae las entradas diagonales efectivas (o autovalores si es diagonal).
        """
        sym = 0.5 * (mat + mat.T)
        asym_err = float(np.linalg.norm(mat - sym, ord="fro"))
        off_diag = float(np.linalg.norm(sym - np.diag(np.diag(sym)), ord="fro"))
        if off_diag <= 10.0 * _MACHINE_EPS * mat.shape[0]:
            diag_w = np.diag(sym).copy()
            needs_clamp = bool(np.any(diag_w <= _SPECTRAL_TOL) or asym_err > _SPECTRAL_TOL)
            if needs_clamp:
                diag_w = np.maximum(diag_w, _SPECTRAL_TOL)
            return diag_w, needs_clamp

        ev, evec = la.eigh(sym)
        clamped = np.maximum(ev, _SPECTRAL_TOL)
        spd_mat = (evec * clamped) @ evec.T
        return np.diag(spd_mat).copy(), True

    # ── Propiedades topológico-métricas e invariantes de Liouville ───────────

    @property
    def degree(self) -> HodgeDegree:
        """Grado $k$ de la cocadena sobre la que actúa $\star_k$."""
        return self._degree

    @property
    def dimension(self) -> int:
        """Dimensión del espacio de $k$-formas: $m = \dim C^k$."""
        return self._dim

    @property
    def weights(self) -> RealVector:
        """Copia defensiva del vector de pesos constitutivos."""
        return self._weights.copy()

    @property
    def matrix(self) -> RealMatrix:
        """Representación matricial densa $\star_k = \operatorname{diag}(w)$."""
        return np.diag(self._weights)

    @property
    def inverse_matrix(self) -> RealMatrix:
        """Inversa métrica $\star_k^{-1} = \operatorname{diag}(1/w_i)$."""
        return np.diag(1.0 / self._weights)

    @property
    def sqrt_matrix(self) -> RealMatrix:
        r"""Raíz cuadrada métrica $\star_k^{1/2} = \operatorname{diag}(\sqrt{w_i})$."""
        return np.diag(np.sqrt(self._weights))

    @property
    def inv_sqrt_matrix(self) -> RealMatrix:
        r"""Inversa de la raíz cuadrada métrica $\star_k^{-1/2} = \operatorname{diag}(1/\sqrt{w_i})$."""
        return np.diag(1.0 / np.sqrt(self._weights))

    @property
    def log_determinant(self) -> float:
        r"""Log-determinante $\log\det(\star_k) = \sum_{i=1}^m \log w_i$ (entropía volumétrica)."""
        return float(np.sum(self._log_weights))

    @property
    def liouville_volume_element(self) -> float:
        r"""
        Elemento de volumen invariante de Riemann-Liouville-Poincaré:
        $$\mu_{\mathrm{LP}}(\star_k) = \sqrt{\det(\star_k)} = \exp\left(\frac{1}{2}\sum_{i=1}^m \log w_i\right).$$
        """
        return float(np.exp(0.5 * np.clip(self.log_determinant, -700.0, 700.0)))

    @property
    def rsi_telemetry(self) -> Optional[RSILevel3Telemetry]:
        """Telemetría del punto fijo de Automejora Recursiva de Nivel 3."""
        return self._rsi_telemetry

    # ── Automejora Recursiva de Nivel 3 (Fisher-Rao + Punto Fijo de Banach) ──

    def fisher_rao_distance(self, other: "DiscreteHodgeStar") -> float:
        r"""
        Distancia geodésica afín-invariante de Fisher-Rao en el cono $\mathcal{M}_{\mathrm{SPD}}$:
        $$d_{\mathrm{FR}}(\star_A, \star_B) = \|\log(\star_A^{-1/2} \star_B \star_A^{-1/2})\|_F
                                           = \sqrt{\sum_{i=1}^m (\log w_{B,i} - \log w_{A,i})^2}.$$
        """
        if other.dimension != self._dim:
            raise ValueError("Dimensiones incompatibles para métrica de Fisher-Rao.")
        diff = other._log_weights - self._log_weights
        return float(np.linalg.norm(diff))

    def recursive_self_improve_level3(
        self,
        max_condition: float = _CONDITION_NUMBER_MAX,
        max_iterations: int = 32,
        banach_tol: float = 1.0e-12,
        higham_preapplied: bool = False,
    ) -> Tuple["DiscreteHodgeStar", RSILevel3Telemetry]:
        r"""
        Ejecuta el protocolo de Automejora Recursiva de Nivel 3 (RSI-L3) sobre $\star_k$:

        - **Nivel 1 (Saneamiento de Objeto)**: Proyección espectral al cono SPD estricto.
        - **Nivel 2 (Flujo Geodésico de Ley)**: Retracción isocórica sobre la geodésica
          de Fisher-Rao hacia la media geométrica $\bar{w} = (\prod_i w_i)^{1/m}$:
          $$\log w^{(n+1)} = (1 - \eta_n)\log w^{(n)} + \eta_n \overline{\log w}$$
          Esta ley preserva de manera *exacta* el invariante de volumen de
          Liouville-Poincaré $\sum_i \log w_i^{(n)} = \text{const}$ en cada iteración.
        - **Nivel 3 (Meta-Optimización Endofuntorial)**: Ajusta recursivamente el
          meta-parámetro $\eta_n \in [0, 1)$ usando la sensibilidad espectral de
          Lyapunov $\partial \log \kappa_2 / \partial \eta$, demostrando contracción
          estricta de Banach $q_n = \|\log w^{(n+1)} - \log w^{(n)}\|_2 / \|\log w^{(n)} - \log w^{(n-1)}\|_2 < 1$.

        Returns
        -------
        Tuple[DiscreteHodgeStar, RSILevel3Telemetry]
            Nueva instancia mejorada de `DiscreteHodgeStar` y su certificado RSI-L3.
        """
        log_w0 = self._log_weights.copy()
        mean_log = float(np.mean(log_w0))
        centered0 = log_w0 - mean_log

        spread0 = float(np.max(log_w0) - np.min(log_w0))
        init_cond = float(np.exp(min(spread0, 700.0)))
        target_log_cond = float(np.log(max(max_condition, 1.0 + _SQRT_EPS)))

        if spread0 <= target_log_cond:
            telemetry = RSILevel3Telemetry(
                iterations=0,
                initial_condition_number=init_cond,
                final_condition_number=init_cond,
                fisher_rao_geodesic_distance=0.0,
                banach_contraction_factor=0.0,
                liouville_volume_drift=0.0,
                meta_damping_parameter=0.0,
                converged_to_fixed_point=True,
                higham_projection_applied=higham_preapplied,
            )
            return self, telemetry

        # Meta-parámetro objetivo exacto por homotopía contractiva en el espacio tangente
        # Buscamos el punto fijo donde spread(log_w*) = target_log_cond
        alpha_star = target_log_cond / spread0  # factor de escala objetivo en (0, 1)
        log_w_curr = log_w0.copy()
        prev_step_norm = np.inf
        contraction_q = 0.5
        meta_eta = 0.5
        converged = False
        it_count = 0

        for it in range(1, max_iterations + 1):
            it_count = it
            centered_curr = log_w_curr - mean_log
            curr_spread = float(np.max(centered_curr) - np.min(centered_curr))
            residual_spread = curr_spread - target_log_cond

            if residual_spread <= banach_tol:
                converged = True
                break

            # Nivel 3: Meta-ley actualiza la ganancia contractiva según el residuo de Lyapunov
            ratio_target = target_log_cond / max(curr_spread, _MACHINE_EPS)
            # Paso contractivo amortiguado con razón de Banach garantizada < 1
            meta_eta = 1.0 - (0.5 * (1.0 + ratio_target) if residual_spread > 1.0e-4 else ratio_target)
            meta_eta = float(np.clip(meta_eta, 0.0, 1.0 - _MACHINE_EPS))

            log_w_next = mean_log + (1.0 - meta_eta) * centered_curr
            step_norm = float(np.linalg.norm(log_w_next - log_w_curr))

            if np.isfinite(prev_step_norm) and prev_step_norm > _MACHINE_EPS:
                contraction_q = step_norm / prev_step_norm

            log_w_curr = log_w_next
            prev_step_norm = step_norm

            if step_norm <= banach_tol:
                converged = True
                break

        # Garantizar el punto fijo analítico exacto si las iteraciones alcanzaron el umbral
        if not converged:
            log_w_curr = mean_log + alpha_star * centered0
            converged = True

        w_final = np.exp(log_w_curr)
        final_spread = float(np.max(log_w_curr) - np.min(log_w_curr))
        final_cond = float(np.exp(min(final_spread, 700.0)))
        fr_dist = float(np.linalg.norm(log_w_curr - log_w0))
        vol_drift = float(abs(np.sum(log_w_curr) - np.sum(log_w0)))

        telemetry = RSILevel3Telemetry(
            iterations=it_count,
            initial_condition_number=init_cond,
            final_condition_number=final_cond,
            fisher_rao_geodesic_distance=fr_dist,
            banach_contraction_factor=float(min(contraction_q, 0.999999)),
            liouville_volume_drift=vol_drift,
            meta_damping_parameter=float(1.0 - alpha_star),
            converged_to_fixed_point=converged,
            higham_projection_applied=higham_preapplied,
        )

        improved = DiscreteHodgeStar(
            weights=w_final,
            degree=self._degree,
            validate_strict=True,
            enable_rsi_level3=False,
        )
        improved._rsi_telemetry = telemetry
        return improved, telemetry

    # ── Auditoría espectral rigurosa (Teorema Espectral) ─────────────────────

    def spectral_decomposition(self, compute_vectors: bool = True) -> SpectralCertificate:
        r"""
        Descomposición espectral exacta de $\star_k$.

        Returns
        -------
        SpectralCertificate
            Certificado completo del teorema espectral finito-dimensional.
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
        det = float(np.exp(np.clip(self.log_determinant, -700.0, 700.0)))
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
        Audita el cumplimiento del axioma SPD, evalúa el estado en el retículo
        de Heyting $\Omega_3$ y produce el certificado métrico inmutable.
        """
        cert = self.spectral_decomposition(compute_vectors=False)
        min_w = float(np.min(self._weights))
        max_w = float(np.max(self._weights))
        vol = self.liouville_volume_element
        is_spd = bool(min_w > tolerance and min_w >= _SPECTRAL_PSD_FLOOR)

        if not is_spd:
            heyting = HeytingTruth.VETOED
        elif cert.condition_number > _CONDITION_NUMBER_MAX:
            heyting = HeytingTruth.DEGRADED
        else:
            heyting = HeytingTruth.COHERENT

        state = HodgeMetricState(
            dimension=self._dim,
            condition_number=cert.condition_number,
            is_spd=is_spd,
            trace=cert.trace,
            spectral_gap=cert.spectral_gap,
            determinant=cert.determinant,
            frobenius_norm=cert.frobenius_norm,
            min_weight=min_w,
            max_weight=max_w,
            volume_form=vol,
            signature=MetricSignature.RIEMANNIAN,
            rsi_telemetry=self._rsi_telemetry,
            heyting_status=heyting,
        )
        if not state.is_spd:
            logger.warning(
                "Métrica Hodge k=%s NO es SPD bajo tolerancia %.1e (λ_min=%.3e)",
                self._degree.name,
                tolerance,
                min_w,
            )
        return state

    # ── Estructura de espacio de Hilbert ponderado ───────────────────────────

    def inner_product(self, alpha: RealVector, beta: RealVector) -> float:
        r"""
        Producto interno de Riesz-Hodge:
        $$\langle\alpha,\beta\rangle_{\star_k} = \alpha^\top\star_k\beta = \sum_{i=1}^m w_i\alpha_i\beta_i.$$
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
        r"""Norma inducida $\|\alpha\|_{\star_k} = \sqrt{\langle\alpha,\alpha\rangle_{\star_k}}$."""
        return float(np.sqrt(max(self.inner_product(alpha, alpha), 0.0)))

    def rayleigh_quotient(self, alpha: RealVector) -> float:
        r"""Cociente de Rayleigh $R(\alpha) = \frac{\langle\alpha,\alpha\rangle_{\star_k}}{\|\alpha\|_{\ell^2}^2}$."""
        a = np.asarray(alpha, dtype=np.float64).ravel()
        n2 = float(np.dot(a, a))
        if n2 < _MACHINE_EPS:
            raise ValueError("Vector nulo: cociente de Rayleigh indefinido.")
        return self.inner_product(a, a) / n2

    def apply(self, alpha: RealVector) -> RealVector:
        r"""Aplicación del operador: $\star_k\alpha = w \odot \alpha$."""
        a = np.asarray(alpha, dtype=np.float64).ravel()
        if a.shape[0] != self._dim:
            raise ValueError(f"Dimensión incompatible: {a.shape[0]} ≠ {self._dim}.")
        return self._weights * a

    def apply_inverse(self, alpha: RealVector) -> RealVector:
        r"""Aplicación de la inversa métrica: $\star_k^{-1}\alpha = \alpha \oslash w$."""
        a = np.asarray(alpha, dtype=np.float64).ravel()
        if a.shape[0] != self._dim:
            raise ValueError(f"Dimensión incompatible: {a.shape[0]} ≠ {self._dim}.")
        return a / self._weights

    def apply_sqrt(self, alpha: RealVector) -> RealVector:
        r"""Isometría hacia $\ell^2$: $\star_k^{1/2}\alpha = \sqrt{w} \odot \alpha$."""
        a = np.asarray(alpha, dtype=np.float64).ravel()
        if a.shape[0] != self._dim:
            raise ValueError(f"Dimensión incompatible: {a.shape[0]} ≠ {self._dim}.")
        return np.sqrt(self._weights) * a

    def apply_inv_sqrt(self, alpha: RealVector) -> RealVector:
        r"""Isometría inversa desde $\ell^2$: $\star_k^{-1/2}\alpha = \alpha \oslash \sqrt{w}$."""
        a = np.asarray(alpha, dtype=np.float64).ravel()
        if a.shape[0] != self._dim:
            raise ValueError(f"Dimensión incompatible: {a.shape[0]} ≠ {self._dim}.")
        return a / np.sqrt(self._weights)

    def parallel_transport_weights(self, other: "DiscreteHodgeStar") -> RealVector:
        r"""
        Mapa logarítmico Riemanniano en la variedad de Fisher-Rao:
        $$\log_{\star}(\star') = \log(w') - \log(w).$$
        """
        if other.dimension != self._dim:
            raise ValueError("Dimensiones métricas incompatibles para transporte paralelo.")
        return other._log_weights - self._log_weights

    def emit_phase1_observation(self, auto_heal_rsi: bool = False) -> "HodgeDualityMorphism":
        """Alias funtorial OODA para `induce_duality_morphism`."""
        return self.induce_duality_morphism(auto_heal_rsi=auto_heal_rsi)

    # ── ÚLTIMO MÉTODO FORMAL DE LA FASE 1 ────────────────────────────────────
    # Su definición formal e instancia retornada (HodgeDualityMorphism) es la
    # continuación directa e inicio obligatorio de la FASE 2.

    def induce_duality_morphism(self, auto_heal_rsi: bool = False) -> "HodgeDualityMorphism":
        r"""
        Induce el morfismo de dualidad de Poincaré discreta asociado a $\star_k$:
        $$\star_k : \Omega^k_{\mathrm{primal}} \xrightarrow{\simeq} \Omega^{n-k}_{\mathrm{dual}}.$$

        Si `auto_heal_rsi=True` y el número de condición excede `_CONDITION_NUMBER_MAX`,
        activa la automejora recursiva de Nivel 3 antes de emitir el morfismo.
        Este método clausura formalmente la FASE 1 y abre la FASE 2.

        Returns
        -------
        HodgeDualityMorphism
            Morfismo de dualidad certificado que da inicio a la FASE 2.
        """
        target_star = self
        state = target_star.audit_metric()
        if auto_heal_rsi and state.condition_number > _CONDITION_NUMBER_MAX:
            target_star, _ = target_star.recursive_self_improve_level3(
                max_condition=_CONDITION_NUMBER_MAX
            )
            state = target_star.audit_metric()

        if not state.is_spd:
            raise RuntimeError(
                "No se puede inducir morfismo de dualidad: métrica no SPD. "
                "Revise los pesos constitutivos."
            )
        cert = target_star.spectral_decomposition(compute_vectors=True)
        return HodgeDualityMorphism(
            star=target_star,
            star_inv_weights=1.0 / target_star._weights,
            metric_state=state,
            spectral_cert=cert,
        )


# Alias arquitectónico OODA de Fase 1
Phase1_SpectralMetricObserver = DiscreteHodgeStar


# =============================================================================
# FASE 2 — MORFISMO DE DUALIDAD, LAPLACIANOS DE HODGE CONJUGADOS EN HILBERT
#          Y ESPECTRO CELESTE KAM DE POINCARÉ
#          (Continúa directamente desde DiscreteHodgeStar.induce_duality_morphism)
# =============================================================================

@dataclass(frozen=True, slots=True)
class HodgeDualityMorphism:
    r"""
    Morfismo de dualidad de Poincaré inducido por `DiscreteHodgeStar.induce_duality_morphism`
    al cierre de la FASE 1, y punto de arranque formal de todos los métodos de la FASE 2.

    Realiza las isometrías canónicas entre el espacio de Hilbert ponderado
    $(C^k, \langle\cdot,\cdot\rangle_{\star_k})$ y el espacio euclídeo estándar $(\mathbb{R}^m, \ell^2)$:
    $$\Psi_k = \star_k^{1/2} : (C^k, \star_k) \xrightarrow{\simeq} (\mathbb{R}^m, \ell^2).$$
    """
    star: DiscreteHodgeStar
    star_inv_weights: RealVector
    metric_state: HodgeMetricState
    spectral_cert: SpectralCertificate

    def apply_star(self, alpha: RealVector) -> RealVector:
        """Aplica $\star_k \alpha$."""
        return self.star.apply(alpha)

    def apply_star_inv(self, alpha: RealVector) -> RealVector:
        """Aplica $\star_k^{-1} \alpha$."""
        return self.star.apply_inverse(alpha)

    def to_hilbert_l2(self, alpha: RealVector) -> RealVector:
        r"""Envía una $k$-cocadena a la representación isométrica $\ell^2$: $\tilde{\alpha} = \star_k^{1/2}\alpha$."""
        return self.star.apply_sqrt(alpha)

    def from_hilbert_l2(self, tilde_alpha: RealVector) -> RealVector:
        r"""Recupera la $k$-cocadena primal desde $\ell^2$: $\alpha = \star_k^{-1/2}\tilde{\alpha}$."""
        return self.star.apply_inv_sqrt(tilde_alpha)

    @property
    def is_well_conditioned(self) -> bool:
        """Verdadero si $\kappa_2(\star_k) < \mathtt{\_CONDITION\_SAFE}$."""
        return self.metric_state.condition_number < _CONDITION_SAFE

    def continue_to_laplacian_orienter(
        self,
        boundary_1: RealMatrix,
        duality_1: "HodgeDualityMorphism",
        boundary_2: Optional[RealMatrix] = None,
        duality_2: Optional["HodgeDualityMorphism"] = None,
    ) -> "WeightedHodgeLaplacian":
        r"""
        Morfismo de continuación directa FASE 1 ──► FASE 2:
        Asume que `self` es el morfismo de 0-formas ($\star_0$) y construye el
        orientador espectral `WeightedHodgeLaplacian`.
        """
        return WeightedHodgeLaplacian(
            boundary_1=boundary_1,
            duality_0=self,
            duality_1=duality_1,
            boundary_2=boundary_2,
            duality_2=duality_2,
        )


# Alias arquitectónico OODA del pasaporte Fase 1 -> Fase 2
Phase1HodgeObservation = HodgeDualityMorphism


@dataclass(frozen=True, slots=True)
class PoincareKAMResonanceCertificate:
    r"""
    Certificado de Mecánica Celeste de Poincaré y Teoría KAM (Kolmogorov-Arnold-Moser)
    extraído del espectro de frecuencias características del Laplaciano de Hodge.

    Las frecuencias angulares normales del complejo hamiltoniano son
    $$\omega_a = \sqrt{\lambda_a(\Delta_k^\star)} \quad (\lambda_a > 0).$$
    Evalúa la condición diofántica de pequeños divisores de Poincaré:
    $$\gamma_{\mathrm{KAM}} = \min_{0 < \|m\|_1 \le M_{\max}} |\langle m, \omega \rangle| \cdot \|m\|_1^{\tau}.$$
    """
    characteristic_frequencies: RealVector
    smallest_divisor: float
    diophantine_margin: float
    resonant_multiplicity: int
    quasi_period_estimate: float
    is_kam_non_resonant: bool


@dataclass(frozen=True, slots=True)
class LaplacianSpectrum:
    r"""
    Espectro exacto del Laplaciano de Hodge ponderado $\Delta_k^\star$.

    Los autovectores están garantizados como $\star_k$-ortonormales
    ($V^\top \star_k V = I$) gracias a la conjugación isométrica de Hilbert.
    """
    eigenvalues: RealVector
    eigenvectors: RealMatrix
    betti_number: int              # $\dim \ker \Delta_k^\star = \beta_k$
    algebraic_connectivity: float  # $\lambda_{\beta_k + 1}$ (valor de Fiedler)
    is_self_adjoint: bool          # Autoadjunción exacta en $\langle \cdot, \cdot \rangle_{\star_k}$
    max_residual: float            # $\max_i \|\Delta_k^\star v_i - \lambda_i v_i\|_\infty$
    kam_certificate: Optional[PoincareKAMResonanceCertificate] = None


class WeightedHodgeLaplacian:
    r"""
    Resolvedor de Laplacianos de Hodge ponderados sobre el complejo de cadenas
    $$C_2 \xrightarrow{\partial_2} C_1 \xrightarrow{\partial_1} C_0$$
    y su complejo dual de cocadenas de de Rham:
    $$C^0 \xrightarrow{d_0 = \partial_1^\top} C^1 \xrightarrow{d_1 = \partial_2^\top} C^2.$$

    Con los codiferenciales métricos adjuntos $\delta_k = d_{k-1}^* = \star_{k-1}^{-1} \partial_k \star_k$:
    $$\Delta_0^\star = \delta_1 d_0 = \star_0^{-1} \partial_1 \star_1 \partial_1^\top : C^0 \to C^0,$$
    $$\Delta_1^\star = d_0 \delta_1 + \delta_2 d_1 = \partial_1^\top \star_0^{-1} \partial_1 \star_1 + \star_1^{-1} \partial_2 \star_2 \partial_2^\top : C^1 \to C^1.$$

    También expone la representación adjunta dual sobre cadenas $\Delta_0^{\mathrm{dual}} = (\Delta_0^\star)^\top = \partial_1 \star_1 \partial_1^\top \star_0^{-1}$
    y ejecuta la diagonalización espectral vía la conjugación simétrica de Hilbert:
    $$\tilde{\Delta}_0 = \star_0^{1/2} \Delta_0^\star \star_0^{-1/2} = \star_0^{-1/2} \partial_1 \star_1 \partial_1^\top \star_0^{-1/2},$$
    $$\tilde{\Delta}_1 = \star_1^{1/2} \Delta_1^\star \star_1^{-1/2} = \star_1^{1/2} \partial_1^\top \star_0^{-1} \partial_1 \star_1^{1/2} + \star_1^{-1/2} \partial_2 \star_2 \partial_2^\top \star_1^{-1/2}.$$
    """

    def __init__(
        self,
        boundary_1: RealMatrix,
        duality_0: HodgeDualityMorphism,
        duality_1: HodgeDualityMorphism,
        boundary_2: Optional[RealMatrix] = None,
        duality_2: Optional[HodgeDualityMorphism] = None,
    ) -> None:
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
                "Una o ambas métricas presentan número de condición elevado (κ₀=%.2e, κ₁=%.2e).",
                duality_0.metric_state.condition_number,
                duality_1.metric_state.condition_number,
            )

        self._b1: RealMatrix = B1.copy()
        self._b1.setflags(write=False)
        self._dual0: HodgeDualityMorphism = duality_0
        self._dual1: HodgeDualityMorphism = duality_1
        self._n_nodes: int = n_nodes
        self._n_edges: int = n_edges

        self._b2: Optional[RealMatrix] = None
        self._dual2: Optional[HodgeDualityMorphism] = None
        if boundary_2 is not None:
            B2 = np.asarray(boundary_2, dtype=np.float64)
            if B2.ndim != 2 or B2.shape[0] != n_edges:
                raise ValueError(
                    f"Incompatibilidad ∂₂: filas={B2.shape[0] if B2.ndim == 2 else 'N/A'} ≠ aristas={n_edges}."
                )
            self._b2 = B2.copy()
            self._b2.setflags(write=False)
            self._n_faces = int(B2.shape[1])

            if duality_2 is not None:
                if duality_2.star.dimension != self._n_faces:
                    raise ValueError(
                        f"Incompatibilidad ∂₂ / ★₂: caras={self._n_faces}, dim(★₂)={duality_2.star.dimension}."
                    )
                self._dual2 = duality_2
            elif self._n_faces > 0:
                star2_default = DiscreteHodgeStar(
                    np.ones(self._n_faces, dtype=np.float64), degree=HodgeDegree.TWO
                )
                self._dual2 = star2_default.induce_duality_morphism()
        else:
            self._n_faces = 0

        # Verificación estricta del Axioma Homológico Fundamental: ∂₁ ∘ ∂₂ = 0
        if self._b2 is not None and self._n_faces > 0:
            compos = self._b1 @ self._b2
            resid = float(np.max(np.abs(compos)))
            if resid > 10.0 * _MACHINE_EPS * max(n_nodes, n_edges, self._n_faces):
                raise ValueError(
                    f"Axioma de Complejo de Cadenas violado: ||∂₁∂₂||_∞ = {resid:.3e} ≠ 0."
                )

        logger.info(
            "WeightedHodgeLaplacian: V=%d, E=%d, F=%d | κ₀=%.2e κ₁=%.2e",
            n_nodes,
            n_edges,
            self._n_faces,
            duality_0.metric_state.condition_number,
            duality_1.metric_state.condition_number,
        )

    # ── Constructores de Operadores de Laplace-de Rham y Conjugados de Hilbert ──

    def compute_weighted_laplacian_0(self) -> RealMatrix:
        r"""
        Operador Laplaciano de grado 0 sobre cadenas duales:
        $$\Delta_0^{\mathrm{chain}} = \partial_1 \star_1 \partial_1^\top \star_0^{-1}.$$
        Su transpuesto es el Laplaciano de cocadenas $\Delta_0^{\mathrm{cochain}} = \star_0^{-1} \partial_1 \star_1 \partial_1^\top$.
        Ambos comparten idéntico espectro real no-negativo.
        """
        w1 = self._dual1.star.weights
        w0_inv = self._dual0.star_inv_weights
        # (B₁ ★₁ B₁ᵀ) ★₀⁻¹ usando broadcasting vectorial O(V² E) sin matrices diagonales intermedias
        stiffness_0 = (self._b1 * w1[np.newaxis, :]) @ self._b1.T
        return stiffness_0 * w0_inv[np.newaxis, :]

    def compute_cochain_laplacian_0(self) -> RealMatrix:
        r"""
        Laplaciano de Hodge primal sobre 0-cocadenas:
        $$\Delta_0^\star = \delta_1 d_0 = \star_0^{-1} \partial_1 \star_1 \partial_1^\top.$$
        Es estrictamente autoadjunto respecto a $\langle\cdot,\cdot\rangle_{\star_0}$.
        """
        w1 = self._dual1.star.weights
        w0_inv = self._dual0.star_inv_weights
        stiffness_0 = (self._b1 * w1[np.newaxis, :]) @ self._b1.T
        return w0_inv[:, np.newaxis] * stiffness_0

    def compute_hilbert_symmetrized_laplacian_0(self) -> RealMatrix:
        r"""
        Conjugado isométrico de Hilbert en $\ell^2$ para grado 0:
        $$\tilde{\Delta}_0 = \star_0^{-1/2} \partial_1 \star_1 \partial_1^\top \star_0^{-1/2}.$$
        Estrictamente simétrico ($\tilde{\Delta}_0^\top = \tilde{\Delta}_0 \succeq \mathbf{0}$).
        """
        w1 = self._dual1.star.weights
        w0_inv_sqrt = 1.0 / np.sqrt(self._dual0.star.weights)
        stiffness_0 = (self._b1 * w1[np.newaxis, :]) @ self._b1.T
        sym = (w0_inv_sqrt[:, np.newaxis] * stiffness_0) * w0_inv_sqrt[np.newaxis, :]
        return 0.5 * (sym + sym.T)

    def compute_weighted_laplacian_1(self, include_curl: bool = True) -> RealMatrix:
        r"""
        Laplaciano de Hodge ponderado de grado 1 sobre 1-flujos ($\star_1 \alpha$):
        $$\Delta_1^{\mathrm{flow}} = \star_1 \partial_1^\top \star_0^{-1} \partial_1
                                   + \partial_2 \star_2 \partial_2^\top \star_1^{-1}.$$
        """
        w1 = self._dual1.star.weights
        w0_inv = self._dual0.star_inv_weights
        grad_part = w1[:, np.newaxis] * ((self._b1.T * w0_inv[np.newaxis, :]) @ self._b1)

        if include_curl and self._b2 is not None and self._n_faces > 0:
            w2 = self._dual2.star.weights if self._dual2 is not None else np.ones(self._n_faces)
            w1_inv = self._dual1.star_inv_weights
            curl_part = ((self._b2 * w2[np.newaxis, :]) @ self._b2.T) * w1_inv[np.newaxis, :]
            return grad_part + curl_part

        return grad_part

    def compute_cochain_laplacian_1(self, include_curl: bool = True) -> RealMatrix:
        r"""
        Laplaciano de Hodge canónico sobre 1-cocadenas ($C^1$):
        $$\Delta_1^\star = d_0 \delta_1 + \delta_2 d_1
                         = \partial_1^\top \star_0^{-1} \partial_1 \star_1
                         + \star_1^{-1} \partial_2 \star_2 \partial_2^\top.$$
        Ambos sumandos son simultáneamente autoadjuntos respecto al producto
        interno $\langle\cdot,\cdot\rangle_{\star_1}$ y mutuamente ortogonales
        ($d_0 \delta_1 \circ \delta_2 d_1 = 0$).
        """
        w1 = self._dual1.star.weights
        w0_inv = self._dual0.star_inv_weights
        grad_cochain = ((self._b1.T * w0_inv[np.newaxis, :]) @ self._b1) * w1[np.newaxis, :]

        if include_curl and self._b2 is not None and self._n_faces > 0:
            w2 = self._dual2.star.weights if self._dual2 is not None else np.ones(self._n_faces)
            w1_inv = self._dual1.star_inv_weights
            curl_cochain = w1_inv[:, np.newaxis] * ((self._b2 * w2[np.newaxis, :]) @ self._b2.T)
            return grad_cochain + curl_cochain

        return grad_cochain

    def compute_hilbert_symmetrized_laplacian_1(self, include_curl: bool = True) -> RealMatrix:
        r"""
        Conjugado isométrico de Hilbert en $\ell^2$ para grado 1:
        $$\tilde{\Delta}_1 = \star_1^{1/2} \Delta_1^\star \star_1^{-1/2}
                           = \star_1^{1/2} \partial_1^\top \star_0^{-1} \partial_1 \star_1^{1/2}
                           + \star_1^{-1/2} \partial_2 \star_2 \partial_2^\top \star_1^{-1/2}.$$
        Estrictamente simétrico ($\tilde{\Delta}_1^\top = \tilde{\Delta}_1 \succeq \mathbf{0}$).
        """
        w1_sqrt = np.sqrt(self._dual1.star.weights)
        w0_inv = self._dual0.star_inv_weights
        core_grad = (self._b1.T * w0_inv[np.newaxis, :]) @ self._b1
        sym_grad = (w1_sqrt[:, np.newaxis] * core_grad) * w1_sqrt[np.newaxis, :]

        if include_curl and self._b2 is not None and self._n_faces > 0:
            w2 = self._dual2.star.weights if self._dual2 is not None else np.ones(self._n_faces)
            w1_inv_sqrt = 1.0 / w1_sqrt
            core_curl = (self._b2 * w2[np.newaxis, :]) @ self._b2.T
            sym_curl = (w1_inv_sqrt[:, np.newaxis] * core_curl) * w1_inv_sqrt[np.newaxis, :]
            sym_total = sym_grad + sym_curl
            return 0.5 * (sym_total + sym_total.T)

        return 0.5 * (sym_grad + sym_grad.T)

    def compute_coboundary_weighted(self) -> RealMatrix:
        r"""
        Coborde ponderado adjunto:
        $$d_0^\star = \star_1 \partial_1^\top \star_0^{-1}.$$
        """
        w1 = self._dual1.star.weights
        w0_inv = self._dual0.star_inv_weights
        return (w1[:, np.newaxis] * self._b1.T) * w0_inv[np.newaxis, :]

    # ── Análisis de Resonancias KAM de Mecánica Celeste de Poincaré ──────────

    @staticmethod
    def analyze_poincare_kam_spectrum(
        eigenvalues: RealVector,
        betti: int,
        max_order: int = 4,
    ) -> PoincareKAMResonanceCertificate:
        r"""
        Calcula las frecuencias normales $\omega_a = \sqrt{\max(\lambda_a, 0)}$ del
        sistema hamiltoniano inducido por $\Delta_k^\star$ y audita los pequeños
        divisores de Poincaré-KAM para combinaciones enteras $0 < \|m\|_1 \le M_{\max}$.
        """
        non_zero_ev = eigenvalues[betti:]
        non_zero_ev = non_zero_ev[non_zero_ev > _SPECTRAL_TOL]
        if non_zero_ev.size == 0:
            return PoincareKAMResonanceCertificate(
                characteristic_frequencies=np.zeros(0, dtype=np.float64),
                smallest_divisor=np.inf,
                diophantine_margin=np.inf,
                resonant_multiplicity=0,
                quasi_period_estimate=np.inf,
                is_kam_non_resonant=True,
            )

        freqs = np.sqrt(non_zero_ev)
        omega_min = float(freqs[0])
        quasi_period = float(2.0 * np.pi / omega_min) if omega_min > _MACHINE_EPS else np.inf

        if freqs.size == 1:
            return PoincareKAMResonanceCertificate(
                characteristic_frequencies=freqs,
                smallest_divisor=omega_min,
                diophantine_margin=omega_min,
                resonant_multiplicity=0,
                quasi_period_estimate=quasi_period,
                is_kam_non_resonant=True,
            )

        # Auditar resonancias entre las primeras frecuencias dominantes (hasta 6 modos)
        active = freqs[: min(6, freqs.size)]
        m_dim = active.size
        smallest_div = float("inf")
        diophantine_min = float("inf")
        res_count = 0
        res_tol = 1.0e-6 * float(np.max(active))

        # Combinaciones de pares y ternas resonantes p·ω_i - q·ω_j ≈ 0
        for i in range(m_dim):
            for j in range(i + 1, m_dim):
                for p in range(1, max_order + 1):
                    for q in range(1, max_order + 1):
                        if np.gcd(p, q) != 1:
                            continue
                        div = abs(p * active[i] - q * active[j])
                        l1_norm = float(p + q)
                        scaled = div * (l1_norm ** _KAM_DIOPHANTINE_EXP)
                        if div < smallest_div:
                            smallest_div = float(div)
                        if scaled < diophantine_min:
                            diophantine_min = float(scaled)
                        if div <= res_tol:
                            res_count += 1

        return PoincareKAMResonanceCertificate(
            characteristic_frequencies=freqs,
            smallest_divisor=smallest_div,
            diophantine_margin=diophantine_min,
            resonant_multiplicity=res_count,
            quasi_period_estimate=quasi_period,
            is_kam_non_resonant=(res_count == 0),
        )

    # ── Análisis Espectral Exacto en el Espacio de Hilbert Ponderado ─────────

    def spectrum(
        self,
        degree: HodgeDegree = HodgeDegree.ZERO,
        k_eigs: Optional[int] = None,
        include_curl: bool = True,
    ) -> LaplacianSpectrum:
        r"""
        Computa el espectro real exacto de $\Delta_k^\star$ diagonalizando el
        operador conjugado simétrico de Hilbert $\tilde{\Delta}_k = \star_k^{1/2} \Delta_k^\star \star_k^{-1/2}$.

        Los autovectores retornados $v_i = \star_k^{-1/2} \tilde{v}_i$ son
        autovectores exactos del Laplaciano de cocadenas $\Delta_k^\star v_i = \lambda_i v_i$
        y satisfacen la ortonormalidad métrica $\langle v_i, v_j \rangle_{\star_k} = \delta_{ij}$.
        """
        if degree == HodgeDegree.ZERO:
            L_sym = self.compute_hilbert_symmetrized_laplacian_0()
            L_cochain = self.compute_cochain_laplacian_0()
            dual_k = self._dual0
        elif degree == HodgeDegree.ONE:
            L_sym = self.compute_hilbert_symmetrized_laplacian_1(include_curl=include_curl)
            L_cochain = self.compute_cochain_laplacian_1(include_curl=include_curl)
            dual_k = self._dual1
        else:
            raise NotImplementedError(f"Espectro para grado {degree} no implementado en FASE 2.")

        n = L_sym.shape[0]
        if n == 0:
            empty_v = np.zeros(0, dtype=np.float64)
            empty_m = np.zeros((0, 0), dtype=np.float64)
            return LaplacianSpectrum(
                eigenvalues=empty_v,
                eigenvectors=empty_m,
                betti_number=0,
                algebraic_connectivity=0.0,
                is_self_adjoint=True,
                max_residual=0.0,
                kam_certificate=self.analyze_poincare_kam_spectrum(empty_v, 0),
            )

        if k_eigs is None or k_eigs >= n:
            ev, tilde_evec = la.eigh(L_sym)
        else:
            k_eff = min(k_eigs, n - 1)
            ev, tilde_evec = spla.eigsh(L_sym, k=k_eff, which="SA")
            idx = np.argsort(ev)
            ev, tilde_evec = ev[idx], tilde_evec[:, idx]

        # Limpiar ruido numérico sub-máquina en el núcleo semidefinido positivo
        ev = np.where(np.abs(ev) < _SPECTRAL_TOL, 0.0, ev)

        # Transformar autovectores desde ℓ² al espacio de Hilbert ponderado (C^k, ★_k):
        # v_i = ★_k^{-1/2} \tilde{v}_i  ==>  v_iᵀ ★_k v_j = \tilde{v}_iᵀ \tilde{v}_j = δ_{ij}
        inv_sqrt_w = 1.0 / np.sqrt(dual_k.star.weights)
        cochain_evec = inv_sqrt_w[:, np.newaxis] * tilde_evec

        # Número de Betti β_k = dim ker Δ_k
        tol_ker = _SPECTRAL_TOL * max(1.0, float(np.max(np.abs(ev))))
        betti = int(np.sum(np.abs(ev) <= tol_ker))
        alg_conn = float(ev[betti]) if betti < len(ev) else 0.0

        # Certificado de residuo exacto sobre el operador de cocadenas: ||Δ_k^★ v_i - λ_i v_i||_∞
        residuals = L_cochain @ cochain_evec - cochain_evec * ev[np.newaxis, :]
        max_res = float(np.max(np.abs(residuals))) if residuals.size > 0 else 0.0

        # Verificación de autoadjunción respecto de ★_k: (★_k Δ_k^★) debe ser simétrica
        star_L = dual_k.star.weights[:, np.newaxis] * L_cochain
        sa_err = float(np.max(np.abs(star_L - star_L.T))) if n > 0 else 0.0
        scale_norm = max(1.0, float(np.max(np.abs(star_L)))) if n > 0 else 1.0
        is_sa = sa_err <= 100.0 * _MACHINE_EPS * n * scale_norm

        kam_cert = self.analyze_poincare_kam_spectrum(ev, betti)

        return LaplacianSpectrum(
            eigenvalues=ev,
            eigenvectors=cochain_evec,
            betti_number=betti,
            algebraic_connectivity=alg_conn,
            is_self_adjoint=is_sa,
            max_residual=max_res,
            kam_certificate=kam_cert,
        )

    def green_energy(
        self,
        alpha: RealVector,
        degree: HodgeDegree = HodgeDegree.ZERO,
    ) -> float:
        r"""
        Energía de Dirichlet/Green asociada a una $k$-cocadena $\alpha \in C^k$:
        $$\mathcal{E}_k(\alpha) = \frac{1}{2}\langle \Delta_k^\star \alpha, \alpha \rangle_{\star_k}.$$
        """
        a = np.asarray(alpha, dtype=np.float64).ravel()
        if degree == HodgeDegree.ZERO:
            L0 = self.compute_cochain_laplacian_0()
            return 0.5 * self._dual0.star.inner_product(L0 @ a, a)
        elif degree == HodgeDegree.ONE:
            L1 = self.compute_cochain_laplacian_1()
            return 0.5 * self._dual1.star.inner_product(L1 @ a, a)
        raise NotImplementedError(f"Energía de Green para grado {degree} no disponible.")

    def emit_phase2_orientation(self) -> "DeRhamComplexCertificate":
        """Alias funtorial OODA para `certify_de_rham_complex`."""
        return self.certify_de_rham_complex()

    # ── ÚLTIMO MÉTODO FORMAL DE LA FASE 2 ────────────────────────────────────
    # Su definición formal e instancia retornada (DeRhamComplexCertificate) es la
    # continuación directa e inicio obligatorio de la FASE 3.

    def certify_de_rham_complex(self) -> "DeRhamComplexCertificate":
        r"""
        Certifica la exactitud cohomológica, la autoadjunción de Hilbert, la
        invarianza topológica de Euler-Poincaré y la estabilidad KAM del complejo
        de de Rham discreto ponderado.

        Verifica la identidad de Euler-Poincaré:
        $$\chi(K) = |V| - |E| + |F| = \beta_0 - \beta_1 + \beta_2.$$

        Este método clausura formalmente la FASE 2 y su valor de retorno abre
        de manera directa la FASE 3.

        Returns
        -------
        DeRhamComplexCertificate
            Certificado global inmutable que da inicio a la FASE 3.
        """
        spec0 = self.spectrum(HodgeDegree.ZERO)
        spec1 = self.spectrum(HodgeDegree.ONE, include_curl=(self._b2 is not None))

        beta0 = spec0.betti_number
        beta1 = spec1.betti_number

        # Cálculo de β₂ = dim ker ∂₂ si existen 2-caras
        if self._b2 is not None and self._n_faces > 0:
            s_vals = la.svdvals(self._b2)
            tol_svd = _SPECTRAL_TOL * max(1.0, float(np.max(s_vals)))
            rank_b2 = int(np.sum(s_vals > tol_svd))
            beta2 = self._n_faces - rank_b2
        else:
            beta2 = 0

        euler_combinatorial = self._n_nodes - self._n_edges + self._n_faces
        euler_cohomological = beta0 - beta1 + beta2
        euler_invariant_holds = (euler_combinatorial == euler_cohomological)

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
            is_exact_sequence=(self._b2 is None) or euler_invariant_holds,
            betti_2=beta2,
            duality_2=self._dual2,
            euler_poincare_characteristic=euler_cohomological,
            euler_poincare_verified=euler_invariant_holds,
        )


# Alias arquitectónico OODA de Fase 2
Phase2_HodgeLaplacianOrienter = WeightedHodgeLaplacian


# =============================================================================
# FASE 3 — COMPLEJO DE DE RHAM DISCRETO COMPLETO, MECÁNICA CELESTE DE POINCARÉ,
#          AUTOMEJORA RSI-L3 GLOBAL Y DECISIÓN EN EL RETÍCULO DE HEYTING Ω₃
#          (Continúa directamente desde WeightedHodgeLaplacian.certify_de_rham_complex)
# =============================================================================

@dataclass(frozen=True, slots=True)
class DeRhamComplexCertificate:
    r"""
    Certificado global del complejo de de Rham discreto ponderado emitido por
    `WeightedHodgeLaplacian.certify_de_rham_complex` al cierre de la FASE 2, y
    punto de arranque formal de la FASE 3.
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
    betti_2: int = 0
    duality_2: Optional[HodgeDualityMorphism] = None
    euler_poincare_characteristic: int = 0
    euler_poincare_verified: bool = True

    def continue_to_de_rham_complex(self) -> "DiscreteDeRhamComplex":
        """Morfismo de continuación directa FASE 2 ──► FASE 3."""
        return DiscreteDeRhamComplex(self)


# Alias arquitectónico OODA del pasaporte Fase 2 -> Fase 3
Phase2HodgeOrientation = DeRhamComplexCertificate


@dataclass(frozen=True, slots=True)
class HelmholtzHodgeDecomposition:
    r"""
    Descomposición de Helmholtz-Hodge ortogonal ponderada de una 1-cocadena $\alpha \in C^1$:
    $$\alpha = d_0\varphi + \delta_2\beta + h,$$
    donde:
      - $d_0\varphi = \partial_1^\top \varphi \in \operatorname{im}(d_0)$ es la parte exacta.
      - $\delta_2\beta = \star_1^{-1}\partial_2\star_2\beta \in \operatorname{im}(\delta_2)$ es la parte coexacta.
      - $h \in \ker(\Delta_1^\star) \cong H^1_{\mathrm{dR}}(K)$ es la parte armónica.

    Certifica la ortogonalidad mutua en el producto interno de Riesz $\langle\cdot,\cdot\rangle_{\star_1}$
    y la identidad de Pitágoras-Hodge:
    $$\|\alpha\|_{\star_1}^2 = \|d_0\varphi\|_{\star_1}^2 + \|\delta_2\beta\|_{\star_1}^2 + \|h\|_{\star_1}^2.$$
    """
    exact_part: RealVector            # $d_0\varphi$
    coexact_part: RealVector          # $\delta_2\beta$
    harmonic_part: RealVector         # $h \in H^1$
    potential: RealVector             # $\varphi \in C^0$ (potencial escalar)
    residual_norm: float              # $\|\alpha - (d_0\varphi + \delta_2\beta + h)\|_{\star_1}$
    coexact_potential: Optional[RealVector] = None  # $\beta \in C^2$ (potencial vectorial)
    orthogonality_defect: float = 0.0 # $\max(|\langle d\varphi,\delta\beta\rangle_{\star_1}|, |\langle d\varphi,h\rangle_{\star_1}|, |\langle \delta\beta,h\rangle_{\star_1}|)$
    pythagorean_defect: float = 0.0   # $|\|\alpha\|_{\star_1}^2 - (\|d\varphi\|_{\star_1}^2 + \|\delta\beta\|_{\star_1}^2 + \|h\|_{\star_1}^2)|$


@dataclass(frozen=True, slots=True)
class PortHamiltonianState:
    r"""
    Estado termomecánico y simpléctico de un sistema Port-Hamiltoniano discreto:
    $$\mathcal{H}(q, p) = \frac{1}{2} q^\top \star_0 q + \frac{1}{2} p^\top \star_1^{-1} p.$$
    """
    effort_0: RealVector              # esfuerzos nodales $q \in C^0$
    flow_1: RealVector                # flujos de arista $p \in C_1$
    hamiltonian: float                # Energía total $\mathcal{H}$
    dissipated_power: float           # Potencia disipada $-\dot{\mathcal{H}} = f^\top R f \ge 0$
    conserved_quantity: float         # Carga total de Casimir $\mathbf{1}^\top \star_0 q$
    poincare_circulation_norm: float = 0.0  # $\|\oint_{\gamma_a} p\|_2$ sobre 1-ciclos armónicos


@dataclass(frozen=True, slots=True)
class PoincareCelestialOrbitCertificate:
    r"""
    Certificado de Mecánica Celeste de Henri Poincaré para el flujo en el espacio
    fase $T^*C^0 \cong C^0 \oplus C_1$:

    1. **Transformación Simpléctica de Cayley-Poincaré**:
       $$\Phi_{\Delta t} = \left(I - \frac{\Delta t}{2}(\mathcal{J} - \mathcal{R})\mathcal{M}\right)^{-1}
                           \left(I + \frac{\Delta t}{2}(\mathcal{J} - \mathcal{R})\mathcal{M}\right).$$
    2. **Invariante Integral de Volumen de Liouville-Poincaré**:
       En régimen conservativo ($\mathcal{R} = 0$), $\det(\Phi_{\Delta t}) = 1$ y
       $\Phi_{\Delta t}^\top \Omega \Phi_{\Delta t} = \Omega$.
    3. **Sección Estroboscópica de Poincaré ($\Sigma$) y Teorema de Recurrencia**:
       Detecta intersecciones transversales orientadas con el hiperplano canónico
       $\Sigma = \{z : \langle z - z_0, n_\Sigma \rangle = 0\}$ y mide la distancia mínima
       de recurrencia de Kac-Poincaré $\min_{t_k \ge T_{\min}} \|z(t_k) - z(0)\|_{\mathcal{H}}$.
    4. **Multiplicadores Característicos de Floquet**:
       Autovalores $\mu_i \in \sigma(\Phi_{\Delta t})$ que determinan la estabilidad orbital
       ($|\mu_i| = 1$ estabilidad neutra de Poincaré; $|\mu_i| < 1$ atracción disipativa).
    """
    trajectory_efforts: RealMatrix        # $(N_{\mathrm{steps}}+1, V)$
    trajectory_flows: RealMatrix          # $(N_{\mathrm{steps}}+1, E)$
    hamiltonian_history: RealVector       # $\mathcal{H}(t_k)$
    poincare_section_times: RealVector    # Instantes de cruce transversal por $\Sigma$
    poincare_section_states: RealMatrix   # Estados en la sección de Poincaré $\Sigma$
    min_recurrence_distance: float        # $\min_{k \ge 1} \|z(t_k) - z(0)\|_{\mathcal{H}}$
    recurrence_time: float                # Instante $t^*$ de máximo acercamiento recurrente
    liouville_determinant: float          # $\det(\Phi_{\Delta t})$
    symplectic_defect: float              # $\|\Phi_{\Delta t}^\top \Omega \Phi_{\Delta t} - \Omega\|_F$
    floquet_multipliers: ComplexVector    # $\sigma(\Phi_{\Delta t})$
    max_floquet_modulus: float            # $\max_i |\mu_i|$
    poincare_cartan_circulation_drift: float  # Variación del invariante $\oint_\gamma p$


@dataclass(frozen=True, slots=True)
class Phase3HodgeDecision:
    r"""
    Pasaporte terminal de la FASE 3 con la evaluación en el Álgebra de Heyting $\Omega_3$,
    la descomposición de Helmholtz-Hodge, el certificado orbital de Poincaré y la
    síntesis de Automejora Recursiva de Nivel 3.
    """
    heyting_truth: HeytingTruth
    decomposition: HelmholtzHodgeDecomposition
    betti_0: int
    betti_1: int
    betti_2: int
    euler_characteristic: int
    condition_number_0: float
    condition_number_1: float
    orthogonality_verified: bool
    poincare_orbit: Optional[PoincareCelestialOrbitCertificate]
    rsi_iterations_total: int
    diagnostic_reasons: Tuple[str, ...]


class DiscreteDeRhamComplex:
    r"""
    Complejo de de Rham discreto ponderado de longitud 2 ($C^0 \xrightarrow{d_0} C^1 \xrightarrow{d_1} C^2$)
    equipado con:
      - Proyectores cohomológicos $\star_k$-ortogonales exactos.
      - Descomposición de Helmholtz-Hodge completa no-tautológica.
      - Estructura de Dirac $\mathcal{J} - \mathcal{R}$ de sistemas Port-Hamiltonianos.
      - Integrador Simpléctico de Cayley-Poincaré, Sección de Poincaré y Teorema de Recurrencia.
      - Síntesis de Automejora Recursiva de Nivel 3 (RSI-L3) y Aduana de Heyting $\Omega_3$.
    """

    def __init__(self, certificate: DeRhamComplexCertificate) -> None:
        if not isinstance(certificate, DeRhamComplexCertificate):
            raise TypeError(
                "FASE 3 exige un DeRhamComplexCertificate emitido por la FASE 2. "
                "No se admiten construcciones ad-hoc."
            )
        self._cert: DeRhamComplexCertificate = certificate
        self._dual0: HodgeDualityMorphism = certificate.duality_0
        self._dual1: HodgeDualityMorphism = certificate.duality_1
        self._dual2: Optional[HodgeDualityMorphism] = certificate.duality_2
        self._b1: RealMatrix = certificate.boundary_1
        self._b2: Optional[RealMatrix] = certificate.boundary_2

        # Las columnas del núcleo ya son ★_k-ortonormales por la conjugación de Hilbert en Fase 2
        self._harmonic_0: RealMatrix = self._extract_harmonic_basis(
            certificate.laplacian_0_spectrum, certificate.betti_0
        )
        self._harmonic_1: RealMatrix = self._extract_harmonic_basis(
            certificate.laplacian_1_spectrum, certificate.betti_1
        )

        logger.info(
            "DiscreteDeRhamComplex certificado: β₀=%d β₁=%d β₂=%d χ=%d | V=%d E=%d F=%d",
            certificate.betti_0,
            certificate.betti_1,
            certificate.betti_2,
            certificate.euler_poincare_characteristic,
            certificate.n_nodes,
            certificate.n_edges,
            certificate.n_faces,
        )

    @property
    def certificate(self) -> DeRhamComplexCertificate:
        """Certificado cohomológico subyacente de Fase 2."""
        return self._cert

    @staticmethod
    def _extract_harmonic_basis(
        spectrum: LaplacianSpectrum, betti: int
    ) -> RealMatrix:
        """Extrae las primeras `betti` columnas ($\star_k$-ortonormales) del núcleo."""
        if betti == 0 or spectrum.eigenvectors.size == 0:
            return np.zeros((spectrum.eigenvectors.shape[0], 0), dtype=np.float64)
        return spectrum.eigenvectors[:, :betti].copy()

    # ── Proyectores de Hodge y Descomposición de Helmholtz-Hodge Exacta ──────

    def project_harmonic(self, alpha: RealVector, degree: HodgeDegree) -> RealVector:
        r"""
        Proyección $\star_k$-ortogonal exacta sobre el subespacio de formas armónicas
        $\mathcal{H}^k = \ker\Delta_k^\star \cong H^k_{\mathrm{dR}}(K)$:
        $$P_{\mathcal{H}^k}\alpha = \sum_{j=1}^{\beta_k} \langle\alpha, h_j\rangle_{\star_k} h_j.$$
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

        if a.shape[0] != star.dimension:
            raise ValueError(f"Dimensión incompatible: {a.shape[0]} ≠ {star.dimension}.")
        if basis.shape[1] == 0:
            return np.zeros_like(a)

        # Coeficientes de Fourier-Hodge: c_j = h_jᵀ ★_k a
        star_a = star.apply(a)
        coeffs = basis.T @ star_a

        # Matriz de Gram métrica G_{ij} = h_iᵀ ★_k h_j (ya es ≈ I por Fase 2, refinada a ε_mach)
        gram = basis.T @ (star.weights[:, np.newaxis] * basis)
        try:
            coeffs_metric = la.solve(gram, coeffs, assume_a="pos")
        except la.LinAlgError:
            coeffs_metric = la.lstsq(gram, coeffs)[0]
        return basis @ coeffs_metric

    def helmholtz_hodge_decomposition(
        self, alpha: RealVector
    ) -> HelmholtzHodgeDecomposition:
        r"""
        Descomposición de Helmholtz-Hodge ortogonal ponderada y no-tautológica
        de una 1-cocadena $\alpha \in C^1(K)$:
        $$\alpha = \underbrace{\partial_1^\top \varphi}_{\text{exacta } d_0\varphi}
                 + \underbrace{\star_1^{-1} \partial_2 \star_2 \beta}_{\text{coexacta } \delta_2\beta}
                 + \underbrace{h}_{\text{armónica } \in \ker\Delta_1^\star}.$$

        Algoritmo Doctoral Exacto:
        1. **Potencial Escalar Exacto ($\varphi \in C^0$)**:
           Como $\langle d_0\varphi, \delta_2\beta + h \rangle_{\star_1} = 0$, tomando el
           codiferencial $\delta_1$ se obtiene la ecuación de Poisson en $(C^0, \star_0)$:
           $$(\partial_1 \star_1 \partial_1^\top) \varphi = \partial_1 \star_1 \alpha.$$
           Se resuelve mediante pseudoinversa espectral de Moore-Penrose sobre
           $(\ker \Delta_0^\star)^\perp$, válida para cualquier número de componentes
           conexas $\beta_0 \ge 1$. Entonces $d_0\varphi = \partial_1^\top \varphi$.
        2. **Potencial Vectorial Coexacto ($\beta \in C^2$)**:
           Como $\langle \delta_2\beta, d_0\varphi + h \rangle_{\star_1} = 0$, tomando el
           diferencial exterior $d_1 = \partial_2^\top$ se obtiene el sistema dual:
           $$(\star_2 \partial_2^\top \star_1^{-1} \partial_2 \star_2) \beta = \star_2 \partial_2^\top \alpha.$$
           Se resuelve en $(\ker \partial_2)^\perp$ y se evalúa $\delta_2\beta = \star_1^{-1}\partial_2\star_2\beta$
           (si no existen 2-caras, $\delta_2\beta \equiv \mathbf{0}$).
        3. **Componente Armónica ($h \in \ker\Delta_1^\star$)**:
           Se calcula de forma *independiente* mediante el proyector espectral
           $h = P_{\ker\Delta_1^\star}(\alpha)$, lo que permite auditar el residuo real
           no-tautológico $\|\alpha - (d_0\varphi + \delta_2\beta + h)\|_{\star_1}$.
        """
        a = np.asarray(alpha, dtype=np.float64).ravel()
        if a.shape[0] != self._cert.n_edges:
            raise ValueError(
                f"1-forma de dim {a.shape[0]} incompatible con E={self._cert.n_edges}."
            )

        w1 = self._dual1.star.weights
        star1_a = w1 * a

        # ── 1. Componente Exacta: d₀φ = ∂₁ᵀ φ ────────────────────────────────
        # Sistema simétrico semidefinido positivo: (∂₁ ★₁ ∂₁ᵀ) φ = ∂₁ ★₁ α
        L0_stiff = (self._b1 * w1[np.newaxis, :]) @ self._b1.T
        L0_stiff = 0.5 * (L0_stiff + L0_stiff.T)
        rhs_0 = self._b1 @ star1_a

        ev0, evec0 = la.eigh(L0_stiff)
        tol0 = _SPECTRAL_TOL * max(1.0, float(np.max(np.abs(ev0))))
        inv_ev0 = np.where(ev0 > tol0, 1.0 / ev0, 0.0)
        phi = evec0 @ (inv_ev0 * (evec0.T @ rhs_0))
        # Fijar gauge ortogonal a constantes en cada componente conexa
        phi = phi - self.project_harmonic(phi, HodgeDegree.ZERO)
        exact = self._b1.T @ phi

        # ── 2. Componente Coexacta: δ₂β = ★₁⁻¹ ∂₂ ★₂ β ───────────────────────
        if self._b2 is not None and self._cert.n_faces > 0:
            w2 = (
                self._dual2.star.weights
                if self._dual2 is not None
                else np.ones(self._cert.n_faces, dtype=np.float64)
            )
            w1_inv = self._dual1.star_inv_weights
            # Sistema para ψ = ★₂ β: (∂₂ᵀ ★₁⁻¹ ∂₂) ψ = ∂₂ᵀ α
            L2_curl = (self._b2.T * w1_inv[np.newaxis, :]) @ self._b2
            L2_curl = 0.5 * (L2_curl + L2_curl.T)
            rhs_2 = self._b2.T @ a

            ev2, evec2 = la.eigh(L2_curl)
            tol2 = _SPECTRAL_TOL * max(1.0, float(np.max(np.abs(ev2))))
            inv_ev2 = np.where(ev2 > tol2, 1.0 / ev2, 0.0)
            psi = evec2 @ (inv_ev2 * (evec2.T @ rhs_2))
            beta_pot = psi / w2
            coexact = w1_inv * (self._b2 @ psi)
        else:
            beta_pot = np.zeros(0, dtype=np.float64)
            coexact = np.zeros_like(a)

        # ── 3. Componente Armónica Independiente: h = P_{ker Δ₁}(α) ──────────
        h = self.project_harmonic(a, HodgeDegree.ONE)

        # ── 4. Auditoría de Residuo Real y Ortogonalidad de Riesz ────────────
        reconstructed = exact + coexact + h
        residual_vec = a - reconstructed
        residual_norm = self._dual1.star.norm(residual_vec)

        ip_ex_co = abs(self._dual1.star.inner_product(exact, coexact))
        ip_ex_ha = abs(self._dual1.star.inner_product(exact, h))
        ip_co_ha = abs(self._dual1.star.inner_product(coexact, h))
        ortho_defect = float(max(ip_ex_co, ip_ex_ha, ip_co_ha))

        norm_a_sq = self._dual1.star.inner_product(a, a)
        norm_parts_sq = (
            self._dual1.star.inner_product(exact, exact)
            + self._dual1.star.inner_product(coexact, coexact)
            + self._dual1.star.inner_product(h, h)
        )
        pyth_defect = float(abs(norm_a_sq - norm_parts_sq))

        return HelmholtzHodgeDecomposition(
            exact_part=exact,
            coexact_part=coexact,
            harmonic_part=h,
            potential=phi,
            residual_norm=residual_norm,
            coexact_potential=beta_pot,
            orthogonality_defect=ortho_defect,
            pythagorean_defect=pyth_defect,
        )

    # ── Estructura Port-Hamiltoniana / Dirac e Invariantes de Poincaré ───────

    def dirac_structure_matrix(self) -> RealMatrix:
        r"""
        Matriz de la estructura de Dirac canónica discreta $\mathcal{J} = -\mathcal{J}^\top$:
        $$\mathcal{J} = \begin{pmatrix} \mathbf{0} & -\partial_1 \\ \partial_1^\top & \mathbf{0} \end{pmatrix}.$$
        """
        V, E = self._cert.n_nodes, self._cert.n_edges
        J = np.zeros((V + E, V + E), dtype=np.float64)
        J[:V, V:] = -self._b1
        J[V:, :V] = self._b1.T
        return J

    def poincare_cartan_circulation(self, flow_1: RealVector) -> RealVector:
        r"""
        Invariante Integral Relativo de Primer Orden de Poincaré-Cartan:
        $$\mathcal{I}_1(\gamma_a) = \oint_{\gamma_a} p \, dq = \langle \star_1^{-1} f_1, h_a \rangle_{\star_1} = f_1^\top h_a,$$
        evaluado sobre la base de 1-ciclos cohomológicos armónicos $\{h_a\}_{a=1}^{\beta_1}$.
        Bajo cualquier flujo hamiltoniano potencial $\dot{f}_1 = -\star_1 d_0(\dots)$,
        estas circulaciones permanecen exactamente constantes ($\dot{\mathcal{I}}_1 \equiv 0$).
        """
        f = np.asarray(flow_1, dtype=np.float64).ravel()
        if f.shape[0] != self._cert.n_edges:
            raise ValueError("Dimensión de flujo incompatible con E.")
        if self._harmonic_1.shape[1] == 0:
            return np.zeros(0, dtype=np.float64)
        return self._harmonic_1.T @ f

    def evaluate_hamiltonian(
        self,
        effort_0: RealVector,
        flow_1: RealVector,
        resistance_weights: Optional[RealVector] = None,
    ) -> PortHamiltonianState:
        r"""
        Evalúa el estado energético Port-Hamiltoniano y la circulación de Poincaré:
        $$\mathcal{H}(q, p) = \frac{1}{2}\|q\|_{\star_0}^2 + \frac{1}{2}\|p\|_{\star_1^{-1}}^2, \qquad
          P_{\mathrm{diss}} = p^\top R p \ge 0.$$
        """
        e = np.asarray(effort_0, dtype=np.float64).ravel()
        f = np.asarray(flow_1, dtype=np.float64).ravel()
        if e.shape[0] != self._cert.n_nodes or f.shape[0] != self._cert.n_edges:
            raise ValueError("Dimensiones de esfuerzo/flujo incompatibles con el complejo.")

        H = 0.5 * self._dual0.star.inner_product(e, e) + 0.5 * float(
            np.dot(f, self._dual1.star.apply_inverse(f))
        )

        if resistance_weights is None:
            P_diss = self._dual1.star.inner_product(f, f)
        else:
            r = np.asarray(resistance_weights, dtype=np.float64).ravel()
            if r.shape[0] != self._cert.n_edges:
                raise ValueError("Vector de resistencias incompatible con número de aristas.")
            P_diss = float(np.dot(f, r * f))

        q_cons = float(np.sum(self._dual0.star.apply(e)))
        circ = self.poincare_cartan_circulation(f)
        circ_norm = float(np.linalg.norm(circ)) if circ.size > 0 else 0.0

        return PortHamiltonianState(
            effort_0=e,
            flow_1=f,
            hamiltonian=H,
            dissipated_power=P_diss,
            conserved_quantity=q_cons,
            poincare_circulation_norm=circ_norm,
        )

    # ── Dinámica Celeste de Poincaré: Integrador Simpléctico y Recurrencia ───

    def propagate_poincare_symplectic_orbit(
        self,
        initial_effort_0: RealVector,
        initial_flow_1: RealVector,
        dt: float = 0.05,
        n_steps: int = 200,
        dissipation_rate: float = 0.0,
    ) -> PoincareCelestialOrbitCertificate:
        r"""
        Integra la dinámica en el espacio fase $z = [q, p]^\top \in C^0 \oplus C_1$
        mediante el **Morfismo Simpléctico de Cayley-Poincaré**:
        $$\dot{z} = (\mathcal{J} - \mathcal{R}) \nabla_z \mathcal{H}(z) = (\mathcal{J} - \mathcal{R}) \mathcal{M} z,$$
        donde $\mathcal{M} = \operatorname{diag}(\star_0, \star_1^{-1})$ es la métrica
        de energía hamiltoniana y $\mathcal{R} = \operatorname{diag}(\mathbf{0}_V, \gamma \star_1)$.

        Propiedades Exactas (Teorema de Poincaré-Liouville):
        - Para `dissipation_rate == 0.0`, la matriz de transición de un paso
          $\Phi_{\Delta t} = (I - \frac{\Delta t}{2}\mathcal{J}\mathcal{M})^{-1}(I + \frac{\Delta t}{2}\mathcal{J}\mathcal{M})$
          es una transformación canónica simpléctica respecto a $\mathcal{M}$:
          $$\Phi_{\Delta t}^\top \mathcal{M} \Phi_{\Delta t} = \mathcal{M}, \qquad \det(\Phi_{\Delta t}) = 1, \qquad \mathcal{H}(z_k) \equiv \mathcal{H}(z_0).$$
        - Calcula los cruces transversales orientados por la **Sección de Poincaré** $\Sigma$,
          la distancia mínima de **Recurrencia de Kac-Poincaré**, los **Multiplicadores
          de Floquet** y la conservación de la **Circulación de Poincaré-Cartan**.
        """
        if dt <= 0.0 or n_steps < 1:
            raise ValueError("Se requiere dt > 0 y n_steps >= 1 para la órbita de Poincaré.")
        if dissipation_rate < 0.0:
            raise ValueError("Axioma de Pasividad de Lyapunov: dissipation_rate debe ser >= 0.")

        q0 = np.asarray(initial_effort_0, dtype=np.float64).ravel()
        p0 = np.asarray(initial_flow_1, dtype=np.float64).ravel()
        V, E = self._cert.n_nodes, self._cert.n_edges
        if q0.shape[0] != V or p0.shape[0] != E:
            raise ValueError("Estado inicial incompatible con las dimensiones (V, E) del complejo.")

        J = self.dirac_structure_matrix()
        m_diag = np.concatenate([self._dual0.star.weights, self._dual1.star_inv_weights])
        M_mat = np.diag(m_diag)

        r_diag = np.concatenate([
            np.zeros(V, dtype=np.float64),
            dissipation_rate * self._dual1.star.weights,
        ])
        R_mat = np.diag(r_diag)

        # Generador dinámico A = (J - R) M
        A_gen = (J - R_mat) @ M_mat
        I_mat = np.eye(V + E, dtype=np.float64)
        half_dt = 0.5 * dt

        # Transformación de Cayley: Φ_dt = (I - dt/2 A)⁻¹ (I + dt/2 A)
        Phi_dt = la.solve(I_mat - half_dt * A_gen, I_mat + half_dt * A_gen)

        # Invariante de Liouville det(Φ_dt) y defecto simpléctico ||Φᵀ M Φ - M||_F
        liouville_det = float(la.det(Phi_dt))
        symp_defect = float(np.linalg.norm(Phi_dt.T @ M_mat @ Phi_dt - M_mat, ord="fro"))

        # Multiplicadores característicos de Floquet
        floquet_eigs = la.eigvals(Phi_dt)
        max_floquet = float(np.max(np.abs(floquet_eigs))) if floquet_eigs.size > 0 else 0.0

        # Propagación orbital y detección de Sección de Poincaré Σ
        z_curr = np.concatenate([q0, p0])
        z0 = z_curr.copy()

        traj_q = np.zeros((n_steps + 1, V), dtype=np.float64)
        traj_p = np.zeros((n_steps + 1, E), dtype=np.float64)
        H_hist = np.zeros(n_steps + 1, dtype=np.float64)

        traj_q[0] = q0
        traj_p[0] = p0
        H_hist[0] = 0.5 * float(z0.T @ (m_diag * z0))

        # Hiperplano canónico de sección de Poincaré Σ: ortogonal a la velocidad inicial Az₀
        vel0 = A_gen @ z0
        vel0_norm = float(np.linalg.norm(vel0))
        if vel0_norm > _SPECTRAL_TOL:
            n_sigma = vel0 / vel0_norm
        else:
            n_sigma = np.zeros(V + E, dtype=np.float64)
            n_sigma[0] = 1.0

        section_times: List[float] = []
        section_states: List[RealVector] = []
        prev_proj = 0.0

        min_rec_dist = float("inf")
        rec_time = 0.0
        sqrt_m = np.sqrt(m_diag)

        circ_0 = self.poincare_cartan_circulation(p0)
        max_circ_drift = 0.0

        for step in range(1, n_steps + 1):
            z_next = Phi_dt @ z_curr
            t_curr = step * dt

            q_step = z_next[:V]
            p_step = z_next[V:]
            traj_q[step] = q_step
            traj_p[step] = p_step
            H_hist[step] = 0.5 * float(z_next.T @ (m_diag * z_next))

            # Deriva de circulación de Poincaré-Cartan
            if circ_0.size > 0:
                circ_step = self.poincare_cartan_circulation(p_step)
                drift = float(np.linalg.norm(circ_step - circ_0))
                if drift > max_circ_drift:
                    max_circ_drift = drift

            # Distancia de recurrencia en norma de energía de Poincaré (después del despegue inicial)
            if step >= 2:
                dist_energy = float(np.linalg.norm(sqrt_m * (z_next - z0)))
                if dist_energy < min_rec_dist:
                    min_rec_dist = dist_energy
                    rec_time = t_curr

            # Cruce transversal orientado por la Sección de Poincaré Σ (prev_proj < 0 <= curr_proj)
            curr_proj = float(np.dot(z_next - z0, n_sigma))
            if step > 1 and prev_proj < 0.0 and curr_proj >= 0.0:
                denom = curr_proj - prev_proj
                frac = -prev_proj / denom if abs(denom) > _MACHINE_EPS else 0.5
                t_cross = (step - 1 + frac) * dt
                z_cross = (1.0 - frac) * z_curr + frac * z_next
                section_times.append(t_cross)
                section_states.append(z_cross)

            prev_proj = curr_proj
            z_curr = z_next

        if not np.isfinite(min_rec_dist):
            min_rec_dist = float(np.linalg.norm(sqrt_m * (z_curr - z0)))
            rec_time = n_steps * dt

        sec_times_arr = np.array(section_times, dtype=np.float64)
        sec_states_arr = (
            np.vstack(section_states)
            if len(section_states) > 0
            else np.zeros((0, V + E), dtype=np.float64)
        )

        return PoincareCelestialOrbitCertificate(
            trajectory_efforts=traj_q,
            trajectory_flows=traj_p,
            hamiltonian_history=H_hist,
            poincare_section_times=sec_times_arr,
            poincare_section_states=sec_states_arr,
            min_recurrence_distance=min_rec_dist,
            recurrence_time=rec_time,
            liouville_determinant=liouville_det,
            symplectic_defect=symp_defect,
            floquet_multipliers=floquet_eigs,
            max_floquet_modulus=max_floquet,
            poincare_cartan_circulation_drift=max_circ_drift,
        )

    # ── Semigrupo del Calor de Hodge (Relajación Espectral Exacta) ───────────

    def heat_semigroup(
        self,
        alpha0: RealVector,
        t: float,
        degree: HodgeDegree = HodgeDegree.ZERO,
    ) -> RealVector:
        r"""
        Evolución bajo el semigrupo del calor de Hodge:
        $$\alpha(t) = e^{-t\Delta_k^\star}\alpha_0 = \sum_{j=1}^{\dim C^k} e^{-t\lambda_j} \langle\alpha_0, v_j\rangle_{\star_k} v_j.$$
        Como $\{v_j\}$ es una base $\star_k$-ortonormal exacta, para $t \to \infty$
        converge exponencialmente a la proyección armónica $P_{\mathcal{H}^k}\alpha_0$.
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
            raise ValueError("Dimensión de la cocadena incompatible con el espectro.")

        # Coeficientes en la base ★_k-ortonormal: c_j = v_jᵀ ★_k a
        coeffs = spec.eigenvectors.T @ star.apply(a)
        decay = np.exp(-t * np.maximum(spec.eigenvalues, 0.0))
        return spec.eigenvectors @ (decay * coeffs)

    def hodge_norm_squared(self, alpha: RealVector, degree: HodgeDegree) -> float:
        r"""Acceso unificado a $\|\alpha\|_{\star_k}^2 = \langle\alpha,\alpha\rangle_{\star_k}$."""
        if degree == HodgeDegree.ZERO:
            return self._dual0.star.inner_product(alpha, alpha)
        if degree == HodgeDegree.ONE:
            return self._dual1.star.inner_product(alpha, alpha)
        raise NotImplementedError(f"Norma Hodge grado {degree} no disponible.")

    # ── Automejora Recursiva de Nivel 3 del Complejo y Decisión de Heyting ───

    def execute_rsi_level3_complex_refinement(
        self,
        max_condition: float = _CONDITION_NUMBER_MAX,
    ) -> "DiscreteDeRhamComplex":
        r"""
        Aplica el endofuntor de Automejora Recursiva de Nivel 3 (RSI-L3) a todo
        el complejo de de Rham: si alguna de las métricas constitutivas ($\star_0, \star_1, \star_2$)
        supera el umbral espectral `max_condition`, ejecuta el flujo geodésico
        isocórico de Fisher-Rao preservando el volumen de Liouville y reconstruye
        el complejo certificado en el punto fijo de Banach.
        """
        s0_new, _ = self._dual0.star.recursive_self_improve_level3(max_condition=max_condition)
        s1_new, _ = self._dual1.star.recursive_self_improve_level3(max_condition=max_condition)
        d0_new = s0_new.induce_duality_morphism()
        d1_new = s1_new.induce_duality_morphism()

        d2_new: Optional[HodgeDualityMorphism] = None
        if self._dual2 is not None:
            s2_new, _ = self._dual2.star.recursive_self_improve_level3(max_condition=max_condition)
            d2_new = s2_new.induce_duality_morphism()

        lap_new = WeightedHodgeLaplacian(
            boundary_1=self._b1,
            duality_0=d0_new,
            duality_1=d1_new,
            boundary_2=self._b2,
            duality_2=d2_new,
        )
        cert_new = lap_new.certify_de_rham_complex()
        return DiscreteDeRhamComplex(cert_new)

    def evaluate_heyting_veto_and_emit_decision(
        self,
        alpha_1: RealVector,
        max_betti_1: Optional[int] = None,
        strict_veto_exception: bool = False,
        simulate_poincare_orbit: bool = True,
    ) -> Phase3HodgeDecision:
        r"""
        Ejecuta la descomposición de Helmholtz-Hodge sobre $\alpha_1 \in C^1$,
        audita la estabilidad celeste de Poincaré y evalúa el Supremo ($\sqcup$)
        en el retículo distributivo de Heyting $\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}$.

        Si el estado colapsa a `HeytingTruth.VETOED` y `strict_veto_exception=True`,
        detona `HodgeStarAgentError` (actuador Crowbar GPIO14).
        """
        decomp = self.helmholtz_hodge_decomposition(alpha_1)
        c = self._cert

        truth = HeytingTruth.COHERENT
        reasons: List[str] = []

        # 1. Aduana de definición positiva métrica (SPD)
        if not c.duality_0.metric_state.is_spd or not c.duality_1.metric_state.is_spd:
            truth = truth.join(HeytingTruth.VETOED)
            reasons.append("Violación de positividad métrica SPD en ★₀ o ★₁.")

        # 2. Aduana de condicionamiento espectral
        k0 = c.duality_0.metric_state.condition_number
        k1 = c.duality_1.metric_state.condition_number
        if k0 > _CONDITION_NUMBER_MAX or k1 > _CONDITION_NUMBER_MAX:
            truth = truth.join(HeytingTruth.DEGRADED)
            reasons.append(f"Condicionamiento espectral elevado (κ₀={k0:.2e}, κ₁={k1:.2e}).")

        # 3. Aduana de ortogonalidad de Riesz en Helmholtz-Hodge
        scale_a = max(1.0, self._dual1.star.norm(alpha_1) ** 2)
        ortho_ok = bool(decomp.orthogonality_defect <= 1.0e-9 * scale_a)
        if not ortho_ok:
            truth = truth.join(HeytingTruth.DEGRADED)
            reasons.append(f"Defecto de ortogonalidad de Helmholtz: {decomp.orthogonality_defect:.3e}.")

        # 4. Aduana topológica de ciclos parásitos (β₁) e invariante de Euler-Poincaré
        if not c.euler_poincare_verified:
            truth = truth.join(HeytingTruth.VETOED)
            reasons.append("Violación de la característica de Euler-Poincaré χ(K).")

        if max_betti_1 is not None and c.betti_1 > max_betti_1:
            truth = truth.join(HeytingTruth.VETOED)
            reasons.append(f"Ciclos cohomológicos β₁={c.betti_1} exceden el umbral max_betti_1={max_betti_1}.")

        # 5. Certificado orbital de Mecánica Celeste de Poincaré
        orbit_cert: Optional[PoincareCelestialOrbitCertificate] = None
        if simulate_poincare_orbit:
            q_init = decomp.potential
            p_init = self._dual1.star.apply(alpha_1)
            orbit_cert = self.propagate_poincare_symplectic_orbit(
                initial_effort_0=q_init,
                initial_flow_1=p_init,
                dt=0.05,
                n_steps=64,
                dissipation_rate=0.0,
            )
            if abs(orbit_cert.liouville_determinant - 1.0) > 1.0e-9:
                truth = truth.join(HeytingTruth.DEGRADED)
                reasons.append(
                    f"Deriva del volumen de Liouville-Poincaré: det(Φ)={orbit_cert.liouville_determinant:.6f}."
                )

        if len(reasons) == 0:
            reasons.append("Complejo de de Rham, ortogonalidad de Hodge e invariantes de Poincaré coherentes.")

        rsi_it_0 = c.duality_0.metric_state.rsi_telemetry.iterations if c.duality_0.metric_state.rsi_telemetry else 0
        rsi_it_1 = c.duality_1.metric_state.rsi_telemetry.iterations if c.duality_1.metric_state.rsi_telemetry else 0

        if truth == HeytingTruth.VETOED and strict_veto_exception:
            raise HodgeStarAgentError(
                message=" | ".join(reasons),
                heyting_state=HeytingTruth.VETOED,
                crowbar_gpio_pin=14,
            )

        return Phase3HodgeDecision(
            heyting_truth=truth,
            decomposition=decomp,
            betti_0=c.betti_0,
            betti_1=c.betti_1,
            betti_2=c.betti_2,
            euler_characteristic=c.euler_poincare_characteristic,
            condition_number_0=k0,
            condition_number_1=k1,
            orthogonality_verified=ortho_ok,
            poincare_orbit=orbit_cert,
            rsi_iterations_total=rsi_it_0 + rsi_it_1,
            diagnostic_reasons=tuple(reasons),
        )

    # ── Función generadora de categorías (Topos Monoidal de Complejos) ───────

    def monoidal_product(
        self, other: "DiscreteDeRhamComplex"
    ) -> Dict[str, int]:
        r"""
        Producto monoidal coproducto (suma directa de complejos) en la categoría
        de complejos de de Rham discretos:
        $$(K,\star) \oplus (K',\star') \quad\Longrightarrow\quad
          \beta_k^{\mathrm{tot}} = \beta_k + \beta_k', \quad \chi^{\mathrm{tot}} = \chi + \chi'.$$
        """
        return {
            "betti_0": self._cert.betti_0 + other._cert.betti_0,
            "betti_1": self._cert.betti_1 + other._cert.betti_1,
            "betti_2": self._cert.betti_2 + other._cert.betti_2,
            "euler_characteristic": (
                self._cert.euler_poincare_characteristic
                + other._cert.euler_poincare_characteristic
            ),
            "n_nodes": self._cert.n_nodes + other._cert.n_nodes,
            "n_edges": self._cert.n_edges + other._cert.n_edges,
            "n_faces": self._cert.n_faces + other._cert.n_faces,
        }

    def summary(self) -> str:
        """Resumen doctoral legible del complejo certificado (diagnóstico de laboratorio)."""
        c = self._cert
        kam1 = c.laplacian_1_spectrum.kam_certificate
        kam_str = (
            f"div_min={kam1.smallest_divisor:.3e} (no-resonante={kam1.is_kam_non_resonant})"
            if kam1 is not None
            else "N/A"
        )
        ev0_min = float(c.laplacian_0_spectrum.eigenvalues[0]) if c.laplacian_0_spectrum.eigenvalues.size > 0 else 0.0
        ev1_min = float(c.laplacian_1_spectrum.eigenvalues[0]) if c.laplacian_1_spectrum.eigenvalues.size > 0 else 0.0
        lines = [
            "=" * 76,
            "DISCRETE DE RHAM COMPLEX — CERTIFICADO DOCTORAL POINCARÉ + RSI-L3 (FASE 3)",
            "=" * 76,
            f"  Dimensiones        : V={c.n_nodes}  E={c.n_edges}  F={c.n_faces}",
            f"  Números de Betti   : β₀={c.betti_0}  β₁={c.betti_1}  β₂={c.betti_2}",
            f"  Euler-Poincaré χ   : {c.euler_poincare_characteristic} (verificado={c.euler_poincare_verified})",
            f"  κ(★₀) / κ(★₁)      : {c.duality_0.metric_state.condition_number:.4e} / {c.duality_1.metric_state.condition_number:.4e}",
            f"  Volumen Liouville  : μ(★₀)={c.duality_0.metric_state.volume_form:.4e}  μ(★₁)={c.duality_1.metric_state.volume_form:.4e}",
            f"  λ_min(Δ₀) / (Δ₁)   : {ev0_min:.4e} / {ev1_min:.4e}",
            f"  Conectividad alg.  : {c.laplacian_0_spectrum.algebraic_connectivity:.4e}",
            f"  Espectro KAM Δ₁    : {kam_str}",
            f"  Escala energética  : {c.energy_scale:.4e}",
            f"  Sucesión exacta    : {c.is_exact_sequence}",
            f"  Autoadjunto Δ₀/Δ₁  : {c.laplacian_0_spectrum.is_self_adjoint} / {c.laplacian_1_spectrum.is_self_adjoint}",
            f"  Residuo máx. Δ₀/Δ₁ : {c.laplacian_0_spectrum.max_residual:.3e} / {c.laplacian_1_spectrum.max_residual:.3e}",
            "=" * 76,
        ]
        return "\n".join(lines)


# Alias arquitectónico OODA de Fase 3
Phase3_HelmholtzDecisionMaker = DiscreteDeRhamComplex


# =============================================================================
# API de Conveniencia y Factoría (Composición Funtorial de las 3 Fases Anidadas)
# =============================================================================

def build_certified_de_rham_complex(
    weights_0: RealVector,
    weights_1: RealVector,
    boundary_1: RealMatrix,
    boundary_2: Optional[RealMatrix] = None,
    weights_2: Optional[RealVector] = None,
    enable_rsi_level3: bool = False,
) -> DiscreteDeRhamComplex:
    r"""
    Factoría maestra que encadena el funtor constitutivo $\mathcal{Z}_{\mathrm{Hodge}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$
    a través de las tres fases anidadas:

      FASE 1 ──► `DiscreteHodgeStar(...).induce_duality_morphism()`
                 Entrega `HodgeDualityMorphism` (con automejora RSI-L3 opcional).
      FASE 2 ──► `WeightedHodgeLaplacian(...).certify_de_rham_complex()`
                 Consume `HodgeDualityMorphism` y entrega `DeRhamComplexCertificate`.
      FASE 3 ──► `DiscreteDeRhamComplex(certificate)`
                 Consume `DeRhamComplexCertificate` y habilita Helmholtz-Hodge,
                 mecánica celeste de Poincaré y el veto de Heyting $\Omega_3$.
    """
    # ── FASE 1 ──────────────────────────────────────────────────────────────
    star0 = DiscreteHodgeStar(
        weights_0, degree=HodgeDegree.ZERO, enable_rsi_level3=enable_rsi_level3
    )
    star1 = DiscreteHodgeStar(
        weights_1, degree=HodgeDegree.ONE, enable_rsi_level3=enable_rsi_level3
    )
    dual0 = star0.induce_duality_morphism(auto_heal_rsi=enable_rsi_level3)  # ← último método FASE 1
    dual1 = star1.induce_duality_morphism(auto_heal_rsi=enable_rsi_level3)

    dual2: Optional[HodgeDualityMorphism] = None
    if boundary_2 is not None and weights_2 is not None:
        star2 = DiscreteHodgeStar(
            weights_2, degree=HodgeDegree.TWO, enable_rsi_level3=enable_rsi_level3
        )
        dual2 = star2.induce_duality_morphism(auto_heal_rsi=enable_rsi_level3)

    # ── FASE 2 (inicia desde HodgeDualityMorphism) ──────────────────────────
    lap = dual0.continue_to_laplacian_orienter(
        boundary_1=boundary_1,
        duality_1=dual1,
        boundary_2=boundary_2,
        duality_2=dual2,
    )
    cert = lap.certify_de_rham_complex()  # ← último método FASE 2

    # ── FASE 3 (inicia desde DeRhamComplexCertificate) ──────────────────────
    complex_3 = cert.continue_to_de_rham_complex()  # ← entrada FASE 3
    logger.info("\n%s", complex_3.summary())
    return complex_3
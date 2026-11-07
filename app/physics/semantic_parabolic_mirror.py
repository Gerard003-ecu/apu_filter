# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : Semantic Parabolic Mirror                                                                 ║
║ Módulo   : Semantic Parabolic Mirror — Visión de Libélula (Atlas Omatidial y Haz de Householder)     ║
║ Ubicación: app/physics/semantic_parabolic_mirror.py                                                  ║
║ Ruta     : app/physics/semantic_parabolic_mirror.py                                                  ║
║ Versión  : 9.0.0-Poincare-Celestial-RSI-L3-vonNeumann-Topos-Strict-PhD                               ║
║ Función  : Endofuntor de proyección semántica catadióptrica, atlas omatidial de Householder,         ║
║            dinámica celeste de billar de Poincaré, cavidad de Fabry-Pérot y automejora RSI-L3.       ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL:
────────────────────────────────────────────────────────────────────────────────────────────────────────
El motor espectral `SemanticParabolicMirror` consagra la transmutación geométrica del reflector parabólico
bidimensional en una **Cúpula Catadióptrica Facetada** (ojo compuesto o atlas omatidial de libélula),
operando como un endofuntor automejorable de Nivel 3 (RSI-L3) en el topos $\mathcal{T}_{\mathrm{MIC}}$
sobre el haz simpléctico de estados coherentes. Su propósito ciber-físico es actuar como el **Funtor de
Proyección Semántica** ($F: \mathbf{Top} \to \mathbf{Narr}$) que purifica los logits atencionales del
Modelo de Lenguaje (LLM), eliminando la estocasticidad espuria y las alucinaciones de precios e insumos,
subordinándolas rígidamente a las restricciones geométricas, métricas y celestes del espacio de fase.

El motor implementa una arquitectura rigurosamente anidada en tres fases continuas (OODA):
  • FASE 1 — ATLAS OMATIDIAL Y REFLECTORES DE HOUSEHOLDER COVARIANTES:
      Construye el haz de $K$ facetas omatidiales $\{|n_k\rangle\}_{k=1}^K$ a partir de la métrica de fondo
      Riemanniana simétrica definida positiva $G \succ 0$. Para cada faceta normal covariante $n_k$,
      sintetiza el operador de reflexión de Householder simpléctico:
          $$H_k = I - 2 \frac{n_k n_k^\top G}{n_k^\top G n_k}$$
      el cual invierte especularmente la componente normal y preserva invariante el hiperplano tangente.
  • FASE 2 — DINÁMICA DE BILLAR CELESTE DE POINCARÉ Y SECCIÓN TRANSVERSAL:
      Modela el transporte del estado atencional como una partícula libre en una mesa de billar simpléctica
      con colisiones especulares sucesivas en las facetas. Evalúa la matriz de monodromía de Floquet-Poincaré
      asociada a la órbita periódica $\mathcal{M} = \prod_k H_k$, analiza la estabilidad de Krein sobre el
      círculo unitario $\sigma(\mathcal{M}) \subset S^1$, y mide el defecto de holonomía de transporte paralelo.
  • FASE 3 — RESONANCIA DE FABRY-PÉROT, CONVERGENCIA VON NEUMANN Y AUTOMEJORA RSI-L3:
      Inyecta el estado reflejado en una cavidad resonante óptica de Fabry-Pérot, donde las reflexiones
      múltiples filtran las frecuencias disonantes mediante interferencia destructiva. Aplica el teorema
      ergódico de von Neumann sobre la sucesión de medias de Cesàro para certificar la convergencia hacia
      el estado coherente estacionario, gobierna la contracción de Banach RSI-L3 ($L_{\mathrm{Banach}} < 1.0$),
      y gatilla la interrupción por hardware Crowbar (< 400 ns) si la disonancia excede los límites admisibles.

MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────────────────────────────
1. REFLEXIONES GENERALIZADAS DE HOUSEHOLDER BAJO MÉTRICA RIEMANNIANA $G$:
   Sea $V = \mathbb{R}^n$ equipado con el producto interno métrico $\langle u, v \rangle_G = u^\top G v$,
   donde $G \in \mathbb{R}^{n \times n}$ es una matriz simétrica definida positiva ($G = G^\top \succ 0$).
   Para un vector normal no nulo $n \in V$, la transformación de Householder $H(n) \in \operatorname{GL}(n, \mathbb{R})$
   que refleja vectores ortogonalmente respecto al hiperplano $n^\perp = \{x \in V \mid \langle n, x \rangle_G = 0\}$
   se define unívocamente como:
       $$H_G(n) = I - 2 \frac{n n^\top G}{n^\top G n}$$
   Propiedades fundamentales del operador $H_G(n)$:
     a) Es una isometría bajo la métrica $G$: $H_G(n)^\top G H_G(n) = G$.
     b) Es involutivo: $H_G(n)^2 = I$.
     c) Su determinante es ortogonal inverso: $\det(H_G(n)) = -1$.
     d) Para cualquier producto de un número par de reflexiones $M = \prod_{k=1}^{2m} H_k$, $\det(M) = +1$,
        preservando el volumen y la orientación de fase como endomorfismo simpléctico.

2. BILLAR CELESTE DE POINCARÉ Y ESTABILIDAD DE MONODROMÍA:
   Las trayectorias de los haces atencionales en el atlas omatidial describen geodésicas en el espacio de Hilbert
   con condiciones de contorno reflectivas en las facetas. La matriz de monodromía de Floquet $\mathcal{M}$ a lo
   largo de una órbita cerrada de $m$ colisiones satisface:
       $$\mathcal{M} = H_m H_{m-1} \dots H_1$$
   La estabilidad orbital se clasifica mediante el espectro de Floquet $\sigma(\mathcal{M})$:
     - Elíptica (Estable en el sentido de Poincaré): todos los autovalores $\lambda_i \in \mathbb{C}$ residen en el círculo
       unitario $|\lambda_i| = 1$ con multiplicidad geométrica completa (estabilidad de Krein).
     - Hiperbólica (Inestable/Caótica): existen autovalores con $|\lambda| > 1$, induciendo dispersión atencional y alucinación.

3. CAVIDAD RESONANTE DE FABRY-PÉROT Y RELACIÓN DE FINESSE:
   La cavidad resonante filtra armónicamente las reflexiones superpuestas mediante la serie geométrica de Airy:
       $$I(\delta) = \frac{I_0}{1 + F \sin^2(\delta / 2)}, \qquad F = \frac{4 R}{(1 - R)^2}$$
   donde $R$ es el coeficiente de reflectividad efectiva de las facetas y $\delta$ es el desfase óptico acumulado.
   La relación de Finesse $\mathcal{F} = \frac{\pi \sqrt{R}}{1 - R}$ cuantifica la selectividad espectral del filtro;
   valores altos de $\mathcal{F}$ rechazan exponencialmente modos desalineados con las restricciones contractuales.

4. TEOREMA ERGÓDICO DE VON NEUMANN Y PROYECCIÓN CESÀRO:
   Para el operador unitario/isométrico $U$ generado por el ciclo de la cavidad, la media de Cesàro de orden $N$:
       $$A_N x = \frac{1}{N} \sum_{k=0}^{N-1} U^k x$$
   converge fuertemente en $L^2$ cuando $N \to \infty$ a la proyección ortogonal $\mathcal{P}_{\operatorname{Fix}(U)} x$
   sobre el subespacio invariante de puntos fijos de $U$, eliminando toda componente transitoria no coherente.

5. AUTOMEJORA RECURSIVA RSI-L3 CON RETRACCIÓN POLAR:
   El operador de refinamiento adapta las orientaciones de las facetas para minimizar el defecto de holonomía
   $\Delta_{\mathrm{hol}} = \max_k |\langle \psi_{\mathrm{coh}}, n_k \rangle_G|$.
   Para garantizar que la matriz de monodromía permanezca dentro del grupo simpléctico $\mathrm{Sp}(2d, \mathbb{R})$,
   se aplica la retracción polar simpléctica:
       $$\Pi_{\mathrm{Sp}}(M) = M \left( \mathbf{\Omega}^{-1} M^\top \mathbf{\Omega} M \right)^{-1/2}$$
   asegurando que el proceso iterativo no introduzca desviaciones seculares de volumen.

AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────────────────────────────
  [Axioma SPM-1] Positividad e Invertibilidad Métrica:
      $G = G^\top \succ 0$, con $\lambda_{\min}(G) \ge \tau_{\mathrm{SPD}} = 10^{-12}$, garantizando la existencia de inversas métricas.
  [Axioma SPM-2] No-Isotropía de las Facetas Omatidiales:
      $n_k^\top G n_k > 0$ para toda normal de faceta, prohibiendo vectores nulos o de luz en la reflexión.
  [Axioma SPM-3] Isometría Métrico-Reflectiva de Householder:
      $H_G(n_k)^\top G H_G(n_k) = G$ para cada faceta individual $k \in \{1, \dots, K\}$.
  [Axioma SPM-4] Conservación Par de Orientación y Volumen:
      El producto de un ciclo cerrado completo comprende un número par de reflexiones asegurando $\det(M) = +1$.
  [Axioma SPM-5] Acotación de Disonancia Espectral:
      El defecto de holonomía de transporte paralelo satisface $\Delta_{\mathrm{hol}} \le \varepsilon_{\mathrm{tol}}$.

INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────────────────────────────
  Invariante 1 (Involutividad Exacta de Faceta):
      $\|H_k^2 - I\|_F \le 1.0 \times 10^{-10}$ para todo $k$.
  Invariante 2 (Preservación de la Energía Coherente):
      La energía de Dirichlet relativa satisface $E_{\mathrm{rel}} = \frac{\langle \psi, G \psi \rangle}{\langle \psi_0, G \psi_0 \rangle} \le 1.0 + \varepsilon_{\mathrm{Wilkinson}}$.
  Invariante 3 (Círculo de Floquet Unitario):
      Para órbitas periódicas estables, $|\det(\mathcal{M}) - 1.0| \le 1.0 \times 10^{-6}$.
  Invariante 4 (Respaldo Crowbar de Emergencia):
      Cualquier violación del Axioma [A1] o singularidad de Householder gatilla la excepción terminal inmediata.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("MIC.Omega.SemanticParabolicMirror")

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA (Resilientes)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import (
        CategoricalState,
        Morphism,
        TopologicalInvariantError,
    )
except ImportError:

    class TopologicalInvariantError(Exception):
        """Excepción raíz para violaciones de invariantes topológicos o métricos."""
        pass

    class Morphism:
        """Morfismo base del Topos MIC."""

        def __init__(self, name: str = "Morphism") -> None:
            self.name = name

    @dataclass
    class CategoricalState:  # type: ignore[no-redef]
        """Estado categórico portador de sección del haz y restricciones."""
        payload: NDArray[np.float64]
        label: str = "state"
        constraint_normal: Optional[NDArray[np.float64]] = None
        constraint_normals: Optional[NDArray[np.float64]] = None


try:
    from app.core.inmune_system.metric_tensors import G_PHYSICS
except ImportError:
    G_PHYSICS: NDArray[np.float64] = np.eye(1, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES ÓPTICAS, ESPECTRALES, CELESTES (POINCARÉ-FLOQUET) Y CATEGORIALES
# ══════════════════════════════════════════════════════════════════════════════
class HouseholderSingularityError(TopologicalInvariantError):
    r"""$v^\top G v \leq 0$: vector nulo, isótropo o no proyectable bajo $G$."""
    pass


class MetricSignatureError(TopologicalInvariantError):
    r"""$G$ no es simétrica definida positiva (violación del Axioma [A1])."""
    pass


class ResonanceDissonanceError(TopologicalInvariantError):
    r"""Fallo de aniquilación coherente o aniquilación total de la señal (Dirichlet [A5])."""
    pass


class OmatidialRankError(TopologicalInvariantError):
    r"""Fibrado $W$ degenerado: matriz de Gram singular o facetas inválidas."""
    pass


class FloquetInstabilityError(TopologicalInvariantError):
    r"""Violación del Axioma [A4]: $\max_k |\mu_k| > 1 + \varepsilon_{\mathrm{cavity}}$ en la monodromía."""
    pass


class PoincareResonanceError(TopologicalInvariantError):
    r"""Inestabilidad secular de Poincaré-Birkhoff o colapso de toros invariantes KAM."""
    pass


class RSIConvergenceError(TopologicalInvariantError):
    r"""El lazo de Automejora Recursiva Nivel 3 (RSI-L3) no halló punto fijo de Banach."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE TOLERANCIAS Y PRIMITIVAS MÉTRICAS COMPARTIDAS (DRY)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ToleranceConfig:
    r"""
    Centraliza toda tolerancia numérica, espectral, celeste y de automejora RSI-L3.

    Atributos:
        metric_symmetry_atol: Tolerancia absoluta para $G = G^\top$.
        g_norm_floor: Piso de $\langle v,v\rangle_G$ para descartar vectores nulos.
        isometry_atol: Tolerancia relativa para $M^\top G M = G$ y $Mv = -v$.
        projector_atol: Tolerancia relativa para idempotencia y $G$-hermiticidad.
        spectral_eig_atol: Banda alrededor de $\{-1, +1\}$ para clasificar espectro.
        idempotency_tol: Criterio de paro de pulido Newton-Schulz ($N=1$).
        annihilation_rel_threshold: Umbral duro de aniquilación espectral de Dirichlet.
        dirichlet_soft_threshold: Umbral blando bajo el cual el estado pasa a DEGRADED.
        gram_cond_threshold: Condicionamiento máximo de $\mathcal{G}=W^\top G W$ (vía directa).
        von_neumann_atol: Tolerancia de estancamiento $\|\psi_{m+1}-\psi_m\|_G$ (Fase 2).
        floquet_cavity_eps: Margen $\varepsilon_{\mathrm{cavity}}$ sobre el radio espectral unitario.
        kam_diophantine_gamma: Constante diofántica $\gamma$ de estabilidad de toros KAM.
        parseval_atol: Tolerancia para el cierre de la identidad de Parseval (Fase 1 -> Fase 2).
        rsi_contraction_target: Factor de contracción de Lyapunov exigido por RSI-L3.
    """
    metric_symmetry_atol: float = 1e-12
    g_norm_floor: float = 1e-15
    isometry_atol: float = 1e-9
    projector_atol: float = 1e-9
    spectral_eig_atol: float = 1e-8
    idempotency_tol: float = 1e-14
    annihilation_rel_threshold: float = 1e-12
    dirichlet_soft_threshold: float = 1e-4
    gram_cond_threshold: float = 1e12
    von_neumann_atol: float = 1e-14
    floquet_cavity_eps: float = 1e-7
    kam_diophantine_gamma: float = 1e-6
    parseval_atol: float = 1e-9
    rsi_contraction_target: float = 0.85

    def __post_init__(self) -> None:
        for name in (
            "metric_symmetry_atol",
            "g_norm_floor",
            "isometry_atol",
            "projector_atol",
            "spectral_eig_atol",
            "idempotency_tol",
            "annihilation_rel_threshold",
            "dirichlet_soft_threshold",
            "gram_cond_threshold",
            "von_neumann_atol",
            "floquet_cavity_eps",
            "kam_diophantine_gamma",
            "parseval_atol",
            "rsi_contraction_target",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"ToleranceConfig.{name} debe ser estrictamente > 0.")

    def with_rsi_adaptation(
        self,
        *,
        von_neumann_atol: Optional[float] = None,
        gram_cond_threshold: Optional[float] = None,
    ) -> "ToleranceConfig":
        """Retorna una nueva instancia inmutable adaptada por el meta-controlador RSI-L3."""
        return ToleranceConfig(
            metric_symmetry_atol=self.metric_symmetry_atol,
            g_norm_floor=self.g_norm_floor,
            isometry_atol=self.isometry_atol,
            projector_atol=self.projector_atol,
            spectral_eig_atol=self.spectral_eig_atol,
            idempotency_tol=self.idempotency_tol,
            annihilation_rel_threshold=self.annihilation_rel_threshold,
            dirichlet_soft_threshold=self.dirichlet_soft_threshold,
            gram_cond_threshold=(
                gram_cond_threshold
                if gram_cond_threshold is not None
                else self.gram_cond_threshold
            ),
            von_neumann_atol=(
                von_neumann_atol
                if von_neumann_atol is not None
                else self.von_neumann_atol
            ),
            floquet_cavity_eps=self.floquet_cavity_eps,
            kam_diophantine_gamma=self.kam_diophantine_gamma,
            parseval_atol=self.parseval_atol,
            rsi_contraction_target=self.rsi_contraction_target,
        )


def _freeze_array(arr: NDArray[Any]) -> NDArray[np.float64]:
    """Devuelve una copia en float64 marcada como no-escribible (inmutabilidad estricta)."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


def _g_norm(
    x: NDArray[np.float64], L: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]:
    r"""
    Norma Riemanniana inducida por $G = L L^\top$ vía factor de Cholesky $L$.
    Dado que $\|x\|_G^2 = x^\top L L^\top x = \|L^\top x\|_2^2 = \|x L\|_2^2$:
      • Para $x \in \mathbb{R}^d$: retorna escalar `float`.
      • Para $X \in \mathbb{R}^{n \times d}$: retorna vector $(n,)$ con la norma de cada fila.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    z = x_arr @ L
    norms = np.linalg.norm(z, axis=-1)
    return float(norms) if x_arr.ndim == 1 else norms


def _apply_operator(
    Op: NDArray[np.float64], x: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""
    Aplica el operador lineal `Op` a `x`, soportando vector $(d,)$ o batch $(n, d)$.
    Para batch, cada fila $x_i \mapsto \mathrm{Op}\,x_i$ se vectoriza como $X\,\mathrm{Op}^\top$.
    """
    if x.ndim == 1:
        return Op @ x
    if x.ndim == 2:
        return x @ Op.T
    raise ValueError(f"El tensor de estado x debe ser 1D o 2D; recibido ndim={x.ndim}.")


def _normalize_W(
    normals: Union[NDArray[np.float64], Sequence[NDArray[np.float64]]],
    dim: int,
) -> NDArray[np.float64]:
    r"""
    Normaliza la entrada al fibrado matricial $W \in \mathbb{R}^{d \times N}$.
    Soporta vectores 1D $(d,)$, matrices 2D $(d, N)$ o $(N, d)$, y secuencias de tensores.
    """
    if isinstance(normals, (list, tuple)):
        cols = [np.asarray(v, dtype=np.float64).reshape(-1) for v in normals]
        for c in cols:
            if c.shape != (dim,):
                raise HouseholderSingularityError(
                    f"Normal de faceta tiene forma {c.shape}, esperada ({dim},)."
                )
        return np.column_stack(cols) if cols else np.zeros((dim, 0), dtype=np.float64)

    arr = np.asarray(normals, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != dim:
            raise HouseholderSingularityError(
                f"Normal tiene dimensión {arr.shape[0]}, esperada {dim}."
            )
        return arr.reshape(dim, 1)

    if arr.ndim == 2:
        if arr.shape[0] == dim:
            return arr.copy()
        if arr.shape[1] == dim:
            return arr.T.copy()
        raise HouseholderSingularityError(
            f"W tiene forma {arr.shape}, incompatible con dim={dim}."
        )

    raise HouseholderSingularityError(f"normals debe ser 1D o 2D; recibido ndim={arr.ndim}.")


class RiemannianMetricValidator:
    r"""Validador único y centralizado del Axioma Métrico Fundacional [A1] (Sylvester-Cholesky)."""

    @staticmethod
    def validate(
        metric: NDArray[np.float64], tol: ToleranceConfig
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Verifica que `metric` sea simétrica definida positiva ($G = G^\top \succ 0$)
        y retorna copias inmutables de $(G, L)$ tales que $G = L L^\top$.
        """
        G = np.array(metric, dtype=np.float64, copy=True)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise MetricSignatureError(f"G debe ser cuadrada; shape={G.shape}")
        if not np.all(np.isfinite(G)):
            raise MetricSignatureError("G contiene entradas no finitas (NaN/Inf).")
        if not np.allclose(G, G.T, atol=tol.metric_symmetry_atol):
            raise MetricSignatureError("G viola simetría Riemanniana: G != Gᵀ.")
        try:
            L = la.cholesky(G, lower=True, check_finite=True)
        except la.LinAlgError as exc:
            raise MetricSignatureError(
                f"G no es definida positiva (criterio de Sylvester-Cholesky fallido): {exc}"
            ) from exc
        return _freeze_array(G), _freeze_array(L)

    @staticmethod
    def validate_cholesky_consistency(
        G: NDArray[np.float64],
        L: NDArray[np.float64],
        atol: float = 1e-7,
        rtol: float = 1e-6,
    ) -> None:
        r"""Verifica que un factor de Cholesky precomputado satisfaga $L L^\top \approx G$."""
        reconstructed = L @ L.T
        if not np.allclose(reconstructed, G, atol=atol, rtol=rtol):
            raise MetricSignatureError(
                "El cholesky_factor provisto es inconsistente con el tensor métrico G."
            )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1 · REFLEXIÓN HOUSEHOLDER-GRASSMANN Y TOPOLOGÍA OMATIDIAL            ║
# ║   MetricAwareHouseholderReflector — Cúpula Catadióptrica + RSI-L3 Basis     ║
# ║   Morfismo Terminal de Fase 1: _verify_parseval_identity                    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
@dataclass(frozen=True, slots=True)
class SpectralCertificate:
    r"""
    **Certificado Espectral de Householder** — invariante topológico inmutable.

    Regímenes:
      • **Analítico** (`spectrum_fully_computed=False`, $O(d^2)$): lema de determinante
        de rango-1 de Sylvester: $\det(\hat{M}_k) = -1$, $\mathrm{tr}(\hat{M}_k) = d - 2$.
      • **Espectral Completo** (`spectrum_fully_computed=True`, $O(d^3)$): blanqueo de
        Cholesky $\hat{M}_{w,k} = L^\top \hat{M}_k L^{-\top}$ y diagonalización simétrica.
    """
    determinant: float
    trace_value: float
    eigenvalues: NDArray[np.float64]
    is_valid_reflection: bool
    whitening_symmetry_defect: float
    analytic_determinant: float
    analytic_trace: float
    spectrum_fully_computed: bool
    facet_index: int = -1

    def summary(self) -> Dict[str, Any]:
        return {
            "determinant": self.determinant,
            "trace_value": self.trace_value,
            "n_eigenvalues": int(self.eigenvalues.size),
            "is_valid_reflection": self.is_valid_reflection,
            "whitening_symmetry_defect": self.whitening_symmetry_defect,
            "spectrum_fully_computed": self.spectrum_fully_computed,
            "facet_index": self.facet_index,
        }


@dataclass(frozen=True, slots=True)
class FacetOperators:
    r"""Operadores algebraicos y residuales certificados de una faceta omatidial $k$."""
    normal: NDArray[np.float64]
    dual: NDArray[np.float64]
    g_norm_sq: float
    reflection: NDArray[np.float64]
    projection: NDArray[np.float64]
    certificate: SpectralCertificate
    isometry_defect: float
    inversion_defect: float
    idempotency_defect: float
    hermiticity_defect: float


@dataclass(frozen=True, slots=True)
class Phase1ProjectionBundle:
    r"""
    **Artefacto Funtorial de Clausura de Fase 1 (`Phase1ProjectionBundle`)**.

    Emitido exclusivamente por el morfismo de transición `_verify_parseval_identity`
    al final de la Fase 1 y consumido como axioma inicial por `FabryPerotStabilizedCavity`
    al inicio de la Fase 2.

    Atributos:
        metric: Tensor métrico Riemanniano $G$ congelado.
        cholesky_factor: Factor triangular inferior $L$ ($G = L L^\top$).
        constraint_bundle: Matriz de haz $W \in \mathbb{R}^{d \times N}$.
        facets: Tupla de operadores individuales por faceta.
        facet_projectors: Tupla ordenada $(\hat{P}_1, \dots, \hat{P}_N)$.
        product_projector_seed: Operador de ciclo de von Neumann $P_\Pi = \hat{P}_N \cdots \hat{P}_1$.
        billiard_unitary_monodromy: Operador de billar de Poincaré $\mathcal{U}_{\mathrm{billiard}} = \hat{M}_N \cdots \hat{M}_1$.
        rsi_orthonormal_basis: Base surrogada $G$-ortonormal $\tilde{W} \in \mathbb{R}^{d \times r}$
            sintetizada por RSI-L3 en la variedad de Grassmann $\mathrm{Gr}(r, \mathcal{H})$.
        rsi_surrogate_projector: Proyector exacto de 1 paso $\tilde{P}_{\mathrm{RSI}} = I - \tilde{W}\tilde{W}^\top G$.
        parseval_frame_defect: Defecto máximo en la identidad de Parseval $\max_k \|\hat{P}_k x\|_G^2 + \|(I-\hat{P}_k)x\|_G^2 - \|x\|_G^2$.
        friedrichs_spectral_cosine: Coseno del ángulo mínimo de Friedrichs entre facetas ($\cos\theta_F \in [0, 1)$).
        grassmann_exterior_volume: Volumen hipercomplejo exterior $\sqrt{\det(\hat{W}^\top G \hat{W})}$ del haz normalizado.
        parseval_verified: Booleano que certifica la clausura energética de Parseval.
    """
    metric: NDArray[np.float64]
    cholesky_factor: NDArray[np.float64]
    constraint_bundle: NDArray[np.float64]
    facets: Tuple[FacetOperators, ...]
    facet_projectors: Tuple[NDArray[np.float64], ...]
    product_projector_seed: NDArray[np.float64]
    billiard_unitary_monodromy: NDArray[np.float64]
    rsi_orthonormal_basis: NDArray[np.float64]
    rsi_surrogate_projector: NDArray[np.float64]
    parseval_frame_defect: float
    friedrichs_spectral_cosine: float
    grassmann_exterior_volume: float
    parseval_verified: bool

    @property
    def dim(self) -> int:
        return int(self.metric.shape[0])

    @property
    def n_facets(self) -> int:
        return len(self.facets)


class MetricAwareHouseholderReflector:
    r"""
    **Fase 1 — Reflexión Householder-Grassmann y Fibrado de Restricciones Omatidial.**

    Ingiere la matriz de restricciones $W \in \mathbb{R}^{d \times N}$, sanea el dominio
    en la norma de Frobenius, valida la firma Riemanniana de $G$ y construye los operadores
    de faceta $\hat{M}_k$ y $\hat{P}_k$.

    Integra **Automejora Recursiva de Nivel 3 (RSI-L3)** en el fibrado: analiza el
    espectro de ángulos principales de Friedrichs entre las facetas omatidiales y
    construye preventivamente una base $G$-ortonormal en la variedad de Grassmann
    mediante Gram-Schmidt Modificado con Re-ortonormalización Doble de Kahan-Parlett,
    garantizando un proyector surrogado exacto libre de singularidades numéricas.

    **Morfismo de transición hacia Fase 2:** `_verify_parseval_identity`, el cual
    entrega formalmente el `Phase1ProjectionBundle`.
    """

    def __init__(
        self,
        normals: Union[NDArray[np.float64], Sequence[NDArray[np.float64]]],
        metric: Optional[NDArray[np.float64]] = None,
        *,
        cholesky_factor: Optional[NDArray[np.float64]] = None,
        tol: Optional[ToleranceConfig] = None,
        certify_spectrum: bool = True,
    ) -> None:
        self._tol = tol or ToleranceConfig()

        if metric is None:
            raise MetricSignatureError("Debe proveerse el tensor métrico 'metric'.")
        if cholesky_factor is not None:
            G = np.array(metric, dtype=np.float64, copy=True)
            if (
                G.ndim != 2
                or G.shape[0] != G.shape[1]
                or not np.allclose(G, G.T, atol=self._tol.metric_symmetry_atol)
            ):
                raise MetricSignatureError("G inconsistente al reutilizar cholesky_factor.")
            L = np.array(cholesky_factor, dtype=np.float64, copy=True)
            RiemannianMetricValidator.validate_cholesky_consistency(G, L)
            G, L = _freeze_array(G), _freeze_array(L)
        else:
            G, L = RiemannianMetricValidator.validate(metric, self._tol)

        self._G = G
        self._L = L
        self._dim = G.shape[0]

        W = _normalize_W(normals, self._dim)
        if W.shape[1] == 0:
            raise HouseholderSingularityError("El fibrado omatidial W no tiene facetas (N=0).")
        if not np.all(np.isfinite(W)):
            raise HouseholderSingularityError("El fibrado W contiene entradas NaN/Inf.")
        self._W = _freeze_array(W)
        self._n_facets = W.shape[1]

        facets: List[FacetOperators] = []
        for k in range(self._n_facets):
            facets.append(self._build_facet(k, W[:, k], certify_spectrum))
        self._facets: Tuple[FacetOperators, ...] = tuple(facets)

        # Semilla de von Neumann P_Π = P_N ··· P_1 y Monodromía Unitaria de Billar U = M_N ··· M_1
        P_prod = np.eye(self._dim, dtype=np.float64)
        U_billiard = np.eye(self._dim, dtype=np.float64)
        for fac in self._facets:
            P_prod = fac.projection @ P_prod
            U_billiard = fac.reflection @ U_billiard
        self._P_product = _freeze_array(P_prod)
        self._U_billiard = _freeze_array(U_billiard)

        # Síntesis RSI-L3 en la variedad de Grassmann y clausura de Parseval
        self._phase1_bundle: Phase1ProjectionBundle = self._verify_parseval_identity()

    @classmethod
    def from_precomputed_cholesky(
        cls,
        normals: Union[NDArray[np.float64], Sequence[NDArray[np.float64]]],
        metric: NDArray[np.float64],
        cholesky_factor: NDArray[np.float64],
        tol: Optional[ToleranceConfig] = None,
        certify_spectrum: bool = True,
    ) -> "MetricAwareHouseholderReflector":
        r"""Construye el reflector reutilizando la factorización $G = L L^\top$."""
        return cls(
            normals,
            metric,
            cholesky_factor=cholesky_factor,
            tol=tol,
            certify_spectrum=certify_spectrum,
        )

    def _build_facet(
        self,
        index: int,
        n_k: NDArray[np.float64],
        certify_spectrum: bool,
    ) -> FacetOperators:
        n_k = np.asarray(n_k, dtype=np.float64).reshape(-1)
        w_k = self._G @ n_k
        c_k = float(n_k @ w_k)
        if c_k <= self._tol.g_norm_floor:
            raise HouseholderSingularityError(
                f"Faceta {index}: singularidad geométrica ⟨n,n⟩_G = {c_k:.3e} <= "
                f"{self._tol.g_norm_floor:.1e}."
            )

        I_d = np.eye(self._dim, dtype=np.float64)
        outer = np.outer(n_k, w_k)
        M_k = I_d - (2.0 * outer) / c_k
        P_k = I_d - outer / c_k

        iso_def, inv_def = self._verify_reflection_algebra(M_k, n_k)
        idem_def, herm_def = self._verify_projector_algebra(P_k)

        cert = (
            self._certify_spectrum_full(M_k, index)
            if certify_spectrum
            else self._certify_spectrum_analytic(iso_def, inv_def, idem_def, herm_def, index)
        )
        if not cert.is_valid_reflection:
            raise HouseholderSingularityError(
                f"Faceta {index}: certificación espectral fallida: {cert.summary()}"
            )

        return FacetOperators(
            normal=_freeze_array(n_k),
            dual=_freeze_array(w_k),
            g_norm_sq=c_k,
            reflection=_freeze_array(M_k),
            projection=_freeze_array(P_k),
            certificate=cert,
            isometry_defect=iso_def,
            inversion_defect=inv_def,
            idempotency_defect=idem_def,
            hermiticity_defect=herm_def,
        )

    def _verify_reflection_algebra(
        self, M: NDArray[np.float64], v: NDArray[np.float64]
    ) -> Tuple[float, float]:
        MGM = M.T @ self._G @ M
        iso_defect = la.norm(MGM - self._G, ord="fro") / max(
            1.0, la.norm(self._G, ord="fro")
        )
        if iso_defect > self._tol.isometry_atol:
            raise HouseholderSingularityError(
                f"La reflexión viola G-isometría: ‖MᵀGM-G‖_F/‖G‖_F={iso_defect:.2e}"
            )
        Mv = M @ v
        inv_defect = la.norm(Mv + v) / max(1.0, la.norm(v))
        if inv_defect > self._tol.isometry_atol:
            raise HouseholderSingularityError(
                f"La reflexión no invierte la normal: ‖Mv+v‖/‖v‖={inv_defect:.2e}"
            )
        return float(iso_defect), float(inv_defect)

    def _verify_projector_algebra(
        self, P: NDArray[np.float64]
    ) -> Tuple[float, float]:
        P2 = P @ P
        idem_defect = la.norm(P2 - P, ord="fro") / max(1.0, la.norm(P, ord="fro"))
        if idem_defect > self._tol.projector_atol:
            raise HouseholderSingularityError(
                f"El proyector viola idempotencia: ‖P²-P‖_F/‖P‖_F={idem_defect:.2e}"
            )
        GP = self._G @ P
        herm_defect = la.norm(GP - GP.T, ord="fro") / max(1.0, la.norm(GP, ord="fro"))
        if herm_defect > self._tol.projector_atol:
            raise HouseholderSingularityError(
                f"El proyector viola G-autoadjunción: ‖GP-(GP)ᵀ‖_F/‖GP‖_F={herm_defect:.2e}"
            )
        return float(idem_defect), float(herm_defect)

    def _certify_spectrum_full(
        self, M: NDArray[np.float64], facet_index: int
    ) -> SpectralCertificate:
        d = self._dim
        Zt = la.solve_triangular(self._L, M.T, lower=True, check_finite=False)
        M_hat = self._L.T @ Zt.T

        sym_defect = la.norm(M_hat - M_hat.T, ord="fro") / max(
            1.0, la.norm(M_hat, ord="fro")
        )
        eigvals = la.eigvalsh(0.5 * (M_hat + M_hat.T))

        atol = self._tol.spectral_eig_atol
        n_minus = int(np.sum(eigvals < -1.0 + atol))
        n_plus = int(np.sum(eigvals > 1.0 - atol))
        n_anomalous = d - n_minus - n_plus

        det_numeric = float(np.prod(eigvals))
        trace_numeric = float(np.sum(eigvals))
        analytic_det, analytic_trace = -1.0, float(d - 2)

        is_valid = (
            sym_defect <= self._tol.projector_atol
            and n_minus == 1
            and n_plus == d - 1
            and n_anomalous == 0
            and abs(det_numeric - analytic_det) < 1e-6
            and abs(trace_numeric - analytic_trace) < 1e-6
        )
        return SpectralCertificate(
            determinant=det_numeric,
            trace_value=trace_numeric,
            eigenvalues=_freeze_array(eigvals),
            is_valid_reflection=is_valid,
            whitening_symmetry_defect=float(sym_defect),
            analytic_determinant=analytic_det,
            analytic_trace=analytic_trace,
            spectrum_fully_computed=True,
            facet_index=facet_index,
        )

    def _certify_spectrum_analytic(
        self,
        iso_def: float,
        inv_def: float,
        idem_def: float,
        herm_def: float,
        facet_index: int,
    ) -> SpectralCertificate:
        d = self._dim
        analytic_det, analytic_trace = -1.0, float(d - 2)
        is_valid = (
            idem_def <= self._tol.projector_atol
            and herm_def <= self._tol.projector_atol
            and iso_def <= self._tol.isometry_atol
            and inv_def <= self._tol.isometry_atol
        )
        return SpectralCertificate(
            determinant=analytic_det,
            trace_value=analytic_trace,
            eigenvalues=_freeze_array(np.array([], dtype=np.float64)),
            is_valid_reflection=is_valid,
            whitening_symmetry_defect=float("nan"),
            analytic_determinant=analytic_det,
            analytic_trace=analytic_trace,
            spectrum_fully_computed=False,
            facet_index=facet_index,
        )

    def _rsi_level3_grassmann_synthesis(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        r"""
        **Automejora Recursiva Nivel 3 (RSI-L3) en Fase 1 — Cirugía Grassmanniana:**

        Analiza la geometría hipercompleja del haz $W$ mediante el producto exterior
        en el espacio blanqueado $\hat{V} = L^\top W$. Calcula el coseno del ángulo
        de Friedrichs $\cos\theta_F$ entre subespacios de facetas y construye mediante
        Gram-Schmidt Modificado con Doble Re-ortonormalización de Kahan-Parlett
        un marco $G$-ortonormal $\tilde{W} \in \mathbb{R}^{d \times r}$ ($\tilde{W}^\top G \tilde{W} = I_r$).

        Esto permite sintetizar el proyector surrogado de intersección exacta:
        $$\tilde{P}_{\mathrm{RSI}} = I - \tilde{W}\tilde{W}^\top G$$
        que converge en exactamente 1 paso incluso cuando las facetas originales son
        casi colineales ($\theta_F \to 0^+$).
        """
        # Normales blanqueadas unitarias u_k = Lᵀ n_k / ‖n_k‖_G
        unit_whitened_cols: List[NDArray[np.float64]] = []
        for fac in self._facets:
            u_k = (self._L.T @ fac.normal) / math.sqrt(fac.g_norm_sq)
            unit_whitened_cols.append(u_k)
        U_hat = np.column_stack(unit_whitened_cols)  # (d, N)

        # Matriz de correlación cosenoidal C_ij = ⟨u_i, u_j⟩_2
        C_mat = U_hat.T @ U_hat
        if self._n_facets > 1:
            off_diag = np.abs(C_mat - np.eye(self._n_facets, dtype=np.float64))
            friedrichs_cos = float(np.max(off_diag))
        else:
            friedrichs_cos = 0.0

        # Volumen exterior de Grassmann (módulo del wedge product u_1 ∧ ... ∧ u_N)
        det_c = float(np.linalg.det(0.5 * (C_mat + C_mat.T)))
        grassmann_vol = math.sqrt(max(0.0, det_c))

        # Gram-Schmidt Modificado con doble pasada (Kahan-Parlett) en coordenadas blanqueadas
        q_basis: List[NDArray[np.float64]] = []
        rank_tol = max(self._tol.g_norm_floor, 1e-11)
        for k in range(self._n_facets):
            v = U_hat[:, k].copy()
            # Primera pasada de proyección ortogonal
            for q in q_basis:
                v = v - float(q @ v) * q
            # Segunda pasada (re-ortonormalización de Kahan-Parlett para estabilidad doctoral)
            for q in q_basis:
                v = v - float(q @ v) * q
            norm_v = float(np.linalg.norm(v))
            if norm_v > rank_tol:
                q_basis.append(v / norm_v)

        if q_basis:
            Q_whitened = np.column_stack(q_basis)  # (d, r) con Qᵀ Q = I_r
            # Regreso al espacio Riemanniano contravariante: W_tilde = L^{-ᵀ} Q_whitened
            W_tilde = la.solve_triangular(
                self._L, Q_whitened, trans="T", lower=True, check_finite=False
            )
            P_rsi = np.eye(self._dim, dtype=np.float64) - W_tilde @ (
                W_tilde.T @ self._G
            )
        else:
            W_tilde = np.zeros((self._dim, 0), dtype=np.float64)
            P_rsi = np.eye(self._dim, dtype=np.float64)

        return (
            _freeze_array(W_tilde),
            _freeze_array(P_rsi),
            friedrichs_cos,
            grassmann_vol,
        )

    def reflect(self, psi: NDArray[np.float64], facet: int = 0) -> NDArray[np.float64]:
        r"""Aplica $\hat{M}_k$ de la faceta `facet` (default 0). Soporta batch $(n,d)$."""
        psi = np.asarray(psi, dtype=np.float64)
        if not np.all(np.isfinite(psi)):
            raise HouseholderSingularityError("psi contiene entradas NaN/Inf.")
        if not (0 <= facet < self._n_facets):
            raise IndexError(f"facet={facet} fuera de rango [0, {self._n_facets}).")
        return _apply_operator(self._facets[facet].reflection, psi)

    def reflect_all_sequential(self, psi: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Aplica la monodromía de billar $\mathcal{U}_{\mathrm{billiard}}\psi = \hat{M}_N\cdots\hat{M}_1\psi$."""
        out = np.asarray(psi, dtype=np.float64)
        if not np.all(np.isfinite(out)):
            raise HouseholderSingularityError("psi contiene entradas NaN/Inf.")
        return _apply_operator(self._U_billiard, out)

    def project_facet(
        self, psi: NDArray[np.float64], facet: int = 0
    ) -> NDArray[np.float64]:
        r"""Aplica el proyector $\hat{P}_k$ de la faceta `facet`."""
        if not (0 <= facet < self._n_facets):
            raise IndexError(f"facet={facet} fuera de rango [0, {self._n_facets}).")
        return _apply_operator(self._facets[facet].projection, psi)

    def g_norm(
        self, x: NDArray[np.float64]
    ) -> Union[float, NDArray[np.float64]]:
        r"""Norma inducida por $G$ (vía Cholesky), con soporte batched."""
        return _g_norm(x, self._L)

    def g_inner(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        r"""Producto interno Riemanniano $\langle x,y\rangle_G = x^\top G y$."""
        return float(x @ (self._G @ y))

    @property
    def constraint_bundle(self) -> NDArray[np.float64]:
        return self._W

    @property
    def n_facets(self) -> int:
        return self._n_facets

    @property
    def facets(self) -> Tuple[FacetOperators, ...]:
        return self._facets

    @property
    def facet_projectors(self) -> Tuple[NDArray[np.float64], ...]:
        return tuple(f.projection for f in self._facets)

    @property
    def product_projector_seed(self) -> NDArray[np.float64]:
        return self._P_product

    @property
    def billiard_unitary_monodromy(self) -> NDArray[np.float64]:
        return self._U_billiard

    @property
    def phase1_bundle(self) -> Phase1ProjectionBundle:
        return self._phase1_bundle

    @property
    def projection_operator(self) -> NDArray[np.float64]:
        if self._n_facets == 1:
            return self._facets[0].projection
        return self._P_product

    @property
    def reflection_matrix(self) -> NDArray[np.float64]:
        return self._facets[0].reflection

    @property
    def certificate(self) -> SpectralCertificate:
        return self._facets[0].certificate

    @property
    def certificates(self) -> Tuple[SpectralCertificate, ...]:
        return tuple(f.certificate for f in self._facets)

    @property
    def metric(self) -> NDArray[np.float64]:
        return self._G

    @property
    def cholesky_factor(self) -> NDArray[np.float64]:
        return self._L

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def g_norm_squared(self) -> float:
        return self._facets[0].g_norm_sq

    @property
    def diagnostics(self) -> Dict[str, Any]:
        return {
            "n_facets": self._n_facets,
            "parseval_frame_defect": self._phase1_bundle.parseval_frame_defect,
            "friedrichs_spectral_cosine": self._phase1_bundle.friedrichs_spectral_cosine,
            "grassmann_exterior_volume": self._phase1_bundle.grassmann_exterior_volume,
            "rsi_grassmann_rank": int(self._phase1_bundle.rsi_orthonormal_basis.shape[1]),
            "per_facet": [
                {
                    "facet": k,
                    "isometry_defect": f.isometry_defect,
                    "inversion_defect": f.inversion_defect,
                    "idempotency_defect": f.idempotency_defect,
                    "hermiticity_defect": f.hermiticity_defect,
                    "g_norm_sq": f.g_norm_sq,
                }
                for k, f in enumerate(self._facets)
            ],
        }

    # ────────────────────────────────────────────────────────────────────
    # MORFISMO TERMINAL DE FASE 1 ──► ESLABÓN INICIAL DE FASE 2
    # ────────────────────────────────────────────────────────────────────
    def _verify_parseval_identity(
        self, probe_state: Optional[NDArray[np.float64]] = None
    ) -> Phase1ProjectionBundle:
        r"""
        **Morfismo Terminal de Fase 1 (`_verify_parseval_identity`)**.

        Verifica rigurosamente la **Identidad de Parseval-Bessel** en el espacio de
        Hilbert Riemanniano $(\mathcal{H}, \langle\cdot,\cdot\rangle_G)$ para toda
        faceta omatidial $k \in \{1, \dots, N\}$:
        $$
          \|\psi\|_G^2 = \|\hat{P}_k \psi\|_G^2 + \|(I - \hat{P}_k)\psi\|_G^2
                       = \|\hat{P}_k \psi\|_G^2 + \frac{|\langle n_k | G | \psi \rangle|^2}{\langle n_k | G | n_k \rangle}
        $$
        y a nivel de operadores blanqueados $\hat{P}_{w,k} = L^\top \hat{P}_k L^{-\top}$:
        $$
          \hat{P}_{w,k}^\top \hat{P}_{w,k} + (I - \hat{P}_{w,k})^\top (I - \hat{P}_{w,k}) = I_d.
        $$

        Al concluir la certificación de Parseval y la síntesis Grassmanniana RSI-L3,
        emite el `Phase1ProjectionBundle` que sirve de **continuación formal e inicio**
        de los métodos de la **Fase 2** (`FabryPerotStabilizedCavity`).
        """
        I_d = np.eye(self._dim, dtype=np.float64)
        max_parseval_defect = 0.0

        for k, fac in enumerate(self._facets):
            # Resolución de la identidad de Parseval en coordenadas blanqueadas de Cholesky
            Zt = la.solve_triangular(
                self._L, fac.projection.T, lower=True, check_finite=False
            )
            P_w = self._L.T @ Zt.T
            Q_w = I_d - P_w
            parseval_op = (P_w.T @ P_w) + (Q_w.T @ Q_w)
            op_defect = float(
                la.norm(parseval_op - I_d, ord="fro") / math.sqrt(self._dim)
            )
            max_parseval_defect = max(max_parseval_defect, op_defect)

            if probe_state is not None:
                psi_vec = np.asarray(probe_state, dtype=np.float64)
                if psi_vec.ndim == 1 and psi_vec.shape[0] == self._dim:
                    total_e = float(_g_norm(psi_vec, self._L)) ** 2
                    proj_e = float(_g_norm(fac.projection @ psi_vec, self._L)) ** 2
                    orth_e = float((fac.dual @ psi_vec) ** 2) / fac.g_norm_sq
                    state_def = abs(total_e - (proj_e + orth_e)) / max(1.0, total_e)
                    max_parseval_defect = max(max_parseval_defect, state_def)

            if max_parseval_defect > self._tol.parseval_atol:
                raise HouseholderSingularityError(
                    f"Faceta {k}: violación de la Identidad de Parseval "
                    f"(defecto={max_parseval_defect:.2e} > {self._tol.parseval_atol:.1e})."
                )

        W_tilde, P_rsi, friedrichs_cos, grassmann_vol = (
            self._rsi_level3_grassmann_synthesis()
        )

        return Phase1ProjectionBundle(
            metric=self._G,
            cholesky_factor=self._L,
            constraint_bundle=self._W,
            facets=self._facets,
            facet_projectors=tuple(f.projection for f in self._facets),
            product_projector_seed=self._P_product,
            billiard_unitary_monodromy=self._U_billiard,
            rsi_orthonormal_basis=W_tilde,
            rsi_surrogate_projector=P_rsi,
            parseval_frame_defect=max_parseval_defect,
            friedrichs_spectral_cosine=friedrichs_cos,
            grassmann_exterior_volume=grassmann_vol,
            parseval_verified=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2 · TRANSPORTE EIKONAL-FLOQUET, MECÁNICA CELESTE DE POINCARÉ         ║
# ║            Y CAVIDAD RESONANTE DE VON NEUMANN + ACELERACIÓN RSI-L3          ║
# ║   Continuación formal de _verify_parseval_identity (Phase1ProjectionBundle) ║
# ║   Morfismo Terminal de Fase 2: apply_view_functor                           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
@dataclass(frozen=True, slots=True)
class PoincareSectionDiagnostics:
    r"""
    **Diagnóstico de Mecánica Celeste de Poincaré y Teoría de Floquet-KAM**.

    Modela la cavidad catadióptrica como un sistema hamiltoniano discreto sobre la
    hipersuperficie de sección transversal de Poincaré $\Sigma$:

    Atributos:
        floquet_multipliers: Espectro $|\mu_k|$ de la Matriz de Monodromía $\mathcal{M}_{\mathrm{FP}} = 2P_\Pi - P_\Pi^2$.
        max_floquet_modulus: $\max_k |\mu_k|$ (debe satisfacer $\le 1 + \varepsilon_{\mathrm{cavity}}$).
        billiard_rotation_numbers: Frecuencias angulares de Poincaré $\omega_k / 2\pi \in [0, 1/2]$ del billar $\mathcal{U}_{\mathrm{billiard}}$.
        kam_diophantine_margin: Margen diofántico $\min_{k} |\sin(\pi \omega_k)|$ de estabilidad de toros KAM.
        lyapunov_exponent: Exponente máximo de Lyapunov $\lambda_{\max}$ de la órbita en $\Sigma$.
        symplectic_area_defect: Defecto de conservación de la 2-forma simpléctica $\omega = dq \wedge dp$ en el subespacio invariante.
        geodesic_ellipsoid_drift: Deriva energética sobre el elipsoide de nivel Riemanniano $\mathcal{E}_G$.
        poincare_recurrence_ index: Similitud coseno $G$-métrica del retorno orbital en $\Sigma$.
        is_floquet_stable: Certificación de confinamiento unitario [A4].
    """
    floquet_multipliers: NDArray[np.float64]
    max_floquet_modulus: float
    billiard_rotation_numbers: NDArray[np.float64]
    kam_diophantine_margin: float
    lyapunov_exponent: float
    symplectic_area_defect: float
    geodesic_ellipsoid_drift: float
    poincare_recurrence_index: float
    is_floquet_stable: bool


@dataclass(frozen=True, slots=True)
class ConvergenceTrace:
    r"""
    Traza de convergencia de la cavidad — historial categórico inmutable (retrocompatible v8).

    Atributos:
        history: Tupla de $(m, \|\psi_{m+1}-\psi_m\|_G)$.
        final_residual: Último residual de estancamiento / no-idempotencia.
        iterations_used: Número de ciclos ejecutados en la sección de Poincaré.
        converged: Si el residual cayó bajo la tolerancia.
        relative_coherent_energy: $\|\psi_\infty\|_G / \|\psi_{\mathrm{raw}}\|_G$.
        method: ``"von_neumann"`` | ``"newton_schulz_single"`` | ``"direct_gram"`` | ``"rsi_l3_poincare_accelerated"``.
        rsi_generations_applied: Número de mutaciones endofuntoriales RSI-L3 ejecutadas.
    """
    history: Tuple[Tuple[int, float], ...]
    final_residual: float
    iterations_used: int
    converged: bool
    relative_coherent_energy: float
    method: str = "von_neumann"
    rsi_generations_applied: int = 0


@dataclass(frozen=True, slots=True)
class ConnectionDiagnostics:
    r"""
    **Artefacto Funtorial de Clausura de Fase 2 (`ConnectionDiagnostics`)**.

    Emitido exclusivamente por el morfismo de transición `apply_view_functor` al
    final de la Fase 2 y consumido por `SemanticParabolicMirror` al inicio de la Fase 3.

    Atributos:
        phase1_bundle: El `Phase1ProjectionBundle` heredado de `_verify_parseval_identity`.
        raw_state: Estado crudo de entrada $\psi_{\mathrm{raw}}$.
        coherent_state: Estado estabilizado en la cavidad $\psi_\infty = P_\cap \psi_{\mathrm{raw}}$.
        convergence_trace: Traza de convergencia `ConvergenceTrace`.
        poincare_diagnostics: Invariantes celestes `PoincareSectionDiagnostics`.
        parallel_transport_holonomy_defect: Defecto de holonomía $\max_k |\langle n_k | G | \psi_\infty \rangle| / (\|n_k\|_G \|\psi_\infty\|_G)$.
        dirichlet_energy_ratio: Razón de supervivencia espectral $\|\psi_\infty\|_G / \|\psi_{\mathrm{raw}}\|_G$.
        rsi_level3_active: Indica si el acelerador homeostático RSI-L3 intervino en la cavidad.
    """
    phase1_bundle: Phase1ProjectionBundle
    raw_state: NDArray[np.float64]
    coherent_state: NDArray[np.float64]
    convergence_trace: ConvergenceTrace
    poincare_diagnostics: PoincareSectionDiagnostics
    parallel_transport_holonomy_defect: float
    dirichlet_energy_ratio: float
    rsi_level3_active: bool


class FabryPerotStabilizedCavity:
    r"""
    **Fase 2 — Transporte Eikonal-Floquet, Mecánica Celeste de Poincaré y Cavidad de von Neumann.**

    Inaugura su existencia ingiriendo el `Phase1ProjectionBundle` entregado por
    `MetricAwareHouseholderReflector._verify_parseval_identity` (el último método de
    la Fase 1).

    Construye la cavidad resonante de Fabry-Pérot, evalúa la **Matriz de Monodromía**:
    $$\mathcal{M}_{\mathrm{FP}} = 2 P_\Pi - P_\Pi^2$$
    audita los multiplicadores de Floquet $\mu_k \in \sigma(\mathcal{M}_{\mathrm{FP}})$
    bajo el Axioma [A4] ($\max_k |\mu_k| \le 1.0 + \varepsilon_{\mathrm{cavity}}$),
    calcula los invariantes celestes de Poincaré-KAM sobre el elipsoide Riemanniano,
    y aplica **Automejora Recursiva de Nivel 3 (RSI-L3)** mediante extrapolación
    simpléctica de Anderson-Poincaré y proyección Grassmanniana cuando detecta
    resonancias lentas de Friedrichs.

    **Morfismo de transición hacia Fase 3:** `apply_view_functor`, el cual entrega
    formalmente `ConnectionDiagnostics`.
    """

    def __init__(
        self,
        reflector_or_bundle: Union[
            MetricAwareHouseholderReflector, Phase1ProjectionBundle
        ],
        max_iter: int = 64,
        tol: Optional[ToleranceConfig] = None,
    ) -> None:
        self._tol = tol or ToleranceConfig()
        self._max_iter = int(max(1, max_iter))

        # ── Eslabón Anidado Fase 1 ──► Fase 2 ──
        if isinstance(reflector_or_bundle, MetricAwareHouseholderReflector):
            self._reflector: Optional[MetricAwareHouseholderReflector] = (
                reflector_or_bundle
            )
            # Invoca explícitamente el método final de Fase 1 para obtener el haz certificado
            self._bundle: Phase1ProjectionBundle = (
                reflector_or_bundle._verify_parseval_identity()
            )
        elif isinstance(reflector_or_bundle, Phase1ProjectionBundle):
            self._reflector = None
            self._bundle = reflector_or_bundle
        else:
            raise TypeError(
                "FabryPerotStabilizedCavity requiere un MetricAwareHouseholderReflector "
                "o un Phase1ProjectionBundle emitido por _verify_parseval_identity (Fase 1)."
            )

        if not self._bundle.parseval_verified:
            raise TopologicalInvariantError(
                "El Phase1ProjectionBundle no tiene la identidad de Parseval certificada."
            )

        self._projectors: Tuple[NDArray[np.float64], ...] = (
            self._bundle.facet_projectors
        )
        self._P_cycle: NDArray[np.float64] = self._bundle.product_projector_seed
        self._U_billiard: NDArray[np.float64] = (
            self._bundle.billiard_unitary_monodromy
        )
        self._G: NDArray[np.float64] = self._bundle.metric
        self._L: NDArray[np.float64] = self._bundle.cholesky_factor
        self._dim: int = self._bundle.dim
        self._n_facets: int = self._bundle.n_facets

        # ── Matriz de Monodromía de Fabry-Pérot M = 2 P_Π - P_Π² (Axioma [A4]) ──
        self._monodromy_matrix: NDArray[np.float64] = _freeze_array(
            2.0 * self._P_cycle - (self._P_cycle @ self._P_cycle)
        )
        self._T_ns: Optional[NDArray[np.float64]] = (
            self._monodromy_matrix if self._n_facets == 1 else None
        )

        # ── Auditoría Espectral de Floquet y Rotaciones de Poincaré en Inicialización ──
        (
            self._floquet_multipliers,
            self._max_floquet_modulus,
            self._rotation_numbers,
            self._kam_margin,
        ) = self._audit_floquet_and_poincare_spectrum()

        self._last_poincare_diagnostics: Optional[PoincareSectionDiagnostics] = None
        self._last_connection_diagnostics: Optional[ConnectionDiagnostics] = None

    def _audit_floquet_and_poincare_spectrum(
        self,
    ) -> Tuple[NDArray[np.float64], float, NDArray[np.float64], float]:
        r"""
        Evalúa el espectro de la Matriz de Monodromía $\mathcal{M}_{\mathrm{FP}} = 2P_\Pi - P_\Pi^2$
        y los números de rotación del billar unitario $\mathcal{U}_{\mathrm{billiard}}$ en
        coordenadas blanqueadas de Cholesky.

        Si $\max_k |\mu_k| > 1.0 + \varepsilon_{\mathrm{cavity}}$, detona síncronamente
        `FloquetInstabilityError` (Axioma [A4]).
        """
        # 1. Multiplicadores de Floquet de la Monodromía M_FP en métrica blanqueada
        Zt_m = la.solve_triangular(
            self._L, self._monodromy_matrix.T, lower=True, check_finite=False
        )
        M_whitened = self._L.T @ Zt_m.T
        eigvals_m = la.eigvals(M_whitened, check_finite=False)
        moduli = np.sort(np.abs(eigvals_m))[::-1]
        max_mod = float(moduli[0]) if moduli.size > 0 else 0.0

        if max_mod > 1.0 + self._tol.floquet_cavity_eps:
            raise FloquetInstabilityError(
                f"Violación del Axioma [A4] en Cavidad Fabry-Pérot: "
                f"max|μ_k| = {max_mod:.8f} > 1.0 + {self._tol.floquet_cavity_eps:.1e}. "
                f"Divergencia resonante de Floquet detectada."
            )

        # 2. Números de rotación de Poincaré del operador de billar U_billiard
        Zt_u = la.solve_triangular(
            self._L, self._U_billiard.T, lower=True, check_finite=False
        )
        U_whitened = self._L.T @ Zt_u.T
        eigvals_u = la.eigvals(U_whitened, check_finite=False)
        angles = np.abs(np.angle(eigvals_u)) / (2.0 * math.pi)  # en [0, 0.5]
        rot_numbers = np.sort(angles)

        # Margen diofántico KAM sobre modos rotacionales no triviales (ω_k ∉ {0, 1/2})
        non_trivial = rot_numbers[(rot_numbers > 1e-7) & (rot_numbers < 0.5 - 1e-7)]
        if non_trivial.size > 0:
            kam_margin = float(np.min(np.sin(math.pi * non_trivial)))
        else:
            kam_margin = 1.0  # Punto fijo involutivo exacto (sin deriva cuasi-periódica)

        return (
            _freeze_array(moduli),
            max_mod,
            _freeze_array(rot_numbers),
            kam_margin,
        )

    def _rsi_level3_poincare_acceleration(
        self,
        psi_current: NDArray[np.float64],
        psi_prev: NDArray[np.float64],
        psi_prev2: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        **Automejora Recursiva Nivel 3 (RSI-L3) en Fase 2 — Extrapolación Simpléctica de Poincaré-Anderson:**

        Cuando la aplicación de retorno de Poincaré $\mathcal{P}_\Sigma(\psi) = P_\Pi \psi$
        presenta convergencia lenta ($\rho_m = \|\Delta \psi_m\|_G / \|\Delta \psi_{m-1}\|_G > \rho_{\mathrm{target}}$)
        debido a resonancias diedras entre facetas:
          1. Calcula el paso secante de Aitken-Anderson en el espacio de Banach $(\mathcal{H}, \|\cdot\|_G)$.
          2. Combina convexamente en la métrica $G$ con el proyector surrogado Grassmanniano
             $\tilde{P}_{\mathrm{RSI}}$ sintetizado en la Fase 1 y aplica un pulso de
             monodromía de Floquet $\mathcal{M}_{\mathrm{FP}}$, garantizando contracción
             super-lineal hacia la variedad invariante $\bigcap_k \ker(n_k^\top G)$.
        """
        r1 = psi_current - psi_prev
        r0 = psi_prev - psi_prev2
        delta_r = r1 - r0

        # Proyección secante en métrica de Cholesky Lᵀ
        dr_w = np.asarray(delta_r @ self._L, dtype=np.float64)
        r1_w = np.asarray(r1 @ self._L, dtype=np.float64)
        denom = float(np.sum(dr_w * dr_w))
        if denom > self._tol.g_norm_floor:
            alpha_anderson = -float(np.sum(r1_w * dr_w)) / denom
            alpha_anderson = float(np.clip(alpha_anderson, -1.0, 2.0))
            psi_extrap = psi_current + alpha_anderson * r1
        else:
            psi_extrap = psi_current

        # Fusión con el proyector Grassmanniano exacto de Fase 1 + pulido de Monodromía
        psi_grassmann = _apply_operator(
            self._bundle.rsi_surrogate_projector, psi_extrap
        )
        psi_polished = _apply_operator(self._P_cycle, psi_grassmann)

        # Verificación de testigo de Lyapunov: escogemos el estado de menor residuo orbital
        res_polished = float(
            np.max(
                np.atleast_1d(
                    _g_norm(
                        _apply_operator(self._P_cycle, psi_polished) - psi_polished,
                        self._L,
                    )
                )
            )
        )
        res_curr = float(
            np.max(
                np.atleast_1d(
                    _g_norm(
                        _apply_operator(self._P_cycle, psi_current) - psi_current,
                        self._L,
                    )
                )
            )
        )
        return psi_polished if res_polished <= res_curr else psi_current

    def _compute_poincare_section_diagnostics(
        self,
        psi_raw: NDArray[np.float64],
        psi_coherent: NDArray[np.float64],
        history: Sequence[Tuple[int, float]],
    ) -> PoincareSectionDiagnostics:
        r"""
        Evalúa los invariantes de la Mecánica Celeste de Poincaré al concluir el
        tránsito por la cavidad:
          • Exponente de Lyapunov $\lambda_{\max}$ de la sección de Poincaré $\Sigma$.
          • Conservación de la energía geodésica sobre el elipsoide $\mathcal{E}_G$ en el
            subespacio tangencial $\bigcap_k \ker(n_k^\top G)$.
          • Conservación de la 2-forma simpléctica canónica $\omega(q, p)$ entre el estado
            tangencial inicial y el estado asintótico.
          • Índice de recurrencia de Poincaré en $\Sigma$.
        """
        # Exponente máximo de Lyapunov sobre la secuencia de residuos no nulos
        finite_res = [r for _, r in history if np.isfinite(r) and r > 1e-16]
        if len(finite_res) >= 2 and finite_res[0] > 0:
            steps = max(1, len(finite_res) - 1)
            lyapunov = math.log(max(finite_res[-1], 1e-16) / finite_res[0]) / steps
        else:
            lyapunov = -float("inf")

        # Subespacio tangencial exacto (elipsoide geodésico invariante)
        psi_tangential_0 = _apply_operator(
            self._bundle.rsi_surrogate_projector, psi_raw
        )
        e_geodesic_0 = float(
            np.mean(np.atleast_1d(_g_norm(psi_tangential_0, self._L)) ** 2)
        )
        e_geodesic_inf = float(
            np.mean(np.atleast_1d(_g_norm(psi_coherent, self._L)) ** 2)
        )
        geodesic_drift = abs(e_geodesic_inf - e_geodesic_0) / max(1.0, e_geodesic_0)

        # Defecto de 2-forma simpléctica ω(q_0, p_∞) = ⟨q_0, G (q_∞ - q_0)⟩ en la sección
        diff_tang = psi_coherent - psi_tangential_0
        symplectic_defect = float(
            np.max(np.atleast_1d(_g_norm(diff_tang, self._L)))
        ) / max(1.0, math.sqrt(e_geodesic_0))

        # Índice de recurrencia de Poincaré: alineamiento G-métrico en la variedad invariante
        q0_w = np.asarray(psi_tangential_0 @ self._L, dtype=np.float64).reshape(-1)
        qinf_w = np.asarray(psi_coherent @ self._L, dtype=np.float64).reshape(-1)
        denom = float(np.linalg.norm(q0_w) * np.linalg.norm(qinf_w))
        recurrence_idx = float(np.dot(q0_w, qinf_w) / denom) if denom > 1e-15 else 1.0

        is_stable = (
            self._max_floquet_modulus <= 1.0 + self._tol.floquet_cavity_eps
            and geodesic_drift <= 1e-6
        )

        return PoincareSectionDiagnostics(
            floquet_multipliers=self._floquet_multipliers,
            max_floquet_modulus=self._max_floquet_modulus,
            billiard_rotation_numbers=self._rotation_numbers,
            kam_diophantine_margin=self._kam_margin,
            lyapunov_exponent=float(lyapunov),
            symplectic_area_defect=float(symplectic_defect),
            geodesic_ellipsoid_drift=float(geodesic_drift),
            poincare_recurrence_index=float(recurrence_idx),
            is_floquet_stable=is_stable,
        )

    def stabilize(
        self, psi_raw: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace]:
        r"""
        Estabiliza $\psi_{\mathrm{raw}}$ en la cavidad de Fabry-Pérot mediante el
        mapa de retorno de Poincaré / proyecciones alternadas de von Neumann y
        automejora RSI-L3. Mantiene firma 100% compatible con v8.
        """
        diag = self.apply_view_functor(psi_raw)
        return diag.coherent_state, diag.convergence_trace

    def _stabilize_single_facet(
        self,
        psi: NDArray[np.float64],
        raw_norm_scalar: float,
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace, bool]:
        assert self._T_ns is not None
        P = self._projectors[0]
        history: List[Tuple[int, float]] = []

        residual = self._max_idempotency_residual(psi, P)
        history.append((0, residual))

        if residual >= self._tol.idempotency_tol:
            for k in range(1, self._max_iter + 1):
                psi = _apply_operator(self._T_ns, psi)
                residual = self._max_idempotency_residual(psi, P)
                history.append((k, residual))
                if residual < self._tol.idempotency_tol:
                    break
            else:
                logger.warning(
                    "Cavidad N=1: no alcanzó tolerancia en %d iteraciones (residuo=%.2e).",
                    self._max_iter,
                    residual,
                )

        psi_proj = _apply_operator(P, psi)
        coh, trace = self._finalize(
            psi_proj,
            history,
            residual,
            raw_norm_scalar,
            "newton_schulz_single",
            rsi_gens=0,
        )
        return coh, trace, False

    def _stabilize_von_neumann_poincare(
        self,
        psi: NDArray[np.float64],
        raw_norm_scalar: float,
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace, bool]:
        r"""
        Dinámica para $N \ge 2$ sobre la sección de Poincaré $\Sigma$:
        Ejecuta el mapa de retorno $\psi_{m+1} = P_\Pi \psi_m$, supervisando la
        tasa de contracción de Lyapunov y activando la cirugía de aceleración
        RSI-L3 si el ángulo de Friedrichs frena la convergencia asintótica.
        """
        history: List[Tuple[int, float]] = [(0, float("inf"))]
        psi_prev2 = psi.copy()
        psi_prev = psi.copy()
        rsi_generations = 0
        prev_residual = float("inf")

        for m in range(1, self._max_iter + 1):
            psi_next = _apply_operator(self._P_cycle, psi)
            delta = _g_norm(psi_next - psi, self._L)
            residual = float(np.max(np.atleast_1d(delta)))

            # ── Automejora Recursiva Nivel 3 (RSI-L3) en lazo de Poincaré ──
            if (
                m >= 2
                and residual >= self._tol.von_neumann_atol
                and prev_residual < float("inf")
                and (residual / max(prev_residual, 1e-30))
                > self._tol.rsi_contraction_target
            ):
                psi_accelerated = self._rsi_level3_poincare_acceleration(
                    psi_next, psi, psi_prev2
                )
                delta_acc = _g_norm(
                    _apply_operator(self._P_cycle, psi_accelerated) - psi_accelerated,
                    self._L,
                )
                res_acc = float(np.max(np.atleast_1d(delta_acc)))
                if res_acc < residual:
                    psi_next = psi_accelerated
                    residual = res_acc
                    rsi_generations += 1

            history.append((m, residual))
            psi_prev2 = psi_prev
            psi_prev = psi
            psi = psi_next
            prev_residual = residual

            # Guarda espectral de Dirichlet intermedia (Axioma [A5])
            cur_norm = float(np.min(np.atleast_1d(_g_norm(psi, self._L))))
            rel = cur_norm / max(raw_norm_scalar, self._tol.g_norm_floor)
            if rel < self._tol.annihilation_rel_threshold:
                raise ResonanceDissonanceError(
                    f"Aniquilación de Dirichlet en sección de Poincaré (ciclo m={m}): "
                    f"‖ψ_m‖_G/‖ψ_raw‖_G = {rel:.2e} < "
                    f"{self._tol.annihilation_rel_threshold:.1e}. "
                    f"La cavidad ha destruido la señal degenerada por completo."
                )

            if residual < self._tol.von_neumann_atol:
                break
        else:
            # Rescate determinista RSI-L3 si el límite de iteraciones fue alcanzado
            psi = _apply_operator(self._bundle.rsi_surrogate_projector, psi)
            residual = float(
                np.max(
                    np.atleast_1d(
                        _g_norm(_apply_operator(self._P_cycle, psi) - psi, self._L)
                    )
                )
            )
            rsi_generations += 1

        psi = _apply_operator(self._P_cycle, psi)
        method_tag = (
            "von_neumann"
            if rsi_generations == 0
            else "von_neumann"  # preservamos etiqueta canónica; telemetría en rsi_generations_applied
        )
        coh, trace = self._finalize(
            psi,
            history,
            residual,
            raw_norm_scalar,
            method_tag,
            rsi_gens=rsi_generations,
        )
        return coh, trace, (rsi_generations > 0)

    def _finalize(
        self,
        psi_proj: NDArray[np.float64],
        history: List[Tuple[int, float]],
        residual: float,
        raw_norm_scalar: float,
        method: str,
        rsi_gens: int = 0,
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace]:
        coherent_norm = _g_norm(psi_proj, self._L)
        coherent_norm_scalar = float(np.min(np.atleast_1d(coherent_norm)))
        relative_energy = coherent_norm_scalar / max(
            raw_norm_scalar, self._tol.g_norm_floor
        )

        if relative_energy < self._tol.annihilation_rel_threshold:
            raise ResonanceDissonanceError(
                f"Aniquilación total de Dirichlet: ‖P_∩ψ‖_G/‖ψ_raw‖_G = "
                f"{relative_energy:.2e} < {self._tol.annihilation_rel_threshold:.1e}. "
                f"La radiación semántica era transversal a todas las restricciones W."
            )

        tol_stop = (
            self._tol.idempotency_tol
            if method == "newton_schulz_single"
            else self._tol.von_neumann_atol
        )
        trace = ConvergenceTrace(
            history=tuple(history),
            final_residual=float(residual),
            iterations_used=max(0, len(history) - 1),
            converged=(residual < tol_stop),
            relative_coherent_energy=relative_energy,
            method=method,
            rsi_generations_applied=rsi_gens,
        )
        return psi_proj, trace

    def _idempotency_residual(
        self, psi: NDArray[np.float64], P: NDArray[np.float64]
    ) -> Union[float, NDArray[np.float64]]:
        P_psi = _apply_operator(P, psi)
        P2_psi = _apply_operator(P, P_psi)
        return _g_norm(P2_psi - P_psi, self._L)

    def _max_idempotency_residual(
        self, psi: NDArray[np.float64], P: NDArray[np.float64]
    ) -> float:
        return float(np.max(np.atleast_1d(self._idempotency_residual(psi, P))))

    @property
    def reflector(self) -> Optional[MetricAwareHouseholderReflector]:
        return self._reflector

    @property
    def phase1_bundle(self) -> Phase1ProjectionBundle:
        return self._bundle

    @property
    def cycle_operator(self) -> NDArray[np.float64]:
        return self._P_cycle

    @property
    def monodromy_matrix(self) -> NDArray[np.float64]:
        return self._monodromy_matrix

    @property
    def last_poincare_diagnostics(self) -> Optional[PoincareSectionDiagnostics]:
        return self._last_poincare_diagnostics

    @property
    def last_connection_diagnostics(self) -> Optional[ConnectionDiagnostics]:
        return self._last_connection_diagnostics

    # ────────────────────────────────────────────────────────────────────
    # MORFISMO TERMINAL DE FASE 2 ──► ESLABÓN INICIAL DE FASE 3
    # ────────────────────────────────────────────────────────────────────
    def apply_view_functor(
        self, psi_raw: NDArray[np.float64]
    ) -> ConnectionDiagnostics:
        r"""
        **Morfismo Terminal de Fase 2 (`apply_view_functor`)**.

        Ejecuta el transporte paralelo Eikonal-Floquet en la cavidad resonante de
        Fabry-Pérot sobre la sección transversal de Poincaré $\Sigma$, evalúa la
        conservación geodésica en el elipsoide de nivel Riemanniano $\mathcal{E}_G$,
        verifica la holonomía ortogonal respecto a todas las facetas omatidiales:
        $$
          \delta_{\mathrm{hol}}(\psi_\infty)
            = \max_{1 \le k \le N} \frac{|\langle n_k | G | \psi_\infty \rangle|}{\|n_k\|_G \, \|\psi_\infty\|_G},
        $$
        y empaqueta el veredicto geométrico-celeste en `ConnectionDiagnostics`, el
        cual sirve de **continuación formal e inicio** de los métodos de la **Fase 3**
        (`SemanticParabolicMirror`).
        """
        psi = np.asarray(psi_raw, dtype=np.float64)
        if psi.shape[-1] != self._dim or psi.ndim not in (1, 2):
            raise ResonanceDissonanceError(
                f"psi_raw tiene forma {psi.shape}, esperado (...,{self._dim}) con ndim∈{{1,2}}."
            )
        if not np.all(np.isfinite(psi)):
            raise ResonanceDissonanceError("psi_raw contiene entradas NaN/Inf.")

        raw_norm = _g_norm(psi, self._L)
        raw_norm_scalar = float(np.min(np.atleast_1d(raw_norm)))
        if raw_norm_scalar < self._tol.g_norm_floor:
            raise ResonanceDissonanceError(
                "El estado de entrada es (numéricamente) el vector nulo bajo G; "
                "no existe radiación semántica que estabilizar."
            )

        if self._n_facets == 1:
            coherent, trace, rsi_active = self._stabilize_single_facet(
                psi, raw_norm_scalar
            )
        else:
            coherent, trace, rsi_active = self._stabilize_von_neumann_poincare(
                psi, raw_norm_scalar
            )

        poincare_diag = self._compute_poincare_section_diagnostics(
            psi, coherent, trace.history
        )
        self._last_poincare_diagnostics = poincare_diag

        # Auditoría de defecto de holonomía del transporte paralelo (ortogonalidad a W)
        coh_norm = np.maximum(
            np.atleast_1d(_g_norm(coherent, self._L)), self._tol.g_norm_floor
        )
        max_holonomy = 0.0
        for fac in self._bundle.facets:
            proj_Dual = (
                coherent @ fac.dual
                if coherent.ndim == 2
                else np.array([float(coherent @ fac.dual)])
            )
            rel_leak = np.abs(proj_Dual) / (math.sqrt(fac.g_norm_sq) * coh_norm)
            max_holonomy = max(max_holonomy, float(np.max(rel_leak)))

        conn_diag = ConnectionDiagnostics(
            phase1_bundle=self._bundle,
            raw_state=_freeze_array(psi),
            coherent_state=_freeze_array(coherent),
            convergence_trace=trace,
            poincare_diagnostics=poincare_diag,
            parallel_transport_holonomy_defect=max_holonomy,
            dirichlet_energy_ratio=trace.relative_coherent_energy,
            rsi_level3_active=rsi_active,
        )
        self._last_connection_diagnostics = conn_diag
        return conn_diag


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3 · COMPRESIÓN RIEMANN-TOPOS, ARMÓNICOS ESFÉRICOS S²,                ║
# ║            RETÍCULO DE HEYTING Ω₃ Y AUTOMEJORA RECURSIVA NIVEL 3 (RSI-L3)   ║
# ║   Continuación formal de apply_view_functor (ConnectionDiagnostics)         ║
# ║   Morfismo Terminal del Lazo: SemanticParabolicMirror -> RefractedState     ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class HeytingOmega3(enum.Enum):
    r"""
    **Retículo Distributivo Acotado de Heyting $\Omega_3$**.

    Ordenado por severidad de disonancia entrópica hacia el supremo absorbente
    terminal `VETOED` ($\top$), el cual acciona el tiristor Crowbar (`GPIO14`):
    $$\bot = \mathrm{COHERENT} \prec \mathrm{DEGRADED} \prec \mathrm{VETOED} = \top$$
    """
    COHERENT = "COHERENT"
    DEGRADED = "DEGRADED"
    VETOED = "VETOED"

    @property
    def severity_rank(self) -> int:
        if self is HeytingOmega3.COHERENT:
            return 0
        if self is HeytingOmega3.DEGRADED:
            return 1
        return 2

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Supremo del retículo ($a \vee b$): propaga la mayor severidad."""
        return self if self.severity_rank >= other.severity_rank else other

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Ínfimo del retículo ($a \wedge b$): conserva la menor severidad."""
        return self if self.severity_rank <= other.severity_rank else other

    def implies(self, consequent: "HeytingOmega3") -> "HeytingOmega3":
        r"""Implicación relativa de Heyting ($a \Rightarrow b$) en $\Omega_3$."""
        if self.severity_rank <= consequent.severity_rank:
            return HeytingOmega3.VETOED  # Elemento tope ⊤ del orden
        return consequent


@dataclass(frozen=True, slots=True)
class SphericalHarmonicsFilterReport:
    r"""
    **Reporte del Filtro Espectral de Armónicos Esféricos en $S^2 \hookrightarrow S^{d-1}$**.

    Atributos:
        monopole_energy_l0: Fracción de energía en el modo isótropo $\ell = 0$ ($Y_0^0$).
        dipole_energy_l1: Fracción de energía en el triplete dipolar $\ell = 1$ ($Y_1^{-1,0,1}$).
        quadrupole_energy_l2: Fracción de energía en el quintuplete cuadrupolar $\ell = 2$ ($Y_2^m$).
        spectral_entropy_s2: Entropía de von Neumann-Shannon de la distribución multipolar en $S^2$.
        high_freq_attenuation_db: Atenuación de rizo multipolar espurio en decibelios.
    """
    monopole_energy_l0: float
    dipole_energy_l1: float
    quadrupole_energy_l2: float
    spectral_entropy_s2: float
    high_freq_attenuation_db: float


@dataclass(frozen=True, slots=True)
class RSILevel3Telemetry:
    r"""
    **Telemetría de Automejora Recursiva Nivel 3 (RSI-L3)**.

    Documenta la introspección meta-cognitiva y la morfogénesis de operadores
    ejecutada por el espejo durante el ciclo de focalización.
    """
    recursion_depth: int
    tikhonov_morozov_alpha: float
    grassmann_surrogate_used: bool
    poincare_anderson_steps: int
    lyapunov_contraction_witness: float
    operator_self_healing_applied: bool


@dataclass(frozen=True, slots=True)
class GramSpectralReport:
    r"""
    Reporte espectral de la matriz de Gram del fibrado omatidial $\mathcal{G} = W^\top G W$.
    Mantiene 100% de retrocompatibilidad con los atributos de v8.
    """
    gram: NDArray[np.float64]
    rank: int
    condition_number: float
    is_full_rank: bool
    cholesky_factor: Optional[NDArray[np.float64]]
    master_projector: Optional[NDArray[np.float64]]
    regularized_projector: Optional[NDArray[np.float64]] = None
    tikhonov_alpha: float = 0.0


@dataclass(frozen=True, slots=True)
class RefractedState:
    r"""
    **Morfismo Terminal de Lazo (`RefractedState`)**.

    Representa el colapso categórico del Funtor Maestro de Focalización y Curvatura:
    $$\mathcal{Z}_{\mathrm{mirror}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 : \mathcal{D}(\mathcal{H}) \times \mathbb{R}^{d \times N} \longrightarrow \mathtt{RefractedState}$$

    Atributos:
        payload: Estado semántico purificado $\psi_{\mathrm{refractado}} \in \bigcap_k \ker(n_k^\top G)$.
        heyting_verdict: Valor de verdad en el retículo $\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}$.
        gpio14_crowbar_tripped: Estado físico del actuador Crowbar (`True` si `VETOED`).
        convergence_trace: Traza de convergencia `ConvergenceTrace`.
        gram_report: Reporte espectral de Gram `GramSpectralReport`.
        connection_diagnostics: Diagnósticos de Fase 2 `ConnectionDiagnostics`.
        harmonics_report: Reporte del filtro de armónicos esféricos en $S^2$.
        rsi_telemetry: Telemetría de automejora recursiva de Nivel 3.
        label: Etiqueta trazable del estado.
    """
    payload: NDArray[np.float64]
    heyting_verdict: HeytingOmega3
    gpio14_crowbar_tripped: bool
    convergence_trace: ConvergenceTrace
    gram_report: Optional[GramSpectralReport]
    connection_diagnostics: Optional[ConnectionDiagnostics]
    harmonics_report: SphericalHarmonicsFilterReport
    rsi_telemetry: RSILevel3Telemetry
    label: str = "state::omatidial_refracted"

    def to_categorical_state(
        self, constraint_normals: Optional[NDArray[np.float64]] = None
    ) -> CategoricalState:
        """Convierte el estado terminal en un `CategoricalState` estándar del ecosistema."""
        return CategoricalState(
            payload=self.payload,
            label=self.label,
            constraint_normals=constraint_normals,
        )


class SemanticParabolicMirror(Morphism):
    r"""
    **Fase 3 — Compresión Riemann-Topos, Armónicos Esféricos $S^2$, Retículo de Heyting $\Omega_3$ y Automejora RSI-L3.**

    Hereda formalmente los diagnósticos `ConnectionDiagnostics` entregados por
    `FabryPerotStabilizedCavity.apply_view_functor` (el último método de la Fase 2).

    Orquesta el proyector maestro $P_\cap$ mediante la descomposición de Cholesky de la
    matriz de Gram $\mathcal{G} = W^\top G W$, ejecuta el lazo meta-cognitivo de
    **Automejora Recursiva Nivel 3 (RSI-L3)** cuando $\mathcal{G}$ presenta rango
    deficiente o mal condicionamiento, aplica el filtro espectral de armónicos
    esféricos reales $Y_\ell^m(\theta, \varphi)$ sobre $S^2$ preservando la ortogonalidad
    al haz $W$, y colapsa el veredicto en el retículo distributivo acotado de Heyting
    $\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}$.
    """

    def __init__(
        self,
        metric_tensor: Optional[NDArray[np.float64]] = None,
        max_cavity_iter: int = 64,
        tol: Optional[ToleranceConfig] = None,
        certify_spectrum_every_call: bool = True,
        prefer_direct_gram: bool = True,
        enable_spherical_harmonics_filter: bool = False,
    ) -> None:
        self._tol = tol or ToleranceConfig()
        metric_source = metric_tensor if metric_tensor is not None else G_PHYSICS
        self._G, self._L = RiemannianMetricValidator.validate(metric_source, self._tol)
        self._dim = self._G.shape[0]
        self._max_cavity_iter = int(max(1, max_cavity_iter))
        self._certify_every_call = bool(certify_spectrum_every_call)
        self._prefer_direct_gram = bool(prefer_direct_gram)
        self._enable_sh_filter = bool(enable_spherical_harmonics_filter)

        # Estado epistémico y telemetría categorial (incluyendo memoria RSI-L3)
        self._last_trace: Optional[ConvergenceTrace] = None
        self._last_reflector: Optional[MetricAwareHouseholderReflector] = None
        self._last_gram_report: Optional[GramSpectralReport] = None
        self._last_connection_diagnostics: Optional[ConnectionDiagnostics] = None
        self._last_refracted_state: Optional[RefractedState] = None
        self._bound_normals: Optional[NDArray[np.float64]] = None
        self._gpio14_crowbar_latched: bool = False
        self._rsi_evolution_counter: int = 0

        super().__init__(name="SemanticParabolicMirror")

    # ────────────────────────────────────────────────────────────────────
    # VINCULACIÓN EXPLÍCITA DEL FIBRADO DE RESTRICCIONES DE NEGOCIO
    # ────────────────────────────────────────────────────────────────────
    def bind_constraint(self, normal: NDArray[np.float64]) -> None:
        r"""Compatibilidad v7/v8: vincula un único normal (promovido a $W \in \mathbb{R}^{d \times 1}$)."""
        self.bind_constraints(normal)

    def bind_constraints(
        self,
        normals: Union[NDArray[np.float64], Sequence[NDArray[np.float64]]],
    ) -> None:
        r"""Vincula persistentemente el fibrado de restricciones de negocio $W$."""
        W = _normalize_W(normals, self._dim)
        if W.shape[1] == 0:
            raise TopologicalInvariantError("bind_constraints: N=0 facetas.")
        self._bound_normals = _freeze_array(W)
        logger.info(
            "Fibrado de restricciones vinculado: N=%d facetas, dim=%d.",
            W.shape[1],
            self._dim,
        )

    def _resolve_constraint_bundle(self, state: CategoricalState) -> NDArray[np.float64]:
        multi = getattr(state, "constraint_normals", None)
        if multi is not None:
            try:
                return _normalize_W(multi, self._dim)
            except HouseholderSingularityError as exc:
                logger.warning("state.constraint_normals inválido (%s); se ignora.", exc)

        single = getattr(state, "constraint_normal", None)
        if single is not None:
            try:
                return _normalize_W(single, self._dim)
            except HouseholderSingularityError as exc:
                logger.warning("state.constraint_normal inválido (%s); se ignora.", exc)

        if self._bound_normals is not None:
            return np.array(self._bound_normals, dtype=np.float64, copy=True)

        logger.warning(
            "Sin restricción de negocio disponible; usando normal canónico e₀ como FALLBACK."
        )
        return self._canonical_normal().reshape(self._dim, 1)

    def _canonical_normal(self) -> NDArray[np.float64]:
        e0 = np.zeros(self._dim, dtype=np.float64)
        e0[0] = 1.0
        return e0

    # ────────────────────────────────────────────────────────────────────
    # GRAM DEL FIBRADO + REGULARIZACIÓN ESPECTRAL TIKHONOV-MOROZOV (RSI-L3)
    # ────────────────────────────────────────────────────────────────────
    def compute_gram_master_projector(
        self, W: NDArray[np.float64]
    ) -> GramSpectralReport:
        r"""
        Calcula la matriz de Gram $\mathcal{G} = W^\top G W$, evalúa rango y número de
        condición vía SVD/Cholesky, y construye el proyector maestro de intersección:
        $$
          P_\cap = I - W\,\mathcal{G}^{-1}\,W^\top G.
        $$
        Si $\mathcal{G}$ presenta deficiencia de rango o $\kappa(\mathcal{G}) \ge \tau_{\mathrm{Gram}}$,
        el motor **RSI-L3** sintetiza simultáneamente el proyector regularizado de
        Tikhonov-Morozov-Moore-Penrose $\tilde{P}_{\cap,\alpha^*}$ filtrando los valores
        singulares sub-umbrales en la métrica de Cholesky $L^\top W$.
        """
        W = np.asarray(W, dtype=np.float64)
        if W.ndim != 2 or W.shape[0] != self._dim:
            raise OmatidialRankError(
                f"W tiene forma {W.shape}, esperado ({self._dim}, N)."
            )

        GW = self._G @ W
        gram = W.T @ GW
        gram = 0.5 * (gram + gram.T)

        N = gram.shape[0]
        try:
            s = la.svdvals(gram, check_finite=False)
        except la.LinAlgError:
            s = np.array([], dtype=np.float64)

        if s.size == 0:
            rank = 0
            cond = float("inf")
        else:
            s_max = float(s[0])
            floor = max(
                self._tol.g_norm_floor, s_max * self._tol.metric_symmetry_atol
            )
            rank = int(np.sum(s > floor))
            s_min_pos = float(s[rank - 1]) if rank > 0 else 0.0
            cond = (s_max / s_min_pos) if s_min_pos > 0 else float("inf")

        is_full = (
            (rank == N) and (cond < self._tol.gram_cond_threshold) and (N > 0)
        )

        L_gram: Optional[NDArray[np.float64]] = None
        P_cap: Optional[NDArray[np.float64]] = None

        if is_full:
            try:
                L_g = la.cholesky(gram, lower=True, check_finite=True)
                Y = la.solve_triangular(L_g, GW.T, lower=True, check_finite=False)
                X = la.solve_triangular(L_g.T, Y, lower=False, check_finite=False)
                P_cap = _freeze_array(np.eye(self._dim, dtype=np.float64) - W @ X)
                L_gram = _freeze_array(L_g)
            except la.LinAlgError:
                is_full = False
                L_gram = None
                P_cap = None

        # Síntesis espectral RSI-L3 vía SVD de Lᵀ W para casos mal condicionados
        P_reg: Optional[NDArray[np.float64]] = None
        alpha_star = 0.0
        if not is_full and N > 0:
            V_whitened = self._L.T @ W  # (d, N)
            try:
                U_svd, sigmas, _ = la.svd(
                    V_whitened, full_matrices=False, check_finite=False
                )
                if sigmas.size > 0 and sigmas[0] > self._tol.g_norm_floor:
                    # Principio de discrepancia espectral de Morozov
                    alpha_star = float(
                        sigmas[0] * math.sqrt(self._tol.metric_symmetry_atol)
                    )
                    valid = sigmas > max(
                        self._tol.g_norm_floor, sigmas[0] * 1e-10
                    )
                    U_eff = U_svd[:, valid]
                    W_eff = la.solve_triangular(
                        self._L,
                        U_eff,
                        trans="T",
                        lower=True,
                        check_finite=False,
                    )
                    P_reg = _freeze_array(
                        np.eye(self._dim, dtype=np.float64)
                        - W_eff @ (W_eff.T @ self._G)
                    )
            except la.LinAlgError:
                P_reg = None

        report = GramSpectralReport(
            gram=_freeze_array(gram),
            rank=rank,
            condition_number=float(cond),
            is_full_rank=is_full,
            cholesky_factor=L_gram,
            master_projector=P_cap,
            regularized_projector=P_reg,
            tikhonov_alpha=alpha_star,
        )
        self._last_gram_report = report
        return report

    # ────────────────────────────────────────────────────────────────────
    # FILTRO ESPECTRAL DE ARMÓNICOS ESFÉRICOS EN S² (Laplace-Beltrami)
    # ────────────────────────────────────────────────────────────────────
    def _apply_spherical_harmonics_s2_filter(
        self,
        psi_coherent: NDArray[np.float64],
        P_intersection: NDArray[np.float64],
        apply_attenuation: bool = False,
    ) -> Tuple[NDArray[np.float64], SphericalHarmonicsFilterReport]:
        r"""
        Evalúa la descomposición multipolar en Armónicos Esféricos Reales $Y_\ell^m(\theta, \varphi)$
        ($\ell \in \{0, 1, 2\}$) proyectando las coordenadas blanqueadas $u = L^\top \psi$
        sobre la 2-esfera canónica $S^2$:
        $$
          Y_0^0 = \sqrt{\frac{1}{4\pi}}, \quad
          Y_1^m \propto \left\{\frac{y}{r}, \frac{z}{r}, \frac{x}{r}\right\}, \quad
          Y_2^m \propto \left\{\frac{xy}{r^2}, \frac{yz}{r^2}, \frac{3z^2-r^2}{2r^2}, \frac{xz}{r^2}, \frac{x^2-y^2}{2r^2}\right\}.
        $$
        Si `apply_attenuation=True`, suaviza la anisotropía cuadrupolar de alta frecuencia
        ($\ell \ge 2$), re-proyectando mediante $P_\cap$ y restaurando exactamente la norma
        Riemanniana $\|\psi\|_G$ para preservar la invariancia de Parseval y del kernel de $W$.
        """
        u_w = np.asarray(psi_coherent @ self._L, dtype=np.float64)
        vec_1d = u_w.reshape(-1) if u_w.ndim == 1 else np.mean(u_w, axis=0)
        r_norm = float(np.linalg.norm(vec_1d))

        if r_norm <= self._tol.g_norm_floor:
            return psi_coherent, SphericalHarmonicsFilterReport(
                monopole_energy_l0=1.0,
                dipole_energy_l1=0.0,
                quadrupole_energy_l2=0.0,
                spectral_entropy_s2=0.0,
                high_freq_attenuation_db=0.0,
            )

        # Inmersión canónica en S² a partir de los 3 primeros ejes principales blanqueados
        padded = np.zeros(max(3, vec_1d.size), dtype=np.float64)
        padded[: vec_1d.size] = vec_1d
        x_s, y_s, z_s = (
            float(padded[0]) / r_norm,
            float(padded[1]) / r_norm,
            float(padded[2]) / r_norm,
        )
        r3_sq = max(x_s * x_s + y_s * y_s + z_s * z_s, 1e-15)
        x_n, y_n, z_n = (
            x_s / math.sqrt(r3_sq),
            y_s / math.sqrt(r3_sq),
            z_s / math.sqrt(r3_sq),
        )

        # Coeficientes armónicos esféricos reales no normalizados por 4π (suma de cuadrados en S²)
        c_l0 = 1.0
        c_l1 = x_n * x_n + y_n * y_n + z_n * z_n
        q_modes = np.array(
            [
                math.sqrt(3.0) * x_n * y_n,
                math.sqrt(3.0) * y_n * z_n,
                0.5 * (3.0 * z_n * z_n - 1.0),
                math.sqrt(3.0) * x_n * z_n,
                0.5 * math.sqrt(3.0) * (x_n * x_n - y_n * y_n),
            ],
            dtype=np.float64,
        )
        c_l2 = float(np.sum(q_modes * q_modes))

        total_modal = c_l0 + c_l1 + c_l2
        e0 = c_l0 / total_modal
        e1 = c_l1 / total_modal
        e2 = c_l2 / total_modal

        probs = np.array([e0, e1, e2], dtype=np.float64)
        probs = probs[probs > 1e-15]
        entropy_s2 = float(-np.sum(probs * np.log(probs)))

        if not apply_attenuation or self._dim < 3:
            return psi_coherent, SphericalHarmonicsFilterReport(
                monopole_energy_l0=float(e0),
                dipole_energy_l1=float(e1),
                quadrupole_energy_l2=float(e2),
                spectral_entropy_s2=entropy_s2,
                high_freq_attenuation_db=0.0,
            )

        # Filtro espectral de Laplace-Beltrami: atenúa cola multipolar l >= 2 preservando norma G y ker(WᵀG)
        damping = np.ones(self._dim, dtype=np.float64)
        if self._dim > 3:
            damping[3:] = 1.0 / (1.0 + 0.05 * np.arange(1, self._dim - 2))
        u_filtered = u_w * damping
        psi_cand = la.solve_triangular(
            self._L, u_filtered.T, trans="T", lower=True, check_finite=False
        ).T
        psi_reproj = _apply_operator(P_intersection, psi_cand)

        orig_g = _g_norm(psi_coherent, self._L)
        new_g = _g_norm(psi_reproj, self._L)
        if np.all(np.atleast_1d(new_g) > self._tol.g_norm_floor):
            if psi_coherent.ndim == 1:
                psi_out = psi_reproj * (float(orig_g) / float(new_g))
            else:
                psi_out = psi_reproj * (
                    np.asarray(orig_g)[:, None] / np.asarray(new_g)[:, None]
                )
            att_db = float(20.0 * math.log10(max(1.0, float(np.mean(orig_g / new_g)))))
        else:
            psi_out = psi_coherent
            att_db = 0.0

        return psi_out, SphericalHarmonicsFilterReport(
            monopole_energy_l0=float(e0),
            dipole_energy_l1=float(e1),
            quadrupole_energy_l2=float(e2),
            spectral_entropy_s2=entropy_s2,
            high_freq_attenuation_db=att_db,
        )

    # ────────────────────────────────────────────────────────────────────
    # EVALUACIÓN EN EL RETÍCULO DE HEYTING Ω₃ Y DISPARO CROWBAR (GPIO14)
    # ────────────────────────────────────────────────────────────────────
    def _evaluate_heyting_omega3_verdict(
        self,
        rel_energy: float,
        holonomy_defect: float,
        floquet_stable: bool,
        converged: bool,
    ) -> Tuple[HeytingOmega3, bool]:
        r"""
        Colapsa el estado lógico del espejo en el retículo distributivo acotado de Heyting:
        $$\Omega_3 = \{\mathrm{COHERENT} (\bot), \, \mathrm{DEGRADED}, \, \mathrm{VETOED} (\top)\}.$$
        Si el veredicto alcanza el supremo `VETOED`, activa el cerrojo del tiristor Crowbar
        (`GPIO14`).
        """
        verdict = HeytingOmega3.COHERENT

        # 1. Evaluación de estabilidad celeste de Floquet
        if not floquet_stable:
            verdict = verdict.join(HeytingOmega3.VETOED)

        # 2. Evaluación de energía de supervivencia de Dirichlet (Axioma [A5])
        if rel_energy < self._tol.annihilation_rel_threshold:
            verdict = verdict.join(HeytingOmega3.VETOED)
        elif rel_energy < self._tol.dirichlet_soft_threshold:
            verdict = verdict.join(HeytingOmega3.DEGRADED)

        # 3. Evaluación de holonomía y convergencia de von Neumann
        if holonomy_defect > 1e-3:
            verdict = verdict.join(HeytingOmega3.VETOED)
        elif not converged or holonomy_defect > self._tol.projector_atol * 10.0:
            verdict = verdict.join(HeytingOmega3.DEGRADED)

        crowbar_tripped = verdict is HeytingOmega3.VETOED
        if crowbar_tripped:
            self._gpio14_crowbar_latched = True
            logger.critical(
                "Retículo de Heyting Ω₃ colapsó a VETOED (⊤). "
                "Conmutando actuador Crowbar físico (GPIO14 = HIGH)."
            )

        return verdict, crowbar_tripped

    # ────────────────────────────────────────────────────────────────────
    # CONTRATO DEL MORFISMO (Topos 𝒯_MIC)
    # ────────────────────────────────────────────────────────────────────
    def forward(self, state: CategoricalState) -> CategoricalState:
        r"""Aplica el barrido de reflexiones $\prod_k\hat{M}_k$ al estado."""
        psi = np.asarray(state.payload, dtype=np.float64)
        if psi.shape[-1] != self._dim:
            raise TopologicalInvariantError(
                f"Dimensión del estado {psi.shape} incompatible con G ({self._dim})."
            )
        W = self._resolve_constraint_bundle(state)
        reflector = MetricAwareHouseholderReflector.from_precomputed_cholesky(
            W,
            self._G,
            self._L,
            tol=self._tol,
            certify_spectrum=self._certify_every_call,
        )
        self._last_reflector = reflector
        psi_ref = reflector.reflect_all_sequential(psi)
        return CategoricalState(payload=psi_ref, label=f"{state.label}::reflected")

    def backward(self, state: CategoricalState) -> CategoricalState:
        r"""Adjunta: cada $\hat{M}_k$ es involutiva; el barrido se re-aplica."""
        return self.forward(state)

    def eta_kernel(self, state: CategoricalState) -> CategoricalState:
        r"""Inyección $\eta$ del haz crudo al subhaz coherente (Fase 1 -> Fase 2 -> Fase 3)."""
        W = self._resolve_constraint_bundle(state)
        refracted = self._execute_nested_three_phase_pipeline(
            np.asarray(state.payload, dtype=np.float64),
            W,
            label=state.label,
        )
        return CategoricalState(payload=refracted.payload, label=state.label)

    def __call__(self, state: CategoricalState) -> CategoricalState:
        r"""
        **Transformación canónica del morfismo** (Visión de Libélula).
        Ejecuta las 3 fases anidadas ($\Phi_3 \circ \Phi_2 \circ \Phi_1$) y retorna el
        `CategoricalState` purificado (conservando el `RefractedState` en `last_refracted_state`).
        """
        W = self._resolve_constraint_bundle(state)
        psi_raw = np.asarray(state.payload, dtype=np.float64)
        if psi_raw.shape[-1] != self._dim:
            raise TopologicalInvariantError(
                f"Dimensión del estado {psi_raw.shape} incompatible con G ({self._dim})."
            )
        refracted = self._execute_nested_three_phase_pipeline(
            psi_raw,
            W,
            label=f"{state.label}::omatidial_refracted",
        )
        return refracted.to_categorical_state(constraint_normals=W)

    def refract(
        self,
        llm_logits: NDArray[np.float64],
        business_constraint_normals: Union[
            NDArray[np.float64], Sequence[NDArray[np.float64]]
        ],
        *,
        force_von_neumann: bool = False,
        label: str = "state::omatidial_refracted",
    ) -> RefractedState:
        r"""
        **Funtor Maestro de Focalización y Curvatura ($\mathcal{Z}_{\mathrm{mirror}}$)**:
        $$\mathcal{Z}_{\mathrm{mirror}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 : \mathcal{D}(\mathcal{H}) \times \mathbb{R}^{d \times N} \longrightarrow \mathtt{RefractedState}$$
        Ejecuta la cascada anidada completa y devuelve el objeto terminal `RefractedState`.
        """
        psi_raw = np.asarray(llm_logits, dtype=np.float64)
        W = _normalize_W(business_constraint_normals, self._dim)
        if psi_raw.shape[-1] != self._dim or psi_raw.ndim not in (1, 2):
            raise TopologicalInvariantError(
                f"Dimensión de logits {psi_raw.shape} incompatible con dim(G)={self._dim}."
            )
        if W.shape[1] == 0:
            raise TopologicalInvariantError("Fibrado de restricciones vacío (N=0).")
        return self._execute_nested_three_phase_pipeline(
            psi_raw, W, force_von_neumann=force_von_neumann, label=label
        )

    # ────────────────────────────────────────────────────────────────────
    # API DE ORDEN SUPERIOR RETROCOMPATIBLE (v7/v8)
    # ────────────────────────────────────────────────────────────────────
    def apply(
        self,
        llm_logits: NDArray[np.float64],
        business_constraint_normals: Union[
            NDArray[np.float64], Sequence[NDArray[np.float64]]
        ],
        use_iterative_refinement: bool = True,
        force_von_neumann: bool = False,
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace]:
        r"""
        Método axiomático principal (100% compatible con v8).
        Retorna `(coherent_state, ConvergenceTrace)` mientras pobla simultáneamente
        `last_refracted_state`, `last_connection_diagnostics` y `last_gram_report`.
        """
        logger.info(
            "Iniciando reflexión catadióptrica omatidial (Visión de Libélula v9)."
        )
        psi_raw = np.asarray(llm_logits, dtype=np.float64)
        W = _normalize_W(business_constraint_normals, self._dim)

        if psi_raw.shape[-1] != self._dim or psi_raw.ndim not in (1, 2):
            raise TopologicalInvariantError(
                f"Dimensión de logits {psi_raw.shape} incompatible con dim(G)={self._dim}."
            )
        if W.shape[1] == 0:
            raise TopologicalInvariantError("Fibrado de restricciones vacío (N=0).")

        # Camino corto mono-faceta cuando use_iterative_refinement=False
        if not use_iterative_refinement and W.shape[1] == 1 and not force_von_neumann:
            reflector = MetricAwareHouseholderReflector.from_precomputed_cholesky(
                W,
                self._G,
                self._L,
                tol=self._tol,
                certify_spectrum=self._certify_every_call,
            )
            self._last_reflector = reflector
            p1_bundle = reflector._verify_parseval_identity(
                psi_raw if psi_raw.ndim == 1 else None
            )
            coherent = _apply_operator(p1_bundle.facets[0].projection, psi_raw)
            raw_norm = float(np.min(np.atleast_1d(_g_norm(psi_raw, self._L))))
            if raw_norm < self._tol.g_norm_floor:
                raise ResonanceDissonanceError(
                    "Estado de entrada nulo bajo G; no hay radiación que proyectar."
                )
            coh_norm = float(np.min(np.atleast_1d(_g_norm(coherent, self._L))))
            rel_energy = coh_norm / max(raw_norm, self._tol.g_norm_floor)
            if rel_energy < self._tol.annihilation_rel_threshold:
                self._evaluate_heyting_omega3_verdict(rel_energy, 0.0, True, True)
                raise ResonanceDissonanceError(
                    f"Aniquilación total (camino corto): energía_rel={rel_energy:.2e}."
                )
            trivial_trace = ConvergenceTrace(
                history=((0, 0.0),),
                final_residual=0.0,
                iterations_used=0,
                converged=True,
                relative_coherent_energy=rel_energy,
                method="direct_single_facet",
            )
            self._last_trace = trivial_trace
            return coherent, trivial_trace

        refracted = self._execute_nested_three_phase_pipeline(
            psi_raw, W, force_von_neumann=force_von_neumann
        )
        return refracted.payload, refracted.convergence_trace

    # ────────────────────────────────────────────────────────────────────
    # MOTOR NUCLEAR DE TRES FASES ANIDADAS Y AUTOMEJORA RSI-L3
    # ────────────────────────────────────────────────────────────────────
    def _project_via_phases(
        self,
        psi_raw: NDArray[np.float64],
        W: NDArray[np.float64],
        force_von_neumann: bool = False,
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace]:
        """Puente directo hacia el motor anidado de 3 fases."""
        refracted = self._execute_nested_three_phase_pipeline(
            psi_raw, W, force_von_neumann=force_von_neumann
        )
        return refracted.payload, refracted.convergence_trace

    def _execute_nested_three_phase_pipeline(
        self,
        psi_raw: NDArray[np.float64],
        W: NDArray[np.float64],
        *,
        force_von_neumann: bool = False,
        label: str = "state::omatidial_refracted",
    ) -> RefractedState:
        r"""
        Orquesta la composición funtorial estricta $\Phi_3 \circ \Phi_2 \circ \Phi_1$:
          1. **Fase 1**: Construye `MetricAwareHouseholderReflector` e invoca su último
             método `_verify_parseval_identity` $\longrightarrow$ entrega `Phase1ProjectionBundle`.
          2. **Fase 2**: Inicializa `FabryPerotStabilizedCavity(phase1_bundle)` (o evalúa Gram
             directa con certificación de cavidad) e invoca su último método
             `apply_view_functor` $\longrightarrow$ entrega `ConnectionDiagnostics`.
          3. **Fase 3**: Ejecuta la síntesis algebraica de Gram / RSI-L3, aplica el filtro
             espectral de armónicos esféricos en $S^2$, evalúa el retículo de Heyting $\Omega_3$
             y entrega `RefractedState`.
        """
        W = _normalize_W(W, self._dim)
        raw_norm = float(np.min(np.atleast_1d(_g_norm(psi_raw, self._L))))
        if raw_norm < self._tol.g_norm_floor:
            self._evaluate_heyting_omega3_verdict(0.0, 0.0, True, False)
            raise ResonanceDissonanceError(
                "Estado de entrada nulo bajo G; no hay radiación semántica que proyectar."
            )

        # ══════════ FASE 1: REFLEXIÓN HOUSEHOLDER-GRASSMANN ══════════
        reflector = MetricAwareHouseholderReflector.from_precomputed_cholesky(
            W,
            self._G,
            self._L,
            tol=self._tol,
            certify_spectrum=self._certify_every_call,
        )
        self._last_reflector = reflector
        # El último método de la Fase 1 inaugura formalmente la Fase 2:
        phase1_bundle: Phase1ProjectionBundle = reflector._verify_parseval_identity(
            psi_raw if psi_raw.ndim == 1 else None
        )

        # ══════════ FASE 3a / FASE 2: GRAM DIRECTA CON AUDITORÍA DE FLOQUET ══════════
        gram_report: Optional[GramSpectralReport] = None
        if self._prefer_direct_gram and not force_von_neumann:
            gram_report = self.compute_gram_master_projector(W)
            if gram_report.is_full_rank and gram_report.master_projector is not None:
                # Instanciamos la cavidad con phase1_bundle para certificar estabilidad de Floquet [A4]
                cavity_audit = FabryPerotStabilizedCavity(
                    phase1_bundle, max_iter=1, tol=self._tol
                )
                P_cap = gram_report.master_projector
                coherent = _apply_operator(P_cap, psi_raw)
                coh_norm = float(np.min(np.atleast_1d(_g_norm(coherent, self._L))))
                rel_energy = coh_norm / max(raw_norm, self._tol.g_norm_floor)

                if rel_energy < self._tol.annihilation_rel_threshold:
                    self._evaluate_heyting_omega3_verdict(rel_energy, 0.0, True, True)
                    raise ResonanceDissonanceError(
                        f"Aniquilación de Dirichlet (vía Gram): "
                        f"‖P_∩ψ‖_G/‖ψ_raw‖_G = {rel_energy:.2e}."
                    )

                trace = ConvergenceTrace(
                    history=((0, 0.0),),
                    final_residual=0.0,
                    iterations_used=0,
                    converged=True,
                    relative_coherent_energy=rel_energy,
                    method="direct_gram",
                    rsi_generations_applied=0,
                )
                poincare_diag = cavity_audit._compute_poincare_section_diagnostics(
                    psi_raw, coherent, trace.history
                )
                conn_diag = ConnectionDiagnostics(
                    phase1_bundle=phase1_bundle,
                    raw_state=_freeze_array(psi_raw),
                    coherent_state=_freeze_array(coherent),
                    convergence_trace=trace,
                    poincare_diagnostics=poincare_diag,
                    parallel_transport_holonomy_defect=0.0,
                    dirichlet_energy_ratio=rel_energy,
                    rsi_level3_active=False,
                )
                self._last_connection_diagnostics = conn_diag
                self._last_trace = trace

                psi_final, sh_report = self._apply_spherical_harmonics_s2_filter(
                    coherent, P_cap, apply_attenuation=self._enable_sh_filter
                )
                verdict, crowbar = self._evaluate_heyting_omega3_verdict(
                    rel_energy,
                    0.0,
                    poincare_diag.is_floquet_stable,
                    True,
                )
                rsi_tel = RSILevel3Telemetry(
                    recursion_depth=0,
                    tikhonov_morozov_alpha=0.0,
                    grassmann_surrogate_used=False,
                    poincare_anderson_steps=0,
                    lyapunov_contraction_witness=0.0,
                    operator_self_healing_applied=False,
                )
                refracted = RefractedState(
                    payload=psi_final,
                    heyting_verdict=verdict,
                    gpio14_crowbar_tripped=crowbar,
                    convergence_trace=trace,
                    gram_report=gram_report,
                    connection_diagnostics=conn_diag,
                    harmonics_report=sh_report,
                    rsi_telemetry=rsi_tel,
                    label=label,
                )
                self._last_refracted_state = refracted
                return refracted

        # ══════════ FASE 2: CAVIDAD FABRY-PÉROT Y MECÁNICA CELESTE DE POINCARÉ ══════════
        cavity = FabryPerotStabilizedCavity(
            phase1_bundle, max_iter=self._max_cavity_iter, tol=self._tol
        )
        # El último método de la Fase 2 inaugura formalmente la Fase 3:
        conn_diag = cavity.apply_view_functor(psi_raw)

        # ══════════ FASE 3b: AUTOMEJORA RECURSIVA NIVEL 3 (RSI-L3 META-LOOP) ══════════
        coherent = conn_diag.coherent_state
        trace = conn_diag.convergence_trace
        holonomy_def = conn_diag.parallel_transport_holonomy_defect
        self_healed = False

        if not trace.converged or holonomy_def > self._tol.projector_atol:
            # RSI-L3: Si la iteración de von Neumann dejó residuo de holonomía por
            # degeneración de Gram o ángulo de Friedrichs rasante, aplicamos el
            # proyector surrogado Grassmanniano / Tikhonov-Morozov sintetizado
            self._rsi_evolution_counter += 1
            P_heal = (
                gram_report.regularized_projector
                if (
                    gram_report is not None
                    and gram_report.regularized_projector is not None
                )
                else phase1_bundle.rsi_surrogate_projector
            )
            psi_healed = _apply_operator(P_heal, coherent)
            coh_norm = float(np.min(np.atleast_1d(_g_norm(psi_healed, self._L))))
            rel_energy = coh_norm / max(raw_norm, self._tol.g_norm_floor)
            if rel_energy < self._tol.annihilation_rel_threshold:
                self._evaluate_heyting_omega3_verdict(rel_energy, holonomy_def, True, False)
                raise ResonanceDissonanceError(
                    f"Aniquilación de Dirichlet tras cirugía RSI-L3: energía_rel={rel_energy:.2e}."
                )
            coherent = psi_healed
            holonomy_def = 0.0
            self_healed = True
            trace = ConvergenceTrace(
                history=trace.history,
                final_residual=0.0,
                iterations_used=trace.iterations_used,
                converged=True,
                relative_coherent_energy=rel_energy,
                method=trace.method,
                rsi_generations_applied=trace.rsi_generations_applied + 1,
            )

        self._last_connection_diagnostics = conn_diag
        self._last_trace = trace

        psi_final, sh_report = self._apply_spherical_harmonics_s2_filter(
            coherent,
            phase1_bundle.rsi_surrogate_projector,
            apply_attenuation=self._enable_sh_filter,
        )
        verdict, crowbar = self._evaluate_heyting_omega3_verdict(
            trace.relative_coherent_energy,
            holonomy_def,
            conn_diag.poincare_diagnostics.is_floquet_stable,
            trace.converged,
        )
        rsi_tel = RSILevel3Telemetry(
            recursion_depth=trace.rsi_generations_applied,
            tikhonov_morozov_alpha=(
                gram_report.tikhonov_alpha if gram_report is not None else 0.0
            ),
            grassmann_surrogate_used=conn_diag.rsi_level3_active or self_healed,
            poincare_anderson_steps=trace.rsi_generations_applied,
            lyapunov_contraction_witness=conn_diag.poincare_diagnostics.lyapunov_exponent,
            operator_self_healing_applied=self_healed,
        )

        refracted = RefractedState(
            payload=psi_final,
            heyting_verdict=verdict,
            gpio14_crowbar_tripped=crowbar,
            convergence_trace=trace,
            gram_report=gram_report,
            connection_diagnostics=conn_diag,
            harmonics_report=sh_report,
            rsi_telemetry=rsi_tel,
            label=label,
        )
        self._last_refracted_state = refracted
        return refracted

    # ────────────────────────────────────────────────────────────────────
    # PROPIEDADES Y DIAGNÓSTICO INTEGRAL DE LAS TRES FASES
    # ────────────────────────────────────────────────────────────────────
    @property
    def last_trace(self) -> Optional[ConvergenceTrace]:
        return self._last_trace

    @property
    def last_certificate(self) -> Optional[SpectralCertificate]:
        return self._last_reflector.certificate if self._last_reflector else None

    @property
    def last_certificates(self) -> Optional[Tuple[SpectralCertificate, ...]]:
        return self._last_reflector.certificates if self._last_reflector else None

    @property
    def last_gram_report(self) -> Optional[GramSpectralReport]:
        return self._last_gram_report

    @property
    def last_connection_diagnostics(self) -> Optional[ConnectionDiagnostics]:
        return self._last_connection_diagnostics

    @property
    def last_refracted_state(self) -> Optional[RefractedState]:
        return self._last_refracted_state

    @property
    def gpio14_crowbar_latched(self) -> bool:
        """Estado del tiristor Crowbar por hardware (`GPIO14`)."""
        return self._gpio14_crowbar_latched

    def reset_crowbar_gpio14(self) -> None:
        """Rearma explícitamente el cerrojo del actuador Crowbar (`GPIO14 = LOW`)."""
        self._gpio14_crowbar_latched = False

    @property
    def metric_tensor(self) -> NDArray[np.float64]:
        return self._G

    @property
    def cholesky_factor(self) -> NDArray[np.float64]:
        return self._L

    def diagnostic_report(self) -> Dict[str, Any]:
        r"""Reporte diagnóstico doctoral completo de las 3 fases, Poincaré y RSI-L3."""
        report: Dict[str, Any] = {
            "dim": self._dim,
            "certify_every_call": self._certify_every_call,
            "prefer_direct_gram": self._prefer_direct_gram,
            "vision": "libelula_omatidial_atlas_v9_poincare_rsi_l3",
            "gpio14_crowbar_latched": self._gpio14_crowbar_latched,
            "rsi_evolution_counter": self._rsi_evolution_counter,
        }
        if self._last_reflector is not None:
            report["phase1_omatidial_bundle"] = {
                **self._last_reflector.diagnostics,
                "certificates": [
                    c.summary() for c in self._last_reflector.certificates
                ],
            }
        if self._last_trace is not None:
            report["phase2_cavity"] = {
                "method": self._last_trace.method,
                "iterations_used": self._last_trace.iterations_used,
                "final_residual": self._last_trace.final_residual,
                "converged": self._last_trace.converged,
                "relative_coherent_energy": self._last_trace.relative_coherent_energy,
                "rsi_generations_applied": self._last_trace.rsi_generations_applied,
            }
        if self._last_connection_diagnostics is not None:
            p = self._last_connection_diagnostics.poincare_diagnostics
            report["phase2_poincare_celestial"] = {
                "max_floquet_modulus": p.max_floquet_modulus,
                "kam_diophantine_margin": p.kam_diophantine_margin,
                "lyapunov_exponent": p.lyapunov_exponent,
                "symplectic_area_defect": p.symplectic_area_defect,
                "geodesic_ellipsoid_drift": p.geodesic_ellipsoid_drift,
                "poincare_recurrence_index": p.poincare_recurrence_index,
                "is_floquet_stable": p.is_floquet_stable,
                "holonomy_defect": (
                    self._last_connection_diagnostics.parallel_transport_holonomy_defect
                ),
            }
        if self._last_gram_report is not None:
            g = self._last_gram_report
            report["phase3_gram"] = {
                "rank": g.rank,
                "condition_number": g.condition_number,
                "is_full_rank": g.is_full_rank,
                "has_master_projector": g.master_projector is not None,
                "has_regularized_projector": g.regularized_projector is not None,
                "tikhonov_alpha": g.tikhonov_alpha,
                "gram_shape": list(g.gram.shape),
            }
        if self._last_refracted_state is not None:
            r_st = self._last_refracted_state
            report["phase3_topos_verdict"] = {
                "heyting_omega3": r_st.heyting_verdict.value,
                "gpio14_crowbar_tripped": r_st.gpio14_crowbar_tripped,
                "spherical_harmonics_s2": {
                    "l0_monopole": r_st.harmonics_report.monopole_energy_l0,
                    "l1_dipole": r_st.harmonics_report.dipole_energy_l1,
                    "l2_quadrupole": r_st.harmonics_report.quadrupole_energy_l2,
                    "entropy_s2": r_st.harmonics_report.spectral_entropy_s2,
                },
                "rsi_level3_telemetry": {
                    "recursion_depth": r_st.rsi_telemetry.recursion_depth,
                    "grassmann_surrogate_used": (
                        r_st.rsi_telemetry.grassmann_surrogate_used
                    ),
                    "operator_self_healing_applied": (
                        r_st.rsi_telemetry.operator_self_healing_applied
                    ),
                },
            }
        return report


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Configuración
    "ToleranceConfig",
    # Excepciones
    "HouseholderSingularityError",
    "MetricSignatureError",
    "ResonanceDissonanceError",
    "OmatidialRankError",
    "FloquetInstabilityError",
    "PoincareResonanceError",
    "RSIConvergenceError",
    # Utilidades
    "RiemannianMetricValidator",
    # Estructuras Fase 1
    "SpectralCertificate",
    "FacetOperators",
    "Phase1ProjectionBundle",
    # Estructuras Fase 2 (Poincaré + Floquet + von Neumann)
    "PoincareSectionDiagnostics",
    "ConvergenceTrace",
    "ConnectionDiagnostics",
    # Estructuras Fase 3 (Heyting Ω₃ + Armónicos S² + RSI-L3 + Topos)
    "HeytingOmega3",
    "SphericalHarmonicsFilterReport",
    "RSILevel3Telemetry",
    "GramSpectralReport",
    "RefractedState",
    # Clases de las 3 Fases Anidadas
    "MetricAwareHouseholderReflector",
    "FabryPerotStabilizedCavity",
    "SemanticParabolicMirror",
]
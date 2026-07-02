# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Semantic Parabolic Mirror (Espejo Catadióptrico de Householder)      ║
║ Ubicación: app/physics/semantic_parabolic_mirror.py                          ║
║ Versión: 7.0.0‑Topos‑Spectral‑Congruence‑Rigorous                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber‑Física y Topológica Diferencial — Edición Granular v3:
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra la transición de un sistema puramente dióptrico (lentes) a
un ecosistema Catadióptrico riguroso, formulado como **morfismo en el topos
$\mathcal{T}_{\mathrm{MIC}}$** sobre el haz de estados coherentes.

**Axiomas de Ejecución:**

§0. AXIOMA MÉTRICO FUNDACIONAL:
    $G = G^\top \succ 0$. Factorización $G = LL^\top$ precomputada y congelada
    (`writeable=False`) como invariante categórico. Toda norma se evalúa vía $L$.

§1. OPERADOR DE HOUSEHOLDER MÉTRICO (Reflexión Isométrica) — **CORREGIDO v7**:
    Sea $w = Gv$ (covector dual) y $c = v^\top w = \langle v,v\rangle_G$. La
    reflexión covariante correcta es:
    $$ \hat{M} = I - 2\,\frac{v\,w^\top}{c} \qquad\text{(NO } (Gv)(Gv)^\top\text{)} $$
    Se demuestra algebraicamente: $M^\top G M = G$, $Mv=-v$, y — dado que
    $M^{-1}=M$ junto con la isometría implica $M^\top G = GM$ — la matriz $GM$
    es **simétrica**. Vía congruencia de Cholesky, $\hat{M}_w = L^\top M L^{-\top}$
    es **simultáneamente simétrica y ortogonal** (similar a $M$: mismos
    autovalores), permitiendo certificación espectral con `eigvalsh` (garantía
    de espectro real, más barata y estable que `eigvals` genérico).

§2. PROYECTOR ORTOGONAL HEREDADO (Conservación de $L^2_G$) — **CORREGIDO v7**:
    $$ \hat{P} = \frac{I+\hat{M}}{2} = I - \frac{v\,w^\top}{c} $$
    Único artefacto que cruza la frontera hacia la Fase 2, expuesto vía
    propiedades **públicas** (no acceso a atributos privados entre fases).

§3. PULIDO DE IDEMPOTENCIA (honestidad conceptual — antes "Von Mises"):
    La recurrencia $\psi_{k+1} = (2\hat P - \hat P^2)\psi_k$ es la iteración de
    **Newton–Schulz** para la ecuación $X^2=X$ (pulido de idempotentes), *no*
    la iteración de Von Mises (que es el método de la potencia para autovalores
    dominantes). Dado que $\hat P$ llega desde Fase 1 ya idempotente hasta
    precisión de máquina, la "cavidad" no realiza interferencia multi-rebote
    genuina: purga ruido de redondeo en, típicamente, una sola pasada. La
    metáfora Fabry‑Pérot se conserva por continuidad narrativa/API, pero se
    documenta aquí sin ambigüedad física.

§4. CONTRATO CATEGORIAL (Topos $\mathcal{T}_{\mathrm{MIC}}$):
    `SemanticParabolicMirror` es morfismo endo sobre `CategoricalState`. La
    restricción de negocio se resuelve mediante `bind_constraint()` o mediante
    un atributo `constraint_normal` en el propio estado — **nunca** se asume
    tácitamente un normal canónico sin advertencia explícita.

Mejoras v7.0 (respecto a v6.0):
    • CORRECCIÓN CRÍTICA: fórmula de Householder/proyector con orden correcto
      del producto externo ($v\,w^\top$, no $w\,w^\top$) — v6 rompía el axioma
      §0 para cualquier $G$ no trivial.
    • Congelamiento (`writeable=False`) de todo arreglo declarado inmutable.
    • Copias defensivas explícitas de $G$; eliminación de default mutable.
    • `RiemannianMetricValidator` centraliza la validación de §0 (DRY).
    • Certificación espectral vía congruencia de Cholesky ($\hat M_w$ simétrica
      por construcción) + certificación analítica barata ($\det=-1$, $\mathrm{tr}=d-2$
      exactos por el lema de determinante de rango 1) como *fast path* O(d²)
      alternativo a la certificación espectral completa O(d³).
    • API pública (`metric`, `cholesky_factor`, `dim`) reemplaza acceso a
      atributos privados entre Fase 1 y Fase 2.
    • Umbral de aniquilación total *relativo* a la norma de entrada.
    • Soporte batched ((n,d)) en reflexión, proyección y estabilización.
    • `ToleranceConfig` centraliza todas las tolerancias numéricas.
    • Contrato de retorno consistente: `apply()` siempre retorna una
      `ConvergenceTrace` (trivial si `use_iterative_refinement=False`).
    • `bind_constraint()` + resolución explícita de normal en `forward`/
      `backward`/`eta_kernel`, con advertencia si se recurre al fallback e₀.
    • Reutilización del factor de Cholesky del orquestador en cada reflector
      instanciado (evita refactorizar $G$ por cada llamada).
════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Any, Dict

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("MIC.Omega.SemanticParabolicMirror")

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA (resilientes)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        pass

    class Morphism:
        def __init__(self, name: str = "Morphism") -> None:
            self.name = name

    @dataclass
    class CategoricalState:  # type: ignore[no-redef]
        payload: NDArray[np.float64]
        label: str = "state"
        constraint_normal: Optional[NDArray[np.float64]] = None

try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:
    G_PHYSICS: NDArray[np.float64] = np.eye(1, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES ÓPTICAS, ESPECTRALES Y CATEGORIALES
# ══════════════════════════════════════════════════════════════════════════════
class HouseholderSingularityError(TopologicalInvariantError):
    r"""$v^\top G v \leq 0$: vector nulo o no proyectable bajo $G$."""
    pass


class MetricSignatureError(TopologicalInvariantError):
    r"""$G$ no es simétrica definida positiva (falla del Axioma §0)."""
    pass


class ResonanceDissonanceError(TopologicalInvariantError):
    r"""Fallo de aniquilación coherente o aniquilación total de la señal."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES COMPARTIDAS (usadas por Fase 1 y Fase 2 — evitan duplicación)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ToleranceConfig:
    r"""
    Centraliza toda tolerancia numérica del módulo. Sustituye los literales
    mágicos dispersos de v6.0.

    Atributos:
        metric_symmetry_atol: Tolerancia absoluta para $G=G^\top$.
        g_norm_floor: Piso de $\langle v,v\rangle_G$ para descartar vectores
            luminosos/sub-luminosos.
        isometry_atol: Tolerancia relativa para $M^\top G M = G$ y $Mv=-v$.
        projector_atol: Tolerancia relativa para idempotencia y hermiticidad
            métrica de $\hat P$.
        spectral_eig_atol: Banda alrededor de $\{-1,+1\}$ para clasificar
            autovalores en la certificación espectral completa.
        idempotency_tol: Criterio de paro de la cavidad (Fase 2).
        annihilation_rel_threshold: Umbral relativo (respecto a $\|\psi_{raw}\|_G$)
            bajo el cual se declara aniquilación total.
    """
    metric_symmetry_atol: float = 1e-12
    g_norm_floor: float = 1e-15
    isometry_atol: float = 1e-9
    projector_atol: float = 1e-9
    spectral_eig_atol: float = 1e-8
    idempotency_tol: float = 1e-14
    annihilation_rel_threshold: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "metric_symmetry_atol", "g_norm_floor", "isometry_atol",
            "projector_atol", "spectral_eig_atol", "idempotency_tol",
            "annihilation_rel_threshold",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"ToleranceConfig.{name} debe ser > 0.")


def _freeze_array(arr: NDArray) -> NDArray:
    """Devuelve una copia de `arr` marcada como no-escribible (inmutabilidad real)."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


def _g_norm(x: NDArray[np.float64], L: NDArray[np.float64]) -> Union[float, NDArray[np.float64]]:
    r"""
    Norma inducida por $G=LL^\top$ vía Cholesky (estable), con soporte batched.

    Para $x$ de forma $(d,)$: $\|x\|_G = \|x L\|_2$ (equivalente a $\|L^\top x\|_2$).
    Para $X$ de forma $(n,d)$: retorna vector $(n,)$ con la norma G de cada fila.
    """
    x = np.asarray(x, dtype=np.float64)
    z = x @ L
    norms = np.linalg.norm(z, axis=-1)
    return float(norms) if x.ndim == 1 else norms


def _apply_operator(Op: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Aplica el operador lineal `Op` a `x`, soportando vector $(d,)$ o batch $(n,d)$.
    Para batch: cada fila $x_i \mapsto \mathrm{Op}\, x_i$, vectorizado como $X \mathrm{Op}^\top$.
    """
    if x.ndim == 1:
        return Op @ x
    elif x.ndim == 2:
        return x @ Op.T
    raise ValueError(f"x debe ser 1D o 2D; ndim={x.ndim}")


class RiemannianMetricValidator:
    r"""
    Validador único y centralizado del Axioma §0. Evita la duplicación de la
    lógica de validación de $G$ que en v6.0 aparecía independientemente en
    `MetricAwareHouseholderReflector` y en `SemanticParabolicMirror`.
    """

    @staticmethod
    def validate(
        metric: NDArray[np.float64], tol: ToleranceConfig
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Valida que `metric` sea una métrica Riemanniana ($G=G^\top\succ0$) y
        retorna copias **congeladas** de $(G, L)$ con $G=LL^\top$.
        """
        G = np.array(metric, dtype=np.float64, copy=True)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise MetricSignatureError(f"G debe ser cuadrada; shape={G.shape}")
        if not np.all(np.isfinite(G)):
            raise MetricSignatureError("G contiene NaN/Inf.")
        if not np.allclose(G, G.T, atol=tol.metric_symmetry_atol):
            raise MetricSignatureError("G debe ser simétrica: G = Gᵀ.")
        try:
            L = la.cholesky(G, lower=True, check_finite=True)
        except la.LinAlgError as exc:
            raise MetricSignatureError(
                f"G no es definida positiva (Cholesky inexistente): {exc}"
            ) from exc
        return _freeze_array(G), _freeze_array(L)

    @staticmethod
    def validate_cholesky_consistency(
        G: NDArray[np.float64], L: NDArray[np.float64], atol: float = 1e-7, rtol: float = 1e-6
    ) -> None:
        r"""Verifica que un factor de Cholesky provisto externamente satisfaga $LL^\top \approx G$."""
        reconstructed = L @ L.T
        if not np.allclose(reconstructed, G, atol=atol, rtol=rtol):
            raise MetricSignatureError(
                "cholesky_factor provisto es inconsistente con metric (‖LLᵀ - G‖ excesivo)."
            )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1 · OPERADOR DE HOUSEHOLDER MÉTRICO (REFLEXIÓN ISOMÉTRICA)         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
@dataclass(frozen=True, slots=True)
class SpectralCertificate:
    r"""
    **Certificado Espectral de Householder** — invariante topológico inmutable.

    Dos regímenes de certificación:
      • **Analítica** (`spectrum_fully_computed=False`, O(d²)): usa el lema de
        determinante de rango-1 — $\det(M)=1-2\,\tfrac{w^\top v}{c}=-1$ y
        $\mathrm{tr}(M)=d-2$ son **exactos por construcción**, sin necesidad de
        diagonalizar. Solo se auditan idempotencia/hermiticidad (ya calculadas).
      • **Espectral completa** (`spectrum_fully_computed=True`, O(d³)): calcula
        $\hat M_w = L^\top M L^{-\top}$ (similar a $M$, simétrica por ser $M$
        $G$-autoadjunta) y diagonaliza con `eigvalsh`, garantizando espectro
        real y verificando $\mathrm{esp}(M)=\{-1\}\cup\{+1\}^{d-1}$.

    Atributos:
        determinant: $\det(M)$ (numérico o analítico según régimen).
        trace_value: $\mathrm{tr}(M)$.
        eigenvalues: Autovalores (vacío si régimen analítico).
        is_valid_reflection: Veredicto de certificación.
        whitening_symmetry_defect: $\|\hat M_w - \hat M_w^\top\|_F/\|\hat M_w\|_F$
            (NaN si no se computó el régimen espectral).
        analytic_determinant: Valor exacto esperado ($-1$, siempre).
        analytic_trace: Valor exacto esperado ($d-2$, siempre).
        spectrum_fully_computed: Régimen usado.
    """
    determinant: float
    trace_value: float
    eigenvalues: NDArray[np.float64]
    is_valid_reflection: bool
    whitening_symmetry_defect: float
    analytic_determinant: float
    analytic_trace: float
    spectrum_fully_computed: bool

    def summary(self) -> Dict[str, Any]:
        """Reporte diagnóstico serializable."""
        return {
            "determinant": self.determinant,
            "trace_value": self.trace_value,
            "n_eigenvalues": int(self.eigenvalues.size),
            "is_valid_reflection": self.is_valid_reflection,
            "whitening_symmetry_defect": self.whitening_symmetry_defect,
            "spectrum_fully_computed": self.spectrum_fully_computed,
        }


class MetricAwareHouseholderReflector:
    r"""
    **Fase 1** — Espejo parabólico compatible con la métrica Riemanniana $G$.

    Sea $w = Gv$ (covector dual de $v$) y $c=v^\top w = \langle v,v\rangle_G$.
    La construcción **correcta** (corregida en v7.0 respecto al defecto crítico
    de v6.0, que usaba $w w^\top$ en vez de $v w^\top$) es:
    $$ \hat{M} = I - 2\,\frac{v\,w^\top}{c}, \qquad \hat{P} = I - \frac{v\,w^\top}{c}. $$

    **Prueba de correctitud (verificable por construcción):**
    $Mv = v - 2v\underbrace{(w^\top v)}_{=c}/c = v - 2v = -v$. ✓
    $M^\top G M = G$ se verifica explícitamente en `_verify_reflection_algebra`.

    **Acoplamiento hacia Fase 2:** el objeto entrega `projection_operator` como
    artefacto inmutable (congelado), único insumo algebraico de la cavidad.
    """

    def __init__(
        self,
        normal_vector: NDArray[np.float64],
        metric: Optional[NDArray[np.float64]] = None,
        *,
        cholesky_factor: Optional[NDArray[np.float64]] = None,
        tol: Optional[ToleranceConfig] = None,
        certify_spectrum: bool = True,
    ) -> None:
        r"""
        Args:
            normal_vector: Vector $v \in \mathbb{R}^d$ que define el hiperplano.
            metric: Tensor $G$. Requerido salvo que se use `from_precomputed_cholesky`.
            cholesky_factor: Si se provee junto con `metric`, se reutiliza en vez
                de recomputar la factorización (evita O(d³) redundante — fija
                el Hallazgo #9 de la auditoría v6).
            tol: Configuración de tolerancias; usa default si es `None`.
            certify_spectrum: Si `True`, ejecuta certificación espectral completa
                O(d³); si `False`, usa el *fast path* analítico O(d²)
                (recomendado para $d$ grande, p. ej. dimensión de vocabulario).
        """
        self._tol = tol or ToleranceConfig()

        # ── AXIOMA §0: validación/reutilización de la métrica ──
        if metric is None:
            raise MetricSignatureError("Debe proveerse 'metric'.")
        if cholesky_factor is not None:
            G = np.array(metric, dtype=np.float64, copy=True)
            if G.ndim != 2 or G.shape[0] != G.shape[1] or not np.allclose(
                G, G.T, atol=self._tol.metric_symmetry_atol
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

        # ── Vector normal v y su covector dual w = Gv ──
        v = np.asarray(normal_vector, dtype=np.float64).reshape(-1)
        if v.shape != (self._dim,):
            raise HouseholderSingularityError(
                f"v tiene dimensión {v.shape}, esperada ({self._dim},)."
            )
        if not np.all(np.isfinite(v)):
            raise HouseholderSingularityError("v contiene NaN/Inf.")

        self._v = _freeze_array(v)
        dual_v = self._G @ v
        self._dual_v = _freeze_array(dual_v)
        g_norm_sq = float(v @ dual_v)
        if g_norm_sq <= self._tol.g_norm_floor:
            raise HouseholderSingularityError(
                f"Singularidad geométrica: ⟨v,v⟩_G = {g_norm_sq:.3e} ≤ "
                f"{self._tol.g_norm_floor:.1e} (v luminoso/sub-luminoso bajo G)."
            )
        self._g_norm_sq = g_norm_sq

        # ── AXIOMA §1: construcción covariante correcta ──
        self._M = _freeze_array(self._build_reflection_matrix())
        self._P = _freeze_array(self._build_projection_operator())

        # ── Certificación algebraica obligatoria (O(d²), siempre) ──
        self._iso_defect, self._inv_defect = self._verify_reflection_algebra()
        self._idempotency_defect, self._hermiticity_defect = self._verify_projector_algebra()

        # ── Certificación espectral (O(d³) opcional, o fast-path analítico) ──
        self._certificate: SpectralCertificate = (
            self._certify_spectrum_full() if certify_spectrum else self._certify_spectrum_analytic()
        )
        if not self._certificate.is_valid_reflection:
            raise HouseholderSingularityError(
                f"Certificación fallida: {self._certificate.summary()}"
            )

        logger.debug(
            "Reflector Householder certificado: dim=%d, det=%.6f, iso_defect=%.2e, "
            "idem_defect=%.2e, herm_defect=%.2e, espectro_completo=%s",
            self._dim, self._certificate.determinant, self._iso_defect,
            self._idempotency_defect, self._hermiticity_defect,
            self._certificate.spectrum_fully_computed,
        )

    # ────────────────────────────────────────────────────────────────────
    # CONSTRUCTOR ALTERNO — REUTILIZACIÓN DE CHOLESKY (Fase 3 → Fase 1)
    # ────────────────────────────────────────────────────────────────────
    @classmethod
    def from_precomputed_cholesky(
        cls,
        normal_vector: NDArray[np.float64],
        metric: NDArray[np.float64],
        cholesky_factor: NDArray[np.float64],
        tol: Optional[ToleranceConfig] = None,
        certify_spectrum: bool = True,
    ) -> "MetricAwareHouseholderReflector":
        r"""Evita recomputar $G=LL^\top$ cuando el orquestador ya la posee."""
        return cls(
            normal_vector, metric,
            cholesky_factor=cholesky_factor, tol=tol, certify_spectrum=certify_spectrum,
        )

    # ────────────────────────────────────────────────────────────────────
    # CONSTRUCCIÓN ALGEBRAICA (CORREGIDA)
    # ────────────────────────────────────────────────────────────────────
    def _build_reflection_matrix(self) -> NDArray[np.float64]:
        r"""$\hat{M} = I - 2\,\dfrac{v\,w^\top}{c}$, con $w=Gv$, $c=v^\top w$."""
        I = np.eye(self._dim, dtype=np.float64)
        return I - (2.0 * np.outer(self._v, self._dual_v)) / self._g_norm_sq

    def _build_projection_operator(self) -> NDArray[np.float64]:
        r"""$\hat{P} = \dfrac{I+\hat M}{2} = I - \dfrac{v\,w^\top}{c}$."""
        I = np.eye(self._dim, dtype=np.float64)
        return I - np.outer(self._v, self._dual_v) / self._g_norm_sq

    # ────────────────────────────────────────────────────────────────────
    # VERIFICACIÓN ALGEBRAICA (O(d²)/O(d³) — siempre ejecutada)
    # ────────────────────────────────────────────────────────────────────
    def _verify_reflection_algebra(self) -> Tuple[float, float]:
        r"""
        Verifica isometría $M^\top G M = G$ e inversión $Mv=-v$, retornando
        los residuales relativos como diagnóstico (no solo booleanos).
        """
        MGM = self._M.T @ self._G @ self._M
        iso_defect = la.norm(MGM - self._G, ord='fro') / max(1.0, la.norm(self._G, ord='fro'))
        if iso_defect > self._tol.isometry_atol:
            raise HouseholderSingularityError(
                f"La reflexión no preserva G: ‖MᵀGM-G‖_F/‖G‖_F={iso_defect:.2e}"
            )

        Mv = self._M @ self._v
        inv_defect = la.norm(Mv + self._v) / max(1.0, la.norm(self._v))
        if inv_defect > self._tol.isometry_atol:
            raise HouseholderSingularityError(
                f"La reflexión no invierte v: ‖Mv+v‖/‖v‖={inv_defect:.2e}"
            )
        return float(iso_defect), float(inv_defect)

    def _verify_projector_algebra(self) -> Tuple[float, float]:
        r"""
        Verifica idempotencia ($P^2=P$) y **autoadjunción métrica** ($GP=(GP)^\top$,
        forma algebraicamente equivalente y más directa que $P^\top GP=GP$ usada
        en v6.0). Ambas propiedades son necesarias para la convergencia en un
        solo paso del pulido de Fase 2.
        """
        P2 = self._P @ self._P
        idem_defect = la.norm(P2 - self._P, ord='fro') / max(1.0, la.norm(self._P, ord='fro'))
        if idem_defect > self._tol.projector_atol:
            raise HouseholderSingularityError(
                f"El proyector no es idempotente: ‖P²-P‖_F/‖P‖_F={idem_defect:.2e}"
            )

        GP = self._G @ self._P
        herm_defect = la.norm(GP - GP.T, ord='fro') / max(1.0, la.norm(GP, ord='fro'))
        if herm_defect > self._tol.projector_atol:
            raise HouseholderSingularityError(
                f"El proyector no es G-autoadjunto: ‖GP-(GP)ᵀ‖_F/‖GP‖_F={herm_defect:.2e}"
            )
        return float(idem_defect), float(herm_defect)

    # ────────────────────────────────────────────────────────────────────
    # CERTIFICACIÓN ESPECTRAL — DOS RÉGIMENES
    # ────────────────────────────────────────────────────────────────────
    def _certify_spectrum_full(self) -> SpectralCertificate:
        r"""
        Régimen completo O(d³): construye $\hat M_w = L^\top M L^{-\top}$ vía
        resolución de sistemas triangulares (evita inversión explícita) y
        diagonaliza con `eigvalsh` (garantía de espectro real).

        Demostración de que $\hat M_w$ es simétrica: dado que $M$ es isometría
        involutiva ($M^{-1}=M$, $M^\top GM=G$), se sigue $M^\top G = GM^{-1}=GM$,
        i.e. $GM$ es simétrica. Con $G=LL^\top$: $M^\top LL^\top = LL^\top M
        \Rightarrow L^{-1}M^\top L = L^\top M L^{-\top} = \hat M_w$, y por
        simetría de $GM$, $\hat M_w^\top = L^{-1}M^\top L = \hat M_w$. ∎
        """
        d = self._dim
        # Resuelve L·Zᵀ = Mᵀ  ⟹  Zᵀ = L⁻¹Mᵀ  ⟹  Z = M L⁻ᵀ
        Zt = la.solve_triangular(self._L, self._M.T, lower=True, check_finite=False)
        Z = Zt.T
        M_hat = self._L.T @ Z  # = LᵀM L⁻ᵀ

        sym_defect = la.norm(M_hat - M_hat.T, ord='fro') / max(1.0, la.norm(M_hat, ord='fro'))
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
            and n_minus == 1 and n_plus == d - 1 and n_anomalous == 0
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
        )

    def _certify_spectrum_analytic(self) -> SpectralCertificate:
        r"""
        Régimen rápido O(d²): $\det(M)=1-2c/c=-1$ y $\mathrm{tr}(M)=d-2$ son
        **exactos por el lema de determinante de rango-1**, sin diagonalizar.
        Recomendado cuando $d$ es grande (p. ej. logits de vocabulario) y ya
        se ejecutaron las verificaciones algebraicas O(d²) obligatorias.
        """
        d = self._dim
        analytic_det, analytic_trace = -1.0, float(d - 2)
        is_valid = (
            self._idempotency_defect <= self._tol.projector_atol
            and self._hermiticity_defect <= self._tol.projector_atol
            and self._iso_defect <= self._tol.isometry_atol
            and self._inv_defect <= self._tol.isometry_atol
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
        )

    # ────────────────────────────────────────────────────────────────────
    # API PÚBLICA — CONSUMIDA POR FASE 2 Y FASE 3 (sin atributos privados)
    # ────────────────────────────────────────────────────────────────────
    def reflect(self, psi: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Aplica $\hat M$: soporta $\psi$ de forma $(d,)$ o batch $(n,d)$."""
        psi = np.asarray(psi, dtype=np.float64)
        if not np.all(np.isfinite(psi)):
            raise HouseholderSingularityError("psi contiene NaN/Inf.")
        return _apply_operator(self._M, psi)

    def g_norm(self, x: NDArray[np.float64]) -> Union[float, NDArray[np.float64]]:
        r"""Norma inducida por $G$ (vía Cholesky), con soporte batched."""
        return _g_norm(x, self._L)

    def g_inner(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        r"""Producto interno métrico: $\langle x,y\rangle_G = x^\top G y$."""
        return float(x @ (self._G @ y))

    @property
    def reflection_matrix(self) -> NDArray[np.float64]:
        """Copia de $\hat M$ (ya inmutable por congelamiento, se retorna vista compartida segura)."""
        return self._M

    @property
    def projection_operator(self) -> NDArray[np.float64]:
        r"""
        Proyector $\hat P = I - v w^\top/c$ (congelado, sin copia adicional
        necesaria: `writeable=False` ya garantiza inmutabilidad).
        **Artefacto que inicia la Fase 2.**
        """
        return self._P

    @property
    def certificate(self) -> SpectralCertificate:
        """Certificado espectral inmutable."""
        return self._certificate

    @property
    def metric(self) -> NDArray[np.float64]:
        """Tensor $G$ (congelado) — API pública, reemplaza acceso a `_G`."""
        return self._G

    @property
    def cholesky_factor(self) -> NDArray[np.float64]:
        """Factor $L$ tal que $G=LL^\top$ (congelado) — API pública, reemplaza `_L`."""
        return self._L

    @property
    def dim(self) -> int:
        """Dimensión del espacio de estados."""
        return self._dim

    @property
    def g_norm_squared(self) -> float:
        r"""$\|v\|_G^2 = v^\top G v$ (cacheado)."""
        return self._g_norm_sq

    @property
    def diagnostics(self) -> Dict[str, float]:
        """Residuales algebraicos de la certificación (para auditoría externa)."""
        return {
            "isometry_defect": self._iso_defect,
            "inversion_defect": self._inv_defect,
            "idempotency_defect": self._idempotency_defect,
            "hermiticity_defect": self._hermiticity_defect,
        }


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2 · CAVIDAD DE PULIDO DE IDEMPOTENCIA (Newton–Schulz)              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
@dataclass(frozen=True, slots=True)
class ConvergenceTrace:
    r"""
    Traza de convergencia de la cavidad — historial categórico inmutable.

    Atributos:
        history: Tupla de $(k, \|P^2\psi_k-P\psi_k\|_G)$; para batch, el
            residual reportado es el máximo sobre las muestras (peor caso).
        final_residual: Último residual de no-idempotencia.
        iterations_used: Número de iteraciones ejecutadas.
        converged: Si el residual cayó bajo tolerancia.
        relative_coherent_energy: $\|\hat P\psi\|_G / \|\psi_{\text{raw}}\|_G$
            (mínimo sobre batch) — mide cuánta energía sobrevive a la proyección.
    """
    history: Tuple[Tuple[int, float], ...]
    final_residual: float
    iterations_used: int
    converged: bool
    relative_coherent_energy: float


class FabryPerotStabilizedCavity:
    r"""
    **Fase 2** — Pulido de idempotencia por iteración de **Newton–Schulz**
    (renombrado conceptualmente respecto a v6.0, que la llamaba erróneamente
    "iteración de Von Mises" — ver §3 del preámbulo del módulo).

    Para un proyector exacto $\hat P$ ($P^2=P$), el esquema
    $$ \psi_{k+1} = (2\hat P - \hat P^2)\,\psi_k $$
    coincide con $\hat P\psi_0$ en un solo paso. Bajo aritmética de punto
    flotante, $P^2 = P + \mathcal{O}(\varepsilon_{\text{mach}})$, de modo que la
    iteración **purga el defecto de idempotencia residual**:
    $$ \psi_{k+1} - \hat P\psi_k = (2\hat P-\hat P^2-\hat P)(\psi_k-\hat P\psi_k). $$

    El criterio de paro es $\|P^2\psi-P\psi\|_G<\varepsilon$: mide exactamente
    la energía de la componente que viola $P^2=P$.

    **Acoplamiento hacia Fase 3:** `stabilize` retorna `(coherent_state, ConvergenceTrace)`.
    """

    def __init__(
        self,
        reflector: MetricAwareHouseholderReflector,
        max_iter: int = 8,
        tol: Optional[ToleranceConfig] = None,
    ) -> None:
        if not isinstance(reflector, MetricAwareHouseholderReflector):
            raise TypeError("La cavidad requiere un MetricAwareHouseholderReflector certificado.")
        self._reflector = reflector
        self._max_iter = int(max(1, max_iter))
        self._tol = tol or ToleranceConfig()

        # ── Inyección del artefacto de Fase 1 vía API PÚBLICA (no _P privado) ──
        self._P: NDArray[np.float64] = reflector.projection_operator
        self._T: NDArray[np.float64] = 2.0 * self._P - self._P @ self._P
        self._L: NDArray[np.float64] = reflector.cholesky_factor
        self._dim: int = reflector.dim

    # ────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL — RECIBE ψ_raw DE FASE 3
    # ────────────────────────────────────────────────────────────────────
    def stabilize(
        self, psi_raw: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace]:
        r"""
        Induce el pulido de idempotencia sobre $\psi_{\text{raw}}$, convergiendo
        a $\hat P\psi$. Soporta $\psi_{\text{raw}}$ de forma $(d,)$ o $(n,d)$.

        Retorna:
            coherent_state: $\hat P\psi$ (misma forma que la entrada).
            trace: Historial de convergencia para auditoría categórica.
        """
        psi = np.asarray(psi_raw, dtype=np.float64)
        if psi.shape[-1] != self._dim or psi.ndim not in (1, 2):
            raise ResonanceDissonanceError(
                f"psi_raw tiene forma {psi.shape}, esperado (...,{self._dim}) con ndim∈{{1,2}}."
            )
        if not np.all(np.isfinite(psi)):
            raise ResonanceDissonanceError("psi_raw contiene NaN/Inf.")

        raw_norm = _g_norm(psi, self._L)
        raw_norm_scalar = float(np.min(np.atleast_1d(raw_norm)))
        if raw_norm_scalar < self._tol.g_norm_floor:
            raise ResonanceDissonanceError(
                "El estado de entrada es (numéricamente) el vector nulo bajo G; "
                "no existe radiación semántica que estabilizar."
            )

        history: List[Tuple[int, float]] = []
        residual = self._max_idempotency_residual(psi)
        history.append((0, residual))

        if residual < self._tol.idempotency_tol:
            logger.debug("Cavidad: estado ya idempotente (k=0), sin iteración.")
        else:
            for k in range(1, self._max_iter + 1):
                psi = _apply_operator(self._T, psi)
                residual = self._max_idempotency_residual(psi)
                history.append((k, residual))
                if residual < self._tol.idempotency_tol:
                    logger.debug("Cavidad estabilizada en k=%d, residuo=%.2e", k, residual)
                    break
            else:
                logger.warning(
                    "Cavidad no alcanzó tolerancia en %d iteraciones (residuo=%.2e).",
                    self._max_iter, residual,
                )

        psi_proj = _apply_operator(self._P, psi)
        coherent_norm = _g_norm(psi_proj, self._L)
        coherent_norm_scalar = float(np.min(np.atleast_1d(coherent_norm)))

        # ── Umbral de aniquilación RELATIVO (fija Hallazgo #6 de v6.0) ──
        relative_energy = coherent_norm_scalar / max(raw_norm_scalar, self._tol.g_norm_floor)
        if relative_energy < self._tol.annihilation_rel_threshold:
            raise ResonanceDissonanceError(
                f"Aniquilación total: ‖Pψ‖_G/‖ψ_raw‖_G = {relative_energy:.2e} < "
                f"{self._tol.annihilation_rel_threshold:.1e}. La radiación semántica era "
                f"paralela a la restricción — sin componente coherente."
            )

        self._verify_coherence(psi_proj)

        trace = ConvergenceTrace(
            history=tuple(history),
            final_residual=float(residual),
            iterations_used=len(history) - 1,
            converged=(residual < self._tol.idempotency_tol),
            relative_coherent_energy=relative_energy,
        )
        return psi_proj, trace

    # ────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS DE MEDICIÓN Y VERIFICACIÓN
    # ────────────────────────────────────────────────────────────────────
    def _idempotency_residual(self, psi: NDArray[np.float64]) -> Union[float, NDArray[np.float64]]:
        r"""$\|P^2\psi - P\psi\|_G$, con soporte batched (retorna $(n,)$ si aplica)."""
        P_psi = _apply_operator(self._P, psi)
        P2_psi = _apply_operator(self._P, P_psi)
        return _g_norm(P2_psi - P_psi, self._L)

    def _max_idempotency_residual(self, psi: NDArray[np.float64]) -> float:
        r"""Reduce el residual (posiblemente batched) al peor caso, para el criterio de paro."""
        return float(np.max(np.atleast_1d(self._idempotency_residual(psi))))

    def _verify_coherence(self, psi_proj: NDArray[np.float64]) -> None:
        r"""Confirma $\hat P\psi_{\text{proj}} = \psi_{\text{proj}}$ dentro de tolerancia laxa."""
        residual = self._max_idempotency_residual(psi_proj)
        threshold = max(1e-10, self._tol.idempotency_tol * 1e3)
        if residual > threshold:
            raise ResonanceDissonanceError(
                f"El estado retornado no es coherente: residual={residual:.2e} > {threshold:.2e}."
            )

    @property
    def reflector(self) -> MetricAwareHouseholderReflector:
        """Referencia al reflector de Fase 1 que originó esta cavidad (solo lectura)."""
        return self._reflector


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3 · MORFISMO CATADIÓPTRICO SUPREMO (ORQUESTADOR CATEGORIAL)       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class SemanticParabolicMirror(Morphism):
    r"""
    **Fase 3** — Morfismo endo en el topos $\mathcal{T}_{\mathrm{MIC}}$.

    Acopla el tensor métrico $G$ del MIC para purificar la radiación del LLM,
    reflejando alucinaciones y proyectándolas sobre el subespacio de las
    restricciones del negocio.

    **Resolución de la restricción de negocio (fija Hallazgo #8 de v6.0):**
    En orden de precedencia: (1) `state.constraint_normal` si está presente y
    tiene la forma correcta; (2) el normal vinculado vía `bind_constraint()`;
    (3) el normal canónico $e_0$ **con advertencia explícita** de fallback —
    nunca se asume tácitamente sin registro.

    **Contrato categórico:**
      • `forward(state)`  : reflexión pura $\hat M$.
      • `backward(state)` : $\hat M^{-1}=\hat M$ (involutiva) ⟹ `backward≡forward`.
      • `eta_kernel(state)`: inyección $\eta$ al subhaz coherente (Fase1→Fase2).
      • `apply(...)`      : punto de entrada externo único (reflexión→proyección).
    """

    def __init__(
        self,
        metric_tensor: Optional[NDArray[np.float64]] = None,
        max_cavity_iter: int = 8,
        tol: Optional[ToleranceConfig] = None,
        certify_spectrum_every_call: bool = True,
    ) -> None:
        r"""
        Args:
            metric_tensor: Tensor $G$; si `None`, usa `G_PHYSICS` (copiado, nunca
                referenciado directamente — fija Hallazgo #2 de v6.0).
            max_cavity_iter: Iteraciones máximas de pulido (Fase 2).
            tol: Configuración de tolerancias compartida por todas las fases.
            certify_spectrum_every_call: Si `False`, usa el *fast path* analítico
                O(d²) en cada reflector instanciado (recomendado para $d$ grande).
        """
        self._tol = tol or ToleranceConfig()
        metric_source = metric_tensor if metric_tensor is not None else G_PHYSICS
        self._G, self._L = RiemannianMetricValidator.validate(metric_source, self._tol)
        self._dim = self._G.shape[0]
        self._max_cavity_iter = int(max(1, max_cavity_iter))
        self._certify_every_call = bool(certify_spectrum_every_call)

        # Bookkeeping categórico
        self._last_trace: Optional[ConvergenceTrace] = None
        self._last_reflector: Optional[MetricAwareHouseholderReflector] = None
        self._bound_normal: Optional[NDArray[np.float64]] = None

        super().__init__(name="SemanticParabolicMirror")

    # ────────────────────────────────────────────────────────────────────
    # VINCULACIÓN EXPLÍCITA DE RESTRICCIONES DE NEGOCIO
    # ────────────────────────────────────────────────────────────────────
    def bind_constraint(self, normal: NDArray[np.float64]) -> None:
        r"""
        Vincula persistentemente un normal de restricción de negocio, usado por
        `forward`/`backward`/`eta_kernel` cuando el `CategoricalState` no porta
        su propio `constraint_normal`.
        """
        v = np.asarray(normal, dtype=np.float64).reshape(-1)
        if v.shape != (self._dim,):
            raise TopologicalInvariantError(
                f"normal vinculado tiene forma {v.shape}, esperada ({self._dim},)."
            )
        self._bound_normal = _freeze_array(v)
        logger.info("Restricción de negocio vinculada explícitamente al morfismo.")

    def _resolve_constraint_normal(self, state: CategoricalState) -> NDArray[np.float64]:
        r"""Resuelve el normal a usar, con orden de precedencia y trazabilidad explícita."""
        candidate = getattr(state, "constraint_normal", None)
        if candidate is not None:
            v = np.asarray(candidate, dtype=np.float64).reshape(-1)
            if v.shape == (self._dim,):
                return v
            logger.warning(
                "state.constraint_normal tiene forma incompatible %s; se ignora.", v.shape
            )
        if self._bound_normal is not None:
            return self._bound_normal
        logger.warning(
            "Sin restricción de negocio disponible (ni en el estado ni vinculada); "
            "usando normal canónico e₀ como FALLBACK explícito. Considere invocar "
            "bind_constraint() o poblar state.constraint_normal."
        )
        return self._canonical_normal()

    # ────────────────────────────────────────────────────────────────────
    # CONTRATO DEL MORFISMO (Topos $\mathcal{T}_{\mathrm{MIC}}$)
    # ────────────────────────────────────────────────────────────────────
    def forward(self, state: CategoricalState) -> CategoricalState:
        r"""Aplica $\hat M$ al estado (reflexión pura, sin proyección)."""
        psi = np.asarray(state.payload, dtype=np.float64)
        if psi.shape[-1] != self._dim:
            raise TopologicalInvariantError(
                f"Dimensión del estado {psi.shape} incompatible con G ({self._dim})."
            )
        normal = self._resolve_constraint_normal(state)
        reflector = MetricAwareHouseholderReflector.from_precomputed_cholesky(
            normal, self._G, self._L, tol=self._tol, certify_spectrum=self._certify_every_call,
        )
        self._last_reflector = reflector
        psi_ref = reflector.reflect(psi)
        return CategoricalState(payload=psi_ref, label=f"{state.label}::reflected")

    def backward(self, state: CategoricalState) -> CategoricalState:
        r"""Adjunta: $\hat M^{-1}=\hat M$ (involutiva) ⟹ `backward` ≡ `forward`."""
        return self.forward(state)

    def eta_kernel(self, state: CategoricalState) -> CategoricalState:
        r"""Inyección $\eta$ del haz crudo al subhaz coherente (Fase1→Fase2 completas)."""
        normal = self._resolve_constraint_normal(state)
        psi_coherent, trace = self._project_via_phases(
            np.asarray(state.payload, dtype=np.float64), normal
        )
        self._last_trace = trace
        return CategoricalState(payload=psi_coherent, label=state.label)

    # ────────────────────────────────────────────────────────────────────
    # API DE ORDEN SUPERIOR — ACOPLA FASE 1 + FASE 2
    # ────────────────────────────────────────────────────────────────────
    def apply(
        self,
        llm_logits: NDArray[np.float64],
        business_constraint_normal: NDArray[np.float64],
        use_iterative_refinement: bool = True,
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace]:
        r"""
        Método axiomático principal. **Punto de entrada externo único.**

        Ejecuta la cadena categórica:
            ψ_raw ──▶ Fase 1 (Householder, certifica M) ──▶ P
                  ──▶ Fase 2 (pulido de idempotencia) ──▶ ψ_coherent

        Contrato de retorno **consistente** (fija Hallazgo #7 de v6.0): siempre
        retorna una `ConvergenceTrace` válida, incluso en el camino corto sin
        refinamiento (trace trivial con `iterations_used=0`).

        Soporta `llm_logits` de forma $(d,)$ o batch $(n,d)$.
        """
        logger.info("Iniciando reflexión catadióptrica métrica de radiación semántica.")
        psi_raw = np.asarray(llm_logits, dtype=np.float64)
        normal = np.asarray(business_constraint_normal, dtype=np.float64).reshape(-1)

        if psi_raw.shape[-1] != self._dim or psi_raw.ndim not in (1, 2):
            raise TopologicalInvariantError(
                f"Dimensión de logits {psi_raw.shape} incompatible con dim(G)={self._dim}."
            )
        if normal.shape != (self._dim,):
            raise TopologicalInvariantError(
                f"Dimensión de restricción {normal.shape} ≠ dim(G)={self._dim}."
            )

        if not use_iterative_refinement:
            reflector = MetricAwareHouseholderReflector.from_precomputed_cholesky(
                normal, self._G, self._L, tol=self._tol, certify_spectrum=self._certify_every_call,
            )
            self._last_reflector = reflector
            coherent = _apply_operator(reflector.projection_operator, psi_raw)

            raw_norm = float(np.min(np.atleast_1d(_g_norm(psi_raw, self._L))))
            coh_norm = float(np.min(np.atleast_1d(_g_norm(coherent, self._L))))
            rel_energy = coh_norm / max(raw_norm, self._tol.g_norm_floor)

            trivial_trace = ConvergenceTrace(
                history=((0, 0.0),), final_residual=0.0,
                iterations_used=0, converged=True,
                relative_coherent_energy=rel_energy,
            )
            self._last_trace = trivial_trace
            logger.info("Proyección directa aplicada (camino corto categórico, sin cavidad).")
            return coherent, trivial_trace

        coherent, trace = self._project_via_phases(psi_raw, normal)
        self._last_trace = trace
        logger.info(
            "Colapso catadióptrico completado: k=%d, residuo=%.2e, convergió=%s, energía_rel=%.4f",
            trace.iterations_used, trace.final_residual, trace.converged,
            trace.relative_coherent_energy,
        )
        return coherent, trace

    # ────────────────────────────────────────────────────────────────────
    # MÉTODO NUCLEAR — PRIVADO, ACOPLA LAS DOS FASES
    # ────────────────────────────────────────────────────────────────────
    def _project_via_phases(
        self, psi_raw: NDArray[np.float64], normal: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace]:
        r"""Acoplamiento formal Fase 1 → Fase 2, reutilizando el Cholesky del orquestador."""
        # ══════════ FASE 1: REFLEXIÓN MÉTRICA ══════════
        reflector = MetricAwareHouseholderReflector.from_precomputed_cholesky(
            normal, self._G, self._L, tol=self._tol, certify_spectrum=self._certify_every_call,
        )
        self._last_reflector = reflector

        # ══════════ FASE 2: PULIDO DE IDEMPOTENCIA ══════════
        cavity = FabryPerotStabilizedCavity(
            reflector, max_iter=self._max_cavity_iter, tol=self._tol,
        )
        return cavity.stabilize(psi_raw)

    # ────────────────────────────────────────────────────────────────────
    # UTILIDADES CATEGORIALES Y DIAGNÓSTICO
    # ────────────────────────────────────────────────────────────────────
    def _canonical_normal(self) -> NDArray[np.float64]:
        r"""Normal canónico ($e_0$), usado únicamente como fallback explícito y advertido."""
        e0 = np.zeros(self._dim, dtype=np.float64)
        e0[0] = 1.0
        return e0

    @property
    def last_trace(self) -> Optional[ConvergenceTrace]:
        """Última traza de convergencia (post-`apply` o post-`eta_kernel`)."""
        return self._last_trace

    @property
    def last_certificate(self) -> Optional[SpectralCertificate]:
        """Certificado espectral del último reflector instanciado."""
        return self._last_reflector.certificate if self._last_reflector else None

    @property
    def metric_tensor(self) -> NDArray[np.float64]:
        """Tensor métrico G (ya inmutable por congelamiento)."""
        return self._G

    @property
    def cholesky_factor(self) -> NDArray[np.float64]:
        """Factor de Cholesky L cacheado a nivel de orquestador."""
        return self._L

    def diagnostic_report(self) -> Dict[str, Any]:
        r"""Reporte diagnóstico completo del último ciclo ejecutado (estilo unificado con Dirac Agent)."""
        report: Dict[str, Any] = {"dim": self._dim, "certify_every_call": self._certify_every_call}
        if self._last_reflector is not None:
            report["phase1_householder"] = {
                **self._last_reflector.diagnostics,
                "certificate": self._last_reflector.certificate.summary(),
                "g_norm_squared": self._last_reflector.g_norm_squared,
            }
        if self._last_trace is not None:
            report["phase2_cavity"] = {
                "iterations_used": self._last_trace.iterations_used,
                "final_residual": self._last_trace.final_residual,
                "converged": self._last_trace.converged,
                "relative_coherent_energy": self._last_trace.relative_coherent_energy,
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
    # Utilidades
    "RiemannianMetricValidator",
    # Estructuras
    "SpectralCertificate",
    "ConvergenceTrace",
    # Fases
    "MetricAwareHouseholderReflector",
    "FabryPerotStabilizedCavity",
    # Orquestador
    "SemanticParabolicMirror",
]
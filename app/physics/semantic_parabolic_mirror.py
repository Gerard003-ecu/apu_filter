# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Semantic Parabolic Mirror — Visión de Libélula (Atlas Omatidial)    ║
║ Ubicación: app/physics/semantic_parabolic_mirror.py                          ║
║ Versión: 8.0.0‑Dragonfly‑Omatidial‑Atlas‑vonNeumann‑Topos                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber‑Física y Topológica Diferencial — Visión de Libélula v8:
────────────────────────────────────────────────────────────────────────────────
Transmutación del reflector dióptrico bidimensional en una **Cúpula Catadióptrica
Facetada** (ojo compuesto / atlas omatidial) formulada como morfismo en el topos
$\mathcal{T}_{\mathrm{MIC}}$ sobre el haz de estados coherentes.

**Axiomas de Ejecución — Visión de Libélula:**

§0. AXIOMA MÉTRICO FUNDACIONAL (invariante):
    $G = G^\top \succ 0$. Factorización $G = LL^\top$ precomputada y congelada
    (`writeable=False`) como invariante categórico. Toda norma se evalúa vía $L$.

§1. FIBRADO DE RESTRICCIONES OMATIDIAL (Fase 1):
    El reflector no recibe un solo vector $|n\rangle$, sino una matriz de haz
    de restricciones $W\in\mathbb{R}^{d\times N}$, donde cada columna $|n_k\rangle$
    es la normal de una faceta del ojo compuesto. Para cada faceta $k$:
    $$
      \hat{M}_k = I - 2\,\frac{|n_k\rangle\langle n_k|G}{\langle n_k|G|n_k\rangle}
               = I - 2\,\frac{n_k\,w_k^\top}{c_k},\quad
      w_k = G n_k,\; c_k = n_k^\top w_k.
    $$
    Proyector ortogonal sobre el subespacio seguro de la faceta:
    $$ \hat{P}_k = \frac{I + \hat{M}_k}{2} = I - \frac{n_k\,w_k^\top}{c_k}. $$
    El estado cuántico del LLM se filtra por el fibrado $\{\hat{P}_k\}_{k=1}^{N}$.

§2. TEOREMA DE PROYECCIONES ALTERNADAS DE VON NEUMANN (Fase 2):
    El vector de intención estocástica $\psi_0$ rebota cíclicamente contra
    todas las facetas del domo catadióptrico:
    $$ \psi_{m+1} = \Bigl(\prod_{k=1}^{N}\hat{P}_k\Bigr)\psi_m. $$
    Por el teorema de von Neumann (extensión de Halperin a $N$ subespacios
    cerrados de un espacio de Hilbert), la iteración converge a la proyección
    ortogonal sobre $\bigcap_k \operatorname{ran}(\hat{P}_k)
    = \bigcap_k \ker(|n_k\rangle)^\perp{}^{G\text{-ort}}$.

§3. CONDICIÓN DE ANIQUILACIÓN DE DIRICHLET (guarda espectral):
    Si $\bigcap_{k=1}^{N}\ker(|n_k\rangle) = \{0\}$ y la señal $\psi$ es
    puramente alucinatoria (viola todas las dimensiones), entonces
    $$ \lim_{m\to\infty}\|\psi_m\|_G < \varepsilon_{\mathrm{mach}}
       \;\Longrightarrow\; \texttt{ResonanceDissonanceError}. $$
    Certifica que la cavidad ha aniquilado la señal degenerada por completo.

§4. PROYECTOR MAESTRO GLOBAL VÍA PSEUDOINVERSA COVARIANTE (Fase 3):
    Matriz de Gram del fibrado omatidial ponderada por $G$:
    $$ \mathcal{G}_{ij} = \langle n_i|G|n_j\rangle = n_i^\top G n_j
       \quad\Leftrightarrow\quad \mathcal{G} = W^\top G W. $$
    Si $\mathcal{G}$ es de rango completo (omatidios ortogonalmente distinguibles),
    se evita la iteración Fabry–Pérot y se instancia la pseudoinversa de
    Moore–Penrose covariante:
    $$ P_\cap = I - W\,\mathcal{G}^{-1}\,W^\top G. $$
    Proyección directa: $\psi_{\mathrm{refractado}} = P_\cap\,\psi_{\mathrm{raw}}$.
    Si $\mathcal{G}$ es singular/mal condicionada, se cae al esquema iterativo
    de von Neumann (Fase 2) con garantías de convergencia.

§5. CONTRATO CATEGORIAL (Topos $\mathcal{T}_{\mathrm{MIC}}$):
    `SemanticParabolicMirror` es morfismo endo sobre `CategoricalState`. La
    restricción de negocio se resuelve mediante `bind_constraints()` (fibrado)
    o mediante un atributo `constraint_normals` / `constraint_normal` en el
    propio estado. El método `__call__` es la transformación canónica.

Mejoras v8.0 (respecto a v7.0 — Visión de Libélula):
    • Fibrado multi‑faceta $W\in\mathbb{R}^{d\times N}$ (ojo compuesto).
    • Proyectores de Householder covariantes por faceta + producto de von Neumann.
    • Proyector maestro $P_\cap$ vía Gram + Cholesky / Moore–Penrose covariante.
    • Condición de aniquilación de Dirichlet con guarda espectral relativa.
    • `__call__` canónico devolviendo `CategoricalState` con homología de
      intersección de restricciones.
    • Compatibilidad hacia atrás: un solo normal se promueve a $W\in\mathbb{R}^{d\times 1}$.
    • Reutilización de Cholesky del orquestador; congelamiento de invariantes;
      `ToleranceConfig` unificado; soporte batched $(n,d)$.
════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Any, Dict, Sequence

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
        constraint_normals: Optional[NDArray[np.float64]] = None

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
    r"""Fallo de aniquilación coherente o aniquilación total de la señal (Dirichlet)."""
    pass


class OmatidialRankError(TopologicalInvariantError):
    r"""Fibrado $W$ degenerado: Gram singular o facetas linealmente dependientes."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES COMPARTIDAS (Fase 1 ↔ Fase 2 ↔ Fase 3 — DRY)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ToleranceConfig:
    r"""
    Centraliza toda tolerancia numérica del módulo.

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
        annihilation_rel_threshold: Umbral relativo (respecto a $\|\psi_{\mathrm{raw}}\|_G$)
            bajo el cual se declara aniquilación total de Dirichlet.
        gram_cond_threshold: Condicionamiento máximo de $\mathcal{G}=W^\top G W$
            para admitir la vía directa Moore–Penrose (Fase 3).
        von_neumann_atol: Tolerancia de estancamiento $\|\psi_{m+1}-\psi_m\|_G$
            en el producto de proyectores alternados.
    """
    metric_symmetry_atol: float = 1e-12
    g_norm_floor: float = 1e-15
    isometry_atol: float = 1e-9
    projector_atol: float = 1e-9
    spectral_eig_atol: float = 1e-8
    idempotency_tol: float = 1e-14
    annihilation_rel_threshold: float = 1e-12
    gram_cond_threshold: float = 1e12
    von_neumann_atol: float = 1e-14

    def __post_init__(self) -> None:
        for name in (
            "metric_symmetry_atol", "g_norm_floor", "isometry_atol",
            "projector_atol", "spectral_eig_atol", "idempotency_tol",
            "annihilation_rel_threshold", "gram_cond_threshold", "von_neumann_atol",
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

    Para $x$ de forma $(d,)$: $\|x\|_G = \|x L\|_2$.
    Para $X$ de forma $(n,d)$: retorna vector $(n,)$ con la norma G de cada fila.
    """
    x = np.asarray(x, dtype=np.float64)
    z = x @ L
    norms = np.linalg.norm(z, axis=-1)
    return float(norms) if x.ndim == 1 else norms


def _apply_operator(Op: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Aplica el operador lineal `Op` a `x`, soportando vector $(d,)$ o batch $(n,d)$.
    Para batch: cada fila $x_i \mapsto \mathrm{Op}\, x_i$, vectorizado como $X\,\mathrm{Op}^\top$.
    """
    if x.ndim == 1:
        return Op @ x
    if x.ndim == 2:
        return x @ Op.T
    raise ValueError(f"x debe ser 1D o 2D; ndim={x.ndim}")


def _normalize_W(
    normals: Union[NDArray[np.float64], Sequence[NDArray[np.float64]]],
    dim: int,
) -> NDArray[np.float64]:
    r"""
    Normaliza la entrada a una matriz de haz $W\in\mathbb{R}^{d\times N}$.

    Acepta:
      • vector $(d,)$ → $W$ de forma $(d,1)$;
      • matriz $(d,N)$ o $(N,d)$ (se infiere orientación por `dim`);
      • secuencia de vectores de longitud $d$.
    """
    if isinstance(normals, (list, tuple)):
        cols = [np.asarray(v, dtype=np.float64).reshape(-1) for v in normals]
        for c in cols:
            if c.shape != (dim,):
                raise HouseholderSingularityError(
                    f"Normal de faceta tiene forma {c.shape}, esperada ({dim},)."
                )
        W = np.column_stack(cols) if cols else np.zeros((dim, 0), dtype=np.float64)
        return W

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

    raise HouseholderSingularityError(f"normals debe ser 1D/2D; ndim={arr.ndim}.")


class RiemannianMetricValidator:
    r"""
    Validador único y centralizado del Axioma §0.
    """

    @staticmethod
    def validate(
        metric: NDArray[np.float64], tol: ToleranceConfig
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Valida que `metric` sea Riemanniana ($G=G^\top\succ0$) y retorna
        copias **congeladas** de $(G, L)$ con $G=LL^\top$.
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
        G: NDArray[np.float64],
        L: NDArray[np.float64],
        atol: float = 1e-7,
        rtol: float = 1e-6,
    ) -> None:
        r"""Verifica que un factor de Cholesky externo satisfaga $LL^\top \approx G$."""
        reconstructed = L @ L.T
        if not np.allclose(reconstructed, G, atol=atol, rtol=rtol):
            raise MetricSignatureError(
                "cholesky_factor provisto es inconsistente con metric (‖LLᵀ - G‖ excesivo)."
            )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1 · FIBRADO DE RESTRICCIONES OMATIDIAL                              ║
# ║   MetricAwareHouseholderReflector — Cúpula Catadióptrica Facetada          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
@dataclass(frozen=True, slots=True)
class SpectralCertificate:
    r"""
    **Certificado Espectral de Householder** — invariante topológico inmutable
    (por faceta o agregado).

    Dos regímenes:
      • **Analítico** (`spectrum_fully_computed=False`, O(d²)): lema de
        determinante de rango‑1 — $\det(M)=-1$, $\mathrm{tr}(M)=d-2$.
      • **Espectral completo** (`spectrum_fully_computed=True`, O(d³)):
        $\hat M_w = L^\top M L^{-\top}$ + `eigvalsh`.
    """
    determinant: float
    trace_value: float
    eigenvalues: NDArray[np.float64]
    is_valid_reflection: bool
    whitening_symmetry_defect: float
    analytic_determinant: float
    analytic_trace: float
    spectrum_fully_computed: bool
    facet_index: int = -1  # -1 = certificado agregado del fibrado

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
    r"""
    Operadores de una faceta individual del ojo compuesto.

    Atributos:
        normal: $|n_k\rangle$ (congelado).
        dual: $w_k = G n_k$ (congelado).
        g_norm_sq: $c_k = \langle n_k|G|n_k\rangle$.
        reflection: $\hat{M}_k$ (congelado).
        projection: $\hat{P}_k$ (congelado).
        certificate: Certificado espectral de la faceta.
        isometry_defect / inversion_defect / idempotency_defect / hermiticity_defect:
            residuales algebraicos de la certificación.
    """
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


class MetricAwareHouseholderReflector:
    r"""
    **Fase 1 — Fibrado de Restricciones Omatidial.**

    Cirugía homotópica: el reflector no recibe un solo vector $|n\rangle$, sino
    una matriz de haz de restricciones $W\in\mathbb{R}^{d\times N}$, donde cada
    columna $|n_k\rangle$ es la normal de una faceta del ojo compuesto.

    Para cada faceta $k=1,\ldots,N$:
    $$
      \hat{M}_k = I - 2\,\frac{n_k\,w_k^\top}{c_k},\qquad
      \hat{P}_k = \frac{I+\hat{M}_k}{2} = I - \frac{n_k\,w_k^\top}{c_k},
    $$
    con $w_k = G n_k$, $c_k = n_k^\top w_k$. Se garantiza la preservación del
    espacio de Hilbert bajo el tensor métrico $G$: $M_k^\top G M_k = G$.

    **Artefacto hacia Fase 2:** la tupla ordenada de proyectores
    $\{\hat{P}_k\}_{k=1}^{N}$ (vía `facet_projectors` / `product_projector_seed`)
    inicia el esquema de proyecciones alternadas de von Neumann.
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
        r"""
        Args:
            normals: Vector $(d,)$, matriz $W\in\mathbb{R}^{d\times N}$, o
                secuencia de vectores — el haz de normales omatidiales.
            metric: Tensor $G$. Requerido salvo reutilización con Cholesky.
            cholesky_factor: Reutiliza $L$ del orquestador (evita O(d³) redundante).
            tol: Configuración de tolerancias.
            certify_spectrum: Régimen espectral completo O(d³) vs. analítico O(d²).
        """
        self._tol = tol or ToleranceConfig()

        # ── AXIOMA §0: validación / reutilización de la métrica ──
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

        # ── Fibrado W ∈ R^{d×N} ──
        W = _normalize_W(normals, self._dim)
        if W.shape[1] == 0:
            raise HouseholderSingularityError("El fibrado omatidial W no tiene facetas (N=0).")
        if not np.all(np.isfinite(W)):
            raise HouseholderSingularityError("W contiene NaN/Inf.")
        self._W = _freeze_array(W)
        self._n_facets = W.shape[1]

        # ── Construcción faceta a faceta ──
        facets: List[FacetOperators] = []
        for k in range(self._n_facets):
            facets.append(self._build_facet(k, W[:, k], certify_spectrum))
        self._facets: Tuple[FacetOperators, ...] = tuple(facets)

        # ── Producto ordenado de proyectores (semilla de von Neumann → Fase 2) ──
        # P_prod = P_N ··· P_2 · P_1  (aplicación de derecha a izquierda sobre ψ)
        P_prod = np.eye(self._dim, dtype=np.float64)
        for fac in self._facets:
            P_prod = fac.projection @ P_prod
        self._P_product = _freeze_array(P_prod)

        logger.debug(
            "Fibrado omatidial certificado: dim=%d, N_facetas=%d, "
            "todos_validos=%s",
            self._dim,
            self._n_facets,
            all(f.certificate.is_valid_reflection for f in self._facets),
        )

    # ────────────────────────────────────────────────────────────────────
    # CONSTRUCTOR ALTERNO — REUTILIZACIÓN DE CHOLESKY (Fase 3 → Fase 1)
    # ────────────────────────────────────────────────────────────────────
    @classmethod
    def from_precomputed_cholesky(
        cls,
        normals: Union[NDArray[np.float64], Sequence[NDArray[np.float64]]],
        metric: NDArray[np.float64],
        cholesky_factor: NDArray[np.float64],
        tol: Optional[ToleranceConfig] = None,
        certify_spectrum: bool = True,
    ) -> "MetricAwareHouseholderReflector":
        r"""Evita recomputar $G=LL^\top$ cuando el orquestador ya la posee."""
        return cls(
            normals,
            metric,
            cholesky_factor=cholesky_factor,
            tol=tol,
            certify_spectrum=certify_spectrum,
        )

    # ────────────────────────────────────────────────────────────────────
    # CONSTRUCCIÓN Y CERTIFICACIÓN POR FACETA
    # ────────────────────────────────────────────────────────────────────
    def _build_facet(
        self,
        index: int,
        n_k: NDArray[np.float64],
        certify_spectrum: bool,
    ) -> FacetOperators:
        r"""
        Construye $\hat{M}_k$, $\hat{P}_k$ y certifica la faceta $k$.

        $$
          \hat{M}_k = I - 2\frac{n_k w_k^\top}{c_k},\quad
          \hat{P}_k = \frac{I+\hat{M}_k}{2}.
        $$
        """
        n_k = np.asarray(n_k, dtype=np.float64).reshape(-1)
        w_k = self._G @ n_k
        c_k = float(n_k @ w_k)
        if c_k <= self._tol.g_norm_floor:
            raise HouseholderSingularityError(
                f"Faceta {index}: singularidad geométrica ⟨n,n⟩_G = {c_k:.3e} ≤ "
                f"{self._tol.g_norm_floor:.1e}."
            )

        I = np.eye(self._dim, dtype=np.float64)
        outer = np.outer(n_k, w_k)
        M_k = I - (2.0 * outer) / c_k
        P_k = I - outer / c_k

        # Verificación algebraica obligatoria
        iso_def, inv_def = self._verify_reflection_algebra(M_k, n_k)
        idem_def, herm_def = self._verify_projector_algebra(P_k)

        cert = (
            self._certify_spectrum_full(M_k, index)
            if certify_spectrum
            else self._certify_spectrum_analytic(iso_def, inv_def, idem_def, herm_def, index)
        )
        if not cert.is_valid_reflection:
            raise HouseholderSingularityError(
                f"Faceta {index}: certificación fallida: {cert.summary()}"
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
        r"""Isometría $M^\top G M = G$ e inversión $Mv=-v$."""
        MGM = M.T @ self._G @ M
        iso_defect = la.norm(MGM - self._G, ord="fro") / max(1.0, la.norm(self._G, ord="fro"))
        if iso_defect > self._tol.isometry_atol:
            raise HouseholderSingularityError(
                f"La reflexión no preserva G: ‖MᵀGM-G‖_F/‖G‖_F={iso_defect:.2e}"
            )
        Mv = M @ v
        inv_defect = la.norm(Mv + v) / max(1.0, la.norm(v))
        if inv_defect > self._tol.isometry_atol:
            raise HouseholderSingularityError(
                f"La reflexión no invierte v: ‖Mv+v‖/‖v‖={inv_defect:.2e}"
            )
        return float(iso_defect), float(inv_defect)

    def _verify_projector_algebra(
        self, P: NDArray[np.float64]
    ) -> Tuple[float, float]:
        r"""Idempotencia $P^2=P$ y autoadjunción métrica $GP=(GP)^\top$."""
        P2 = P @ P
        idem_defect = la.norm(P2 - P, ord="fro") / max(1.0, la.norm(P, ord="fro"))
        if idem_defect > self._tol.projector_atol:
            raise HouseholderSingularityError(
                f"El proyector no es idempotente: ‖P²-P‖_F/‖P‖_F={idem_defect:.2e}"
            )
        GP = self._G @ P
        herm_defect = la.norm(GP - GP.T, ord="fro") / max(1.0, la.norm(GP, ord="fro"))
        if herm_defect > self._tol.projector_atol:
            raise HouseholderSingularityError(
                f"El proyector no es G-autoadjunto: ‖GP-(GP)ᵀ‖_F/‖GP‖_F={herm_defect:.2e}"
            )
        return float(idem_defect), float(herm_defect)

    def _certify_spectrum_full(
        self, M: NDArray[np.float64], facet_index: int
    ) -> SpectralCertificate:
        r"""
        Régimen O(d³): $\hat M_w = L^\top M L^{-\top}$ + `eigvalsh`.
        Demuestra espectro $\{-1\}\cup\{+1\}^{d-1}$.
        """
        d = self._dim
        Zt = la.solve_triangular(self._L, M.T, lower=True, check_finite=False)
        Z = Zt.T
        M_hat = self._L.T @ Z

        sym_defect = la.norm(M_hat - M_hat.T, ord="fro") / max(1.0, la.norm(M_hat, ord="fro"))
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
        r"""Régimen O(d²): $\det=-1$, $\mathrm{tr}=d-2$ exactos por lema de rango‑1."""
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

    # ────────────────────────────────────────────────────────────────────
    # API PÚBLICA — ARTEFACTOS QUE INICIAN LA FASE 2
    # ────────────────────────────────────────────────────────────────────
    def reflect(self, psi: NDArray[np.float64], facet: int = 0) -> NDArray[np.float64]:
        r"""Aplica $\hat{M}_k$ de la faceta `facet` (default 0). Soporta batch $(n,d)$."""
        psi = np.asarray(psi, dtype=np.float64)
        if not np.all(np.isfinite(psi)):
            raise HouseholderSingularityError("psi contiene NaN/Inf.")
        if not (0 <= facet < self._n_facets):
            raise IndexError(f"facet={facet} fuera de rango [0, {self._n_facets}).")
        return _apply_operator(self._facets[facet].reflection, psi)

    def reflect_all_sequential(self, psi: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Aplica el barrido secuencial $\hat{M}_N\cdots\hat{M}_1\psi$
        (reflexión cíclica sobre el domo; no es el proyector de intersección).
        """
        out = np.asarray(psi, dtype=np.float64)
        for fac in self._facets:
            out = _apply_operator(fac.reflection, out)
        return out

    def project_facet(self, psi: NDArray[np.float64], facet: int = 0) -> NDArray[np.float64]:
        r"""Aplica $\hat{P}_k$ de la faceta `facet`."""
        if not (0 <= facet < self._n_facets):
            raise IndexError(f"facet={facet} fuera de rango [0, {self._n_facets}).")
        return _apply_operator(self._facets[facet].projection, psi)

    def g_norm(self, x: NDArray[np.float64]) -> Union[float, NDArray[np.float64]]:
        r"""Norma inducida por $G$ (vía Cholesky), con soporte batched."""
        return _g_norm(x, self._L)

    def g_inner(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        r"""$\langle x,y\rangle_G = x^\top G y$."""
        return float(x @ (self._G @ y))

    @property
    def constraint_bundle(self) -> NDArray[np.float64]:
        r"""Matriz de haz $W\in\mathbb{R}^{d\times N}$ (congelada)."""
        return self._W

    @property
    def n_facets(self) -> int:
        """Número de facetas omatidiales $N$."""
        return self._n_facets

    @property
    def facets(self) -> Tuple[FacetOperators, ...]:
        """Tupla inmutable de operadores por faceta."""
        return self._facets

    @property
    def facet_projectors(self) -> Tuple[NDArray[np.float64], ...]:
        r"""
        Tupla ordenada $(\hat{P}_1,\ldots,\hat{P}_N)$ — **artefacto que inicia
        la Fase 2** (producto de von Neumann).
        """
        return tuple(f.projection for f in self._facets)

    @property
    def product_projector_seed(self) -> NDArray[np.float64]:
        r"""
        Producto precomputado $P_\Pi = \hat{P}_N\cdots\hat{P}_1$ (un ciclo del
        operador de von Neumann). **Puente algebraico Fase 1 → Fase 2.**
        """
        return self._P_product

    @property
    def projection_operator(self) -> NDArray[np.float64]:
        r"""
        Compatibilidad v7: si $N=1$, retorna $\hat{P}_0$; si $N>1$, retorna el
        producto de un ciclo $P_\Pi$ (semilla, no el proyector de intersección).
        """
        if self._n_facets == 1:
            return self._facets[0].projection
        return self._P_product

    @property
    def reflection_matrix(self) -> NDArray[np.float64]:
        """Compatibilidad v7: $\hat{M}_0$ de la primera faceta."""
        return self._facets[0].reflection

    @property
    def certificate(self) -> SpectralCertificate:
        """Certificado de la primera faceta (compatibilidad v7)."""
        return self._facets[0].certificate

    @property
    def certificates(self) -> Tuple[SpectralCertificate, ...]:
        """Certificados de todas las facetas."""
        return tuple(f.certificate for f in self._facets)

    @property
    def metric(self) -> NDArray[np.float64]:
        """Tensor $G$ (congelado)."""
        return self._G

    @property
    def cholesky_factor(self) -> NDArray[np.float64]:
        r"""Factor $L$ tal que $G=LL^\top$ (congelado)."""
        return self._L

    @property
    def dim(self) -> int:
        """Dimensión del espacio de estados."""
        return self._dim

    @property
    def g_norm_squared(self) -> float:
        r"""$\|n_0\|_G^2$ de la primera faceta (compatibilidad v7)."""
        return self._facets[0].g_norm_sq

    @property
    def diagnostics(self) -> Dict[str, Any]:
        """Residuales algebraicos agregados del fibrado."""
        return {
            "n_facets": self._n_facets,
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


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2 · CAVIDAD DE PULIDO Y LÍMITE DE VON NEUMANN                       ║
# ║   FabryPerotStabilizedCavity — Proyecciones Alternadas                     ║
# ║   Continuación formal del product_projector_seed de Fase 1                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
@dataclass(frozen=True, slots=True)
class ConvergenceTrace:
    r"""
    Traza de convergencia de la cavidad — historial categórico inmutable.

    Atributos:
        history: Tupla de $(m, \|\psi_{m+1}-\psi_m\|_G)$ (peor caso si batch).
        final_residual: Último residual de estancamiento / no‑idempotencia.
        iterations_used: Número de ciclos de von Neumann ejecutados.
        converged: Si el residual cayó bajo tolerancia.
        relative_coherent_energy: $\|\psi_\infty\|_G / \|\psi_{\mathrm{raw}}\|_G$
            (mínimo sobre batch).
        method: ``"von_neumann"`` | ``"newton_schulz_single"`` | ``"direct_gram"``.
    """
    history: Tuple[Tuple[int, float], ...]
    final_residual: float
    iterations_used: int
    converged: bool
    relative_coherent_energy: float
    method: str = "von_neumann"


class FabryPerotStabilizedCavity:
    r"""
    **Fase 2 — Cavidad de Pulido y Límite de von Neumann.**

    Eleva la dinámica de la cavidad para satisfacer el **Teorema de las
    Proyecciones Alternadas de von Neumann** (Halperin para $N$ subespacios).

    El vector de intención estocástica $\psi_0$ rebota cíclicamente contra
    todas las facetas del domo catadióptrico, consumiendo el artefacto de
    Fase 1 (`product_projector_seed` / `facet_projectors`):
    $$
      \psi_{m+1}
        = \Bigl(\prod_{k=1}^{N}\hat{P}_k\Bigr)\psi_m
        = P_\Pi\,\psi_m.
    $$
    La iteración converge a $P_\cap\psi_0$, proyección $G$-ortogonal sobre
    $\bigcap_k\operatorname{ran}(\hat{P}_k)$.

    **Condición de Aniquilación de Dirichlet (guarda espectral):**
    Si la intersección de los subespacios de las facetas es trivial y la señal
    es puramente alucinatoria,
    $$
      \lim_{m\to\infty}\|\psi_m\|_G < \varepsilon_{\mathrm{mach}}
      \;\Longrightarrow\; \texttt{ResonanceDissonanceError}.
    $$

    Caso degenerado $N=1$: el producto es un solo proyector idempotente; se
    reduce al pulido Newton–Schulz de v7 (una pasada purga ruido de redondeo).

    **Acoplamiento hacia Fase 3:** `stabilize` retorna
    `(coherent_state, ConvergenceTrace)`.
    """

    def __init__(
        self,
        reflector: MetricAwareHouseholderReflector,
        max_iter: int = 64,
        tol: Optional[ToleranceConfig] = None,
    ) -> None:
        if not isinstance(reflector, MetricAwareHouseholderReflector):
            raise TypeError(
                "La cavidad requiere un MetricAwareHouseholderReflector certificado (Fase 1)."
            )
        self._reflector = reflector
        self._max_iter = int(max(1, max_iter))
        self._tol = tol or ToleranceConfig()

        # ── Inyección del artefacto de Fase 1 vía API PÚBLICA ──
        self._projectors: Tuple[NDArray[np.float64], ...] = reflector.facet_projectors
        self._P_cycle: NDArray[np.float64] = reflector.product_projector_seed
        self._L: NDArray[np.float64] = reflector.cholesky_factor
        self._dim: int = reflector.dim
        self._n_facets: int = reflector.n_facets

        # Operador de pulido Newton–Schulz (solo N=1, compatibilidad)
        if self._n_facets == 1:
            P = self._projectors[0]
            self._T_ns: Optional[NDArray[np.float64]] = 2.0 * P - P @ P
        else:
            self._T_ns = None

    # ────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL — RECIBE ψ_raw DE FASE 3
    # ────────────────────────────────────────────────────────────────────
    def stabilize(
        self, psi_raw: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace]:
        r"""
        Induce el esquema de proyecciones alternadas de von Neumann sobre
        $\psi_{\mathrm{raw}}$, convergiendo a $P_\cap\psi$.

        Soporta $\psi_{\mathrm{raw}}$ de forma $(d,)$ o $(n,d)$.

        Retorna:
            coherent_state: $P_\cap\psi$ (misma forma que la entrada).
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

        if self._n_facets == 1:
            return self._stabilize_single_facet(psi, raw_norm_scalar)
        return self._stabilize_von_neumann(psi, raw_norm_scalar)

    def _stabilize_single_facet(
        self,
        psi: NDArray[np.float64],
        raw_norm_scalar: float,
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace]:
        r"""
        $N=1$: pulido Newton–Schulz $\psi\leftarrow(2P-P^2)\psi$ (v7), con
        guarda de aniquilación de Dirichlet.
        """
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
                    self._max_iter, residual,
                )

        psi_proj = _apply_operator(P, psi)
        return self._finalize(psi_proj, history, residual, raw_norm_scalar, "newton_schulz_single")

    def _stabilize_von_neumann(
        self,
        psi: NDArray[np.float64],
        raw_norm_scalar: float,
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace]:
        r"""
        $N\geq 2$: iteración de von Neumann / Halperin
        $\psi_{m+1} = P_\Pi\psi_m = (\prod_k\hat{P}_k)\psi_m$.

        Criterio de paro: $\|\psi_{m+1}-\psi_m\|_G < \varepsilon$.
        Guarda de Dirichlet: si $\|\psi_m\|_G$ colapsa bajo umbral relativo,
        se detona `ResonanceDissonanceError`.
        """
        history: List[Tuple[int, float]] = []
        history.append((0, float("inf")))

        for m in range(1, self._max_iter + 1):
            psi_next = _apply_operator(self._P_cycle, psi)
            delta = _g_norm(psi_next - psi, self._L)
            residual = float(np.max(np.atleast_1d(delta)))
            history.append((m, residual))
            psi = psi_next

            # Guarda espectral de Dirichlet (aniquilación total intermedia)
            cur_norm = float(np.min(np.atleast_1d(_g_norm(psi, self._L))))
            rel = cur_norm / max(raw_norm_scalar, self._tol.g_norm_floor)
            if rel < self._tol.annihilation_rel_threshold:
                raise ResonanceDissonanceError(
                    f"Aniquilación de Dirichlet (ciclo m={m}): "
                    f"‖ψ_m‖_G/‖ψ_raw‖_G = {rel:.2e} < "
                    f"{self._tol.annihilation_rel_threshold:.1e}. "
                    f"La cavidad ha destruido la señal degenerada por completo "
                    f"(⋂ ker(|n_k⟩) aniquiló la componente alucinatoria)."
                )

            if residual < self._tol.von_neumann_atol:
                logger.debug(
                    "Von Neumann estabilizado en m=%d, ‖Δψ‖_G=%.2e", m, residual
                )
                break
        else:
            logger.warning(
                "Von Neumann no alcanzó tolerancia en %d ciclos (‖Δψ‖_G=%.2e).",
                self._max_iter, residual,
            )

        # Proyección final: un último ciclo garantiza pertenencia numérica a ⋂ ran(P_k)
        psi = _apply_operator(self._P_cycle, psi)
        return self._finalize(psi, history, residual, raw_norm_scalar, "von_neumann")

    def _finalize(
        self,
        psi_proj: NDArray[np.float64],
        history: List[Tuple[int, float]],
        residual: float,
        raw_norm_scalar: float,
        method: str,
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace]:
        r"""Guarda de aniquilación final + ensamblaje de `ConvergenceTrace`."""
        coherent_norm = _g_norm(psi_proj, self._L)
        coherent_norm_scalar = float(np.min(np.atleast_1d(coherent_norm)))
        relative_energy = coherent_norm_scalar / max(raw_norm_scalar, self._tol.g_norm_floor)

        if relative_energy < self._tol.annihilation_rel_threshold:
            raise ResonanceDissonanceError(
                f"Aniquilación total de Dirichlet: ‖P_∩ψ‖_G/‖ψ_raw‖_G = "
                f"{relative_energy:.2e} < {self._tol.annihilation_rel_threshold:.1e}. "
                f"La radiación semántica era transversal a todas las restricciones W."
            )

        tol_stop = (
            self._tol.idempotency_tol if method == "newton_schulz_single"
            else self._tol.von_neumann_atol
        )
        trace = ConvergenceTrace(
            history=tuple(history),
            final_residual=float(residual),
            iterations_used=max(0, len(history) - 1),
            converged=(residual < tol_stop),
            relative_coherent_energy=relative_energy,
            method=method,
        )
        return psi_proj, trace

    def _idempotency_residual(
        self, psi: NDArray[np.float64], P: NDArray[np.float64]
    ) -> Union[float, NDArray[np.float64]]:
        r"""$\|P^2\psi - P\psi\|_G$ (soporte batched)."""
        P_psi = _apply_operator(P, psi)
        P2_psi = _apply_operator(P, P_psi)
        return _g_norm(P2_psi - P_psi, self._L)

    def _max_idempotency_residual(
        self, psi: NDArray[np.float64], P: NDArray[np.float64]
    ) -> float:
        return float(np.max(np.atleast_1d(self._idempotency_residual(psi, P))))

    @property
    def reflector(self) -> MetricAwareHouseholderReflector:
        """Referencia al reflector de Fase 1 que originó esta cavidad."""
        return self._reflector

    @property
    def cycle_operator(self) -> NDArray[np.float64]:
        r"""Operador de un ciclo $P_\Pi = \prod_k\hat{P}_k$."""
        return self._P_cycle


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3 · OPTIMIZACIÓN ALGEBRAICA DEL ORQUESTADOR                         ║
# ║   SemanticParabolicMirror — Proyector Maestro + Gram + __call__            ║
# ║   Continuación formal del stabilize de Fase 2 / Gram del fibrado           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
@dataclass(frozen=True, slots=True)
class GramSpectralReport:
    r"""
    Reporte espectral de la matriz de Gram del fibrado omatidial
    $\mathcal{G} = W^\top G W$.

    Atributos:
        gram: $\mathcal{G}$ (congelada).
        rank: Rango numérico.
        condition_number: $\kappa(\mathcal{G})$ (inf si singular).
        is_full_rank: Si se admite la vía directa Moore–Penrose.
        cholesky_factor: Factor $\mathcal{L}$ con $\mathcal{G}=\mathcal{L}\mathcal{L}^\top$
            (None si no es SPD numérica).
        master_projector: $P_\cap = I - W\mathcal{G}^{-1}W^\top G$ (None si vía iterativa).
    """
    gram: NDArray[np.float64]
    rank: int
    condition_number: float
    is_full_rank: bool
    cholesky_factor: Optional[NDArray[np.float64]]
    master_projector: Optional[NDArray[np.float64]]


class SemanticParabolicMirror(Morphism):
    r"""
    **Fase 3 — Morfismo Catadióptrico Supremo (Orquestador Categorial).**

    Integra el cálculo espectral directo sobre la **Matriz de Gram** del
    fibrado de normales omatidiales ponderada por el tensor de física:
    $$
      \mathcal{G}_{ij}
        = \langle n_i|G|n_j\rangle
        = n_i^\top G n_j
        \quad\Leftrightarrow\quad
      \mathcal{G} = W^\top G W.
    $$

    **Vía directa (rango completo):** se evalúa el condicionamiento vía
    Cholesky de $\mathcal{G}$. Si los omatidios son ortogonalmente
    distinguibles ($\mathcal{G}\succ 0$ y $\kappa(\mathcal{G})$ acotado), se
    instancia la pseudoinversa de Moore–Penrose covariante:
    $$
      P_\cap = I - W\,\mathcal{G}^{-1}\,W^\top G,
    $$
    y se proyecta en un solo paso: $\psi_{\mathrm{refractado}} = P_\cap\psi_{\mathrm{raw}}$.

    **Vía iterativa (rango deficiente / mal condicionada):** se cae al
    esquema de proyecciones alternadas de von Neumann (Fase 1 → Fase 2).

    **Contrato categórico:**
      • `forward(state)`   : reflexión secuencial $\prod_k\hat{M}_k$.
      • `backward(state)`  : involutiva por faceta ⟹ `backward ≡ forward`.
      • `eta_kernel(state)`: inyección $\eta$ al subhaz coherente (Fase1→Fase2/3).
      • `apply(...)`       : punto de entrada externo con traza.
      • `__call__(state)`  : transformación canónica → `CategoricalState`
        cuya homología garantiza que ningún vector estocástico transversal
        a las restricciones $W$ sobreviva a la reflexión.
    """

    def __init__(
        self,
        metric_tensor: Optional[NDArray[np.float64]] = None,
        max_cavity_iter: int = 64,
        tol: Optional[ToleranceConfig] = None,
        certify_spectrum_every_call: bool = True,
        prefer_direct_gram: bool = True,
    ) -> None:
        r"""
        Args:
            metric_tensor: Tensor $G$; si `None`, usa `G_PHYSICS` (copiado).
            max_cavity_iter: Iteraciones máximas de von Neumann (Fase 2).
            tol: Configuración de tolerancias compartida por todas las fases.
            certify_spectrum_every_call: Régimen espectral completo por reflector.
            prefer_direct_gram: Si `True`, intenta $P_\cap$ vía Gram antes de
                caer a la cavidad iterativa.
        """
        self._tol = tol or ToleranceConfig()
        metric_source = metric_tensor if metric_tensor is not None else G_PHYSICS
        self._G, self._L = RiemannianMetricValidator.validate(metric_source, self._tol)
        self._dim = self._G.shape[0]
        self._max_cavity_iter = int(max(1, max_cavity_iter))
        self._certify_every_call = bool(certify_spectrum_every_call)
        self._prefer_direct_gram = bool(prefer_direct_gram)

        # Bookkeeping categórico
        self._last_trace: Optional[ConvergenceTrace] = None
        self._last_reflector: Optional[MetricAwareHouseholderReflector] = None
        self._last_gram_report: Optional[GramSpectralReport] = None
        self._bound_normals: Optional[NDArray[np.float64]] = None  # W vinculado

        super().__init__(name="SemanticParabolicMirror")

    # ────────────────────────────────────────────────────────────────────
    # VINCULACIÓN EXPLÍCITA DEL FIBRADO DE RESTRICCIONES DE NEGOCIO
    # ────────────────────────────────────────────────────────────────────
    def bind_constraint(self, normal: NDArray[np.float64]) -> None:
        r"""Compatibilidad v7: vincula un único normal (promovido a $W\in\mathbb{R}^{d\times 1}$)."""
        self.bind_constraints(normal)

    def bind_constraints(
        self,
        normals: Union[NDArray[np.float64], Sequence[NDArray[np.float64]]],
    ) -> None:
        r"""
        Vincula persistentemente el fibrado de restricciones de negocio $W$,
        usado cuando el `CategoricalState` no porta `constraint_normals` /
        `constraint_normal`.
        """
        W = _normalize_W(normals, self._dim)
        if W.shape[1] == 0:
            raise TopologicalInvariantError("bind_constraints: N=0 facetas.")
        self._bound_normals = _freeze_array(W)
        logger.info(
            "Fibrado de restricciones vinculada: N=%d facetas, dim=%d.",
            W.shape[1], self._dim,
        )

    def _resolve_constraint_bundle(self, state: CategoricalState) -> NDArray[np.float64]:
        r"""
        Resuelve $W$ con orden de precedencia y trazabilidad explícita:
          1. `state.constraint_normals` (fibrado completo);
          2. `state.constraint_normal` (un solo normal → $d\times 1$);
          3. fibrado vinculado vía `bind_constraints()`;
          4. fallback canónico $e_0$ con advertencia.
        """
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
            "Sin restricción de negocio disponible (ni en el estado ni vinculada); "
            "usando normal canónico e₀ como FALLBACK explícito. Considere invocar "
            "bind_constraints() o poblar state.constraint_normals."
        )
        return self._canonical_normal().reshape(self._dim, 1)

    def _canonical_normal(self) -> NDArray[np.float64]:
        e0 = np.zeros(self._dim, dtype=np.float64)
        e0[0] = 1.0
        return e0

    # ────────────────────────────────────────────────────────────────────
    # GRAM DEL FIBRADO + PROYECTOR MAESTRO (núcleo algebraico de Fase 3)
    # ────────────────────────────────────────────────────────────────────
    def compute_gram_master_projector(
        self, W: NDArray[np.float64]
    ) -> GramSpectralReport:
        r"""
        Calcula $\mathcal{G}=W^\top G W$, evalúa rango/condicionamiento vía
        Cholesky, y — si es admisible — construye
        $$
          P_\cap = I - W\,\mathcal{G}^{-1}\,W^\top G.
        $$

        Este es el proyector $G$-ortogonal sobre
        $\bigcap_k\ker(|n_k\rangle)^\perp{}^{G\text{-ort}}
        = (\operatorname{span}\{n_k\})^{\perp_G}$.
        """
        W = np.asarray(W, dtype=np.float64)
        if W.ndim != 2 or W.shape[0] != self._dim:
            raise OmatidialRankError(f"W tiene forma {W.shape}, esperado ({self._dim}, N).")

        # Gram: 𝒢 = Wᵀ G W
        GW = self._G @ W  # (d, N)
        gram = W.T @ GW   # (N, N)
        gram = 0.5 * (gram + gram.T)  # simetrización numérica

        N = gram.shape[0]
        # Rango y condicionamiento vía SVD (robusto)
        try:
            s = la.svdvals(gram, check_finite=False)
        except la.LinAlgError:
            s = np.array([], dtype=np.float64)

        if s.size == 0:
            rank = 0
            cond = float("inf")
        else:
            s_max = float(s[0])
            floor = max(self._tol.g_norm_floor, s_max * self._tol.metric_symmetry_atol)
            rank = int(np.sum(s > floor))
            s_min_pos = float(s[rank - 1]) if rank > 0 else 0.0
            cond = (s_max / s_min_pos) if s_min_pos > 0 else float("inf")

        is_full = (rank == N) and (cond < self._tol.gram_cond_threshold) and (N > 0)

        L_gram: Optional[NDArray[np.float64]] = None
        P_cap: Optional[NDArray[np.float64]] = None

        if is_full:
            try:
                L_g = la.cholesky(gram, lower=True, check_finite=True)
                # Resuelve 𝒢 X = Wᵀ G  ⟺  L Lᵀ X = (Wᵀ G)
                # P_∩ = I - W X = I - W 𝒢⁻¹ Wᵀ G
                Y = la.solve_triangular(L_g, GW.T, lower=True, check_finite=False)
                X = la.solve_triangular(L_g.T, Y, lower=False, check_finite=False)
                P_cap = np.eye(self._dim, dtype=np.float64) - W @ X
                # Simetrización métrica ligera (proyector G-autoadjunto)
                # No forzamos P = Pᵀ euclídeo; la hermiticidad es GP = (GP)ᵀ.
                P_cap = _freeze_array(P_cap)
                L_gram = _freeze_array(L_g)
            except la.LinAlgError:
                is_full = False
                L_gram = None
                P_cap = None
                logger.info(
                    "Cholesky de Gram falló pese a SVD full-rank; "
                    "se caerá a vía iterativa de von Neumann."
                )

        report = GramSpectralReport(
            gram=_freeze_array(gram),
            rank=rank,
            condition_number=float(cond),
            is_full_rank=is_full,
            cholesky_factor=L_gram,
            master_projector=P_cap,
        )
        self._last_gram_report = report
        return report

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
            W, self._G, self._L,
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
        r"""Inyección $\eta$ del haz crudo al subhaz coherente (Fase1→Fase2/3)."""
        W = self._resolve_constraint_bundle(state)
        psi_coherent, trace = self._project_via_phases(
            np.asarray(state.payload, dtype=np.float64), W
        )
        self._last_trace = trace
        return CategoricalState(payload=psi_coherent, label=state.label)

    def __call__(self, state: CategoricalState) -> CategoricalState:
        r"""
        **Transformación canónica del morfismo** (Visión de Libélula).

        Ensambla el operador maestro $P_\cap$ (vía Gram directa o von Neumann)
        y devuelve un `CategoricalState` cuya homología garantiza que ningún
        vector estocástico transversal a las restricciones $W$ sobreviva a la
        reflexión catadióptrica:
        $$
          \psi_{\mathrm{refractado}} = P_\cap\,\psi_{\mathrm{raw}}.
        $$
        """
        W = self._resolve_constraint_bundle(state)
        psi_raw = np.asarray(state.payload, dtype=np.float64)
        if psi_raw.shape[-1] != self._dim:
            raise TopologicalInvariantError(
                f"Dimensión del estado {psi_raw.shape} incompatible con G ({self._dim})."
            )
        psi_coherent, trace = self._project_via_phases(psi_raw, W)
        self._last_trace = trace
        return CategoricalState(
            payload=psi_coherent,
            label=f"{state.label}::omatidial_refracted",
            constraint_normals=W,
        )

    # ────────────────────────────────────────────────────────────────────
    # API DE ORDEN SUPERIOR — ACOPLA FASE 1 + FASE 2 / GRAM DIRECTO
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
        Método axiomático principal. **Punto de entrada externo único.**

        Ejecuta la cadena categórica de la Visión de Libélula:
            ψ_raw ──▶ Gram(W) ──▶ [rango completo?]
                        │ sí                    │ no / force
                        ▼                       ▼
                   P_∩ directo          Fase 1 (fibrado) → Fase 2 (von Neumann)
                        │                       │
                        └───────────┬───────────┘
                                    ▼
                              ψ_coherent

        Contrato de retorno consistente: siempre retorna `ConvergenceTrace`.
        Soporta `llm_logits` de forma $(d,)$ o batch $(n,d)$.

        Args:
            llm_logits: Radiación semántica cruda.
            business_constraint_normals: Fibrado $W$ (vector, matriz o secuencia).
            use_iterative_refinement: Si `False` y N=1, proyección directa de
                una faceta sin cavidad (camino corto).
            force_von_neumann: Si `True`, ignora la vía Gram directa.
        """
        logger.info(
            "Iniciando reflexión catadióptrica omatidial (Visión de Libélula)."
        )
        psi_raw = np.asarray(llm_logits, dtype=np.float64)
        W = _normalize_W(business_constraint_normals, self._dim)

        if psi_raw.shape[-1] != self._dim or psi_raw.ndim not in (1, 2):
            raise TopologicalInvariantError(
                f"Dimensión de logits {psi_raw.shape} incompatible con dim(G)={self._dim}."
            )
        if W.shape[1] == 0:
            raise TopologicalInvariantError("Fibrado de restricciones vacío (N=0).")

        # Camino corto v7: N=1 sin refinamiento
        if not use_iterative_refinement and W.shape[1] == 1 and not force_von_neumann:
            reflector = MetricAwareHouseholderReflector.from_precomputed_cholesky(
                W, self._G, self._L,
                tol=self._tol,
                certify_spectrum=self._certify_every_call,
            )
            self._last_reflector = reflector
            coherent = _apply_operator(reflector.facets[0].projection, psi_raw)
            raw_norm = float(np.min(np.atleast_1d(_g_norm(psi_raw, self._L))))
            coh_norm = float(np.min(np.atleast_1d(_g_norm(coherent, self._L))))
            rel_energy = coh_norm / max(raw_norm, self._tol.g_norm_floor)
            if rel_energy < self._tol.annihilation_rel_threshold:
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
            logger.info("Proyección directa mono‑faceta (camino corto categórico).")
            return coherent, trivial_trace

        coherent, trace = self._project_via_phases(
            psi_raw, W, force_von_neumann=force_von_neumann
        )
        self._last_trace = trace
        logger.info(
            "Colapso omatidial completado: método=%s, k=%d, residuo=%.2e, "
            "convergió=%s, energía_rel=%.4f",
            trace.method, trace.iterations_used, trace.final_residual,
            trace.converged, trace.relative_coherent_energy,
        )
        return coherent, trace

    # ────────────────────────────────────────────────────────────────────
    # MÉTODO NUCLEAR — PRIVADO, ACOPLA GRAM / FASE 1 / FASE 2
    # ────────────────────────────────────────────────────────────────────
    def _project_via_phases(
        self,
        psi_raw: NDArray[np.float64],
        W: NDArray[np.float64],
        force_von_neumann: bool = False,
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace]:
        r"""
        Acoplamiento formal:
          1. Intento de proyector maestro vía Gram (Fase 3 algebraica);
          2. Si no es admisible → Fase 1 (fibrado) → Fase 2 (von Neumann).
        """
        W = _normalize_W(W, self._dim)

        # ══════════ FASE 3a: VÍA DIRECTA GRAM + MOORE–PENROSE ══════════
        if self._prefer_direct_gram and not force_von_neumann:
            report = self.compute_gram_master_projector(W)
            if report.is_full_rank and report.master_projector is not None:
                P_cap = report.master_projector
                coherent = _apply_operator(P_cap, psi_raw)

                raw_norm = float(np.min(np.atleast_1d(_g_norm(psi_raw, self._L))))
                if raw_norm < self._tol.g_norm_floor:
                    raise ResonanceDissonanceError(
                        "Estado de entrada nulo bajo G; no hay radiación que proyectar."
                    )
                coh_norm = float(np.min(np.atleast_1d(_g_norm(coherent, self._L))))
                rel_energy = coh_norm / max(raw_norm, self._tol.g_norm_floor)

                if rel_energy < self._tol.annihilation_rel_threshold:
                    raise ResonanceDissonanceError(
                        f"Aniquilación de Dirichlet (vía Gram): "
                        f"‖P_∩ψ‖_G/‖ψ_raw‖_G = {rel_energy:.2e}. "
                        f"La señal era transversal al fibrado W (⋂ ker = aniquilación)."
                    )

                # Certificar fibrado (auditoría) sin usarlo iterativamente
                reflector = MetricAwareHouseholderReflector.from_precomputed_cholesky(
                    W, self._G, self._L,
                    tol=self._tol,
                    certify_spectrum=self._certify_every_call,
                )
                self._last_reflector = reflector

                trace = ConvergenceTrace(
                    history=((0, 0.0),),
                    final_residual=0.0,
                    iterations_used=0,
                    converged=True,
                    relative_coherent_energy=rel_energy,
                    method="direct_gram",
                )
                logger.debug(
                    "Proyector maestro Gram aplicado: N=%d, rank=%d, κ=%.2e",
                    W.shape[1], report.rank, report.condition_number,
                )
                return coherent, trace

            logger.debug(
                "Gram no admisible (rank=%s, κ=%s); cayendo a von Neumann.",
                getattr(self._last_gram_report, "rank", "?"),
                getattr(self._last_gram_report, "condition_number", "?"),
            )

        # ══════════ FASE 1: FIBRADO OMATIDIAL ══════════
        reflector = MetricAwareHouseholderReflector.from_precomputed_cholesky(
            W, self._G, self._L,
            tol=self._tol,
            certify_spectrum=self._certify_every_call,
        )
        self._last_reflector = reflector

        # ══════════ FASE 2: VON NEUMANN / NEWTON–SCHULZ ══════════
        cavity = FabryPerotStabilizedCavity(
            reflector, max_iter=self._max_cavity_iter, tol=self._tol
        )
        return cavity.stabilize(psi_raw)

    # ────────────────────────────────────────────────────────────────────
    # UTILIDADES CATEGORIALES Y DIAGNÓSTICO
    # ────────────────────────────────────────────────────────────────────
    @property
    def last_trace(self) -> Optional[ConvergenceTrace]:
        """Última traza de convergencia (post-`apply` / `eta_kernel` / `__call__`)."""
        return self._last_trace

    @property
    def last_certificate(self) -> Optional[SpectralCertificate]:
        """Certificado espectral de la primera faceta del último reflector."""
        return self._last_reflector.certificate if self._last_reflector else None

    @property
    def last_certificates(self) -> Optional[Tuple[SpectralCertificate, ...]]:
        """Certificados de todas las facetas del último reflector."""
        return self._last_reflector.certificates if self._last_reflector else None

    @property
    def last_gram_report(self) -> Optional[GramSpectralReport]:
        """Último reporte espectral de Gram (si se intentó la vía directa)."""
        return self._last_gram_report

    @property
    def metric_tensor(self) -> NDArray[np.float64]:
        """Tensor métrico $G$ (inmutable)."""
        return self._G

    @property
    def cholesky_factor(self) -> NDArray[np.float64]:
        """Factor de Cholesky $L$ cacheado a nivel de orquestador."""
        return self._L

    def diagnostic_report(self) -> Dict[str, Any]:
        r"""Reporte diagnóstico completo del último ciclo (estilo Dirac Agent)."""
        report: Dict[str, Any] = {
            "dim": self._dim,
            "certify_every_call": self._certify_every_call,
            "prefer_direct_gram": self._prefer_direct_gram,
            "vision": "libelula_omatidial_atlas_v8",
        }
        if self._last_reflector is not None:
            report["phase1_omatidial_bundle"] = {
                **self._last_reflector.diagnostics,
                "certificates": [c.summary() for c in self._last_reflector.certificates],
            }
        if self._last_trace is not None:
            report["phase2_cavity"] = {
                "method": self._last_trace.method,
                "iterations_used": self._last_trace.iterations_used,
                "final_residual": self._last_trace.final_residual,
                "converged": self._last_trace.converged,
                "relative_coherent_energy": self._last_trace.relative_coherent_energy,
            }
        if self._last_gram_report is not None:
            g = self._last_gram_report
            report["phase3_gram"] = {
                "rank": g.rank,
                "condition_number": g.condition_number,
                "is_full_rank": g.is_full_rank,
                "has_master_projector": g.master_projector is not None,
                "gram_shape": list(g.gram.shape),
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
    # Utilidades
    "RiemannianMetricValidator",
    # Estructuras
    "SpectralCertificate",
    "FacetOperators",
    "ConvergenceTrace",
    "GramSpectralReport",
    # Fases
    "MetricAwareHouseholderReflector",
    "FabryPerotStabilizedCavity",
    # Orquestador
    "SemanticParabolicMirror",
]
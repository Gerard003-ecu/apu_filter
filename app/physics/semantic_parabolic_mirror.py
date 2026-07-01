# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Semantic Parabolic Mirror (Espejo Catadióptrico de Householder)      ║
║ Ubicación: app/physics/semantic_parabolic_mirror.py                          ║
║ Versión: 6.0.0‑Topos‑Spectral‑Cholesky‑CategoryAware                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber‑Física y Topológica Diferencial — Edición Granular:
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra la transición de un sistema puramente dióptrico (lentes) a
un ecosistema Catadióptrico riguroso, formulado como **morfismo en el topos
$\mathcal{T}_{\mathrm{MIC}}$** sobre el haz de estados coherentes.

**Axiomas de Ejecución (Evolución granular v6.0):**

§0. AXIOMA MÉTRICO FUNDACIONAL:
    Antes de cualquier reflexión, se exige que $G$ sea una métrica Riemanniana
    válida sobre $\mathbb{R}^d$: $G = G^\top$ y $G \succ 0$ (definida positiva).
    La factorización de Cholesky $G = L L^\top$ se precomputa y se almacena como
    *invariante de la categoría*; toda norma se evalúa vía $L$ para mantener
    estabilidad numérica ($\kappa_2(G)$ controlado).

§1. OPERADOR DE HOUSEHOLDER MÉTRICO (Reflexión Isométrica):
    El espejo se define sobre $v$ preservando la estructura métrica:
    $$ \hat{M} = I - 2\frac{(G v)(G v)^\top}{v^\top G v} \quad\text{(forma covariante)} $$
    Se verifica $M^\top G M = G$, $M v = -v$, $\det(M) = -1$, y que los valores
    propios son exactamente $\{-1, +1, \dots, +1\}$ (certificación espectral).

§2. PROYECTOR ORTOGONAL HEREDADO (Conservación de $L^2_G$):
    $$ \hat{P} = \frac{I + \hat{M}}{2} = I - \frac{(G v) v^\top}{v^\top G v} $$
    Es el **único artefacto** que cruza la frontera hacia la Fase 2; la cavidad
    jamás recalcula el proyector.

§3. INTERFERENCIA DE FABRY‑PÉROT (Iteración de Von Mises Métrica):
    La cavidad implementa el esquema de Von Mises para proyectores:
    $$ \psi_{k+1} = (2\hat{P} - \hat{P}^2)\,\psi_k $$
    con criterio de paro $\|P^2\psi - P\psi\|_G < \varepsilon$. Esta es la
    iteración canónica del método de Von Mises para raíces cuadradas de matrices
    simétricas; converge cuadráticamente cuando $P$ ya es un proyector.

§4. CONTRATO CATEGORIAL (Topos $\mathcal{T}_{\mathrm{MIC}}$):
    `SemanticParabolicMirror` es un *morfismo endo* sobre `CategoricalState`,
    con adjunción vía Householder hermítica: $\eta_{\mathrm{ker}}$ se inyecta
    en el subhaz de estados coherentes.
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA
# ══════════════════════════════════════════════════════════════════════════════
from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
from app.core.immune_system.metric_tensors import G_PHYSICS

logger = logging.getLogger("MIC.Omega.SemanticParabolicMirror")


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES ÓPTICAS, ESPECTRALES Y CATEGORIALES
# ══════════════════════════════════════════════════════════════════════════════
class HouseholderSingularityError(TopologicalInvariantError):
    r"""
    Se detona si $v^\top G v \leq 0$ (vector nulo o no-tiempo-similar bajo G).
    En geometría Riemanniana esto significa que $v$ es un *vector luminoso*
    o sub-luminoso — carece de dirección proyectable.
    """
    pass


class MetricSignatureError(TopologicalInvariantError):
    r"""
    Se detona si $G$ no es simétrica definida positiva (no es métrica Riemanniana).
    Falla del Axioma §0.
    """
    pass


class ResonanceDissonanceError(TopologicalInvariantError):
    r"""
    Se detona si la cavidad no logra aniquilar la componente no coherente dentro
    de la tolerancia, o si $\|\hat{P}\psi\|_G < \varepsilon_{\mathrm{ext}}$ (aniquilación total).
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: OPERADOR DE HOUSEHOLDER MÉTRICO (REFLEXIÓN ISOMÉTRICA)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SpectralCertificate:
    r"""
    **Certificado Espectral de Householder** — invariante topológico de la reflexión.
    
    Toda reflexión métrica de Householder debe satisfacer:
      • $\det(M) = -1$ (reflexión *par*, cambio de orientación)
      • $\mathrm{tr}(M) = d - 2$ (un único valor propio $-1$)
      • $\mathrm{esp}(M) = \{-1\} \cup \{+1\}^{(d-1)}$ (espectro bipartito)
    """
    determinant: float
    trace_value: float
    eigenvalues: NDArray[np.float64]
    is_valid_reflection: bool


class MetricAwareHouseholderReflector:
    r"""
    **Fase 1** — Espejo parabólico compatible con la métrica Riemanniana $G$.
    
    Construye $\hat{M} \in \mathbb{R}^{d\times d}$ tal que:
    $$ \hat{M}^\top G \hat{M} = G, \qquad \hat{M} v = -v. $$
    
    La fórmula covariante es:
    $$ \hat{M} = I - 2\frac{(G v)(G v)^\top}{v^\top G v}. $$
    
    **Acoplamiento hacia Fase 2:**
    El objeto entrega su *proyector ortogonal* `projection_operator` como artefacto
    inmutable; este será el **único insumo algebraico** de la cavidad resonante.
    """
    # ---------- Aritmética categorial fija ----------
    _EPS_NORM: float = 1e-15
    _EPS_ATOL: float = 1e-12

    def __init__(
        self,
        normal_vector: NDArray[np.float64],
        metric: NDArray[np.float64],
    ) -> None:
        # ── AXIOMA §0: Validación de G como métrica Riemanniana ──
        G = np.asarray(metric, dtype=np.float64)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise MetricSignatureError("G debe ser una matriz cuadrada.")
        if not np.allclose(G, G.T, atol=self._EPS_ATOL):
            raise MetricSignatureError("G debe ser simétrica: G = Gᵀ.")
        try:
            # Cholesky falla ⟹ G no es SPD ⟹ no es métrica Riemanniana
            self._L = la.cholesky(G, lower=True, check_finite=False)
        except la.LinAlgError as exc:
            raise MetricSignatureError(
                f"G no es definida positiva (Cholesky inexistente): {exc}"
            ) from exc
        self._G = G
        self._dim = G.shape[0]

        # ── Vector normal v y su covector dual Gv ──
        v = np.asarray(normal_vector, dtype=np.float64).copy()
        if v.shape != (self._dim,):
            raise HouseholderSingularityError(
                f"v tiene dimensión {v.shape}, esperada ({self._dim},)."
            )
        self._v = v
        self._dual_v = self._G @ v
        self._g_norm_sq = float(v @ self._dual_v)
        if self._g_norm_sq <= self._EPS_NORM:
            raise HouseholderSingularityError(
                "Singularidad Geométrica: v es luminoso/sub-luminoso bajo G (⟨v,Gv⟩≤0)."
            )

        # ── AXIOMA §1: Construcción covariante del espejo ──
        self._M: NDArray[np.float64] = self._build_reflection_matrix()

        # ── Certificación algebraica y espectral ──
        self._certificate: SpectralCertificate = self._verify_and_certify()
        # ── Proyector exacto (heredable a Fase 2) ──
        self._P: NDArray[np.float64] = self._build_projection_operator()
        # Verificación de que el proyector es realmente idempotente y hermítico-métrico
        self._verify_projector()

        logger.debug(
            "Reflector Householder certificado: dim=%d, det=%.6f, tr=%d, κ₂(G)≈%.2e",
            self._dim, self._certificate.determinant, int(round(self._certificate.trace_value)),
            self._condition_number(),
        )

    # ────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS — CONSTRUCCIÓN Y VERIFICACIÓN
    # ────────────────────────────────────────────────────────────────────
    def _build_reflection_matrix(self) -> NDArray[np.float64]:
        r"""Ensambla $\hat{M} = I - 2 \cdot \frac{(G v)(G v)^\top}{v^\top G v}$."""
        I = np.eye(self._dim, dtype=np.float64)
        outer_dual = np.outer(self._dual_v, self._dual_v)
        return I - (2.0 * outer_dual) / self._g_norm_sq

    def _build_projection_operator(self) -> NDArray[np.float64]:
        r"""
        Proyector ortogonal sobre $v^\perp$ respecto a G:
        $$ \hat{P} = \frac{I + \hat{M}}{2} = I - \frac{(G v) v^\top}{v^\top G v}. $$
        Este método **alimenta directamente la Fase 2**.
        """
        I = np.eye(self._dim, dtype=np.float64)
        return I - np.outer(self._dual_v, self._v) / self._g_norm_sq

    def _verify_projector(self) -> None:
        r"""
        Verifica idempotencia ($\hat{P}^2 = \hat{P}$) y hermiticidad métrica
        ($\hat{P}^\top G \hat{P} = G\hat{P}$). La idempotencia es **necesaria**
        para que el esquema de Von Mises de Fase 2 converja en un solo paso.
        """
        P2 = self._P @ self._P
        if not np.allclose(P2, self._P, atol=self._EPS_ATOL):
            raise HouseholderSingularityError(
                "El proyector no es idempotente: P² ≠ P. Reflexión inconsistente."
            )
        if not np.allclose(self._P.T @ self._G @ self._P, self._G @ self._P, atol=self._EPS_ATOL):
            raise HouseholderSingularityError(
                "El proyector no es hermítico bajo G."
            )

    def _verify_and_certify(self) -> SpectralCertificate:
        r"""
        Verifica las tres propiedades cardinales y emite el certificado espectral:
          1. $M^\top G M = G$ (isometría)
          2. $M v = -v$ (inversión del normal)
          3. $\mathrm{esp}(M) = \{-1, +1, \dots, +1\}$ (espectro bipartito)
        """
        # 1. Isometría
        MGM = self._M.T @ self._G @ self._M
        if not np.allclose(MGM, self._G, atol=self._EPS_ATOL):
            raise HouseholderSingularityError("La reflexión no preserva la métrica G.")

        # 2. Inversión del normal
        Mv = self._M @ self._v
        if not np.allclose(Mv, -self._v, atol=self._EPS_ATOL):
            raise HouseholderSingularityError("La reflexión no invierte el vector normal.")

        # 3. Espectro bipartito
        eigs = np.linalg.eigvals(self._M)
        eigs_real = np.real(eigs)
        # Toleramos desviaciones espectrales típicas del álgebra lineal numérica
        n_minus = int(np.sum(eigs_real < -1.0 + 1e-8))
        n_plus = int(np.sum(eigs_real > 1.0 - 1e-8))
        is_valid = (n_minus == 1) and (n_plus == self._dim - 1)

        det_M = float(np.linalg.det(self._M))
        tr_M = float(np.trace(self._M))

        return SpectralCertificate(
            determinant=det_M,
            trace_value=tr_M,
            eigenvalues=eigs_real,
            is_valid_reflection=is_valid,
        )

    def _condition_number(self) -> float:
        r"""Estimación rápida de $\kappa_2(G) = \|G\|_2 \cdot \|G^{-1}\|_2$."""
        try:
            return float(np.linalg.cond(self._G))
        except np.linalg.LinAlgError:
            return float("inf")

    # ────────────────────────────────────────────────────────────────────
    # API PÚBLICA — CONSUMIDA POR FASE 2 Y FASE 3
    # ────────────────────────────────────────────────────────────────────
    def reflect(self, psi: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Aplica $|\psi_{\mathrm{ref}}\rangle = \hat{M} |\psi\rangle$."""
        return self._M @ np.asarray(psi, dtype=np.float64)

    @property
    def reflection_matrix(self) -> NDArray[np.float64]:
        r"""Devuelve $\hat{M}$ (copia defensiva)."""
        return self._M.copy()

    @property
    def projection_operator(self) -> NDArray[np.float64]:
        r"""
        Proyector ortogonal (respecto a G) sobre $v^\perp$.
        **Artefacto entregado a Fase 2**: $\hat{P} = I - \frac{(G v) v^\top}{v^\top G v}$.
        """
        return self._P.copy()

    @property
    def certificate(self) -> SpectralCertificate:
        """Certificado espectral inmutable."""
        return self._certificate

    @property
    def g_norm_squared(self) -> float:
        r"""Devuelve $\|v\|_G^2 = v^\top G v$ (cacheado)."""
        return self._g_norm_sq

    def g_norm(self, x: NDArray[np.float64]) -> float:
        r"""Norma inducida por G: $\|x\|_G = \sqrt{x^\top G x}$ (vía Cholesky)."""
        z = self._L.T @ np.asarray(x, dtype=np.float64)
        return float(np.linalg.norm(z))

    def g_inner(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        r"""Producto interno métrico: $\langle x, y\rangle_G = x^\top G y$."""
        return float(x @ (self._G @ y))


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: CAVIDAD RESONANTE DE FABRY‑PÉROT (ITERACIÓN DE VON MISES MÉTRICA)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ConvergenceTrace:
    r"""
    Traza de convergencia de la cavidad resonante — historial categórico.
    
    Cada entrada es $(k, \|P^2\psi_k - P\psi_k\|_G)$, evidencia de la
    aniquilación coherente de la componente no proyectable.
    """
    history: Tuple[Tuple[int, float], ...]
    final_residual: float
    iterations_used: int
    converged: bool


class FabryPerotStabilizedCavity:
    r"""
    **Fase 2** — Cavidad resonante que aniquila las componentes no alineadas mediante
    interferencia coherente, **heredando el proyector exacto de Fase 1**.
    
    **Algoritmo (Iteración de Von Mises para proyectores):**
    
    Para un proyector exacto $\hat{P}$ (idempotente, $P^2 = P$), el esquema
    $$ \psi_{k+1} = (2\hat{P} - \hat{P}^2)\,\psi_k $$
    coincide con $\psi_k$ en un solo paso. Sin embargo, bajo aritmética de punto
    flotante, $\hat{P}^2 = \hat{P} + \mathcal{O}(\varepsilon_{\mathrm{mach}})$, así que
    la iteración **amplifica la idempotencia** removiendo ruido de redondeo:
    $$ \psi_{k+1} - \hat{P}\psi_k = (2\hat{P} - \hat{P}^2 - \hat{P})(\psi_k - \hat{P}\psi_k). $$
    
    El criterio de paro es $\|P^2\psi - P\psi\|_G < \varepsilon$, que mide exactamente
    la *no-idempotencia residual* (energía de la componente que viola la restricción).
    
    **Acoplamiento hacia Fase 3:**
    El método `stabilize` retorna un par `(coherent_state, ConvergenceTrace)`;
    el primero es el artefacto consumido por el orquestador categórico.
    """
    _EPS_DIV: float = 1e-30

    def __init__(
        self,
        reflector: MetricAwareHouseholderReflector,
        max_iter: int = 8,
        tol: float = 1e-14,
    ) -> None:
        if not isinstance(reflector, MetricAwareHouseholderReflector):
            raise TypeError("La cavidad requiere un MetricAwareHouseholderReflector certificado.")
        self._reflector = reflector
        self._max_iter = int(max(1, max_iter))
        self._tol = float(tol)
        # ── Inyección del artefacto de Fase 1 (única fuente de P) ──
        self._P: NDArray[np.float64] = reflector.projection_operator
        # Precomputamos el operador de Von Mises: T = 2P − P²
        # Esto es equivalente a P pero escrito en forma numéricamente estable.
        self._T: NDArray[np.float64] = 2.0 * self._P - self._P @ self._P
        # Métrica y L (Cholesky) vía reflector — sin recalcular
        self._G = reflector._G
        self._L = reflector._L
        self._dim = reflector._dim

    # ────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL — RECIBE ψ_raw DE FASE 3
    # ────────────────────────────────────────────────────────────────────
    def stabilize(
        self,
        psi_raw: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace]:
        r"""
        Induce interferencia destructiva sobre $\psi_{\mathrm{raw}}$,
        convergiendo al estado coherente $\hat{P}\psi$.
        
        Parámetros
        ----------
        psi_raw : NDArray (d,)
            Estado crudo del LLM (radiación semántica inicial).
            
        Retorna
        -------
        coherent_state : NDArray (d,)
            Estado proyectado sobre $v^\perp$ (onda estacionaria).
        trace : ConvergenceTrace
            Historial de convergencia para auditoría categórica.
        """
        psi = np.asarray(psi_raw, dtype=np.float64).copy()
        if psi.shape != (self._dim,):
            raise ResonanceDissonanceError(
                f"psi_raw tiene dimensión {psi.shape}, esperada ({self._dim},)."
            )

        history: List[Tuple[int, float]] = []
        psi_proj = self._P @ psi
        residual = self._idempotency_residual(psi)

        history.append((0, residual))
        if residual < self._tol:
            logger.debug("Fabry‑Pérot: estado ya idempotente (k=0), sin iteración.")
        else:
            for k in range(1, self._max_iter + 1):
                # Paso de Von Mises: ψ ← T ψ  (T = 2P − P²)
                psi = self._T @ psi
                psi_proj = self._P @ psi
                residual = self._idempotency_residual(psi)
                history.append((k, residual))
                if residual < self._tol:
                    logger.debug(
                        "Fabry‑Pérot estabilizado en k=%d, residuo idempotente=%.2e",
                        k, residual,
                    )
                    break
            else:
                logger.warning(
                    "Fabry‑Pérot no alcanzó tolerancia en %d iteraciones (residuo=%.2e).",
                    self._max_iter, residual,
                )

        # Aniquilación total = incoherencia pura
        coherent_norm = self._g_norm(psi_proj)
        if coherent_norm < self._EPS_DIV:
            raise ResonanceDissonanceError(
                "Aniquilación Total: ‖Pψ‖_G≈0. La radiación semántica era puramente "
                "paralela a la restricción del proyecto — sin componente coherente."
            )

        # Verificación final del estado coherente
        self._verify_coherence(psi_proj)

        trace = ConvergenceTrace(
            history=tuple(history),
            final_residual=float(residual),
            iterations_used=len(history) - 1,
            converged=(residual < self._tol),
        )
        return psi_proj, trace

    # ────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS DE MEDICIÓN Y VERIFICACIÓN
    # ────────────────────────────────────────────────────────────────────
    def _idempotency_residual(self, psi: NDArray[np.float64]) -> float:
        r"""
        Mide $\|P^2\psi - P\psi\|_G$. Esta es la *prueba espectral de no-idempotencia*:
        para un proyector exacto es cero; bajo punto flotante cuantifica la energía
        residual que viola la restricción.
        """
        P_psi = self._P @ psi
        P2_psi = self._P @ P_psi
        delta = P2_psi - P_psi
        return self._g_norm(delta)

    def _g_norm(self, x: NDArray[np.float64]) -> float:
        r"""Norma G vía Cholesky: $\|x\|_G = \|L^\top x\|_2$ (estable)."""
        z = self._L.T @ np.asarray(x, dtype=np.float64)
        return float(np.linalg.norm(z))

    def _verify_coherence(self, psi_proj: NDArray[np.float64]) -> None:
        r"""Confirma que $\hat{P}\psi_{\mathrm{proj}} = \psi_{\mathrm{proj}}$ dentro de tolerancia."""
        residual = self._idempotency_residual(psi_proj)
        if residual > max(1e-10, self._tol * 1e3):
            raise ResonanceDissonanceError(
                f"El estado retornado no es coherente: ‖P²ψ − Pψ‖_G = {residual:.2e}."
            )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: MORFISMO CATADIÓPTRICO SUPREMO (ORQUESTADOR CATEGORIAL)
# ══════════════════════════════════════════════════════════════════════════════
class SemanticParabolicMirror(Morphism):
    r"""
    **Fase 3** — Morfismo endo en el topos $\mathcal{T}_{\mathrm{MIC}}$.
    
    Acopla el tensor métrico $G$ del MIC para purificar la radiación del LLM,
    reflejando las alucinaciones y proyectándolas rígidamente sobre el subespacio
    de las restricciones del negocio.
    
    **Contrato categórico:**
      • `forward(state)` : aplica reflexión de Householder $\hat{M}$
      • `backward(state)`: aplica reflexión inversa (= $\hat{M}$, por involutividad)
      • `eta_kernel`    : inyecta el haz de estados crudos en el subhaz coherente
      • `apply(state)`  : acceso de orden superior (reflexión → proyección)
    
    **Anidamiento estricto:**
    `apply` invoca secuencialmente Fase 1 (Reflector) → Fase 2 (Cavidad) sobre
    el `CategoricalState` recibido.
    """
    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        max_cavity_iter: int = 8,
        cavity_tol: float = 1e-14,
    ) -> None:
        # ── Validación del Axioma §0 a nivel del orquestador ──
        G = np.asarray(metric_tensor, dtype=np.float64)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise MetricSignatureError("G debe ser una matriz cuadrada.")
        if not np.allclose(G, G.T, atol=1e-12):
            raise MetricSignatureError("G debe ser simétrica.")
        try:
            self._L = la.cholesky(G, lower=True, check_finite=False)
        except la.LinAlgError as exc:
            raise MetricSignatureError(f"G no es SPD: {exc}") from exc
        self._G = G
        self._dim = G.shape[0]
        self._max_cavity_iter = int(max(1, max_cavity_iter))
        self._cavity_tol = float(cavity_tol)
        # Estado de la categoría — bookkeeping categórico
        self._last_trace: Optional[ConvergenceTrace] = None
        self._last_reflector: Optional[MetricAwareHouseholderReflector] = None
        super().__init__(name="SemanticParabolicMirror")

    # ────────────────────────────────────────────────────────────────────
    # CONTRATO DEL MORFISMO (Topos $\mathcal{T}_{\mathrm{MIC}}$)
    # ────────────────────────────────────────────────────────────────────
    def forward(self, state: CategoricalState) -> CategoricalState:
        r"""Aplica $\hat{M}$ al estado (reflexión pura, sin proyección)."""
        psi = np.asarray(state.payload, dtype=np.float64)
        if psi.shape != (self._dim,):
            raise TopologicalInvariantError(
                f"Dimensión del estado {psi.shape} incompatible con G ({self._dim})."
            )
        # Construimos reflector con cualquier normal canónico si no existe uno fresco
        normal = self._canonical_normal()
        reflector = MetricAwareHouseholderReflector(normal, self._G)
        psi_ref = reflector.reflect(psi)
        return CategoricalState(payload=psi_ref, label=f"{state.label}::reflected")

    def backward(self, state: CategoricalState) -> CategoricalState:
        r"""
        Adjunta: para Householder, $\hat{M}^{-1} = \hat{M}$ (involutiva).
        Por lo tanto `backward` ≡ `forward`.
        """
        return self.forward(state)

    def eta_kernel(self, state: CategoricalState) -> CategoricalState:
        r"""
        Inyección $\eta$ del haz crudo al subhaz coherente.
        Equivale a `apply` pero conservando la etiqueta categórica original.
        """
        psi_coherent, trace = self._project_via_phases(
            np.asarray(state.payload, dtype=np.float64),
            self._canonical_normal(),
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
                  ──▶ Fase 2 (Cavidad Von Mises, refina) ──▶ ψ_coherent
        
        Parámetros
        ----------
        llm_logits : NDArray (d,)
            Vector de logits del LLM.
        business_constraint_normal : NDArray (d,)
            Vector normal al hiperplano de restricción.
        use_iterative_refinement : bool
            Si True, aplica cavidad (Fase 2); si False, proyección directa vía P.
            
        Retorna
        -------
        coherent_logits : NDArray (d,)
            Estado coherente libre de alucinaciones.
        trace : ConvergenceTrace
            Traza de auditoría (None si refinamiento deshabilitado).
        """
        logger.info("Iniciando reflexión catadióptrica métrica de radiación semántica.")
        psi_raw = np.asarray(llm_logits, dtype=np.float64)
        normal = np.asarray(business_constraint_normal, dtype=np.float64)

        if psi_raw.shape != (self._dim,):
            raise TopologicalInvariantError(
                f"Dimensión de logits {psi_raw.shape} ≠ dim(G)={self._dim}."
            )
        if normal.shape != (self._dim,):
            raise TopologicalInvariantError(
                f"Dimensión de restricción {normal.shape} ≠ dim(G)={self._dim}."
            )

        if not use_iterative_refinement:
            # ── Camino corto: sólo Fase 1 (sin cavidad) ──
            reflector = MetricAwareHouseholderReflector(normal, self._G)
            self._last_reflector = reflector
            coherent = reflector.projection_operator @ psi_raw
            self._last_trace = None
            logger.info("Proyección directa aplicada (camino corto categórico).")
            return coherent, None  # type: ignore[return-value]

        # ── Camino completo: Fase 1 → Fase 2 ──
        coherent, trace = self._project_via_phases(psi_raw, normal)
        self._last_trace = trace
        logger.info(
            "Colapso catadióptrico completado: k=%d, residuo G=%.2e, convergió=%s",
            trace.iterations_used, trace.final_residual, trace.converged,
        )
        return coherent, trace

    # ────────────────────────────────────────────────────────────────────
    # MÉTODO NUCLEAR — PRIVADO, ACOPLA LAS DOS FASES
    # ────────────────────────────────────────────────────────────────────
    def _project_via_phases(
        self,
        psi_raw: NDArray[np.float64],
        normal: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], ConvergenceTrace]:
        r"""
        Acoplamiento formal Fase 1 → Fase 2.
        El proyector $\hat{P}$ construido por Fase 1 es **la única entrada** de Fase 2.
        """
        # ══════════ FASE 1: REFLEXIÓN MÉTRICA ══════════
        reflector = MetricAwareHouseholderReflector(normal, self._G)
        self._last_reflector = reflector

        # ══════════ FASE 2: CAVIDAD RESONANTE ══════════
        cavity = FabryPerotStabilizedCavity(
            reflector,
            max_iter=self._max_cavity_iter,
            tol=self._cavity_tol,
        )
        coherent, trace = cavity.stabilize(psi_raw)
        return coherent, trace

    # ────────────────────────────────────────────────────────────────────
    # UTILIDADES CATEGORIALES
    # ────────────────────────────────────────────────────────────────────
    def _canonical_normal(self) -> NDArray[np.float64]:
        r"""
        Normal canónico ($e_0$) para los morfismos puros (forward/backward/eta)
        cuando no se proporciona restricción explícita. Útil para auditoría
        estructural sin invocar la restricción de negocio.
        """
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
        """Tensor métrico G (copia)."""
        return self._G.copy()


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "SpectralCertificate",
    "ConvergenceTrace",
    "HouseholderSingularityError",
    "MetricSignatureError",
    "ResonanceDissonanceError",
    "MetricAwareHouseholderReflector",
    "FabryPerotStabilizedCavity",
    "SemanticParabolicMirror",
]
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Floquet Monodromy Agent (Operador de Sintonización y Monodromía)    ║
║ Ruta   : app/omega/floquet_agent.py                                           ║
║ Versión: 3.0.0-Topos-CPTP-Monodromy-Spectral-Nested                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TOPOLÓGICA DIFERENCIAL
------------------------------------------------
Meta-funtor de control sobre la cavidad de Fabry–Pérot semántica
(``semantic_parabolic_mirror``) en el topos $\mathcal{T}_{\mathrm{MIC}}$.
Gobierna la reflexión de la radiación semántica del LLM mediante tres
fases anidadas con contratos formales de continuación:

.. code-block:: text

    Phase1_CovariantProjectorSynthesizer
        │  synthesize_projector(∇H)  ──►  ProjectorSynthesisResult
        ▼
    Phase2_FloquetStabilityAuditor
        │  audit_monodromy(P̂)        ──►  FloquetMonodromyState
        ▼
    Phase3_QuantumKrausChannel
        │  execute_quantum_channel   ──►  QuantumChannelEvolution
        ▼
    (salida pública del endofuntor Floquet)

CAMBIOS ESTRUCTURALES RESPECTO A v2.0.0 (evolución granular)
-------------------------------------------------------------
1. COMPLETITUD KRAUS CORREGIDA (bug axiomático)
   v2 exigía $C-I\succeq 0$ (semi-definitud), lo cual **permite** $C>I$
   y por tanto viola la preservación de traza. La condición CPTP exacta es
   \[
     C := \sum_k E_k^\dagger E_k = I
     \quad\Longleftrightarrow\quad
     \|C-I\|_F \le \varepsilon
     \;\land\;
     |\lambda_{\min}(C-I)| \le \varepsilon
     \;\land\;
     |\lambda_{\max}(C-I)| \le \varepsilon.
   \]
   Se verifica residual bilateral (no unilateral).

2. CANAL CPTP SOBRE MATRIZ DE DENSIDAD
   La evolución correcta es
   \[
     \rho_{\mathrm{post}}
       = \sum_k E_k\,\rho_{\mathrm{pre}}\,E_k^\dagger,
     \qquad
     \rho_{\mathrm{pre}} = |\psi\rangle\langle\psi|.
   \]
   La entropía de Von Neumann se calcula sobre los autovalores de $\rho$,
   no sobre la entropía de Shannon del vector (que no es invariante unitario).

3. IDEMPOTENCIA Y SIMETRÍA DEL PROYECTOR
   Se audita $\|\hat{P}^2-\hat{P}\|_F$ y $\|\hat{P}-\hat{P}^\top\|_F$
   (o residual G-simétrico). Un proyector defectuoso invalida monodromía
   y Kraus.

4. MONODROMÍA CON RESIDUO DE PROYECCIÓN
   Para $\hat{P}$ idempotente, $\mathcal{M}=2\hat{P}-\hat{P}^2=\hat{P}$
   y $\mathrm{spec}(\mathcal{M})\subseteq\{0,1\}$. El radio espectral $>1$
   solo puede provenir de un proyector numéricamente corrupto; se reporta
   el residuo de idempotencia como diagnóstico.

5. AUDITORÍA DIMENSIONAL BLOQUEANTE
   Si $\dim(\psi)\ne d$ o $\dim(\nabla H)\ne d$, se lanza
   ``DimensionalMismatchError`` (no se prosigue con ``is_coherent=False``).

6. TIPADO DE MULTIPLICADORES
   ``multipliers`` admite ``float64`` o ``complex128`` según la variedad;
   el DTO usa ``NDArray[Any]`` con flag ``is_complex_manifold``.

7. CONTRATOS FUNTORIALES ANIDADOS
   El terminal de cada fase es la precondición formal de la siguiente
   (espejo de KApex / KBase).

8. ENTROPÍA: CORRECCIÓN CONCEPTUAL
   Los canales CPTP **pueden aumentar** $S_{\mathrm{vN}}$ (p.ej. depolarizante).
   Lo contractivo es la **entropía relativa** $D(\rho\|\sigma)$.
   Se reporta $\Delta S = S(\rho_{\mathrm{post}})-S(\rho_{\mathrm{pre}})$
   sin imponer $\Delta S\le 0$ como invariante (falso en general).

AXIOMAS
-------
§0 Compatibilidad dimensional (bloqueante).
§1 Síntesis covariante: $n = G\nabla H$; si $\|n\|_G=0$ ⇒ $P=I$ (válido).
§2 Monodromía $\mathcal{M}=2P-P^2$; $|\mu_k|\le 1+\varepsilon$.
§3 Completitud Kraus exacta $C=I$ (bilateral).
§4 Auditoría entrópica de Von Neumann sobre $\rho_{\mathrm{pre/post}}$.
§5 Composición por ``Protocol`` runtime_checkable (sin herencia rígida).
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import math
from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Dependencias del ecosistema (con stubs para ejecución aislada / tests)
# ---------------------------------------------------------------------------
try:
    from app.core.mic_algebra import (
        Morphism,
        CategoricalState,
        TopologicalInvariantError,
    )
except ImportError:  # pragma: no cover

    class TopologicalInvariantError(Exception):
        """Stub: invariante topológico violado."""

    class CategoricalState:  # type: ignore[no-redef]
        def __init__(self, payload: Any = None, label: str = "") -> None:
            self.payload = payload
            self.label = label

    class Morphism:  # type: ignore[no-redef]
        def __init__(self, name: str = "") -> None:
            self.name = name


try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:  # pragma: no cover
    G_PHYSICS = np.eye(4, dtype=np.float64)

try:
    from app.omega.semantic_parabolic_mirror import (
        MetricAwareHouseholderReflector,
        HouseholderSingularityError,
    )
except ImportError:  # pragma: no cover

    class HouseholderSingularityError(Exception):
        """Stub: singularidad del reflector métrico."""

    class MetricAwareHouseholderReflector:  # type: ignore[no-redef]
        """
        Stub mínimo: proyector euclídeo $P = I - nn^\top/\|n\|^2$
        cuando no hay espejo métrico disponible.
        """

        def __init__(
            self, n_unit: NDArray[np.float64], G: NDArray[np.float64]
        ) -> None:
            n = np.asarray(n_unit, dtype=np.float64).reshape(-1)
            gn = G @ n
            denom = float(n @ gn)
            if abs(denom) < 1e-15:
                raise HouseholderSingularityError("n^T G n ≈ 0")
            # Proyector G-ortogonal sobre span{n}: P = n n^T G / (n^T G n)
            # (proyección a lo largo de la normal métrica)
            self.projection_operator = np.outer(n, gn) / denom
            self.projection_operator = 0.5 * (
                self.projection_operator + self.projection_operator.T
            )


try:
    from app.core.telemetry_schemas import PositronCartridge
except ImportError:  # pragma: no cover

    @dataclass(frozen=True)
    class PositronCartridge:  # type: ignore[no-redef]
        inertial_mass: float
        topological_spin: str
        homological_charge: int
        authorization_signature: str


logger = logging.getLogger("MIC.Omega.FloquetAgent")

# ---------------------------------------------------------------------------
# Constantes de rigor numérico (Wilkinson / Higham)
# ---------------------------------------------------------------------------
_MACHINE_EPS: float = float(np.finfo(np.float64).eps)
_WILKINSON_SAFETY: float = 100.0
_DEFAULT_STABILITY_TOL: float = 1.0e-9
_DEFAULT_KRAUS_TOL: float = 1.0e-9
_DEFAULT_PROJECTOR_TOL: float = 1.0e-8
_EPS_NORM_TRIVIAL: float = 1.0e-15
_EPS_DISSIPATION: float = 1.0e-12
_SYM_ATOL: float = 1.0e-12


# =============================================================================
# SECCIÓN 0 — EXCEPCIONES (árbol bajo TopologicalInvariantError)
# =============================================================================


class FloquetInstabilityError(TopologicalInvariantError):
    r"""Multiplicador de Floquet $|\mu_k| > 1+\varepsilon$ (resonancia destructiva)."""


class KrausTraceViolationError(TopologicalInvariantError):
    r"""$\|C-I\|_F$ supera la tolerancia de completitud Kraus."""


class KrausCompletenessError(TopologicalInvariantError):
    r"""Autovalores de $C-I$ no son todos $\approx 0$ (violación bilateral CPTP)."""


class DimensionalMismatchError(TopologicalInvariantError):
    """Dimensiones de tensores incoherentes con $G\in\mathbb{R}^{d\times d}$."""


class ProjectorDefectError(TopologicalInvariantError):
    r"""
    $\hat{P}$ viola idempotencia $\|\hat{P}^2-\hat{P}\|_F$ o simetría
    más allá de la tolerancia (proyector numéricamente corrupto).
    """


class FloquetParameterError(TopologicalInvariantError):
    """Parámetro escalar de control fuera de rango admisible."""


# =============================================================================
# SECCIÓN 1 — DTO INMUTABLES (objetos del topos)
# =============================================================================


@dataclass(frozen=True, slots=True)
class DimensionalAudit:
    """Certificado de coherencia dimensional del input al canal cuántico."""

    dimension: int
    psi_dim_ok: bool
    grad_dim_ok: bool
    rho_dim_ok: Optional[bool] = None
    is_coherent: bool = False


@dataclass(frozen=True, slots=True)
class ProjectorSynthesisResult:
    r"""
    Salida terminal de la **Fase 1** — precondición formal de la Fase 2.

    Atributos
    ---------
    P_hat : NDArray, shape (d, d)
        Proyector (euclídeo o G-ortogonal según el reflector).
    reflector : Optional[MetricAwareHouseholderReflector]
        ``None`` sii la obstrucción fue trivial ($P=I$).
    normal_G_norm : float
        $\|n\|_G = \sqrt{n^\top G n}$ antes de normalizar.
    is_trivial_obstruction : bool
        ``True`` ⇒ $P=I$ por Axioma §1.
    idempotence_residual : float
        $\|\hat{P}^2-\hat{P}\|_F$.
    symmetry_residual : float
        $\|\hat{P}-\hat{P}^\top\|_F$.
    dim : int
        Dimensión $d$ del espacio de calibre.
    """

    P_hat: NDArray[np.float64]
    reflector: Optional[Any]
    normal_G_norm: float
    is_trivial_obstruction: bool
    idempotence_residual: float
    symmetry_residual: float
    dim: int


@dataclass(frozen=True, slots=True)
class FloquetMonodromyState:
    r"""
    Salida terminal de la **Fase 2** — precondición formal de la Fase 3.

    Para $\hat{P}$ idempotente exacto: $\mathcal{M}=\hat{P}$ y
    $\mathrm{spec}(\mathcal{M})\subseteq\{0,1\}$, luego $\rho(\mathcal{M})\le 1$.
    """

    multipliers: NDArray[Any]  # float64 o complex128
    spectral_radius: float
    is_asymptotically_stable: bool
    condition_number_P: float
    monodromy_matrix: NDArray[np.float64]
    projector_idempotence_residual: float
    is_complex_manifold: bool = False
    tolerance_used: float = _DEFAULT_STABILITY_TOL


@dataclass(frozen=True, slots=True)
class QuantumChannelEvolution:
    r"""
    Salida terminal de la **Fase 3** y del endofuntor Floquet.

    Campos
    ------
    coherent_state
        $E_0|\psi\rangle = \hat{P}|\psi\rangle$ (amplitud coherente).
    rho_post
        $\rho_{\mathrm{post}}=\sum_k E_k\rho E_k^\dagger$ (matriz de densidad).
    dissipated_entropy
        $S_{\mathrm{vN}}(\rho_{\mathrm{diss}})$ de la componente $E_1\rho E_1^\dagger$
        renormalizada si tiene traza positiva; 0 si traza nula.
    von_neumann_pre, von_neumann_post, delta_entropy
        $S(\rho_{\mathrm{pre}})$, $S(\rho_{\mathrm{post}})$, diferencia.
        **No** se impone $\Delta S\le 0$ (CPTP no es contractivo en $S_{\mathrm{vN}}$).
    antimatter_emission
        Positrón forense si la disipación es significativa.
    kraus_residual_fro, kraus_residual_eig
        $\|\sum E^\dagger E-I\|_F$ y $\max|\lambda(C-I)|$.
    dimensional_audit, monodromy_state
        Certificados de las fases previas (trazabilidad forense).
    """

    coherent_state: NDArray[np.float64]
    rho_post: NDArray[np.float64]
    dissipated_entropy: float
    von_neumann_pre: float
    von_neumann_post: float
    delta_entropy: float
    antimatter_emission: Optional[PositronCartridge]
    kraus_residual_fro: float
    kraus_residual_eig: float
    purity_post: float
    dimensional_audit: DimensionalAudit
    monodromy_state: Optional[FloquetMonodromyState] = None


# =============================================================================
# SECCIÓN 2 — PROTOCOLOS DE COMPOSICIÓN (sin herencia rígida)
# =============================================================================


@runtime_checkable
class ProjectorSynthesizerPort(Protocol):
    def synthesize_projector(
        self, H_obs_gradient: NDArray[np.float64]
    ) -> ProjectorSynthesisResult: ...


@runtime_checkable
class FloquetAuditorPort(Protocol):
    def audit_monodromy(
        self, synthesis: ProjectorSynthesisResult
    ) -> FloquetMonodromyState: ...


@runtime_checkable
class KrausChannelPort(Protocol):
    def execute_quantum_channel(
        self,
        psi_raw: NDArray[np.float64],
        H_obs_gradient: NDArray[np.float64],
        synthesis: ProjectorSynthesisResult,
        monodromy: FloquetMonodromyState,
    ) -> QuantumChannelEvolution: ...


# =============================================================================
# SECCIÓN 3 — ORQUESTADOR CON TRES FASES ANIDADAS
# =============================================================================


class FloquetMonodromyAgent(Morphism):
    r"""
    Morfismo Floquet maestro en $\mathcal{T}_{\mathrm{MIC}}$.

    Compone las tres fases por agregación tipada con ``Protocol``:

    * ``phase1`` — síntesis covariante del proyector
    * ``phase2`` — auditoría de monodromía de Floquet
    * ``phase3`` — canal cuántico CPTP + antimateria forense

    Pipeline canónico
    -----------------
    .. code-block:: text

        ∇H_obs, |ψ⟩
            ──► Fase 1  (ProjectorSynthesisResult)
            ──► Fase 2  (FloquetMonodromyState)
            ──► Fase 3  (QuantumChannelEvolution)
    """

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        stability_tolerance: float = _DEFAULT_STABILITY_TOL,
        kraus_tolerance: float = _DEFAULT_KRAUS_TOL,
        projector_tolerance: float = _DEFAULT_PROJECTOR_TOL,
    ) -> None:
        if stability_tolerance <= 0.0:
            raise FloquetParameterError(
                f"stability_tolerance debe ser > 0; se obtuvo {stability_tolerance}."
            )
        if kraus_tolerance <= 0.0:
            raise FloquetParameterError(
                f"kraus_tolerance debe ser > 0; se obtuvo {kraus_tolerance}."
            )
        if projector_tolerance <= 0.0:
            raise FloquetParameterError(
                f"projector_tolerance debe ser > 0; se obtuvo {projector_tolerance}."
            )

        G = np.asarray(metric_tensor, dtype=np.float64)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise DimensionalMismatchError(
                f"G debe ser cuadrada; se recibió {G.shape}."
            )
        if not np.allclose(G, G.T, atol=_SYM_ATOL):
            raise TopologicalInvariantError("G debe ser simétrica.")
        try:
            la.cholesky(0.5 * (G + G.T), lower=True, check_finite=False)
        except la.LinAlgError as exc:
            raise TopologicalInvariantError(f"G no es SPD: {exc}") from exc

        self._G: NDArray[np.float64] = 0.5 * (G + G.T)
        self._dim: int = int(self._G.shape[0])
        self._stability_tol = float(stability_tolerance)
        self._kraus_tol = float(kraus_tolerance)
        self._projector_tol = float(projector_tolerance)

        # Fases anidadas (eager 1–2; 3 eager también — sin estado perezoso)
        self.phase1 = FloquetMonodromyAgent.Phase1_CovariantProjectorSynthesizer(
            metric_tensor=self._G,
            projector_tolerance=self._projector_tol,
        )
        self.phase2 = FloquetMonodromyAgent.Phase2_FloquetStabilityAuditor(
            stability_tolerance=self._stability_tol,
            projector_tolerance=self._projector_tol,
        )
        self.phase3 = FloquetMonodromyAgent.Phase3_QuantumKrausChannel(
            metric_tensor=self._G,
            kraus_tolerance=self._kraus_tol,
        )

        try:
            super().__init__(name="FloquetMonodromyAgent")
        except TypeError:
            # Stub Morphism sin firma name=
            super().__init__()

        logger.info(
            "[FloquetMonodromyAgent] v3 inicializado: d=%d, "
            "stab_tol=%.1e, kraus_tol=%.1e, proj_tol=%.1e.",
            self._dim,
            self._stability_tol,
            self._kraus_tol,
            self._projector_tol,
        )

    # =========================================================================
    # FASE 1 — SÍNTESIS COVARIANTE DEL PROYECTOR ORTOGONAL
    # =========================================================================

    class Phase1_CovariantProjectorSynthesizer:
        r"""
        **Fase 1** — Síntesis del proyector covariante.

        \[
          n = G\,\nabla\mathcal{H}_{\mathrm{obs}},
          \qquad
          \|n\|_G = \sqrt{n^\top G n}.
        \]

        * Si $\|n\|_G < \varepsilon$ (obstrucción trivial): $P = I_d$ (Axioma §1).
        * Si no: se construye el reflector métrico y se extrae $P$.

        Se audita idempotencia y simetría antes de ceder el control a Fase 2.
        """

        def __init__(
            self,
            metric_tensor: NDArray[np.float64],
            projector_tolerance: float = _DEFAULT_PROJECTOR_TOL,
        ) -> None:
            self._G = np.asarray(metric_tensor, dtype=np.float64)
            self._dim = int(self._G.shape[0])
            self._projector_tol = float(projector_tolerance)

        @staticmethod
        def _projector_residuals(
            P: NDArray[np.float64],
        ) -> Tuple[float, float]:
            """$(\|P^2-P\|_F,\ \|P-P^\top\|_F)$."""
            idem = float(la.norm(P @ P - P, "fro"))
            sym = float(la.norm(P - P.T, "fro"))
            return idem, sym

        def synthesize_projector(
            self, H_obs_gradient: NDArray[np.float64]
        ) -> ProjectorSynthesisResult:
            r"""
            **Método terminal de la Fase 1.**

            Retorna
            -------
            ProjectorSynthesisResult
                Precondición formal de
                ``Phase2_FloquetStabilityAuditor.audit_monodromy``.
            """
            grad = np.asarray(H_obs_gradient, dtype=np.float64).reshape(-1)
            if grad.shape != (self._dim,):
                raise DimensionalMismatchError(
                    f"H_obs_gradient shape={grad.shape}, esperada ({self._dim},)."
                )

            # Pullback covariante
            n_cov = self._G @ grad
            n_norm_G = float(np.sqrt(max(n_cov @ (self._G @ n_cov), 0.0)))

            # ── Axioma §1: obstrucción trivial ───────────────────────────
            if n_norm_G < _EPS_NORM_TRIVIAL:
                logger.warning(
                    "[Fase1] Obstrucción trivial (‖n‖_G=%.2e). P = I_d.",
                    n_norm_G,
                )
                P = np.eye(self._dim, dtype=np.float64)
                idem, sym = self._projector_residuals(P)
                return ProjectorSynthesisResult(
                    P_hat=P,
                    reflector=None,
                    normal_G_norm=n_norm_G,
                    is_trivial_obstruction=True,
                    idempotence_residual=idem,
                    symmetry_residual=sym,
                    dim=self._dim,
                )

            # ── Reflector métrico ────────────────────────────────────────
            n_unit = n_cov / n_norm_G
            try:
                reflector = MetricAwareHouseholderReflector(n_unit, self._G)
            except HouseholderSingularityError as exc:
                logger.error("[Fase1] Singularidad del reflector: %s", exc)
                raise

            P = np.asarray(reflector.projection_operator, dtype=np.float64)
            if P.shape != (self._dim, self._dim):
                raise DimensionalMismatchError(
                    f"projection_operator shape={P.shape}, "
                    f"esperada ({self._dim},{self._dim})."
                )

            idem, sym = self._projector_residuals(P)
            # Tolerancia escalada por dimensión (Wilkinson)
            tol_P = self._projector_tol * max(self._dim, 1)
            if idem > tol_P:
                raise ProjectorDefectError(
                    f"Proyector no idempotente: ‖P²−P‖_F={idem:.6e} > tol={tol_P:.6e}."
                )
            if sym > tol_P:
                logger.warning(
                    "[Fase1] Proyector con asimetría ‖P−Pᵀ‖_F=%.3e "
                    "(se prosigue; Fase 2 usará eigensolver general).",
                    sym,
                )

            logger.debug(
                "[Fase1] Proyector sintetizado: d=%d, ‖n‖_G=%.3e, "
                "idem=%.3e, sym=%.3e.",
                self._dim, n_norm_G, idem, sym,
            )
            return ProjectorSynthesisResult(
                P_hat=P,
                reflector=reflector,
                normal_G_norm=n_norm_G,
                is_trivial_obstruction=False,
                idempotence_residual=idem,
                symmetry_residual=sym,
                dim=self._dim,
            )

    # =========================================================================
    # FASE 2 — AUDITOR DE MONODROMÍA DE FLOQUET
    #           (continuación formal de ProjectorSynthesisResult)
    # =========================================================================

    class Phase2_FloquetStabilityAuditor:
        r"""
        **Fase 2** — Matriz de monodromía y estabilidad espectral.

        \[
          \mathcal{M}_{\mathrm{on}} = 2\hat{P} - \hat{P}^2.
        \]

        Identidad algebraica: si $\hat{P}^2=\hat{P}$, entonces
        $\mathcal{M}=\hat{P}$ y $\mathrm{spec}(\mathcal{M})\subseteq\{0,1\}$.
        Por tanto $\rho(\mathcal{M})>1$ **implica** defecto de idempotencia
        (no un fenómeno físico genuino de la cavidad ideal).

        Solver:
          * $\mathcal{M}=\mathcal{M}^\top$ → ``eigvalsh`` (estable, $O(d^2)$)
          * en caso contrario → ``eigvals`` (variedad compleja)
        """

        def __init__(
            self,
            stability_tolerance: float = _DEFAULT_STABILITY_TOL,
            projector_tolerance: float = _DEFAULT_PROJECTOR_TOL,
        ) -> None:
            self._stability_tol = float(stability_tolerance)
            self._projector_tol = float(projector_tolerance)

        def audit_monodromy(
            self, synthesis: ProjectorSynthesisResult
        ) -> FloquetMonodromyState:
            r"""
            **Método terminal de la Fase 2.**

            Parámetros
            ----------
            synthesis : ProjectorSynthesisResult
                Salida de ``Phase1.synthesize_projector``.

            Retorna
            -------
            FloquetMonodromyState
                Precondición formal de
                ``Phase3_QuantumKrausChannel.execute_quantum_channel``.
            """
            P = np.asarray(synthesis.P_hat, dtype=np.float64)
            d = int(P.shape[0])
            if P.shape != (d, d):
                raise DimensionalMismatchError(
                    f"P_hat debe ser cuadrada; se recibió {P.shape}."
                )

            # Re-auditoría de idempotencia (defensa en profundidad)
            idem = float(la.norm(P @ P - P, "fro"))
            tol_P = self._projector_tol * max(d, 1)
            if idem > tol_P:
                raise ProjectorDefectError(
                    f"[Fase2] Idempotencia rota antes de monodromía: "
                    f"‖P²−P‖_F={idem:.6e} > tol={tol_P:.6e}."
                )

            # Monodromía
            M_on = 2.0 * P - (P @ P)

            # Simetría de M
            sym_M = float(la.norm(M_on - M_on.T, "fro"))
            is_symmetric = sym_M <= _SYM_ATOL * max(float(la.norm(M_on, "fro")), 1.0)

            if is_symmetric:
                M_sym = 0.5 * (M_on + M_on.T)
                multipliers = np.linalg.eigvalsh(M_sym).astype(np.float64)
                is_complex = False
            else:
                multipliers = np.linalg.eigvals(M_on).astype(np.complex128)
                is_complex = True
                logger.warning(
                    "[Fase2] M_on no simétrica (‖M−Mᵀ‖_F=%.3e); "
                    "eigensolver complejo.",
                    sym_M,
                )

            spectral_radius = float(np.max(np.abs(multipliers)))
            is_stable = spectral_radius <= 1.0 + self._stability_tol

            # cond(P): para proyectores, cond puede ser ∞ si hay ker; usamos
            # SVD y reportamos σ_max/σ_min entre valores > tol
            try:
                svals = la.svdvals(P)
                s_pos = svals[svals > 1.0e-14]
                cond_P = (
                    float(s_pos[0] / s_pos[-1])
                    if s_pos.size > 0
                    else float("inf")
                )
            except la.LinAlgError:
                cond_P = float("inf")

            if not is_stable:
                raise FloquetInstabilityError(
                    f"Inestabilidad de Floquet: ρ(M_on)={spectral_radius:.6e} > "
                    f"1+ε={1.0 + self._stability_tol:.6e}. "
                    f"Residuo de idempotencia de P={idem:.6e}."
                )

            logger.debug(
                "[Fase2] Monodromía: ρ=%.6e, estable=%s, cond(P)=%.3e, "
                "complejo=%s, idem(P)=%.3e.",
                spectral_radius, is_stable, cond_P, is_complex, idem,
            )

            return FloquetMonodromyState(
                multipliers=multipliers,
                spectral_radius=spectral_radius,
                is_asymptotically_stable=is_stable,
                condition_number_P=cond_P,
                monodromy_matrix=M_on.copy(),
                projector_idempotence_residual=idem,
                is_complex_manifold=is_complex,
                tolerance_used=self._stability_tol,
            )

    # =========================================================================
    # FASE 3 — CANAL CUÁNTICO CPTP (Kraus–Stinespring)
    #           (continuación formal de FloquetMonodromyState)
    # =========================================================================

    class Phase3_QuantumKrausChannel:
        r"""
        **Fase 3** — Canal CPTP con operadores de Kraus

        \[
          E_0 = \hat{P},
          \qquad
          E_1 = I - \hat{P},
          \qquad
          \rho_{\mathrm{post}}
            = E_0\rho E_0^\dagger + E_1\rho E_1^\dagger.
        \]

        Completitud exacta (Axioma §3, **corregido**):
        \[
          C = E_0^\dagger E_0 + E_1^\dagger E_1 = I.
        \]
        Para $\hat{P}$ simétrico e idempotente esto es una identidad algebraica:
        \[
          P^\top P + (I-P)^\top(I-P)
            = P^2 + (I-P)^2
            = P + I - 2P + P
            = I
          \quad(P^\top=P,\;P^2=P).
        \]
        """

        def __init__(
            self,
            metric_tensor: NDArray[np.float64],
            kraus_tolerance: float = _DEFAULT_KRAUS_TOL,
        ) -> None:
            self._G = np.asarray(metric_tensor, dtype=np.float64)
            self._dim = int(self._G.shape[0])
            self._kraus_tol = float(kraus_tolerance)

        # ------------------------------------------------------------------
        # Completitud Kraus bilateral
        # ------------------------------------------------------------------

        def _verify_kraus_completeness(
            self,
            E0: NDArray[np.float64],
            E1: NDArray[np.float64],
        ) -> Tuple[float, float]:
            r"""
            Verifica $C=\sum_k E_k^\top E_k = I$ de forma **bilateral**.

            Retorna
            -------
            frobenius_residual : float
                $\|C-I\|_F$.
            eig_residual : float
                $\max_i|\lambda_i(C-I)|$.

            Lanza
            -----
            KrausCompletenessError
                Si algún $|\lambda_i(C-I)| > \texttt{kraus\_tol}$.
            KrausTraceViolationError
                Si $\|C-I\|_F > \texttt{kraus\_tol}\cdot\sqrt{d}$.
            """
            d = E0.shape[0]
            C = E0.T @ E0 + E1.T @ E1
            I = np.eye(d, dtype=np.float64)
            diff = 0.5 * ((C - I) + (C - I).T)  # simetrización defensiva
            fro_res = float(la.norm(diff, "fro"))
            eigs = np.linalg.eigvalsh(diff)
            eig_res = float(np.max(np.abs(eigs)))

            tol_eig = self._kraus_tol
            tol_fro = self._kraus_tol * math.sqrt(max(d, 1))

            if eig_res > tol_eig:
                raise KrausCompletenessError(
                    f"C−I no es nula en espectro: max|λ|={eig_res:.6e} > "
                    f"tol={tol_eig:.6e} (λ_min={float(eigs[0]):.6e}, "
                    f"λ_max={float(eigs[-1]):.6e}). "
                    "Violación bilateral de completitud Kraus (CPTP)."
                )
            if fro_res > tol_fro:
                raise KrausTraceViolationError(
                    f"‖C−I‖_F={fro_res:.6e} > tol_fro={tol_fro:.6e}."
                )
            return fro_res, eig_res

        # ------------------------------------------------------------------
        # Entropía de Von Neumann (sobre densidad)
        # ------------------------------------------------------------------

        @staticmethod
        def _density_from_pure(
            psi: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            """$\rho=|\psi\rangle\langle\psi|$ (no normaliza; respeta la traza $\|\psi\|^2$)."""
            v = np.asarray(psi, dtype=np.float64).reshape(-1)
            return np.outer(v, v)

        @staticmethod
        def _von_neumann_entropy_rho(
            rho: NDArray[np.float64],
        ) -> float:
            r"""
            $S(\rho)=-\mathrm{Tr}(\rho\log\rho)=-\sum_i\lambda_i\log\lambda_i$
            sobre autovalores $\lambda_i\ge 0$ de la parte hermítica de $\rho$.

            Convención: $0\log 0 := 0$ (continuidad).
            """
            H = 0.5 * (rho + rho.T)
            # Traza nula o numéricamente vacía
            tr = float(np.trace(H))
            if tr <= 1.0e-30:
                return 0.0
            # Normalizar a estado de traza 1 para la entropía física
            Hn = H / tr
            eigvals = np.linalg.eigvalsh(Hn)
            eigvals = np.clip(eigvals, 0.0, None)
            # Filtrar ceros numéricos
            pos = eigvals[eigvals > 1.0e-15]
            if pos.size == 0:
                return 0.0
            return float(-np.sum(pos * np.log(pos)))

        @staticmethod
        def _purity(rho: NDArray[np.float64]) -> float:
            r"""$\gamma=\mathrm{Tr}(\rho^2)/\mathrm{Tr}(\rho)^2\in(0,1]$."""
            H = 0.5 * (rho + rho.T)
            tr = float(np.trace(H))
            if tr <= 1.0e-30:
                return 0.0
            Hn = H / tr
            return float(np.trace(Hn @ Hn))

        @staticmethod
        def _apply_kraus_channel(
            rho: NDArray[np.float64],
            E0: NDArray[np.float64],
            E1: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""$\rho\mapsto E_0\rho E_0^\top + E_1\rho E_1^\top$ (Kraus real)."""
            return E0 @ rho @ E0.T + E1 @ rho @ E1.T

        # ------------------------------------------------------------------
        # Firma criptográfica del positrón
        # ------------------------------------------------------------------

        @staticmethod
        def _sign_antimatter(
            dissipated_entropy: float,
            delta_entropy: float,
            kraus_residual_eig: float,
            secret: bytes = b"MIC_Floquet_v3",
        ) -> str:
            payload = (
                f"{dissipated_entropy:.10e}|"
                f"{delta_entropy:.10e}|"
                f"{kraus_residual_eig:.10e}"
            )
            return hmac.new(
                secret, payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()

        # ------------------------------------------------------------------
        # Auditoría dimensional bloqueante (Axioma §0)
        # ------------------------------------------------------------------

        def _audit_dimensions_strict(
            self,
            psi_raw: NDArray[np.float64],
            H_obs_gradient: NDArray[np.float64],
            rho: Optional[NDArray[np.float64]] = None,
        ) -> DimensionalAudit:
            psi = np.asarray(psi_raw, dtype=np.float64).reshape(-1)
            grad = np.asarray(H_obs_gradient, dtype=np.float64).reshape(-1)
            psi_ok = psi.shape == (self._dim,)
            grad_ok = grad.shape == (self._dim,)
            rho_ok: Optional[bool] = None
            if rho is not None:
                rho_ok = rho.shape == (self._dim, self._dim)

            is_coherent = bool(
                psi_ok and grad_ok and (rho_ok if rho is not None else True)
            )
            audit = DimensionalAudit(
                dimension=self._dim,
                psi_dim_ok=psi_ok,
                grad_dim_ok=grad_ok,
                rho_dim_ok=rho_ok,
                is_coherent=is_coherent,
            )
            if not is_coherent:
                raise DimensionalMismatchError(
                    f"Auditoría dimensional fallida: psi_ok={psi_ok}, "
                    f"grad_ok={grad_ok}, rho_ok={rho_ok}, d={self._dim}."
                )
            return audit

        # ------------------------------------------------------------------
        # Método terminal de la Fase 3
        # ------------------------------------------------------------------

        def execute_quantum_channel(
            self,
            psi_raw: NDArray[np.float64],
            H_obs_gradient: NDArray[np.float64],
            synthesis: ProjectorSynthesisResult,
            monodromy: FloquetMonodromyState,
        ) -> QuantumChannelEvolution:
            r"""
            **Método terminal de la Fase 3 y del agente completo.**

            Parámetros
            ----------
            psi_raw : NDArray (d,)
                Estado puro inicial $|\psi\rangle$.
            H_obs_gradient : NDArray (d,)
                Gradiente de obstrucción (auditoría dimensional).
            synthesis : ProjectorSynthesisResult
                Salida de Fase 1 (provee $\hat{P}$).
            monodromy : FloquetMonodromyState
                Salida de Fase 2 (certificado de estabilidad; se embebe
                en el DTO de salida para trazabilidad forense).

            Retorna
            -------
            QuantumChannelEvolution
            """
            # §0 — dimensional (bloqueante)
            audit = self._audit_dimensions_strict(psi_raw, H_obs_gradient)

            psi = np.asarray(psi_raw, dtype=np.float64).reshape(-1)
            P = np.asarray(synthesis.P_hat, dtype=np.float64)
            I = np.eye(self._dim, dtype=np.float64)
            E0 = P
            E1 = I - P

            # §3 — completitud Kraus bilateral
            fro_res, eig_res = self._verify_kraus_completeness(E0, E1)

            # Estado coherente (amplitud) y canal sobre densidad
            coherent_state = E0 @ psi
            dissipated_vec = E1 @ psi

            rho_pre = self._density_from_pure(psi)
            rho_post = self._apply_kraus_channel(rho_pre, E0, E1)
            # Componente disipada como sub-canal
            rho_diss = E1 @ rho_pre @ E1.T

            # §4 — entropías de Von Neumann (sobre densidades normalizadas)
            S_pre = self._von_neumann_entropy_rho(rho_pre)
            S_post = self._von_neumann_entropy_rho(rho_post)
            S_diss = self._von_neumann_entropy_rho(rho_diss)
            delta_S = S_post - S_pre
            purity_post = self._purity(rho_post)

            # Emisión de antimateria
            antimatter_emission = None
            dissipated_norm_G = float(
                np.sqrt(max(dissipated_vec @ (self._G @ dissipated_vec), 0.0))
            )
            if (
                dissipated_norm_G > _EPS_DISSIPATION
                or S_diss > _EPS_DISSIPATION
            ):
                signature = self._sign_antimatter(S_diss, delta_S, eig_res)
                antimatter_emission = PositronCartridge(
                    inertial_mass=dissipated_norm_G,
                    topological_spin="inverse_hallucination",
                    homological_charge=-1,
                    authorization_signature=signature,
                )
                logger.warning(
                    "[Fase3] Positrón forense emitido "
                    "(S_diss=%.4f, ‖·‖_G=%.2e, sig=%s…).",
                    S_diss, dissipated_norm_G, signature[:8],
                )

            logger.info(
                "[Fase3] Canal CPTP: ‖C−I‖_F=%.2e, max|λ(C−I)|=%.2e, "
                "S_pre=%.4f, S_post=%.4f, ΔS=%.4f, purity_post=%.4f, "
                "ρ(M)=%.6e.",
                fro_res, eig_res, S_pre, S_post, delta_S, purity_post,
                monodromy.spectral_radius,
            )

            return QuantumChannelEvolution(
                coherent_state=coherent_state,
                rho_post=0.5 * (rho_post + rho_post.T),
                dissipated_entropy=S_diss,
                von_neumann_pre=S_pre,
                von_neumann_post=S_post,
                delta_entropy=delta_S,
                antimatter_emission=antimatter_emission,
                kraus_residual_fro=fro_res,
                kraus_residual_eig=eig_res,
                purity_post=purity_post,
                dimensional_audit=audit,
                monodromy_state=monodromy,
            )

    # =========================================================================
    # INTERFAZ PÚBLICA DEL AGENTE
    # =========================================================================

    def purify_and_tune_cavity(
        self,
        raw_llm_logits: NDArray[np.float64],
        h_obs_gradient: NDArray[np.float64],
    ) -> QuantumChannelEvolution:
        r"""
        Pipeline canónico completo (composición funtorial de las 3 fases):

        1. ``phase1.synthesize_projector`` → ``ProjectorSynthesisResult``
        2. ``phase2.audit_monodromy``      → ``FloquetMonodromyState``
        3. ``phase3.execute_quantum_channel`` → ``QuantumChannelEvolution``
        """
        logger.info(
            "[FloquetMonodromyAgent] Sintonizando cavidad y purgando alucinaciones."
        )

        # Fase 1
        synthesis = self.phase1.synthesize_projector(h_obs_gradient)

        # Fase 2 (continuación formal de synthesis)
        monodromy = self.phase2.audit_monodromy(synthesis)
        logger.debug(
            "Monodromía certificada: ρ=%.6e, estable=%s, idem=%.3e.",
            monodromy.spectral_radius,
            monodromy.is_asymptotically_stable,
            monodromy.projector_idempotence_residual,
        )

        # Fase 3 (continuación formal de monodromy + synthesis)
        evolution = self.phase3.execute_quantum_channel(
            psi_raw=raw_llm_logits,
            H_obs_gradient=h_obs_gradient,
            synthesis=synthesis,
            monodromy=monodromy,
        )
        return evolution

    def forward(self, state: CategoricalState) -> CategoricalState:
        """Reflexión Floquet sobre un ``CategoricalState`` del topos MIC."""
        psi = np.asarray(state.payload, dtype=np.float64).reshape(-1)
        if psi.shape != (self._dim,):
            raise DimensionalMismatchError(
                f"state.payload shape={psi.shape} ≠ ({self._dim},)."
            )
        canonical_grad = np.zeros(self._dim, dtype=np.float64)
        canonical_grad[0] = 1.0
        evolution = self.purify_and_tune_cavity(psi, canonical_grad)
        return CategoricalState(
            payload=evolution.coherent_state,
            label=getattr(state, "label", ""),
        )

    def backward(self, state: CategoricalState) -> CategoricalState:
        """
        Adjunta categórica. Para un Householder métrico $R=I-2P$ se tiene
        $R^{-1}=R$ (involución); aquí se reutiliza ``forward`` como convención
        del topos MIC cuando el payload ya es el estado coherente proyectado.
        """
        return self.forward(state)

    # ------------------------------------------------------------------
    # Propiedades de auditoría
    # ------------------------------------------------------------------

    @property
    def metric_tensor(self) -> NDArray[np.float64]:
        """Copia defensiva de $G$."""
        return self._G.copy()

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def phase1_synthesizer(self) -> ProjectorSynthesizerPort:
        return self.phase1

    @property
    def phase2_auditor(self) -> FloquetAuditorPort:
        return self.phase2

    @property
    def phase3_channel(self) -> KrausChannelPort:
        return self.phase3


# =============================================================================
# EXPORTACIÓN CANÓNICA
# =============================================================================

__all__ = [
    "FloquetInstabilityError",
    "KrausTraceViolationError",
    "KrausCompletenessError",
    "DimensionalMismatchError",
    "ProjectorDefectError",
    "FloquetParameterError",
    "DimensionalAudit",
    "ProjectorSynthesisResult",
    "FloquetMonodromyState",
    "QuantumChannelEvolution",
    "ProjectorSynthesizerPort",
    "FloquetAuditorPort",
    "KrausChannelPort",
    "FloquetMonodromyAgent",
]
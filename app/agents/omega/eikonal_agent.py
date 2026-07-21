# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Eikonal Agent (Operador de Fase de Fresnel y Monodromía Óptica)     ║
║ Ruta   : app/omega/eikonal_agent.py                                          ║
║ Versión: 3.0.0-Topos-WKB-Geodesic-Spectral-Nested                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA DIFERENCIAL (Rigor Categórico):
────────────────────────────────────────────────────────────────────────────────
Este módulo actúa como el Meta-Funtor de Control sobre el `OpticalRiemannLensFibrator` 
en el topos $\mathcal{T}_{\mathrm{MIC}}$. Ejerce su soberanía evaluando el límite 
WKB sobre el fibrado de fases de la intención semántica del LLM, definiendo la 
amplitud de probabilidad como:

$$
\psi(x) = A(x)\, e^{i\mathcal{S}(x)/\hbar}
$$

Este funtor repudia el ruteo estocástico; en su lugar, exige que la radiación 
semántica se propague a través de geodésicas covariantes minimizando el tiempo de
vuelo (Principio de Fermat) sobre una variedad Riemanniana definida por $G_{\mu\nu}$.

AXIOMAS DE EJECUCIÓN Y COMPOSICIÓN FUNTORIAL (Fases Anidadas):
────────────────────────────────────────────────────────────────────────────────
§0. Compatibilidad Dimensional y Física:
    Exige que la matriz de densidad $\rho$ sea estrictamente hermítica, semidefinida 
    positiva (PSD) y de traza unitaria ($\mathrm{Tr}(\rho) = 1$).

§1. Modulación de Apertura (Diafragma Óptico):
    Determina el límite de difracción truncando el espectro de la lente mediante 
    la entropía de von Neumann y la pureza del estado cuántico:
    $$ l = \left\lfloor l_{\max}\exp\left(-\frac{\kappa S_{\mathrm{MAC}}}{\mathrm{Tr}(\rho^2)}\right) \right\rfloor $$

§2. Resolución de la Ecuación Eikonal:
    Garantiza que el gradiente de fase satisfaga el índice de refracción efectivo 
    $n(\sigma^*)$ dictado por el estrés de mercado, incorporando un margen de holgura:
    $$ |\nabla S|_{G^{-1}}^2 = G^{\mu\nu}\partial_\mu S\,\partial_\nu S \ge n^2(\sigma^*)(1-\texttt{slack}) $$

§3. Auditoría del Camino de Fermat y Residuo Geodésico:
    Certifica la integral de acción a lo largo de la trayectoria $\gamma$ aplicando el
    método de Simpson compuesto de orden 4:
    $$ S_{\mathrm{Fermat}} = \int n\| \dot\gamma\|_G\,dt $$
    Computa el residuo geodésico covariante exigiéndolo nulo frente a la tolerancia:
    $$ \|a^{\mathrm{eff}}-a^{\mathrm{geo}}\|_G \le \varepsilon $$

CAMBIOS ESTRUCTURALES RESPECTO A v2.0.0:
────────────────────────────────────────────────────────────────────────────────
1. MetricSignatureError DEFINIDA formalmente para evitar violaciones implícitas.
2. List importado correctamente para garantizar la consistencia en `enforce_geodesic_path`.
3. Fases ANIDADAS en `EikonalAgent` + DTOs de continuación formal (espejo a `KApex` y `Floquet` v3).
4. Fase 1 retorna `ApertureModulationResult` (cutoff + certificado espectral); eliminando auditorías redundantes de $\rho$.
5. Ecuación eikonal con `eikonal_slack`: establece un umbral duro y bandera de margen holgado.
6. Inversión métrica: Cholesky + `solve_triangular` con preacondicionador espectral; consistencia bilateral garantizada: $\|GG^{-1}-I\|_F$ y $\|G^{-1}G-I\|_F \le \varepsilon$.
7. Integración de Simpson compuesto: si $T$ es par se reduce a $T-1$ (orden 4 exacto), evitando la degradación silenciosa a la regla del trapecio sesgada.
8. Residuo geodésico evaluado estrictamente bajo norma $G$ y tolerancia escalada.
9. $\rho$: Proyección hermítica defensiva + recorte PSD + renormalización de traza; se computa la pureza $\mathrm{Tr}(\rho^2) = \sum_i \lambda_i^2$ post-recorte.
10. Parámetros de control validados estrictamente (l_max, κ, slack, κ_max).
═══════════════════════════════════════════════════════════════════════════════

"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import (
    Any,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Dependencias del ecosistema (stubs para ejecución aislada / tests)
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
    from app.core.immune_system.musical_isomorphism_engine import (
        MetricSpectralPreconditioner,
    )
except ImportError:  # pragma: no cover

    class MetricSpectralPreconditioner:  # type: ignore[no-redef]
        """Stub: G_inv = inv(G) con cond via eigvalsh."""

        @dataclass
        class _PM:
            G_inv: NDArray[np.float64]
            condition_number: float

        def precondition(self, G: NDArray[np.float64]) -> "_PM":
            G_sym = 0.5 * (G + G.T)
            G_inv = la.inv(G_sym)
            ev = np.linalg.eigvalsh(G_sym)
            kappa = float(ev[-1] / max(ev[0], 1e-300))
            return self._PM(G_inv=G_inv, condition_number=kappa)


try:
    from app.omega.optical_riemann_lens import (
        OpticalRiemannLensFibrator,
        RefractedState,
    )
except ImportError:  # pragma: no cover

    @dataclass(frozen=True)
    class RefractedState:  # type: ignore[no-redef]
        refracted_logits: NDArray[np.float64]
        kv_compression_ratio: float
        refractive_index: float

    class OpticalRiemannLensFibrator:  # type: ignore[no-redef]
        def __init__(self, G: NDArray[np.float64]) -> None:
            self._G = np.asarray(G, dtype=np.float64)
            self._l_cutoff = 50

        def _compute_fermat_refractive_index(self, sigma: float) -> float:
            return 1.0 + float(np.tanh(0.5 * sigma))

        def refract_attention_logits(
            self, logits: NDArray[np.float64], sigma: float
        ) -> RefractedState:
            n = self._compute_fermat_refractive_index(sigma)
            return RefractedState(
                refracted_logits=np.asarray(logits, dtype=np.float64) * n,
                kv_compression_ratio=max(0.0, 1.0 - 1.0 / max(self._l_cutoff, 1)),
                refractive_index=n,
            )


try:
    from app.omega.levi_civita_agent import (
        LeviCivitaConnectionAgent,
        TangentVector,
    )
except ImportError:  # pragma: no cover

    @dataclass
    class TangentVector:  # type: ignore[no-redef]
        coordinates: NDArray[np.float64]

    class LeviCivitaConnectionAgent:  # type: ignore[no-redef]
        """
        Stub geodésico euclídeo: Γ=0 ⇒ a_geo=0, flujo = v constante.
        """

        def __init__(self, G: NDArray[np.float64]) -> None:
            self._G = np.asarray(G, dtype=np.float64)

        def enforce_geodesic_flow(
            self, v: TangentVector, dt: float
        ) -> TangentVector:
            return TangentVector(coordinates=v.coordinates.copy())

        def geodesic_rhs(self, v_coords: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.zeros_like(v_coords, dtype=np.float64)


logger = logging.getLogger("MIC.Omega.EikonalAgent")

# ---------------------------------------------------------------------------
# Constantes de rigor numérico
# ---------------------------------------------------------------------------
_MACHINE_EPS: float = float(np.finfo(np.float64).eps)
_WILKINSON_SAFETY: float = 100.0
_DEFAULT_KAPPA_MAX: float = 1.0e12
_DEFAULT_EIKONAL_SLACK: float = 0.1
_DEFAULT_CAVITY_TOL: float = 1.0e-10
_EPS_NEG: float = 1.0e-12
_EPS_PURITY: float = 1.0e-12
_SYM_ATOL: float = 1.0e-12
_HERM_ATOL: float = 1.0e-10


# =============================================================================
# SECCIÓN 0 — EXCEPCIONES
# =============================================================================


class QuantumPurityCollapseError(TopologicalInvariantError):
    r"""$\mathrm{Tr}(\rho^2)\to 0$ o ρ no hermitiana / no física."""


class EikonalSingularityError(TopologicalInvariantError):
    r"""$G^{-1}$ no SPD, κ excesivo, o Hamiltoniano eikonal no hiperbólico."""


class FermatOpticalDeviationError(TopologicalInvariantError):
    """Acción de Fermat divergente o residuo geodésico fuera de tolerancia."""


class DimensionalMismatchError(TopologicalInvariantError):
    """Dimensiones de ρ, G o vectores incompatibles (Axioma §0)."""


class MetricSignatureError(TopologicalInvariantError):
    """G no es simétrica o no es SPD."""


class EikonalParameterError(TopologicalInvariantError):
    """Parámetro escalar de control fuera de rango admisible."""


class EikonalRefractionError(TopologicalInvariantError):
    r"""
    $\|\nabla S\|_{G^{-1}}^2 < n^2(1-\mathrm{slack})$:
    frente de onda insuficiente para el índice de refracción.
    """


# =============================================================================
# SECCIÓN 1 — DTO INMUTABLES (objetos del topos)
# =============================================================================


@dataclass(frozen=True, slots=True)
class SpectralDensityAudit:
    """Certificado espectral de $\\rho$ tras proyección hermítica + recorte PSD."""

    purity: float
    von_neumann_entropy: float
    eigenvalues_psd: NDArray[np.float64]
    negative_eigvals_pruned: int
    hermiticity_residual: float
    trace_after_pruning: float
    is_physical: bool
    dim: int


@dataclass(frozen=True, slots=True)
class ApertureModulationResult:
    r"""
    Salida terminal de la **Fase 1** — precondición formal de la Fase 2
    (el cutoff alimenta el lente; el certificado viaja al estado final).
    """

    l_cutoff: int
    attenuation_factor: float
    spectral_certificate: SpectralDensityAudit
    s_mac: float
    kappa_coupling: float
    l_max: int


@dataclass(frozen=True, slots=True)
class EikonalSurfaceResult:
    r"""
    Salida terminal de la **Fase 2** — precondición formal de la Fase 3.

    \[
      \|\nabla S\|_{G^{-1}}^2
        = G^{\mu\nu}\partial_\mu S\,\partial_\nu S,
      \qquad
      \text{target}=n^2.
    \]
    """

    phase_norm_sq: float
    n_refract: float
    n_sq: float
    eikonal_residual: float
    eikonal_threshold: float
    margin_sound: bool
    kappa_G: float
    kappa_G_inv: float
    inverse_residual: float
    dim: int


@dataclass(frozen=True, slots=True)
class FermatPathResult:
    """Salida terminal de la **Fase 3** — trayectoria y acción de Fermat."""

    path_velocities: NDArray[np.float64]
    fermat_action: float
    geodesic_deviation: float
    integration_rule: str  # "simpson" | "trapezoid"
    n_refract: float
    n_steps_integrated: int


@dataclass(frozen=True, slots=True)
class EikonalPhaseState:
    """Estado inmutable de la resolución eikonal completa (salida del endofuntor)."""

    phase_gradient_norm: float
    fermat_action_integral: float
    dynamic_l_cutoff: int
    refracted_state: RefractedState
    geodesic_deviation: float = 0.0
    spectral_certificate: Optional[SpectralDensityAudit] = None
    eikonal_surface: Optional[EikonalSurfaceResult] = None
    fermat_path: Optional[FermatPathResult] = None
    aperture: Optional[ApertureModulationResult] = None


@dataclass(frozen=True, slots=True)
class EikonalControlInput:
    r"""
    Objeto del topos — entrada inmutable de ``execute_optical_guidance``.
    """

    raw_llm_logits: NDArray[np.float64]
    rho_llm: NDArray[np.float64]
    s_mac_entropy: float
    logistic_stress_norm: float
    phase_gradient: NDArray[np.float64]
    path_velocities: NDArray[np.float64]
    use_geodesic_correction: bool = True
    cavity_tol: float = _DEFAULT_CAVITY_TOL


# =============================================================================
# SECCIÓN 2 — PROTOCOLOS
# =============================================================================


@runtime_checkable
class ApertureModulatorPort(Protocol):
    def modulate_aperture(
        self, s_mac: float, rho_llm: NDArray[np.float64]
    ) -> ApertureModulationResult: ...


@runtime_checkable
class EikonalResolverPort(Protocol):
    def resolve_eikonal(
        self, phase_gradient: NDArray[np.float64], n_refract: float
    ) -> EikonalSurfaceResult: ...


@runtime_checkable
class FermatAuditorPort(Protocol):
    def audit_path(
        self,
        path_velocities: NDArray[np.float64],
        n_refract: float,
        *,
        use_geodesic_correction: bool,
        cavity_tol: float,
        dt_fermat: float = 1e-3,
        dt_geo: float = 1.0,
    ) -> FermatPathResult: ...


# =============================================================================
# SECCIÓN 3 — ORQUESTADOR CON TRES FASES ANIDADAS
# =============================================================================


class EikonalAgent(Morphism):
    r"""
    Morfismo Eikonal maestro en $\mathcal{T}_{\mathrm{MIC}}$.

    Compone tres fases anidadas por agregación tipada:

    * ``phase1`` — diafragma cuántico (pureza + cutoff)
    * ``phase2`` — superficie eikonal ($G^{-1}$, hiperbólicidad)
    * ``phase3`` — acción de Fermat + residuo geodésico

    y comanda el ``OpticalRiemannLensFibrator``.
    """

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        l_max_absolute: int = 50,
        kappa_coupling: float = 1.0,
        eikonal_slack: float = _DEFAULT_EIKONAL_SLACK,
        kappa_max: float = _DEFAULT_KAPPA_MAX,
    ) -> None:
        if l_max_absolute < 1:
            raise EikonalParameterError(
                f"l_max_absolute debe ser ≥ 1; se obtuvo {l_max_absolute}."
            )
        if kappa_coupling <= 0.0:
            raise EikonalParameterError(
                f"kappa_coupling debe ser > 0; se obtuvo {kappa_coupling}."
            )
        if not (0.0 <= eikonal_slack < 1.0):
            raise EikonalParameterError(
                f"eikonal_slack debe estar en [0,1); se obtuvo {eikonal_slack}."
            )
        if kappa_max <= 1.0:
            raise EikonalParameterError(
                f"kappa_max debe ser > 1; se obtuvo {kappa_max}."
            )

        G = np.asarray(metric_tensor, dtype=np.float64)
        G = self._validate_metric_spd(G)
        self._G = G
        self._dim = int(G.shape[0])
        self._eikonal_slack = float(eikonal_slack)
        self._kappa_max = float(kappa_max)

        # Fases anidadas
        self.phase1 = EikonalAgent.Phase1_DynamicApertureModulator(
            l_max_absolute=l_max_absolute,
            kappa_coupling=kappa_coupling,
        )
        self.phase2 = EikonalAgent.Phase2_EikonalSurfaceResolver(
            metric_tensor=G,
            kappa_max=kappa_max,
            eikonal_slack=eikonal_slack,
        )
        self.phase3 = EikonalAgent.Phase3_FermatActionAuditor(
            metric_tensor=G,
        )
        self._lens_fibrator = OpticalRiemannLensFibrator(G)

        try:
            super().__init__(name="EikonalAgent")
        except TypeError:
            super().__init__()

        logger.info(
            "[EikonalAgent] v3 inicializado: d=%d, l_max=%d, κ_coup=%.3f, "
            "eikonal_slack=%.3f, κ_max=%.1e.",
            self._dim,
            l_max_absolute,
            kappa_coupling,
            eikonal_slack,
            kappa_max,
        )

    # ------------------------------------------------------------------
    # Validación métrica compartida
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_metric_spd(G: NDArray[np.float64]) -> NDArray[np.float64]:
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise DimensionalMismatchError(
                f"G debe ser cuadrada; se recibió {G.shape}."
            )
        if G.shape[0] == 0:
            raise DimensionalMismatchError("dim(G) no puede ser 0.")
        G_sym = 0.5 * (G + G.T)
        if float(la.norm(G - G.T, "fro")) > _SYM_ATOL * max(
            float(la.norm(G, "fro")), 1.0
        ):
            raise MetricSignatureError(
                f"G no es simétrica: ‖G−Gᵀ‖_F="
                f"{float(la.norm(G - G.T, 'fro')):.6e}."
            )
        try:
            la.cholesky(G_sym, lower=True, check_finite=False)
        except la.LinAlgError as exc:
            raise MetricSignatureError(f"G no es SPD: {exc}") from exc
        return G_sym

    # =========================================================================
    # FASE 1 — DIAFRAGMA DINÁMICO CUÁNTICO
    # =========================================================================

    class Phase1_DynamicApertureModulator:
        r"""
        **Fase 1** — Diafragma espectral.

        \[
          l_{\mathrm{cutoff}}
            = \left\lfloor
                l_{\max}
                \exp\!\Bigl(-\frac{\kappa\, S_{\mathrm{MAC}}}{\mathrm{Tr}(\rho^2)}\Bigr)
              \right\rfloor,
          \qquad l_{\mathrm{cutoff}}\ge 1.
        \]

        Antes de la pureza: proyección hermítica, recorte PSD y renormalización.
        """

        def __init__(
            self, l_max_absolute: int = 50, kappa_coupling: float = 1.0
        ) -> None:
            self._l_max = int(l_max_absolute)
            self._kappa = float(kappa_coupling)

        # ---- auditoría espectral de ρ ------------------------------------

        def audit_density_matrix(
            self, rho: NDArray[np.float64]
        ) -> SpectralDensityAudit:
            r"""
            Axioma §0: hermiticidad, PSD (recorte), traza 1, pureza y $S_{\mathrm{vN}}$.

            Pureza post-recorte:
            \[
              \mathrm{Tr}(\rho^2)=\sum_i\lambda_i^2
              \quad(\lambda_i\ge 0,\;\sum_i\lambda_i=1).
            \]
            """
            rho_arr = np.asarray(rho)
            if rho_arr.ndim != 2 or rho_arr.shape[0] != rho_arr.shape[1]:
                raise DimensionalMismatchError(
                    f"ρ debe ser cuadrada; se recibió {rho_arr.shape}."
                )
            d = int(rho_arr.shape[0])

            # Hermiticidad: residual y proyección defensiva
            if np.iscomplexobj(rho_arr):
                herm_res = float(la.norm(rho_arr - rho_arr.T.conj(), "fro"))
                if herm_res > _HERM_ATOL * max(float(la.norm(rho_arr, "fro")), 1.0):
                    raise QuantumPurityCollapseError(
                        f"ρ no es hermitiana: ‖ρ−ρ†‖_F={herm_res:.6e}."
                    )
                rho_h = 0.5 * (rho_arr + rho_arr.T.conj())
                eigvals = np.linalg.eigvalsh(rho_h).astype(np.float64)
            else:
                rho_f = np.asarray(rho_arr, dtype=np.float64)
                herm_res = float(la.norm(rho_f - rho_f.T, "fro"))
                if herm_res > _HERM_ATOL * max(float(la.norm(rho_f, "fro")), 1.0):
                    raise QuantumPurityCollapseError(
                        f"ρ no es simétrica/hermitiana: ‖ρ−ρᵀ‖_F={herm_res:.6e}."
                    )
                rho_h = 0.5 * (rho_f + rho_f.T)
                eigvals = np.linalg.eigvalsh(rho_h)

            n_negative = int(np.sum(eigvals < -_EPS_NEG))
            if n_negative > 0:
                logger.warning(
                    "[Fase1] ρ: %d autovalores negativos (λ_min=%.3e); recorte PSD.",
                    n_negative, float(eigvals.min()),
                )
            eigvals = np.clip(eigvals, 0.0, None)
            tr = float(np.sum(eigvals))
            if tr <= _EPS_PURITY:
                raise QuantumPurityCollapseError(
                    f"Tr(ρ)≤0 tras recorte PSD (tr={tr:.3e}). Estado no físico."
                )
            eigvals = eigvals / tr

            purity = float(np.sum(eigvals ** 2))
            eig_safe = np.clip(eigvals, 1e-15, None)
            # 0 log 0 := 0
            s_vn = float(-np.sum(eigvals * np.log(eig_safe)))

            return SpectralDensityAudit(
                purity=purity,
                von_neumann_entropy=s_vn,
                eigenvalues_psd=eigvals,
                negative_eigvals_pruned=n_negative,
                hermiticity_residual=herm_res,
                trace_after_pruning=float(np.sum(eigvals)),
                is_physical=(purity >= _EPS_PURITY),
                dim=d,
            )

        # ---- método terminal Fase 1 --------------------------------------

        def modulate_aperture(
            self, s_mac: float, rho_llm: NDArray[np.float64]
        ) -> ApertureModulationResult:
            r"""
            **Método terminal de la Fase 1.**

            Retorna
            -------
            ApertureModulationResult
                Precondición formal hacia el lente / certificado de salida.
            """
            s_mac_f = float(s_mac)
            if s_mac_f < 0.0:
                raise EikonalParameterError(
                    f"s_mac debe ser ≥ 0; se obtuvo {s_mac_f}."
                )

            audit = self.audit_density_matrix(rho_llm)
            if audit.purity < _EPS_PURITY or not audit.is_physical:
                raise QuantumPurityCollapseError(
                    f"Pureza nula tras recorte PSD (Tr(ρ²)={audit.purity:.3e}). "
                    "Caos térmico absoluto del estado generativo."
                )

            attenuation = math.exp(
                -(self._kappa * s_mac_f) / audit.purity
            )
            l_cutoff = max(1, int(math.floor(self._l_max * attenuation)))

            logger.info(
                "[Fase1] Diafragma: l_max=%d → l_cutoff=%d "
                "(κ=%.3f, S_MAC=%.4f, pureza=%.6f, S_vN=%.4f).",
                self._l_max, l_cutoff, self._kappa, s_mac_f,
                audit.purity, audit.von_neumann_entropy,
            )
            return ApertureModulationResult(
                l_cutoff=l_cutoff,
                attenuation_factor=attenuation,
                spectral_certificate=audit,
                s_mac=s_mac_f,
                kappa_coupling=self._kappa,
                l_max=self._l_max,
            )

        # Alias retrocompatible con el Protocol v2
        def compute_dynamic_cutoff(
            self, s_mac: float, rho_llm: NDArray[np.float64]
        ) -> int:
            return self.modulate_aperture(s_mac, rho_llm).l_cutoff

    # =========================================================================
    # FASE 2 — RESOLUTOR DE LA SUPERFICIE EIKONAL
    #           (continuación formal: n_refract del lente + ∇S del control)
    # =========================================================================

    class Phase2_EikonalSurfaceResolver:
        r"""
        **Fase 2** — Campo escalar de fase:

        \[
          G^{\mu\nu}\partial_\mu S\,\partial_\nu S = n^2(\sigma^*).
        \]

        Certifica $G^{-1}\succ 0$ y $\kappa_2(G^{-1})<\kappa_{\max}$.
        Umbral duro con slack:
        \[
          \|\nabla S\|_{G^{-1}}^2
            \ge n^2\,(1-\texttt{eikonal\_slack}).
        \]
        """

        def __init__(
            self,
            metric_tensor: NDArray[np.float64],
            kappa_max: float = _DEFAULT_KAPPA_MAX,
            eikonal_slack: float = _DEFAULT_EIKONAL_SLACK,
        ) -> None:
            self._kappa_max = float(kappa_max)
            self._eikonal_slack = float(eikonal_slack)
            self._G = np.asarray(metric_tensor, dtype=np.float64)
            self._dim = int(self._G.shape[0])

            self._G_inv, self._kappa_G, self._kappa_G_inv, self._inv_residual = (
                self._build_and_certify_inverse(self._G)
            )

            logger.debug(
                "[Fase2] G⁻¹ certificada: κ(G)=%.3e, κ(G⁻¹)=%.3e, "
                "inv_res=%.3e.",
                self._kappa_G, self._kappa_G_inv, self._inv_residual,
            )

        def _build_and_certify_inverse(
            self, G: NDArray[np.float64]
        ) -> Tuple[NDArray[np.float64], float, float, float]:
            """Cholesky → G_inv; fallback preacondicionador; certificación SPD."""
            n = G.shape[0]
            G_sym = 0.5 * (G + G.T)

            # κ(G)
            ev_G = np.linalg.eigvalsh(G_sym)
            lmin_G, lmax_G = float(ev_G[0]), float(ev_G[-1])
            if lmin_G <= _MACHINE_EPS * max(abs(lmax_G), 1.0):
                raise EikonalSingularityError(
                    f"G degenerada: λ_min={lmin_G:.3e}."
                )
            kappa_G = lmax_G / lmin_G
            if kappa_G > self._kappa_max:
                raise EikonalSingularityError(
                    f"κ(G)={kappa_G:.3e} > κ_max={self._kappa_max:.3e}."
                )

            # Inversa vía Cholesky (estable) o preacondicionador
            try:
                L = la.cholesky(G_sym, lower=True)
                # G_inv = L^{-T} L^{-1}
                I = np.eye(n, dtype=np.float64)
                Y = la.solve_triangular(L, I, lower=True, check_finite=False)
                G_inv = Y.T @ Y
                G_inv = 0.5 * (G_inv + G_inv.T)
            except la.LinAlgError:
                logger.warning(
                    "[Fase2] Cholesky falló; usando MetricSpectralPreconditioner."
                )
                pm = MetricSpectralPreconditioner().precondition(G_sym)
                G_inv = 0.5 * (pm.G_inv + pm.G_inv.T)

            # Certificar G_inv SPD + κ
            ev_inv = np.linalg.eigvalsh(G_inv)
            lmin_i, lmax_i = float(ev_inv[0]), float(ev_inv[-1])
            if lmin_i <= 1e-15:
                raise EikonalSingularityError(
                    f"G⁻¹ no SPD: λ_min={lmin_i:.3e}. "
                    "Hamiltoniano eikonal degenerado."
                )
            kappa_G_inv = lmax_i / lmin_i
            if kappa_G_inv > self._kappa_max:
                raise EikonalSingularityError(
                    f"κ(G⁻¹)={kappa_G_inv:.3e} > κ_max={self._kappa_max:.3e}."
                )

            # Consistencia bilateral
            I = np.eye(n, dtype=np.float64)
            r_plus = float(la.norm(G_sym @ G_inv - I, "fro")) / n
            r_minus = float(la.norm(G_inv @ G_sym - I, "fro")) / n
            inv_res = max(r_plus, r_minus)
            tol_inv = kappa_G * _MACHINE_EPS * n * _WILKINSON_SAFETY
            if inv_res > max(tol_inv, 1e-8):
                raise EikonalSingularityError(
                    f"Inconsistencia G·G⁻¹: residual bilateral={inv_res:.3e} "
                    f"> tol={tol_inv:.3e}."
                )

            return G_inv, kappa_G, kappa_G_inv, inv_res

        def resolve_eikonal(
            self,
            phase_gradient: NDArray[np.float64],
            n_refract: float,
        ) -> EikonalSurfaceResult:
            r"""
            **Método terminal de la Fase 2.**

            Retorna
            -------
            EikonalSurfaceResult
                Precondición formal de ``Phase3.audit_path`` (vía n_refract
                y certificación de hiperbolicidad).
            """
            grad = np.asarray(phase_gradient, dtype=np.float64).reshape(-1)
            if grad.shape != (self._dim,):
                raise DimensionalMismatchError(
                    f"∇S shape={grad.shape}, esperada ({self._dim},)."
                )

            n_ref = float(n_refract)
            if n_ref < 0.0:
                raise EikonalParameterError(
                    f"n_refract debe ser ≥ 0; se obtuvo {n_ref}."
                )

            # ‖∇S‖²_{G^{-1}} = ∇Sᵀ G⁻¹ ∇S
            s_norm_sq = float(grad @ self._G_inv @ grad)

            if not np.isfinite(s_norm_sq):
                raise EikonalSingularityError(
                    f"‖∇S‖²_G no finito ({s_norm_sq})."
                )
            if s_norm_sq < 0.0:
                # Numéricamente imposible si G_inv SPD; canario de bug
                raise EikonalSingularityError(
                    f"‖∇S‖²_G={s_norm_sq:.3e} < 0 (G⁻¹ no SPD en la práctica)."
                )

            n_sq = n_ref ** 2
            threshold = n_sq * (1.0 - self._eikonal_slack)
            residual = abs(s_norm_sq - n_sq)

            # Umbral duro (alcanzabilidad del colector)
            if s_norm_sq < threshold:
                raise EikonalRefractionError(
                    f"Fallo Eikonal: ‖∇S‖²_{{G⁻¹}}={s_norm_sq:.6e} < "
                    f"n²(1−slack)={threshold:.6e} "
                    f"(n={n_ref:.6f}, slack={self._eikonal_slack:.3f})."
                )

            soft = n_sq * (1.0 - 0.5 * self._eikonal_slack)
            margin_sound = s_norm_sq >= soft

            if residual > 1e-4 * max(n_sq, 1.0):
                logger.warning(
                    "[Fase2] Desviación eikonal |‖∇S‖²−n²|=%.3e "
                    "(n²=%.6e, ‖∇S‖²=%.6e).",
                    residual, n_sq, s_norm_sq,
                )

            logger.debug(
                "[Fase2] Eikonal OK: ‖∇S‖²=%.6e, n²=%.6e, thr=%.6e, margin=%s.",
                s_norm_sq, n_sq, threshold, margin_sound,
            )
            return EikonalSurfaceResult(
                phase_norm_sq=s_norm_sq,
                n_refract=n_ref,
                n_sq=n_sq,
                eikonal_residual=residual,
                eikonal_threshold=threshold,
                margin_sound=margin_sound,
                kappa_G=self._kappa_G,
                kappa_G_inv=self._kappa_G_inv,
                inverse_residual=self._inv_residual,
                dim=self._dim,
            )

        # Alias v2
        def resolve_eikonal_equation(
            self,
            phase_gradient: NDArray[np.float64],
            n_refract: float,
        ) -> float:
            return self.resolve_eikonal(phase_gradient, n_refract).phase_norm_sq

        @property
        def G_inv(self) -> NDArray[np.float64]:
            return self._G_inv.copy()

    # =========================================================================
    # FASE 3 — ACCIÓN DE FERMAT Y RESIDUO GEODÉSICO
    #           (continuación formal de EikonalSurfaceResult.n_refract)
    # =========================================================================

    class Phase3_FermatActionAuditor:
        r"""
        **Fase 3** — Integral de Fermat y transporte geodésico.

        Simpson compuesto (orden 4), $T$ impar efectivo:

        \[
          \mathcal{A}
            \approx n\cdot\frac{\Delta t}{3}
              \sum_{k=0}^{T-1} w_k\,\|\dot\gamma_k\|_G,
          \qquad
          w=(1,4,2,4,\ldots,4,1).
        \]

        Residuo geodésico covariante medio:

        \[
          \mathcal{R}
            = \frac1N\sum_k
              \Bigl\|
                \frac{v_{k+1}-v_k}{\Delta t}-a^{\mathrm{geo}}_k
              \Bigr\|_G.
        \]
        """

        def __init__(self, metric_tensor: NDArray[np.float64]) -> None:
            self._G = np.asarray(metric_tensor, dtype=np.float64)
            self._dim = int(self._G.shape[0])
            self._levi_civita = LeviCivitaConnectionAgent(self._G)

        def _speed_norms(
            self, V: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            """$\|v_t\|_G=\sqrt{v_t^\top G v_t}$ vectorizado."""
            # max(.,0) evita NaN por redondeo
            quad = np.einsum("ti,ij,tj->t", V, self._G, V)
            return np.sqrt(np.maximum(quad, 0.0))

        def integrate_fermat(
            self,
            path_velocities: NDArray[np.float64],
            n_refract: float,
            dt: float = 1e-3,
        ) -> Tuple[float, str, int]:
            """
            Retorna ``(action, rule, n_pts_used)``.
            """
            V = np.asarray(path_velocities, dtype=np.float64)
            if V.ndim != 2:
                raise DimensionalMismatchError(
                    f"path_velocities debe ser (T,d); se recibió {V.shape}."
                )
            if V.shape[1] != self._dim:
                raise DimensionalMismatchError(
                    f"path_velocities d={V.shape[1]} ≠ {self._dim}."
                )
            if dt <= 0.0:
                raise EikonalParameterError(f"dt debe ser > 0; se obtuvo {dt}.")

            n_pts = int(V.shape[0])
            if n_pts == 0:
                return 0.0, "empty", 0

            norms = self._speed_norms(V)

            if n_pts < 3:
                # Trapecio (o punto único)
                if n_pts == 1:
                    integral = float(norms[0]) * dt
                else:
                    integral = 0.5 * (float(norms[0]) + float(norms[-1])) * dt
                    if n_pts == 2:
                        pass
                    # n_pts==2 only
                rule = "trapezoid"
                used = n_pts
            else:
                # Simpson requiere número impar de nodos: si par, truncar último
                if n_pts % 2 == 0:
                    norms = norms[:-1]
                    used = n_pts - 1
                    logger.debug(
                        "[Fase3] Simpson: T=%d par → se usan %d nodos (orden 4).",
                        n_pts, used,
                    )
                else:
                    used = n_pts
                weights = np.ones(used, dtype=np.float64)
                weights[1:-1:2] = 4.0
                weights[2:-1:2] = 2.0
                integral = float((dt / 3.0) * np.dot(weights, norms))
                rule = "simpson"

            action = float(n_refract) * integral
            if not np.isfinite(action) or action > 1.0e10:
                raise FermatOpticalDeviationError(
                    f"Divergencia en Acción de Fermat: A={action}."
                )
            return action, rule, used

        def enforce_geodesic_path(
            self,
            initial_velocity: TangentVector,
            n_steps: int,
            dt: float = 1.0,
        ) -> Tuple[NDArray[np.float64], float]:
            if n_steps < 1:
                raise EikonalParameterError(
                    f"n_steps debe ser ≥ 1; se obtuvo {n_steps}."
                )
            if dt <= 0.0:
                raise EikonalParameterError(f"dt_geo debe ser > 0; se obtuvo {dt}.")

            velocities: List[NDArray[np.float64]] = []
            total_deviation = 0.0
            v_current = initial_velocity
            velocities.append(np.asarray(v_current.coordinates, dtype=np.float64).copy())

            for _ in range(n_steps):
                v_next = self._levi_civita.enforce_geodesic_flow(v_current, dt)
                v_next_coords = np.asarray(v_next.coordinates, dtype=np.float64)

                a_eff = (v_next_coords - v_current.coordinates) / dt
                a_geo = np.asarray(
                    self._levi_civita.geodesic_rhs(v_current.coordinates),
                    dtype=np.float64,
                )
                delta = a_eff - a_geo
                deviation_step = float(
                    np.sqrt(max(delta @ (self._G @ delta), 0.0))
                )
                total_deviation += deviation_step

                velocities.append(v_next_coords.copy())
                v_current = v_next

            mean_deviation = total_deviation / n_steps
            return np.asarray(velocities, dtype=np.float64), float(mean_deviation)

        def audit_path(
            self,
            path_velocities: NDArray[np.float64],
            n_refract: float,
            *,
            use_geodesic_correction: bool,
            cavity_tol: float,
            dt_fermat: float = 1e-3,
            dt_geo: float = 1.0,
        ) -> FermatPathResult:
            r"""
            **Método terminal de la Fase 3.**

            Opcionalmente corrige la trayectoria por flujo geodésico y
            audita la acción de Fermat sobre la trayectoria resultante.
            """
            V = np.asarray(path_velocities, dtype=np.float64)
            geodesic_deviation = 0.0

            if use_geodesic_correction and V.shape[0] > 0:
                initial_v = TangentVector(coordinates=V[0].copy())
                n_steps = int(V.shape[0])
                V, geodesic_deviation = self.enforce_geodesic_path(
                    initial_v, n_steps, dt=dt_geo
                )
                logger.debug(
                    "[Fase3] Desviación geodésica media: %.3e (tol=%.3e).",
                    geodesic_deviation, cavity_tol,
                )
                if geodesic_deviation > cavity_tol:
                    raise FermatOpticalDeviationError(
                        f"Residuo geodésico {geodesic_deviation:.6e} > "
                        f"tol={cavity_tol:.6e}."
                    )

            action, rule, used = self.integrate_fermat(V, n_refract, dt=dt_fermat)

            logger.debug(
                "[Fase3] Fermat: A=%.6e, rule=%s, pts=%d, R_geo=%.3e.",
                action, rule, used, geodesic_deviation,
            )
            return FermatPathResult(
                path_velocities=V,
                fermat_action=action,
                geodesic_deviation=geodesic_deviation,
                integration_rule=rule,
                n_refract=float(n_refract),
                n_steps_integrated=used,
            )

        # Alias v2
        def audit_fermat_action(
            self,
            path_velocities: NDArray[np.float64],
            n_refract: float,
            dt: float = 1e-3,
        ) -> float:
            action, _, _ = self.integrate_fermat(
                path_velocities, n_refract, dt=dt
            )
            return action

    # =========================================================================
    # INTERFAZ PÚBLICA DEL AGENTE
    # =========================================================================

    def execute_optical_guidance(
        self, control: EikonalControlInput
    ) -> EikonalPhaseState:
        r"""
        Pipeline canónico:

        1. Validación dimensional (§0)
        2. Fase 1 — diafragma → ``ApertureModulationResult``
        3. Índice de refracción del lente
        4. Fase 2 — eikonal → ``EikonalSurfaceResult``
        5. Fase 3 — Fermat/geodésica → ``FermatPathResult``
        6. Inyección en el lente → ``RefractedState``
        """
        logger.info(
            "[EikonalAgent] Resolución WKB y modulación del Lente de Riemann."
        )
        self._validate_control_input(control)

        # Fase 1
        aperture = self.phase1.modulate_aperture(
            control.s_mac_entropy, control.rho_llm
        )

        # Índice de refracción (lente)
        n_refract = float(
            self._lens_fibrator._compute_fermat_refractive_index(
                control.logistic_stress_norm
            )
        )

        # Fase 2
        surface = self.phase2.resolve_eikonal(
            control.phase_gradient, n_refract
        )

        # Fase 3
        fermat = self.phase3.audit_path(
            control.path_velocities,
            n_refract,
            use_geodesic_correction=control.use_geodesic_correction,
            cavity_tol=control.cavity_tol,
        )

        # Lente óptico
        self._lens_fibrator._l_cutoff = aperture.l_cutoff
        refracted_state = self._lens_fibrator.refract_attention_logits(
            control.raw_llm_logits, control.logistic_stress_norm
        )

        logger.info(
            "[EikonalAgent] Consolidado: A=%.4f, l_cutoff=%d, "
            "‖∇S‖²=%.4f, R_geo=%.2e, margin=%s, pureza=%.4f.",
            fermat.fermat_action,
            aperture.l_cutoff,
            surface.phase_norm_sq,
            fermat.geodesic_deviation,
            surface.margin_sound,
            aperture.spectral_certificate.purity,
        )

        return EikonalPhaseState(
            phase_gradient_norm=surface.phase_norm_sq,
            fermat_action_integral=fermat.fermat_action,
            dynamic_l_cutoff=aperture.l_cutoff,
            refracted_state=refracted_state,
            geodesic_deviation=fermat.geodesic_deviation,
            spectral_certificate=aperture.spectral_certificate,
            eikonal_surface=surface,
            fermat_path=fermat,
            aperture=aperture,
        )

    def _validate_control_input(self, control: EikonalControlInput) -> None:
        d = self._dim
        if control.raw_llm_logits.shape != (d,):
            raise DimensionalMismatchError(
                f"raw_llm_logits: {control.raw_llm_logits.shape} ≠ ({d},)."
            )
        if control.rho_llm.shape != (d, d):
            raise DimensionalMismatchError(
                f"rho_llm: {control.rho_llm.shape} ≠ ({d},{d})."
            )
        if control.phase_gradient.shape != (d,):
            raise DimensionalMismatchError(
                f"phase_gradient: {control.phase_gradient.shape} ≠ ({d},)."
            )
        if (
            control.path_velocities.ndim != 2
            or control.path_velocities.shape[1] != d
        ):
            raise DimensionalMismatchError(
                f"path_velocities: {control.path_velocities.shape} "
                f"incompatible con d={d}."
            )
        if control.s_mac_entropy < 0.0:
            raise EikonalParameterError(
                f"s_mac_entropy debe ser ≥ 0; se obtuvo {control.s_mac_entropy}."
            )
        if control.cavity_tol < 0.0:
            raise EikonalParameterError(
                f"cavity_tol debe ser ≥ 0; se obtuvo {control.cavity_tol}."
            )

    def forward(self, state: CategoricalState) -> CategoricalState:
        psi = np.asarray(state.payload, dtype=np.float64).reshape(-1)
        if psi.shape != (self._dim,):
            raise DimensionalMismatchError(
                f"state.payload shape={psi.shape} ≠ ({self._dim},)."
            )
        return CategoricalState(
            payload=psi,
            label=f"{getattr(state, 'label', '')}::eikonal_forward",
        )

    def backward(self, state: CategoricalState) -> CategoricalState:
        return self.forward(state)

    @property
    def metric_tensor(self) -> NDArray[np.float64]:
        return self._G.copy()

    @property
    def metric_inverse(self) -> NDArray[np.float64]:
        return self.phase2.G_inv

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def phase1_modulator(self) -> ApertureModulatorPort:
        return self.phase1

    @property
    def phase2_resolver(self) -> EikonalResolverPort:
        return self.phase2

    @property
    def phase3_auditor(self) -> FermatAuditorPort:
        return self.phase3


# =============================================================================
# EXPORTACIÓN CANÓNICA
# =============================================================================

__all__ = [
    "QuantumPurityCollapseError",
    "EikonalSingularityError",
    "FermatOpticalDeviationError",
    "DimensionalMismatchError",
    "MetricSignatureError",
    "EikonalParameterError",
    "EikonalRefractionError",
    "SpectralDensityAudit",
    "ApertureModulationResult",
    "EikonalSurfaceResult",
    "FermatPathResult",
    "EikonalPhaseState",
    "EikonalControlInput",
    "ApertureModulatorPort",
    "EikonalResolverPort",
    "FermatAuditorPort",
    "EikonalAgent",
]
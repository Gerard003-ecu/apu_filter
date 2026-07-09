# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Eikonal Agent (Operador de Fase de Fresnel y Monodromía Óptica)     ║
║ Ubicación: app/omega/eikonal_agent.py                                        ║
║ Versión: 2.0.0‑Topos‑WKB‑Geodesic‑SpectralAuditor                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber‑Física y Topológica Diferencial — Edición Granular:
────────────────────────────────────────────────────────────────────────────────
Este módulo actúa como el Meta‑Funtor de Control sobre el `OpticalRiemannLensFibrator`
en el topos $\mathcal{T}_{\mathrm{MIC}}$, operando en el límite WKB sobre el fibrado
de fases del espacio de Hilbert logístico:
$$ \psi(x) = A(x)\, e^{i\mathcal{S}(x)/\hbar} $$

**Axiomas de Ejecución (Evolución granular v2.0):**

§0. AXIOMA DE COMPATIBILIDAD DIMENSIONAL:
    Antes de cualquier cálculo, $\rho_{LLM} \in \mathbb{C}^{d\times d}$ debe satisfacer:
      • $\rho = \rho^\dagger$ (hermiticidad)
      • $\rho \succeq 0$ (positividad semidefinida, recortada espectralmente)
      • $\text{Tr}(\rho) = 1$ (normalización)
      • $\dim(\rho) = \dim(G)$ (acoplamiento métrico)

§1. DIAFRAGMA DE ESPECTRO DINÁMICO (Modulación Cuántica Auditada):
    $$ l_{cutoff} = \left\lfloor l_{max} \cdot \exp\left( - \frac{\kappa \cdot S_{MAC}}{\text{Tr}(\rho^2)} \right) \right\rfloor $$
    La pureza se calcula **post-recorte PSD** para garantizar $\rho \succeq 0$ exacto.

§2. ECUACIÓN EIKONAL (Certificación Espectral):
    $$ G^{\mu\nu} \partial_\mu \mathcal{S}\, \partial_\nu \mathcal{S} = n^2(\sigma^*) $$
    Se exige que $G^{-1}$ sea SPD con $\kappa_2(G^{-1}) < \kappa_{\max}$; en caso contrario
    el Hamiltoniano eikonal no es hiperbólico regular y se aborta con EikonalSingularityError.

§3. ACCIÓN DE FERMAT (Integración Simpson Compuesta):
    $$ \mathcal{A} = \int_{\gamma} n(\gamma(t)) \sqrt{G_{\mu\nu} \dot{\gamma}^\mu \dot{\gamma}^\nu}\, dt $$
    Se aproxima por regla de Simpson compuesta vectorizada (orden 4), no suma rectangular.

§4. RESIDUO GEODÉSICO (Auditoría Covariante):
    La desviación geodésica se mide como $\|a^{RK4} - a^{geo}\|_G$ donde:
      • $a^{geo}$ es la aceleración geodésica (Levi‑Civita)
      • $a^{RK4}$ es la aceleración efectiva del método
    Esto es el *test de Noether* para la simetría de reparametrización.

§5. CONTRATO CATEGORIAL (Topos $\mathcal{T}_{\mathrm{MIC}}$):
    `EikonalAgent` es un `Morphism` endo con adjunción vía `forward/backward`,
    consumiendo `EikonalControlInput` inmutable como objeto del topos.
================================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Protocol, runtime_checkable
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA
# ══════════════════════════════════════════════════════════════════════════════
from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
from app.core.immune_system.metric_tensors import G_PHYSICS
from app.core.immune_system.musical_isomorphism_engine import MetricSpectralPreconditioner
from app.omega.optical_riemann_lens import OpticalRiemannLensFibrator, RefractedState
from app.omega.levi_civita_agent import LeviCivitaConnectionAgent, TangentVector

logger = logging.getLogger("MIC.Omega.EikonalAgent")


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES TOPOLÓGICAS, ESPECTRALES Y CUÁNTICAS
# ══════════════════════════════════════════════════════════════════════════════
class QuantumPurityCollapseError(TopologicalInvariantError):
    r"""
    Detonada si $\text{Tr}(\rho^2) \to 0$, indicando un colapso entrópico masivo
    irrecuperable por el lente (estado maximalmente mezclado exacto).
    """
    pass


class EikonalSingularityError(TopologicalInvariantError):
    r"""
    Detonada cuando $G^{-1}$ no es SPD o el residuo Hamiltoniano diverge.
    Implica que el Hamiltoniano eikonal pierde su carácter hiperbólico.
    """
    pass


class FermatOpticalDeviationError(TopologicalInvariantError):
    r"""
    Detonada si la acción de Fermat diverge o el residuo geodésico viola tolerancia.
    Indica violación del Principio de Mínima Acción por parte del LLM.
    """
    pass


class DimensionalMismatchError(TopologicalInvariantError):
    r"""
    Detonada si las dimensiones de $\rho_{LLM}$, $G$, o los vectores no son compatibles.
    Falla del Axioma §0.
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS INMUTABLES DEL ESPACIO DE FASE (OBJETOS DEL TOPOS)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SpectralDensityAudit:
    r"""Certificado espectral de una matriz de densidad $\rho_{LLM}$ procesada."""
    purity: float
    von_neumann_entropy: float
    eigenvalues_psd: NDArray[np.float64]
    negative_eigvals_pruned: int
    trace_after_pruning: float
    is_physical: bool


@dataclass(frozen=True, slots=True)
class EikonalPhaseState:
    r"""Estado inmutable que encapsula la resolución completa de la ecuación eikonal."""
    phase_gradient_norm: float
    fermat_action_integral: float
    dynamic_l_cutoff: int
    refracted_state: RefractedState
    geodesic_deviation: float = 0.0
    spectral_certificate: Optional[SpectralDensityAudit] = None


@dataclass(frozen=True, slots=True)
class EikonalControlInput:
    r"""
    **Objeto del topos $\mathcal{T}_{\mathrm{MIC}}$** — entrada inmutable del morfismo
    `EikonalAgent.execute_optical_guidance`. Garantiza contrato categórico estable.
    """
    raw_llm_logits: NDArray[np.float64]
    rho_llm: NDArray[np.float64]
    s_mac_entropy: float
    logistic_stress_norm: float
    phase_gradient: NDArray[np.float64]
    path_velocities: NDArray[np.float64]
    use_geodesic_correction: bool = True
    cavity_tol: float = 1e-10


# ══════════════════════════════════════════════════════════════════════════════
# PROTOCOLOS DE COMPOSICIÓN (SIN HERENCIA RÍGIDA)
# ══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class ApertureModulatorPort(Protocol):
    """Contrato minimal para el modulador del diafragma (Fase 1)."""
    def compute_dynamic_cutoff(self, s_mac: float, rho_llm: NDArray[np.float64]) -> int: ...


@runtime_checkable
class EikonalResolverPort(Protocol):
    """Contrato minimal para el resolutor eikonal (Fase 2)."""
    def resolve_eikonal_equation(self, phase_gradient: NDArray[np.float64], n_refract: float) -> float: ...


@runtime_checkable
class FermatAuditorPort(Protocol):
    """Contrato minimal para el auditor de Fermat (Fase 3)."""
    def audit_fermat_action(self, path_velocities: NDArray[np.float64], n_refract: float, dt: float = 1e-3) -> float: ...
    def enforce_geodesic_path(self, initial_velocity: TangentVector, n_steps: int, dt: float = 1.0) -> Tuple[NDArray[np.float64], float]: ...


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 1 · DIAFRAGMA DINÁMICO CUÁNTICO (Modulación del Espectrómetro)    │
# │  Evalúa la termodinámica del LLM y de la MAC para cerrar el diafragma   │
# │  óptico, aniquilando armónicos de alta frecuencia (palabrería).         │
# └─────────────────────────────────────────────────────────────────────────┘
class Phase1_DynamicApertureModulator:
    r"""
    **Fase 1** — Diafragma dinámico espectral.
    
    Resuelve:
    $$ l_{cutoff} = \left\lfloor l_{max} \cdot \exp\left( - \frac{\kappa \cdot S_{MAC}}{\text{Tr}(\rho^2)} \right) \right\rfloor $$
    
    Antes del cálculo de pureza, **proyecta $\rho$ sobre el cono PSD** recortando
    autovalores negativos (artefactos numéricos). Esto es la operación de
    *purificación cuántica parcial* del estado del LLM.
    """
    _EPS_NEG: float = 1e-12       # Umbral para considerar autovalor negativo
    _EPS_PURITY: float = 1e-12    # Pureza mínima no-colapsada

    def __init__(self, l_max_absolute: int = 50, kappa_coupling: float = 1.0) -> None:
        if l_max_absolute < 1:
            raise ValueError(f"l_max debe ser ≥ 1, recibido {l_max_absolute}.")
        if kappa_coupling <= 0:
            raise ValueError(f"kappa debe ser > 0, recibido {kappa_coupling}.")
        self._l_max = int(l_max_absolute)
        self._kappa = float(kappa_coupling)

    # ────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL — PUNTO DE ENTRADA DE FASE 1
    # ────────────────────────────────────────────────────────────────────
    def compute_dynamic_cutoff(
        self, s_mac: float, rho_llm: NDArray[np.float64]
    ) -> int:
        r"""
        Resuelve la ecuación de modulación cuántica.
        
        Parámetros
        ----------
        s_mac : float
            Entropía de Von Neumann del sustrato MAC (no negativa).
        rho_llm : NDArray (N, N)
            Matriz de densidad del LLM.
            
        Retorna
        -------
        int
            Nuevo valor de $l_{cutoff}$, al menos 1.
        """
        # ── AXIOMA §0: Validación de la matriz de densidad ──
        audit = self._audit_density_matrix(rho_llm)

        s_mac = float(s_mac)
        if s_mac < 0:
            raise ValueError(f"La entropía de la MAC debe ser ≥ 0, recibido {s_mac}.")

        purity = audit.purity
        logger.debug(
            "Pureza del estado LLM (post-recorte PSD): Tr(ρ²)=%.6f, S_vN=%.4f, "
            "autovalores podados=%d",
            purity, audit.von_neumann_entropy, audit.negative_eigvals_pruned,
        )

        if purity < self._EPS_PURITY:
            raise QuantumPurityCollapseError(
                f"Pureza nula tras recorte PSD (Tr(ρ²)={purity:.2e}). "
                "El estado generativo es caos térmico absoluto."
            )

        # ── Atenuación del diafragma ──
        attenuation_factor = math.exp(- (self._kappa * s_mac) / purity)
        l_cutoff = int(math.floor(self._l_max * attenuation_factor))
        l_cutoff = max(1, l_cutoff)
        logger.info(
            "Diafragma cuántico: l_max=%d → l_cutoff=%d (κ=%.3f, S=%.4f, pureza=%.4f)",
            self._l_max, l_cutoff, self._kappa, s_mac, purity,
        )
        return l_cutoff

    # ────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS — AUDITORÍA DE ρ
    # ────────────────────────────────────────────────────────────────────
    def _audit_density_matrix(self, rho: NDArray[np.float64]) -> SpectralDensityAudit:
        r"""
        **Axioma §0:** Audita y purifica $\rho_{LLM}$ garantizando:
          • Hermiticidad
          • PSD (recorte de autovalores negativos)
          • Traza unitaria
          • Dimensionalidad coherente
        
        Retorna un `SpectralDensityAudit` con los invariantes certificados.
        """
        rho = np.asarray(rho, dtype=np.complex128 if np.iscomplexobj(rho) else np.float64)

        # 1. Hermiticidad
        if not np.allclose(rho, rho.T.conj(), atol=1e-10):
            raise QuantumPurityCollapseError(
                f"ρ no es hermitiana: ‖ρ − ρ†‖ = {la.norm(rho - rho.T.conj()):.2e}."
            )

        # 2. Cuadrada
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise DimensionalMismatchError(
                f"ρ debe ser cuadrada, recibida forma {rho.shape}."
            )

        # 3. Autovalores (hermiticidad garantiza reales) + recorte PSD
        # Usamos eigvalsh para máxima estabilidad numérica
        eigvals = np.linalg.eigvalsh(rho)
        negative_mask = eigvals < -self._EPS_NEG
        n_negative = int(np.sum(negative_mask))

        if n_negative > 0:
            logger.warning(
                "ρ tenía %d autovalores negativos (mín=%.2e); se procede a recorte PSD.",
                n_negative, float(eigvals.min()),
            )
            eigvals = np.clip(eigvals, 0.0, None)

        # 4. Normalización de traza
        eigvals = eigvals / eigvals.sum()

        # 5. Invariantes certificados
        purity = float(np.sum(eigvals ** 2))
        # Entropía de Von Neumann: S = -Tr(ρ log ρ)
        eigvals_safe = np.clip(eigvals, 1e-15, None)
        s_von_neumann = float(-np.sum(eigvals * np.log(eigvals_safe)))

        return SpectralDensityAudit(
            purity=purity,
            von_neumann_entropy=s_von_neumann,
            eigenvalues_psd=eigvals,
            negative_eigvals_pruned=n_negative,
            trace_after_pruning=float(eigvals.sum()),
            is_physical=(purity >= self._EPS_PURITY),
        )


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 2 · RESOLUTOR DE LA ECUACIÓN EIKONAL NO LINEAL                    │
# │  Calcula la norma del gradiente de fase sobre el tensor métrico inverso │
# │  para garantizar que el frente de onda respete la curvatura G_{μν}.     │
# └─────────────────────────────────────────────────────────────────────────┘
class Phase2_EikonalSurfaceResolver:
    r"""
    **Fase 2** — Resuelve el campo escalar de fase:
    $$ G^{\mu\nu} \partial_\mu \mathcal{S}\, \partial_\nu \mathcal{S} = n^2(\sigma^*) $$
    
    **Acoplamiento hacia Fase 3:**
    Certifica la métrica inversa $G^{-1}$ y expone `n_refract` como
    *artefacto algebraico* que viajará a la integración de Fermat.
    """
    _KAPPA_MAX: float = 1e12     # κ₂(G⁻¹) máximo permitido

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        kappa_max: float = _KAPPA_MAX,
    ) -> None:
        self._kappa_max = float(kappa_max)
        self._G = self._validate_and_prepare_metric(metric_tensor)
        # Inversión estabilizada mediante el preacondicionador espectral
        precond = MetricSpectralPreconditioner()
        pm = precond.precondition(self._G)
        self._G_inv = pm.G_inv
        self._condition_number = pm.condition_number
        self._certify_G_inv()
        logger.debug(
            "Métrica invertida con κ₂(G)=%.2e, κ₂(G⁻¹)=%.2e.",
            self._condition_number, self._G_inv_condition_number(),
        )

    # ────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL — PUNTO DE ENTRADA DE FASE 2
    # ────────────────────────────────────────────────────────────────────
    def resolve_eikonal_equation(
        self, phase_gradient: NDArray[np.float64], n_refract: float
    ) -> float:
        r"""
        Verifica el cumplimiento de la ecuación eikonal y retorna $\|\nabla S\|_G^2$.
        
        Parámetros
        ----------
        phase_gradient : NDArray (d,)
            Gradiente espacial de la fase eikonal.
        n_refract : float
            Índice de refracción local.
            
        Retorna
        -------
        float
            Valor calculado de $\|\nabla S\|_G^2$.
            
        Lanza
        -----
        EikonalSingularityError si el Hamiltoniano eikonal colapsa.
        """
        grad = np.asarray(phase_gradient, dtype=np.float64)
        if grad.shape != (self._G.shape[0],):
            raise DimensionalMismatchError(
                f"∇S tiene dimensión {grad.shape}, esperada ({self._G.shape[0]},)."
            )

        # Contracción tensorial: ∂S^T G⁻¹ ∂S
        s_norm_sq = float(np.einsum('i,ij,j->', grad, self._G_inv, grad))

        if not np.isfinite(s_norm_sq) or s_norm_sq <= 0:
            raise EikonalSingularityError(
                f"‖∇S‖²_G = {s_norm_sq}. El frente de onda ha colapsado "
                "termodinámicamente (Hamiltoniano no hiperbólico)."
            )

        target_n_sq = float(n_refract) ** 2
        deviation = abs(s_norm_sq - target_n_sq)
        logger.debug(
            "Eikonal: ‖∇S‖²_G=%.6f, n²=%.6f, residuo=%.4e.",
            s_norm_sq, target_n_sq, deviation,
        )

        if deviation > 1e-4:
            logger.warning(
                "Desviación eikonal significativa: %.2e. "
                "Posible inconsistencia métrica o de refracción.",
                deviation,
            )
        return s_norm_sq

    # ────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS — CERTIFICACIÓN MÉTRICA
    # ────────────────────────────────────────────────────────────────────
    def _validate_and_prepare_metric(self, G: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Valida que $G$ sea SPD — el sustrato del Hamiltoniano eikonal."""
        G = np.asarray(G, dtype=np.float64)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise DimensionalMismatchError(f"G debe ser cuadrada, recibida {G.shape}.")
        if not np.allclose(G, G.T, atol=1e-12):
            raise MetricSignatureError("G debe ser simétrica.")
        try:
            _ = la.cholesky(G, lower=True, check_finite=False)
        except la.LinAlgError as exc:
            raise MetricSignatureError(f"G no es SPD: {exc}") from exc
        return G

    def _certify_G_inv(self) -> None:
        r"""
        **Axioma §2:** Verifica que $G^{-1}$ sea SPD con $\kappa_2(G^{-1}) < \kappa_{\max}$.
        En caso contrario, el Hamiltoniano eikonal no es hiperbólico regular.
        """
        try:
            eigvals_inv = np.linalg.eigvalsh(self._G_inv)
        except np.linalg.LinAlgError as exc:
            raise EikonalSingularityError(
                f"No se pudo calcular el espectro de G⁻¹: {exc}"
            ) from exc
        min_eig = float(eigvals_inv.min())
        if min_eig <= 1e-15:
            raise EikonalSingularityError(
                f"G⁻¹ tiene autovalor mínimo {min_eig:.2e} ≤ 0. "
                "Hamiltoniano eikonal degenerado."
            )
        cond_inv = float(eigvals_inv.max() / min_eig)
        if cond_inv > self._kappa_max:
            raise EikonalSingularityError(
                f"κ₂(G⁻¹) = {cond_inv:.2e} > κ_max = {self._kappa_max:.2e}. "
                "Métrica demasiado mal condicionada para integración numérica estable."
            )

    def _G_inv_condition_number(self) -> float:
        eigvals_inv = np.linalg.eigvalsh(self._G_inv)
        return float(eigvals_inv.max() / eigvals_inv.min())


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 3 · OPTIMIZADOR DE LA ACCIÓN DE FERMAT Y ORQUESTADOR SUPREMO      │
# │  Integra la trayectoria óptica, fuerza geodésicas y comanda al Lente.   │
# └─────────────────────────────────────────────────────────────────────────┘
class Phase3_FermatActionAuditor:
    r"""
    **Fase 3** — Optimizador de la acción de Fermat y transporte geodésico.
    
    **Integración Simpson compuesta (orden 4):**
    $$ \mathcal{A} \approx \frac{\Delta t}{3} \sum_{k=0}^{T-1} w_k \cdot n \sqrt{G_{\mu\nu} \dot\gamma^\mu_k \dot\gamma^\nu_k} $$
    donde $w_k = \{1, 4, 2, 4, \ldots, 4, 1\}$ son los pesos de Simpson.
    
    **Auditoría geodésica covariante:**
    La desviación se mide como $\|a^{RK4} - a^{geo}\|_G$, donde:
      • $a^{geo}$ es la aceleración geodésica exacta (Levi‑Civita)
      • $a^{RK4}$ es la aceleración efectiva del método
    """
    def __init__(self, metric_tensor: NDArray[np.float64] = G_PHYSICS) -> None:
        self._G = np.asarray(metric_tensor, dtype=np.float64)
        self._levi_civita = LeviCivitaConnectionAgent(self._G)

    # ────────────────────────────────────────────────────────────────────
    # AUDITORÍA DE ACCIÓN DE FERMAT (Simpson compuesta)
    # ────────────────────────────────────────────────────────────────────
    def audit_fermat_action(
        self,
        path_velocities: NDArray[np.float64],
        n_refract: float,
        dt: float = 1e-3,
    ) -> float:
        r"""
        Computa la integral de Acción de Fermat con regla de Simpson compuesta.
        """
        V = np.asarray(path_velocities, dtype=np.float64)
        if V.ndim != 2:
            raise ValueError(f"path_velocities debe ser (T,d), recibido {V.shape}.")

        # Norma G de cada vector velocidad
        # norms[t] = sqrt(v^T G v)
        norms = np.sqrt(np.einsum('ti,ij,tj->t', V, self._G, V))

        # Pesos de Simpson: requiere número impar de puntos
        n_pts = norms.shape[0]
        if n_pts < 3 or n_pts % 2 == 0:
            # Fallback a regla del trapecio para casos degenerados
            logger.debug(
                "Simpson requiere nº impar de puntos ≥ 3; usando trapecio (n=%d).",
                n_pts,
            )
            weights = np.ones(n_pts)
            weights[0] = weights[-1] = 0.5
            integral = np.sum(weights * norms) * dt
        else:
            weights = np.ones(n_pts)
            weights[1:-1:2] = 4.0
            weights[2:-1:2] = 2.0
            integral = (dt / 3.0) * np.sum(weights * norms)

        action = float(n_refract) * integral

        if not np.isfinite(action) or action > 1e10:
            raise FermatOpticalDeviationError(
                f"Divergencia en la Acción de Fermat: A={action}. "
                "La trayectoria del LLM viola el Principio de Mínima Acción."
            )
        return float(action)

    # ────────────────────────────────────────────────────────────────────
    # TRANSPORTE GEODÉSICO CON AUDITORÍA COVARIANTE
    # ────────────────────────────────────────────────────────────────────
    def enforce_geodesic_path(
        self,
        initial_velocity: TangentVector,
        n_steps: int,
        dt: float = 1.0,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Genera trayectoria geodésica (RK4) y mide el **residuo geodésico covariante**:
        $$ \mathcal{R} = \frac{1}{N} \sum_{k=1}^{N} \left\| \frac{v_{k+1} - v_k}{\Delta t} - a^{geo}_k \right\|_G $$
        donde $a^{geo}_k = -\Gamma^\mu_{\alpha\beta} v^\alpha v^\beta e_\mu$ es la aceleración
        geodésica exacta.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps debe ser ≥ 1, recibido {n_steps}.")

        velocities: List[NDArray[np.float64]] = []
        total_deviation = 0.0
        v_current = initial_velocity
        velocities.append(v_current.coordinates.copy())

        for k in range(n_steps):
            # Paso RK4 del agente Levi‑Civita
            v_next = self._levi_civita.enforce_geodesic_flow(v_current, dt)
            v_next_coords = v_next.coordinates

            # Aceleración efectiva: (v_next − v_current) / dt
            a_eff = (v_next_coords - v_current.coordinates) / dt

            # Aceleración geodésica exacta
            a_geo = self._levi_civita.geodesic_rhs(v_current.coordinates)

            # Residuo covariante (norma G)
            delta = a_eff - a_geo
            deviation_step = float(np.sqrt(delta @ (self._G @ delta)))
            total_deviation += deviation_step

            velocities.append(v_next_coords)
            v_current = v_next

        mean_deviation = total_deviation / n_steps
        return np.array(velocities), float(mean_deviation)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 4 · ORQUESTADOR SUPREMO (MORFISMO CATEGORIAL)                     │
# │  Integra las tres fases y comanda al Lente Óptico.                      │
# └─────────────────────────────────────────────────────────────────────────┘
class EikonalAgent(Morphism):
    r"""
    **Morfismo Eikonal Maestro** — opera en el topos $\mathcal{T}_{\mathrm{MIC}}$.
    
    Compone las tres fases mediante **agregación tipada** (Protocolos) en lugar de
    herencia rígida. Cada fase es intercambiable si cumple su contrato.
    
    **Contrato categórico:**
      • `forward(state)` : aplica modulación al estado categórico
      • `backward(state)`: adjunta (rollback) — reinyecta estado crudo
      • `execute_optical_guidance(control)` : método axiomático principal
    """
    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        l_max_absolute: int = 50,
        kappa_coupling: float = 1.0,
    ) -> None:
        # ── Validación de G como métrica ──
        G = np.asarray(metric_tensor, dtype=np.float64)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise DimensionalMismatchError(f"G debe ser cuadrada, recibida {G.shape}.")
        if not np.allclose(G, G.T, atol=1e-12):
            raise MetricSignatureError("G debe ser simétrica.")
        try:
            _ = la.cholesky(G, lower=True, check_finite=False)
        except la.LinAlgError as exc:
            raise MetricSignatureError(f"G no es SPD: {exc}") from exc

        # ── Composición por agregación (sin herencia rígida) ──
        self._G = G
        self._phase1: ApertureModulatorPort = Phase1_DynamicApertureModulator(
            l_max_absolute=l_max_absolute, kappa_coupling=kappa_coupling,
        )
        self._phase2: EikonalResolverPort = Phase2_EikonalSurfaceResolver(metric_tensor=G)
        self._phase3: FermatAuditorPort = Phase3_FermatActionAuditor(metric_tensor=G)
        # El lente óptico — el objeto final que el morfismo comanda
        self._lens_fibrator = OpticalRiemannLensFibrator(G)
        super().__init__(name="EikonalAgent")

    # ────────────────────────────────────────────────────────────────────
    # CONTRATO CATEGORIAL (Topos $\mathcal{T}_{\mathrm{MIC}}$)
    # ────────────────────────────────────────────────────────────────────
    def forward(self, state: CategoricalState) -> CategoricalState:
        r"""
        Aplica la reflexión eikonal sobre un `CategoricalState`.
        Implementa la unidad $\eta$ del adjuntor.
        """
        psi = np.asarray(state.payload, dtype=np.float64)
        if psi.shape != (self._G.shape[0],):
            raise DimensionalMismatchError(
                f"state.payload tiene dimensión {psi.shape}, esperada ({self._G.shape[0]},)."
            )
        # En el morfismo directo, simplemente devolvemos el mismo payload con etiqueta actualizada
        # (la reflexión eikonal completa se realiza vía execute_optical_guidance)
        return CategoricalState(payload=psi, label=f"{state.label}::eikonal_forward")

    def backward(self, state: CategoricalState) -> CategoricalState:
        r"""
        Adjunta: rollback. En eikonal, backward = forward (involutividad aproximada).
        """
        return self.forward(state)

    # ────────────────────────────────────────────────────────────────────
    # MÉTODO AXIOMÁTICO PRINCIPAL — PUNTO DE ENTRADA ÚNICO
    # ────────────────────────────────────────────────────────────────────
    def execute_optical_guidance(self, control: EikonalControlInput) -> EikonalPhaseState:
        r"""
        Método axiomático que consolida las tres fases geométricas.
        
        Pipeline categórico:
            control ──▶ Fase 1 (Diafragma) ──▶ l_cutoff
                   ──▶ Fase 2 (Eikonal)   ──▶ ‖∇S‖²_G
                   ──▶ Fase 3 (Fermat)    ──▶ Acción, Geodésica
                   ──▶ Lente Óptico      ──▶ RefractedState
        """
        logger.info("Iniciando resolución del Ansatz WKB y modulación de Lente de Riemann.")

        # ── 0. Validación de coherencia dimensional ──
        self._validate_control_input(control)

        # ── 1. Fase 1 — Diafragma dinámico ──
        l_cutoff_dynamic = self._phase1.compute_dynamic_cutoff(
            control.s_mac_entropy, control.rho_llm
        )

        # ── 2. Índice de refracción del lente ──
        n_refract = self._lens_fibrator._compute_fermat_refractive_index(
            control.logistic_stress_norm
        )

        # ── 3. Fase 2 — Resolución eikonal ──
        phase_norm_sq = self._phase2.resolve_eikonal_equation(
            control.phase_gradient, n_refract
        )

        # ── 4. Fase 3 — Corrección geodésica (si solicitada) ──
        path_velocities = control.path_velocities
        geodesic_deviation = 0.0
        if control.use_geodesic_correction and path_velocities.shape[0] > 0:
            initial_v = TangentVector(coordinates=path_velocities[0].copy())
            n_steps = path_velocities.shape[0]
            path_velocities, geodesic_deviation = self._phase3.enforce_geodesic_path(
                initial_v, n_steps, dt=1.0
            )
            logger.debug("Desviación geodésica covariante media: %.2e", geodesic_deviation)
            if geodesic_deviation > control.cavity_tol:
                raise FermatOpticalDeviationError(
                    f"Residuo geodésico {geodesic_deviation:.2e} > tol={control.cavity_tol:.2e}."
                )

        # ── 5. Acción de Fermat sobre la trayectoria corregida ──
        action_integral = self._phase3.audit_fermat_action(path_velocities, n_refract)

        # ── 6. Inyección en el Lente Óptico ──
        self._lens_fibrator._l_cutoff = l_cutoff_dynamic
        refracted_state = self._lens_fibrator.refract_attention_logits(
            control.raw_llm_logits, control.logistic_stress_norm
        )

        # ── 7. Auditoría de ρ para el certificado de salida ──
        audit = self._phase1._audit_density_matrix(control.rho_llm)

        logger.info(
            "Geodésica Eikonal consolidada: A=%.4f, l_cutoff=%d, "
            "KV=%.2f%%, ‖∇S‖²_G=%.4f, R_geo=%.2e",
            action_integral, l_cutoff_dynamic,
            refracted_state.kv_compression_ratio * 100,
            phase_norm_sq, geodesic_deviation,
        )

        return EikonalPhaseState(
            phase_gradient_norm=phase_norm_sq,
            fermat_action_integral=action_integral,
            dynamic_l_cutoff=l_cutoff_dynamic,
            refracted_state=refracted_state,
            geodesic_deviation=geodesic_deviation,
            spectral_certificate=audit,
        )

    # ────────────────────────────────────────────────────────────────────
    # VALIDACIÓN DE ENTRADA (AXIOMA §0)
    # ────────────────────────────────────────────────────────────────────
    def _validate_control_input(self, control: EikonalControlInput) -> None:
        r"""Axioma §0: coherencia dimensional de todos los tensores."""
        d = self._G.shape[0]
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
        if control.path_velocities.ndim != 2 or control.path_velocities.shape[1] != d:
            raise DimensionalMismatchError(
                f"path_velocities: {control.path_velocities.shape} incompatible con d={d}."
            )
        if control.s_mac_entropy < 0:
            raise ValueError(
                f"s_mac_entropy debe ser ≥ 0, recibido {control.s_mac_entropy}."
            )

    # ────────────────────────────────────────────────────────────────────
    # PROPIEDADES DE AUDITORÍA
    # ────────────────────────────────────────────────────────────────────
    @property
    def metric_tensor(self) -> NDArray[np.float64]:
        r"""Tensor métrico $G$ (copia defensiva)."""
        return self._G.copy()

    @property
    def metric_inverse(self) -> NDArray[np.float64]:
        r"""Tensor métrico inverso $G^{-1}$ (copia defensiva)."""
        return self._phase2._G_inv.copy()  # type: ignore[attr-defined]


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "QuantumPurityCollapseError",
    "EikonalSingularityError",
    "FermatOpticalDeviationError",
    "DimensionalMismatchError",
    "MetricSignatureError",
    "SpectralDensityAudit",
    "EikonalPhaseState",
    "EikonalControlInput",
    "Phase1_DynamicApertureModulator",
    "Phase2_EikonalSurfaceResolver",
    "Phase3_FermatActionAuditor",
    "EikonalAgent",
]
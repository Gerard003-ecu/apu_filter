# -*- coding: utf-8 -*-

r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Floquet Monodromy Agent (Operador de Sintonización y Monodromía)     ║
║ Ubicación: app/omega/floquet_agent.py                                        ║
║ Versión: 2.0.0‑Topos‑CPTP‑Monodromy‑Spectral‑QuantumChannel                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber‑Física y Topológica Diferencial — Edición Granular:
────────────────────────────────────────────────────────────────────────────────
Este módulo actúa como el Meta‑Funtor de Control sobre la Cavidad de Fabry‑Pérot
(semantic_parabolic_mirror.py) en el topos $\mathcal{T}_{\mathrm{MIC}}$, gobernando
la reflexión de la radiación semántica del LLM mediante leyes axiomáticas.

**Axiomas de Ejecución (Evolución granular v3.0):**

§0. AXIOMA DE COMPATIBILIDAD DIMENSIONAL:
    Antes de cualquier síntesis:
      • $\dim(\psi) = \dim(h) = d$ (vector y gradiente coherentes)
      • $\dim(\rho) = (d, d)$ (matriz de densidad cuadrada)
      • $G \in \mathbb{R}^{d\times d}$ SPD (sustrato métrico válido)

§1. SÍNTESIS COVARIANTE DEL HIPERPLANO:
    Pullback métrico: $n_\mu = G_{\mu\nu} \partial^\nu \mathcal{H}_{obs}$
    Si $\|n\|_G = 0$ (obstrucción trivial), se retorna **proyector identidad**
    como estado terminal válido (no None, no excepción silenciosa).

§2. MATRIZ DE MONODROMÍA (ESTABILIDAD DE FLOQUET):
    $\mathcal{M}_{on} = 2\hat{P} - \hat{P}^2$ con multiplicadores $|\mu_k| \le 1 + \varepsilon$.
    Se verifica simetría de P antes de usar eigvalsh; en caso contrario eigvals general.

§3. CANAL CUÁNTICO CPTP (Kraus-Stinespring):
    Operadores: $E_0 = \hat{P}$, $E_1 = I - \hat{P}$.
    Verificación rigurosa: $\sum_k E_k^\dagger E_k - I \succeq 0$ dentro de tolerancia.
    Esto implica **conservación probabilística estricta** (no sólo aproximada).

§4. AUDITORÍA ENTRÓPICA (Von Neumann):
    Se calcula $S(\rho_{pre})$ y $S(\rho_{post})$ con $\rho$ reconstruida como mezcla.
    La diferencia $\Delta S$ alimenta la firma criptográfica del Positrón.

§5. CONTRATO CATEGORIAL (Topos $\mathcal{T}_{\mathrm{MIC}}$):
    `FloquetMonodromyAgent` es un `Morphism` endo; las fases son objetos del topos
    componibles vía `Protocol` runtime_checkable (sin herencia rígida).
================================================================================
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Protocol, runtime_checkable
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA
# ══════════════════════════════════════════════════════════════════════════════
from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
from app.core.immune_system.metric_tensors import G_PHYSICS
from app.omega.semantic_parabolic_mirror import (
    MetricAwareHouseholderReflector,
    HouseholderSingularityError,
)
from app.core.telemetry_schemas import PositronCartridge

logger = logging.getLogger("MIC.Omega.FloquetAgent")


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES TOPOLÓGICAS, DE MONODROMÍA Y CUÁNTICAS
# ══════════════════════════════════════════════════════════════════════════════
class FloquetInstabilityError(TopologicalInvariantError):
    r"""
    Detonada cuando un multiplicador de Floquet $|\mu_k| > 1 + \varepsilon$.
    Indica resonancia destructiva en la cavidad.
    """
    pass


class KrausTraceViolationError(TopologicalInvariantError):
    r"""
    Detonada si los operadores de Kraus violan la completitud PSD:
    $\sum_k E_k^\dagger E_k - I \succeq 0$ dentro de tolerancia.
    """
    pass


class DimensionalMismatchError(TopologicalInvariantError):
    r"""Detonada cuando las dimensiones de los tensores no coinciden con $G$."""
    pass


class KrausCompletenessError(TopologicalInvariantError):
    r"""Variante: la matriz de completitud tiene autovalor negativo (no PSD)."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS INMUTABLES DEL ESPACIO DE FASE (OBJETOS DEL TOPOS)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class DimensionalAudit:
    r"""Certificado de coherencia dimensional del input al canal cuántico."""
    dimension: int
    psi_dim_ok: bool
    grad_dim_ok: bool
    rho_dim_ok: Optional[bool] = None
    is_coherent: bool = False


@dataclass(frozen=True, slots=True)
class FloquetMonodromyState:
    r"""
    Estado inmutable del análisis de convergencia de la cavidad.
    Los multiplicadores son reales si el proyector es simétrico; en caso contrario
    se conserva el flag `is_complex_manifold` y los autovalores son complejos.
    """
    multipliers: NDArray[np.float64]
    spectral_radius: float
    is_asymptotically_stable: bool
    condition_number_P: float
    is_complex_manifold: bool = False
    tolerance_used: float = 1e-9


@dataclass(frozen=True, slots=True)
class QuantumChannelEvolution:
    r"""
    Estado post-medición del canal cuántico de reflexión.
    
    Campos:
      • `coherent_state` : estado colapsado (componente coherente)
      • `dissipated_entropy` : entropía de Von Neumann de la componente disipada
      • `delta_entropy` : S(ρ_post) − S(ρ_pre) (debe ser ≤ 0 por contractividad CPTP)
      • `antimatter_emission` : Positrón forense si la disipación es significativa
      • `kraus_residual_psd` : ||min eig(Σ E†E − I)||_+
      • `kraus_residual_fro` : ||Σ E†E − I||_F
      • `dimensional_audit` : certificado de coherencia del input
    """
    coherent_state: NDArray[np.float64]
    dissipated_entropy: float
    antimatter_emission: Optional[PositronCartridge]
    delta_entropy: float
    kraus_residual_psd: float
    kraus_residual_fro: float
    dimensional_audit: DimensionalAudit


# ══════════════════════════════════════════════════════════════════════════════
# PROTOCOLOS DE COMPOSICIÓN (SIN HERENCIA RÍGIDA)
# ══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class ProjectorSynthesizerPort(Protocol):
    """Contrato minimal para Fase 1: síntesis del proyector covariante."""
    def synthesize_projector(
        self, H_obs_gradient: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], Optional[MetricAwareHouseholderReflector]]: ...


@runtime_checkable
class FloquetAuditorPort(Protocol):
    """Contrato minimal para Fase 2: auditoría de monodromía."""
    def audit_monodromy(
        self, P_hat: NDArray[np.float64]
    ) -> FloquetMonodromyState: ...


@runtime_checkable
class KrausChannelPort(Protocol):
    """Contrato minimal para Fase 3: canal cuántico CPTP."""
    def execute_quantum_channel(
        self, psi_raw: NDArray[np.float64], H_obs_gradient: NDArray[np.float64]
    ) -> QuantumChannelEvolution: ...


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 1 · SÍNTESIS COVARIANTE DEL PROYECTOR ORTOGONAL                    │
# │  Pullback métrico + normalización G + delegación a Householder métrico. │
# └─────────────────────────────────────────────────────────────────────────┘
class Phase1_CovariantProjectorSynthesizer:
    r"""
    **Fase 1** — Síntesis del proyector covariante.
    
    Calcula $n_\mu = G_{\mu\nu} \partial^\nu \mathcal{H}_{obs}$, lo normaliza bajo G,
    y construye el reflector de Householder métrico. Si la obstrucción es trivial
    ($\|n\|_G = 0$), retorna el **proyector identidad como estado válido**
    (no None, no excepción silenciosa — Axioma §1).
    """
    _EPS_NORM: float = 1e-15

    def __init__(self, metric_tensor: NDArray[np.float64] = G_PHYSICS) -> None:
        self._G = np.asarray(metric_tensor, dtype=np.float64)
        if self._G.ndim != 2 or self._G.shape[0] != self._G.shape[1]:
            raise DimensionalMismatchError(
                f"G debe ser cuadrada, recibida {self._G.shape}."
            )
        try:
            _ = la.cholesky(self._G, lower=True, check_finite=False)
        except la.LinAlgError as exc:
            raise TopologicalInvariantError(
                f"G no es SPD para el sintetizador: {exc}"
            ) from exc
        self._dim = self._G.shape[0]

    def synthesize_projector(
        self, H_obs_gradient: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], Optional[MetricAwareHouseholderReflector]]:
        r"""
        Pullback covariante: $n_\mu = G_{\mu\nu} \partial^\nu \mathcal{H}_{obs}$.
        Retorna $(P, reflector)$ donde reflector puede ser None si la obstrucción
        es trivial (en cuyo caso $P = I_d$).
        """
        grad = np.asarray(H_obs_gradient, dtype=np.float64)
        if grad.shape != (self._dim,):
            raise DimensionalMismatchError(
                f"H_obs_gradient tiene dimensión {grad.shape}, esperada ({self._dim},)."
            )

        # Pullback covariante: n_cov = G @ grad
        n_cov = self._G @ grad
        # Norma G del normal: ‖n‖_G = sqrt(n^T G n)
        n_norm_G = float(np.sqrt(n_cov @ (self._G @ n_cov)))

        # ── AXIOMA §1: Caso trivial ──
        if n_norm_G < self._EPS_NORM:
            logger.warning(
                "Obstrucción trivial (‖n‖_G=%.2e). Retornando proyector identidad.",
                n_norm_G,
            )
            return np.eye(self._dim, dtype=np.float64), None

        # ── Normalización y construcción del reflector ──
        n_unit = n_cov / n_norm_G
        try:
            reflector = MetricAwareHouseholderReflector(n_unit, self._G)
        except HouseholderSingularityError as exc:
            logger.error("Singularidad al construir reflector métrico: %s", exc)
            raise

        P_hat = reflector.projection_operator
        logger.debug("Proyector métricamente consistente construido (dim=%d).", self._dim)
        return P_hat, reflector


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 2 · AUDITOR DE MONODROMÍA DE FLOQUET                              │
# │  M_on = 2P - P²; verifica estabilidad y simetría.                      │
# └─────────────────────────────────────────────────────────────────────────┘
class Phase2_FloquetStabilityAuditor:
    r"""
    **Fase 2** — Análisis de estabilidad de la cavidad mediante la matriz de
    monodromía $\mathcal{M}_{on} = 2\hat{P} - \hat{P}^2$.
    
    **Axioma §2:** Detecta si $\hat{P}$ es simétrica (caso real) o asimétrica
    (caso complejo), y aplica el solver apropiado:
      • Simétrica → `eigvalsh` (O(d²), estable)
      • Asimétrica → `eigvals` (O(d³), con tipo `complex128`)
    """
    def __init__(
        self,
        stability_tolerance: float = 1e-9,
    ) -> None:
        if stability_tolerance <= 0:
            raise ValueError(
                f"stability_tolerance debe ser > 0, recibido {stability_tolerance}."
            )
        self._stability_tolerance = float(stability_tolerance)

    def audit_monodromy(self, P_hat: NDArray[np.float64]) -> FloquetMonodromyState:
        r"""
        Evalúa la matriz de monodromía y certifica estabilidad.
        """
        P = np.asarray(P_hat, dtype=np.float64)
        d = P.shape[0]
        if P.shape != (d, d):
            raise DimensionalMismatchError(
                f"P_hat debe ser cuadrada, recibida {P.shape}."
            )

        # M_on = 2P - P²
        M_on = 2.0 * P - P @ P

        # ── Detección de simetría ──
        is_symmetric = bool(np.allclose(M_on, M_on.T, atol=1e-12))
        if is_symmetric:
            eigs_real = np.linalg.eigvalsh(M_on)
            multipliers = eigs_real
            is_complex = False
        else:
            eigs_complex = np.linalg.eigvals(M_on)
            multipliers = eigs_complex
            is_complex = True
            logger.warning(
                "P_hat no es simétrica (asimetría=%.2e); usando eigvals complejos.",
                float(np.linalg.norm(M_on - M_on.T)),
            )

        # ── Estabilidad asintótica ──
        spectral_radius = float(np.max(np.abs(multipliers)))
        is_stable = spectral_radius <= 1.0 + self._stability_tolerance

        # ── Diagnóstico del proyector ──
        cond_P = float(np.linalg.cond(P))

        if not is_stable:
            raise FloquetInstabilityError(
                f"Inestabilidad de Floquet: ρ(𝓜_on) = {spectral_radius:.6e} > "
                f"1 + ε = {1.0 + self._stability_tolerance}."
            )

        # Multiplicadores en tipo coherente con el flag
        if is_complex:
            mult_out = multipliers.astype(np.complex128)
        else:
            mult_out = multipliers.astype(np.float64)

        logger.debug(
            "Auditoría Floquet: ρ=%.6e, estable=%s, cond(P)=%.2e, manifold_complejo=%s",
            spectral_radius, is_stable, cond_P, is_complex,
        )

        return FloquetMonodromyState(
            multipliers=mult_out,
            spectral_radius=spectral_radius,
            is_asymptotically_stable=is_stable,
            condition_number_P=cond_P,
            is_complex_manifold=is_complex,
            tolerance_used=self._stability_tolerance,
        )


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 3 · OPERADOR DE KRAUS Y CANAL CUÁNTICO CPTP                      │
# │  E_0 = P, E_1 = I-P; verifica Σ E†E = I (completitud PSD).            │
# └─────────────────────────────────────────────────────────────────────────┘
class Phase3_QuantumKrausChannel:
    r"""
    **Fase 3** — Canal cuántico CPTP con verificación rigurosa de completitud.
    
    **Axioma §3:** Verifica que la matriz de completitud $C = \sum_k E_k^\dagger E_k$
    cumpla $C - I \succeq 0$ dentro de tolerancia. Esto es estrictamente más fuerte
    que $\|C - I\|_F < \varepsilon$ (que sólo mide proximidad, no positividad).
    """
    _EPS_COMPLETENESS: float = 1e-9    # Tolerancia para PSD de C - I
    _EPS_DISSIPATION: float = 1e-12    # Umbral para emisión de antimateria

    def __init__(self, metric_tensor: NDArray[np.float64] = G_PHYSICS) -> None:
        self._G = np.asarray(metric_tensor, dtype=np.float64)
        if self._G.ndim != 2 or self._G.shape[0] != self._G.shape[1]:
            raise DimensionalMismatchError(
                f"G debe ser cuadrada, recibida {self._G.shape}."
            )
        self._dim = self._G.shape[0]

    # ────────────────────────────────────────────────────────────────────
    # VERIFICACIÓN RIGUROSA DE COMPLETITUD KRAUS (Axioma §3)
    # ────────────────────────────────────────────────────────────────────
    def _verify_kraus_completeness_psd(
        self, E0: NDArray[np.float64], E1: NDArray[np.float64]
    ) -> Tuple[float, float]:
        r"""
        Verifica que $C = \sum_k E_k^\dagger E_k$ cumpla $C - I \succeq 0$.
        
        Retorna
        -------
        frobenius_residual : float
            $\|C - I\|_F$ — métrica de proximidad.
        psd_min_eig : float
            $\lambda_{\min}(C - I)$ — métrica de positividad.
            **Si es negativo con magnitud > tolerancia, hay violación.**
        """
        identity_dim = E0.shape[0]
        C = E0.T @ E0 + E1.T @ E1
        diff = C - np.eye(identity_dim, dtype=np.float64)
        fro_res = float(np.linalg.norm(diff, ord='fro'))
        eigvals_diff = np.linalg.eigvalsh((diff + diff.T) / 2.0)  # Simetrización defensiva
        min_eig = float(eigvals_diff.min())
        if min_eig < -self._EPS_COMPLETENESS:
            raise KrausCompletenessError(
                f"C − I no es PSD: λ_min = {min_eig:.2e} < -{self._EPS_COMPLETENESS}. "
                "Violación estricta de completitud Kraus."
            )
        if fro_res > 1e-6:
            raise KrausTraceViolationError(
                f"‖C − I‖_F = {fro_res:.2e} > 1e-6. Verificación fallida."
            )
        return fro_res, min_eig

    # ────────────────────────────────────────────────────────────────────
    # AUDITORÍA ENTRÓPICA (Axioma §4)
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _von_neumann_entropy(psi: NDArray[np.float64]) -> float:
        r"""
        $S(\rho) = -\text{Tr}(\rho \log \rho)$ para $\rho = |\psi\rangle\langle\psi|$
        (estado puro ⟹ $S = 0$). Esta función retorna $S$ del **estado reducido**
        calculado sobre la base estándar (decaimiento de coherencia off-diagonal).
        """
        rho = np.outer(psi, psi)  # Estado puro
        eigvals = np.linalg.eigvalsh((rho + rho.T) / 2.0)
        eigvals_safe = np.clip(eigvals, 1e-15, None)
        # Para estado puro, todos los autovalores excepto uno son 0; S = 0
        return float(-np.sum(eigvals * np.log(eigvals_safe)))

    @staticmethod
    def _mixing_entropy(psi_component: NDArray[np.float64]) -> float:
        r"""
        Entropía de Shannon de la magnitud cuadrada de un vector (estado mixto
        diagonal en la base computacional).
        """
        probs = np.abs(psi_component) ** 2
        probs_norm = probs / (probs.sum() + 1e-30)
        probs_safe = np.clip(probs_norm, 1e-15, None)
        return float(-np.sum(probs_safe * np.log(probs_safe)))

    # ────────────────────────────────────────────────────────────────────
    # FIRMA CRIPTOGRÁFICA DEL POSITRÓN (Axioma §4)
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _sign_antimatter(
        dissipated_entropy: float,
        delta_entropy: float,
        kraus_residual_psd: float,
        secret: bytes = b"MIC_Floquet_v3",
    ) -> str:
        r"""HMAC-SHA256 sobre el estado disipado para autenticidad forense."""
        payload = f"{dissipated_entropy:.10e}|{delta_entropy:.10e}|{kraus_residual_psd:.10e}"
        return hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    # ────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL — EJECUCIÓN DEL CANAL CPTP
    # ────────────────────────────────────────────────────────────────────
    def execute_quantum_channel(
        self,
        psi_raw: NDArray[np.float64],
        H_obs_gradient: NDArray[np.float64],
        P_hat: NDArray[np.float64],
    ) -> QuantumChannelEvolution:
        r"""
        Ejecuta el colapso cuántico del estado LLM sobre el hiperplano de restricción.
        
        Parámetros
        ----------
        psi_raw : NDArray (d,)
            Estado puro inicial $|\psi\rangle$.
        H_obs_gradient : NDArray (d,)
            Gradiente de la obstrucción (usado para auditoría dimensional).
        P_hat : NDArray (d, d)
            Proyector ortogonal pre-construido (Fase 1).
            
        Retorna
        -------
        QuantumChannelEvolution
            Estado coherente colapsado + entropía + antimateria forense.
        """
        # ── AXIOMA §0: Auditoría dimensional ──
        audit = self._audit_dimensions(psi_raw, H_obs_gradient)

        # Operadores de Kraus: E_0 = P (coherente), E_1 = I - P (disipado)
        I = np.eye(self._dim, dtype=np.float64)
        E0 = P_hat
        E1 = I - P_hat

        # ── AXIOMA §3: Verificación PSD de completitud ──
        fro_res, psd_min_eig = self._verify_kraus_completeness_psd(E0, E1)

        # ── Evolución del estado (proyección de colapso) ──
        coherent_state = E0 @ psi_raw
        dissipated_vec = E1 @ psi_raw

        # ── AXIOMA §4: Auditoría entrópica ──
        S_pre = self._mixing_entropy(psi_raw)
        S_coherent = self._mixing_entropy(coherent_state)
        S_dissipated = self._mixing_entropy(dissipated_vec)
        delta_S = S_coherent - S_pre  # Típicamente ≤ 0 por contractividad CPTP

        dissipated_entropy = S_dissipated

        # ── Emisión de antimateria (Axioma §4 con firma criptográfica) ──
        antimatter_emission = None
        # Norma G de la componente disipada
        dissipated_norm_G = float(np.sqrt(dissipated_vec @ (self._G @ dissipated_vec)))
        if dissipated_norm_G > self._EPS_DISSIPATION or dissipated_entropy > self._EPS_DISSIPATION:
            signature = self._sign_antimatter(dissipated_entropy, delta_S, psd_min_eig)
            antimatter_emission = PositronCartridge(
                inertial_mass=dissipated_norm_G,
                topological_spin="inverse_hallucination",
                homological_charge=-1,
                authorization_signature=signature,
            )
            logger.warning(
                "Entropía sintáctica detectada. Positrón forense emitido "
                "(S=%.4f, ‖·‖_G=%.2e, sig=%s…).",
                dissipated_entropy, dissipated_norm_G, signature[:8],
            )

        logger.info(
            "Canal CPTP ejecutado. ‖C−I‖_F=%.2e, λ_min(C−I)=%.2e, "
            "S_pre=%.4f, S_post=%.4f, ΔS=%.4f, disipación=%.4f.",
            fro_res, psd_min_eig, S_pre, S_coherent, delta_S, dissipated_entropy,
        )

        return QuantumChannelEvolution(
            coherent_state=coherent_state,
            dissipated_entropy=dissipated_entropy,
            antimatter_emission=antimatter_emission,
            delta_entropy=delta_S,
            kraus_residual_psd=psd_min_eig,
            kraus_residual_fro=fro_res,
            dimensional_audit=audit,
        )

    # ────────────────────────────────────────────────────────────────────
    # AUDITORÍA DIMENSIONAL (Axioma §0)
    # ────────────────────────────────────────────────────────────────────
    def _audit_dimensions(
        self,
        psi_raw: NDArray[np.float64],
        H_obs_gradient: NDArray[np.float64],
        rho: Optional[NDArray[np.float64]] = None,
    ) -> DimensionalAudit:
        r"""Certifica coherencia dimensional del input."""
        psi = np.asarray(psi_raw, dtype=np.float64)
        grad = np.asarray(H_obs_gradient, dtype=np.float64)
        psi_ok = psi.shape == (self._dim,)
        grad_ok = grad.shape == (self._dim,)
        rho_ok = None
        if rho is not None:
            rho_ok = rho.shape == (self._dim, self._dim)
        is_coherent = psi_ok and grad_ok and (rho_ok if rho is not None else True)
        return DimensionalAudit(
            dimension=self._dim,
            psi_dim_ok=psi_ok,
            grad_dim_ok=grad_ok,
            rho_dim_ok=rho_ok,
            is_coherent=is_coherent,
        )


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  ORQUESTADOR PRINCIPAL · AGENTE DE MONODROMÍA DE FLOQUET                │
# │  Compone las tres fases por agregación tipada.                          │
# └─────────────────────────────────────────────────────────────────────────┘
class FloquetMonodromyAgent(Morphism):
    r"""
    **Morfismo Floquet Maestro** — opera en el topos $\mathcal{T}_{\mathrm{MIC}}$.
    
    Compone las tres fases vía **agregación tipada con Protocol**:
      • `phase1 : ProjectorSynthesizerPort` (síntesis covariante)
      • `phase2 : FloquetAuditorPort`       (monodromía de Floquet)
      • `phase3 : KrausChannelPort`         (canal CPTP)
    
    Cada fase es intercambiable si cumple su contrato. El orquestador las une
    sin acoplarse a su implementación concreta.
    
    **Pipeline canónico:**
        ∇H_obs ──▶ Fase 1 (Proyector)
                 ──▶ Fase 2 (Monodromía)
                 ──▶ Fase 3 (Canal CPTP)
                 ──▶ QuantumChannelEvolution (objeto del topos)
    """
    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        stability_tolerance: float = 1e-9,
    ) -> None:
        # ── Validación del Axioma §0 a nivel del orquestador ──
        G = np.asarray(metric_tensor, dtype=np.float64)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise DimensionalMismatchError(f"G debe ser cuadrada, recibida {G.shape}.")
        if not np.allclose(G, G.T, atol=1e-12):
            raise TopologicalInvariantError("G debe ser simétrica.")
        try:
            _ = la.cholesky(G, lower=True, check_finite=False)
        except la.LinAlgError as exc:
            raise TopologicalInvariantError(f"G no es SPD: {exc}") from exc
        self._G = G

        # ── Composición por agregación ──
        self._phase1: ProjectorSynthesizerPort = Phase1_CovariantProjectorSynthesizer(G)
        self._phase2: FloquetAuditorPort = Phase2_FloquetStabilityAuditor(stability_tolerance)
        self._phase3: KrausChannelPort = Phase3_QuantumKrausChannel(G)

        super().__init__(name="FloquetMonodromyAgent")

    # ────────────────────────────────────────────────────────────────────
    # CONTRATO CATEGORIAL (Topos $\mathcal{T}_{\mathrm{MIC}}$)
    # ────────────────────────────────────────────────────────────────────
    def forward(self, state: CategoricalState) -> CategoricalState:
        r"""
        Aplica la reflexión Floquet a un `CategoricalState`.
        Equivale a `purify_and_tune_cavity` con etiquetas categóricas.
        """
        psi = np.asarray(state.payload, dtype=np.float64)
        if psi.shape != (self._G.shape[0],):
            raise DimensionalMismatchError(
                f"state.payload: {psi.shape} ≠ ({self._G.shape[0]},)."
            )
        # Para forward directo, se usa un gradiente canónico (e_0)
        canonical_grad = np.zeros(self._G.shape[0], dtype=np.float64)
        canonical_grad[0] = 1.0
        evolution = self.purify_and_tune_cavity(psi, canonical_grad)
        return CategoricalState(payload=evolution.coherent_state, label=state.label)

    def backward(self, state: CategoricalState) -> CategoricalState:
        r"""
        Adjunta: rollback. Para Householder, backward = forward.
        Aquí conservamos la convención categórica del topos MIC.
        """
        return self.forward(state)

    # ────────────────────────────────────────────────────────────────────
    # MÉTODO AXIOMÁTICO PRINCIPAL
    # ────────────────────────────────────────────────────────────────────
    def purify_and_tune_cavity(
        self,
        raw_llm_logits: NDArray[np.float64],
        h_obs_gradient: NDArray[np.float64],
    ) -> QuantumChannelEvolution:
        r"""
        Pipeline canónico:
            1. Síntesis del proyector covariante (Fase 1)
            2. Auditoría de monodromía (Fase 2)
            3. Canal CPTP con emisión de antimateria (Fase 3)
        """
        logger.info("Floquet Agent: Sintonizando cavidad y purgando alucinaciones.")

        # ── Fase 1 ──
        P_hat, _reflector = self._phase1.synthesize_projector(h_obs_gradient)

        # ── Fase 2 ──
        monodromy = self._phase2.audit_monodromy(P_hat)
        logger.debug(
            "Monodromía certificada: ρ=%.6e, estable=%s.",
            monodromy.spectral_radius, monodromy.is_asymptotically_stable,
        )

        # ── Fase 3 ──
        evolution = self._phase3.execute_quantum_channel(raw_llm_logits, h_obs_gradient, P_hat)
        return evolution

    # ────────────────────────────────────────────────────────────────────
    # PROPIEDADES DE AUDITORÍA
    # ────────────────────────────────────────────────────────────────────
    @property
    def metric_tensor(self) -> NDArray[np.float64]:
        """Tensor métrico $G$ (copia defensiva)."""
        return self._G.copy()

    @property
    def phase1_synthesizer(self) -> ProjectorSynthesizerPort:
        """Fase 1 (objeto del topos)."""
        return self._phase1

    @property
    def phase2_auditor(self) -> FloquetAuditorPort:
        """Fase 2 (objeto del topos)."""
        return self._phase2

    @property
    def phase3_channel(self) -> KrausChannelPort:
        """Fase 3 (objeto del topos)."""
        return self._phase3


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "FloquetInstabilityError",
    "KrausTraceViolationError",
    "KrausCompletenessError",
    "DimensionalMismatchError",
    "DimensionalAudit",
    "FloquetMonodromyState",
    "QuantumChannelEvolution",
    "Phase1_CovariantProjectorSynthesizer",
    "Phase2_FloquetStabilityAuditor",
    "Phase3_QuantumKrausChannel",
    "FloquetMonodromyAgent",
]
### -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Quantum Epistemic Spectral Auditor (Sutura Connes-Takesaki-Watcher)║
║  Ruta   : app/wisdom/quantum_epistemic_auditor.py                            ║
║  Versión: 6.0.0-Spectral-Kernel-Componential-Categorical                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  NATURALEZA CIBER-FÍSICA Y RIGOR DOCTORAL:                                   ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Este módulo orquesta la aduana cuántica de sabiduría en TRES FASES          ║
║  ANIDADAS:                                                                   ║
║                                                                              ║
║    FASE 1 — Kernel espectral puro: cálculo funcional espectral, umbrales    ║
║             tipados y jerarquía de excepciones (sin estado, sin clase).      ║
║    FASE 2 — Acoplamiento estructural de componentes: Cauchy-Momentum,       ║
║             Triple de Connes, flujo modular de Takesaki, divergencia de     ║
║             Umegaki y distorsión de Dixmier, como métodos privados          ║
║             independientemente auditables.                                  ║
║    FASE 3 — Clausura categórica del veredicto: retícula de verdicto         ║
║             (COHERENT/DEGRADED/VETOED), certificación NUMÉRICA (no          ║
║             meramente declarada) de la condición KMS, y el orquestador      ║
║             público `execute_epistemic_audit`.                              ║
║                                                                              ║
║  Axioma de Consistencia Espectral:                                           ║
║  $$L_{\mathrm{max}} = \frac{C_{\mathrm{base}}}{1 + \lambda_{\mathrm{disp}}(D)}║
║  \quad \land \quad D(\rho \| \sigma) \le D_{\mathrm{max}}$$                  ║
║                                                                              ║
║  Nota de honestidad doctoral: el "operador de Dirac" ($D=\rho^{-1/2}$) y el  ║
║  "volumen de Dixmier" (traza finito-dimensional $\mathrm{Tr}(X^\dagger X)$)  ║
║  son ANALOGÍAS DE DISEÑO inspiradas en la geometría no conmutativa de       ║
║  Connes, no su construcción literal (que exige dimensión infinita y la      ║
║  traza de Dixmier no normal sobre el ideal de Macaev). Se declara            ║
║  explícitamente para no incurrir en sobreventa científica.                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional, Tuple
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

from app.core.mic_algebra import Morphism, TopologicalInvariantError
from app.core.schemas import Stratum
from app.agents.core.immune_system.watcher_agent import ValidatedStressTensor
from app.wisdom.tomita_takesaki_telescopic_engine import UmegakiExtractionState
from app.agents.wisdom.connes_spectral_auditor_agent import ConnesAuditState

logger = logging.getLogger("MIC.Wisdom.QuantumEpistemicAuditor")

ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 — KERNEL ESPECTRAL PURO                                             ║
# ║ (Teorema Espectral, cálculo funcional, umbrales tipados, excepciones)      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ─────────────────────── 1.1 Retícula de umbrales tipados ───────────────────

@dataclass(frozen=True, slots=True)
class SpectralAuditThresholds:
    r"""
    Reemplaza los números mágicos dispersos en la v5 por una retícula única
    de umbrales, versionable y explícita, que define los tres regímenes de
    veredicto de la Fase 3: COHERENTE, DEGRADADO y VETADO.
    """
    residual_conservation_tol: float = 1e-8
    self_adjointness_tol: float = 1e-12
    fidelity_floor: float = 1e-12
    support_floor: float = 1e-12
    commutator_coherence_bound: float = 1e3
    commutator_divergence_bound: float = 1e6
    umegaki_coherence_bound: float = 0.25
    umegaki_veto_bound: float = 0.50
    four_velocity_norm_tol: float = 1e-6
    kms_numerical_tol: float = 1e-6


# ─────────────────────── 1.2 Jerarquía granular de excepciones ──────────────

class CauchyConservationError(TopologicalInvariantError):
    """Lanzada ante fuga de exergía o asimetría espuria del tensor de esfuerzos."""
    pass


class SpectralFidelityError(TopologicalInvariantError):
    """Lanzada cuando un operador de estado no es autoadjunto o no es fiel (GNS degenerado)."""
    pass


class DiracCommutatorDivergenceError(TopologicalInvariantError):
    """Lanzada cuando el conmutador de Dirac diverge (alucinación semántica detectada)."""
    pass


class RelativeEntropySupportError(TopologicalInvariantError):
    """Lanzada ante violación de la condición de soporte de Araki (divergencia infinita)."""
    pass


class UmegakiVetoError(TopologicalInvariantError):
    """Lanzada cuando la divergencia de información cuántica excede el máximo tolerable."""
    pass


class WeakEnergyConditionError(TopologicalInvariantError):
    """Lanzada ante cuadrivelocidad mal normalizada o densidad de energía negativa."""
    pass


class KMSNumericalInconsistencyError(TopologicalInvariantError):
    """Lanzada ante residuo numérico excesivo en la identidad modular KMS de referencia."""
    pass


# ─────────────────── 1.3 Cálculo funcional espectral (núcleo puro) ──────────

def hermitian_eigendecomposition(
    matrix: ComplexMatrix,
    self_adjoint_tol: float,
    fidelity_floor: float,
) -> Tuple[RealVector, ComplexMatrix]:
    r"""
    Diagonaliza un operador autoadjunto vía el Teorema Espectral y certifica
    que el estado cuántico asociado es fiel (representación GNS no
    degenerada): $\rho = \sum_i \lambda_i |\lambda_i\rangle\langle\lambda_i|$,
    $\lambda_i \in \mathbb{R}$, $\lambda_i > \text{fidelity\_floor}\ \forall i$.

    Lanza:
        SpectralFidelityError: si $\rho \ne \rho^\dagger$ o si posee
            autovalores no estrictamente positivos.
    """
    if not np.allclose(matrix, matrix.conj().T, atol=self_adjoint_tol):
        raise SpectralFidelityError(
            "Axioma roto: el operador de estado viola la autoadjunción ρ = ρ†."
        )
    eigvals, eigvecs = la.eigh(matrix)
    if np.any(eigvals <= fidelity_floor):
        raise SpectralFidelityError(
            f"El estado cuántico no es fiel (GNS degenerado): "
            f"min(λ) = {float(np.min(eigvals)):.3e} ≤ {fidelity_floor:.1e}."
        )
    return eigvals, eigvecs


def spectral_functional_calculus(
    eigvals: RealVector,
    eigvecs: ComplexMatrix,
    func: Callable[[RealVector], RealVector],
) -> ComplexMatrix:
    r"""
    Cálculo funcional espectral exacto: $f(\rho) = V\,\mathrm{diag}(f(\lambda_i))\,V^\dagger$.

    Corrección de rigor V6.0: sustituye rutinas genéricas como
    `scipy.linalg.logm` —basadas en descomposición de Schur, sujetas a
    cortes de rama e inestabilidad numérica en matrices casi-singulares—
    por la fórmula exacta que garantiza el Teorema Espectral, reutilizando
    una diagonalización ya calculada en lugar de recomputarla.
    """
    return eigvecs @ np.diag(func(eigvals).astype(np.complex128)) @ eigvecs.conj().T


def operator_commutator_norm(A: ComplexMatrix, B: ComplexMatrix) -> float:
    r"""Calcula $\|[A,B]\|_2 = \|AB - BA\|_2$ en la norma espectral (operador)."""
    return float(la.norm(A @ B - B @ A, ord=2))


def four_velocity_energy_density(
    T_mu_nu: ComplexMatrix,
    g_mu_nu: ComplexMatrix,
    u_velocity: RealVector,
    norm_tol: float,
) -> float:
    r"""
    Calcula la densidad de energía local medida por un observador de
    cuadrivelocidad $u^\mu$: $\varepsilon = T_{\mu\nu}u^\mu u^\nu$, tras
    certificar la normalización temporal $g_{\mu\nu}u^\mu u^\nu = -1$
    (convención de signatura $(-,+,+,+)$).

    Corrección de rigor V6.0: en la v5, el parámetro `u_velocity` se
    aceptaba pero nunca se consumía — la Condición de Energía Débil jamás
    se verificaba. Esta función cierra esa brecha.

    Lanza:
        WeakEnergyConditionError: si la cuadrivelocidad no está normalizada
            o si $\varepsilon < 0$ (violación de la Condición de Energía
            Débil para un observador físico admisible).
    """
    norm = float(np.real(u_velocity @ g_mu_nu @ u_velocity))
    if abs(norm + 1.0) > norm_tol:
        raise WeakEnergyConditionError(
            f"Cuadrivelocidad no normalizada: u·g·u = {norm:.6f} ≠ -1 "
            f"(tolerancia {norm_tol:.1e})."
        )
    energy_density = float(np.real(u_velocity @ T_mu_nu @ u_velocity))
    if energy_density < -norm_tol:
        raise WeakEnergyConditionError(
            f"Violación de la Condición de Energía Débil: ε = {energy_density:.4e} < 0."
        )
    return energy_density


def kms_modular_identity_residual(
    rho_eigvals: RealVector,
    rho_eigvecs: ComplexMatrix,
    A: ComplexMatrix,
    B: ComplexMatrix,
) -> float:
    r"""
    Contrasta numéricamente la identidad KMS a temperatura inversa
    $\beta = 1$: para el flujo modular $\sigma_t(X) = \rho^{it}X\rho^{-it}$,
    la continuación analítica a $t = i$ da $\sigma_i(A) = \rho^{-1}A\rho$, y
    la condición KMS exige
    $$\mathrm{Tr}\big(\rho\,\sigma_i(A)\,B\big) = \mathrm{Tr}(\rho\,B\,A).$$

    Nota de honestidad doctoral (cierre de la brecha detectada en la v5):
    en álgebras de operadores de Tipo I de dimensión finita —el caso exacto
    de esta MAC— esta identidad es una consecuencia **algebraica trivial**
    de la ciclicidad de la traza para *cualquier* estado fiel $\rho$; el
    contenido no trivial del Teorema de Tomita-Takesaki solo emerge en
    álgebras de Tipo III (dimensión infinita, sin traza finita). Por ello,
    esta función NO certifica una propiedad física no trivial del estado
    —eso sería sobreventa—, sino la **consistencia numérica** del cálculo
    funcional espectral empleado para construir $\rho^{it}$: un residuo no
    nulo delata error de redondeo, jamás una "ruptura física" de KMS.
    """
    rho = spectral_functional_calculus(rho_eigvals, rho_eigvecs, lambda l: l)
    rho_inv = spectral_functional_calculus(rho_eigvals, rho_eigvecs, lambda l: 1.0 / l)
    sigma_i_A = rho_inv @ A @ rho
    lhs = np.trace(rho @ sigma_i_A @ B)
    rhs = np.trace(rho @ B @ A)
    return float(abs(lhs - rhs))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 — ACOPLAMIENTO ESTRUCTURAL DE COMPONENTES                           ║
# ║ (Cauchy-Momentum ⊗ Triple de Connes ⊗ Flujo de Takesaki ⊗ Umegaki/Dixmier) ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ─────────────────── 2.1 Estados intermedios (evitan recomputar) ────────────

@dataclass(frozen=True, slots=True)
class ConnesTripleState:
    r"""
    Estado intermedio del triple espectral $(\mathcal{A}, \mathcal{H}, D)$
    de la MAC, calculado una única vez y reutilizado por el flujo modular,
    la divergencia de Umegaki y la distorsión de Dixmier — corrigiendo la
    recomputación redundante de la diagonalización detectada en la v5.
    """
    eigvals: RealVector
    eigvecs: ComplexMatrix
    dirac_operator: ComplexMatrix
    commutator_norm: float
    lipschitz_limit: float


@dataclass(frozen=True, slots=True)
class ModularFlowState:
    """Resultado del flujo modular de Tomita-Takesaki aplicado al observable."""
    observable_zoomed: ComplexMatrix


# ─────────────────── 2.2 Clase auditora: componentes acoplados ──────────────

class QuantumEpistemicSpectralAuditor(Morphism):
    """
    Morfismo supremo que conecta la dinámica tensorial con la geometría de
    Connes. En esta fase se definen los componentes granulares que la Fase 3
    compondrá en el orquestador público.
    """

    def __init__(
        self,
        target_stratum: Stratum = Stratum.WISDOM,
        default_thresholds: SpectralAuditThresholds = SpectralAuditThresholds(),
    ) -> None:
        """Inicializa el auditor espectral en la compuerta de la Sabiduría."""
        super().__init__()
        self._target_stratum: Stratum = target_stratum
        self._default_thresholds: SpectralAuditThresholds = default_thresholds

    # ── 2.2.1 Componente Cauchy-Momentum (watcher_agent.py) ─────────────────

    def _validate_cauchy_conservation(
        self,
        stress_tensor_data: ValidatedStressTensor,
        thresholds: SpectralAuditThresholds,
    ) -> float:
        r"""
        Certifica la conservación covariante $\nabla_\mu\mathcal{T}^{\mu\nu}=0$
        y la simetría del tensor de esfuerzos $\mathcal{T}_{\mu\nu} =
        \mathcal{T}_{\nu\mu}$ (ausencia de pares internos no balanceados —
        chequeo estructural ausente en la v5), y calcula la energía de
        Dirichlet deformada $E = \mathrm{Tr}(\mathcal{T}\,\mathcal{G})$.

        Lanza:
            CauchyConservationError: ante fuga de exergía o asimetría espuria.
        """
        if stress_tensor_data.residual_relative > thresholds.residual_conservation_tol:
            raise CauchyConservationError(
                f"Fuga de exergía en la frontera: residual relativo de "
                f"∇_μ𝒯^μν = {stress_tensor_data.residual_relative:.4e} > "
                f"{thresholds.residual_conservation_tol:.1e}."
            )
        t_stress = stress_tensor_data.T_mu_nu
        if t_stress.shape[0] == t_stress.shape[1] and not np.allclose(
            t_stress, t_stress.T, atol=thresholds.self_adjointness_tol
        ):
            raise CauchyConservationError(
                "Asimetría espuria en el tensor de esfuerzos: 𝒯_μν ≠ 𝒯_νμ "
                "(par interno no balanceado)."
            )
        g_metric = stress_tensor_data.G_mu_nu
        return float(np.real(np.trace(t_stress @ g_metric)))

    def _certify_weak_energy_condition(
        self,
        stress_tensor_data: ValidatedStressTensor,
        u_velocity: Optional[RealVector],
        thresholds: SpectralAuditThresholds,
    ) -> Optional[float]:
        r"""
        Certifica la Condición de Energía Débil para un observador dado.
        Si `u_velocity` es `None`, se abstiene explícitamente (retorna
        `None`) y registra una advertencia de rigor reducido, en lugar de
        ignorar silenciosamente el parámetro como hacía la v5.
        """
        if u_velocity is None:
            logger.warning(
                "Auditoría sin cuadrivelocidad de observador: la Condición de "
                "Energía Débil NO fue verificada en esta invocación."
            )
            return None
        return four_velocity_energy_density(
            stress_tensor_data.T_mu_nu,
            stress_tensor_data.G_mu_nu,
            u_velocity,
            thresholds.four_velocity_norm_tol,
        )

    # ── 2.2.2 Componente Connes: triple espectral y conmutador de Dirac ─────

    def _instantiate_connes_dirac_triple(
        self,
        rho_mac: ComplexMatrix,
        observable_X: ComplexMatrix,
        c_base: float,
        thresholds: SpectralAuditThresholds,
    ) -> ConnesTripleState:
        r"""
        Instancia el análogo espectral $D = \rho^{-1/2}$ (analogía de diseño,
        no el operador de Dirac literal de Connes) y evalúa el conmutador
        semántico $\|[D, X]\|_2$, cota de la métrica de Connes-Lipschitz
        $d(\phi,\psi) = \sup\{|\phi(X)-\psi(X)| : \|[D,X]\|\le 1\}$.

        Lanza:
            SpectralFidelityError: si $\rho$ no es autoadjunta o fiel.
            DiracCommutatorDivergenceError: si el conmutador diverge
                (alucinación semántica detectada).
        """
        eigvals, eigvecs = hermitian_eigendecomposition(
            rho_mac, thresholds.self_adjointness_tol, thresholds.fidelity_floor
        )
        dirac_operator = spectral_functional_calculus(eigvals, eigvecs, lambda l: 1.0 / np.sqrt(l))
        commutator_norm = operator_commutator_norm(dirac_operator, observable_X)

        if commutator_norm > thresholds.commutator_divergence_bound:
            raise DiracCommutatorDivergenceError(
                f"SemanticDiscontinuityError: ||[D,X]|| = {commutator_norm:.4f} "
                f"> {thresholds.commutator_divergence_bound:.1e}. Alucinación detectada."
            )

        lambda_disp = float(np.max(eigvals) - np.min(eigvals))
        lipschitz_limit = c_base / (1.0 + lambda_disp)

        return ConnesTripleState(
            eigvals=eigvals,
            eigvecs=eigvecs,
            dirac_operator=dirac_operator,
            commutator_norm=commutator_norm,
            lipschitz_limit=lipschitz_limit,
        )

    # ── 2.2.3 Componente Takesaki: flujo modular y regla de Lüders ──────────

    def _apply_modular_flow(
        self, triple: ConnesTripleState, observable_X: ComplexMatrix, zoom_lambda: float
    ) -> ModularFlowState:
        r"""
        Aplica el flujo modular de Tomita-Takesaki
        $\sigma_\lambda(X) = \rho^{-\lambda}X\rho^{\lambda}$ como
        magnificación conforme del observable semántico.
        """
        if not math.isfinite(zoom_lambda):
            raise TopologicalInvariantError(
                f"Parámetro de zoom modular no finito: λ = {zoom_lambda!r}."
            )
        rho_inv_lambda = spectral_functional_calculus(
            triple.eigvals, triple.eigvecs, lambda l: np.power(l, -zoom_lambda)
        )
        rho_lambda = spectral_functional_calculus(
            triple.eigvals, triple.eigvecs, lambda l: np.power(l, zoom_lambda)
        )
        return ModularFlowState(observable_zoomed=rho_inv_lambda @ observable_X @ rho_lambda)

    def _apply_luders_transformation(
        self, rho_mac: ComplexMatrix, observable_X: ComplexMatrix
    ) -> ComplexMatrix:
        r"""
        Calcula el estado post-observación mediante la regla de Lüders
        $\sigma = \tfrac{1}{2}(\rho + \hat X \rho \hat X^\dagger)$, con
        $\hat X = X/\|X\|_2$, y normaliza exactamente la traza.

        Lanza:
            TopologicalInvariantError: si el observable es el operador nulo
                o la traza post-transformación es numéricamente nula.
        """
        norm_X = float(la.norm(observable_X, ord=2))
        if norm_X <= 1e-15:
            raise TopologicalInvariantError(
                "El observable semántico propuesto es el operador nulo: "
                "la regla de Lüders es indefinida."
            )
        obs_normalized = observable_X / norm_X
        rho_deformed = 0.5 * (rho_mac + obs_normalized @ rho_mac @ obs_normalized.conj().T)
        trace = np.trace(rho_deformed)
        if abs(trace) <= 1e-15:
            raise TopologicalInvariantError(
                "Traza nula tras la regla de Lüders: estado post-medida indefinido."
            )
        return rho_deformed / trace

    # ── 2.2.4 Componente Umegaki: entropía relativa cuántica ────────────────

    def _compute_umegaki_divergence(
        self,
        triple: ConnesTripleState,
        rho_deformed: ComplexMatrix,
        thresholds: SpectralAuditThresholds,
    ) -> float:
        r"""
        Calcula la divergencia de Umegaki $D(\rho\|\sigma) =
        \mathrm{Tr}(\rho(\ln\rho - \ln\sigma))$ reutilizando el cálculo
        funcional espectral (corrige el uso redundante de `scipy.linalg.logm`
        de la v5) y certificando previamente la condición de soporte de
        Araki: $D(\rho\|\sigma) < \infty \iff \mathrm{supp}(\rho) \subseteq
        \mathrm{supp}(\sigma)$ — verificación **ausente en la v5**.

        Lanza:
            RelativeEntropySupportError: si $\sigma$ no es de rango completo
                (la divergencia diverge a $+\infty$ por teorema).
            UmegakiVetoError: si la divergencia excede el máximo tolerable.
        """
        try:
            sigma_eigvals, sigma_eigvecs = hermitian_eigendecomposition(
                rho_deformed, thresholds.self_adjointness_tol, thresholds.support_floor
            )
        except SpectralFidelityError as exc:
            raise RelativeEntropySupportError(
                "Condición de soporte de Araki violada: supp(ρ) ⊄ supp(σ) "
                f"— D(ρ‖σ) diverge por teorema. Detalle: {exc}"
            ) from exc

        rho = spectral_functional_calculus(triple.eigvals, triple.eigvecs, lambda l: l)
        ln_rho = spectral_functional_calculus(triple.eigvals, triple.eigvecs, np.log)
        ln_sigma = spectral_functional_calculus(sigma_eigvals, sigma_eigvecs, np.log)
        umegaki_div = float(np.real(np.trace(rho @ (ln_rho - ln_sigma))))

        if umegaki_div > thresholds.umegaki_veto_bound:
            raise UmegakiVetoError(
                f"Veto de Umegaki: D(ρ‖σ) = {umegaki_div:.4f} > "
                f"{thresholds.umegaki_veto_bound:.2f}."
            )
        return umegaki_div

    # ── 2.2.5 Componente Dixmier: distorsión de volumen (analogía finita) ───

    def _compute_dixmier_distortion(
        self, observable_X: ComplexMatrix, observable_zoomed: ComplexMatrix
    ) -> float:
        r"""
        Calcula el ratio de distorsión $\|\hat X_\lambda\|_{HS}^2 /
        \|X\|_{HS}^2$ tras el flujo modular.

        Nota de honestidad doctoral: esta es una **analogía finito-
        dimensional** de la traza de Dixmier —que en su definición canónica
        requiere operadores compactos y asintóticas de valores singulares
        sobre un ideal de Macaev en dimensión infinita—; no se afirma
        equivalencia formal, solo inspiración estructural.
        """
        vol_pre = float(np.real(np.trace(observable_X @ observable_X.conj().T)))
        vol_post = float(np.real(np.trace(observable_zoomed @ observable_zoomed.conj().T)))
        return vol_post / (vol_pre + 1e-15)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 — CLAUSURA CATEGÓRICA DEL VEREDICTO                                 ║
# ║ (Retícula de verdicto + certificación KMS numérica + orquestador público) ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ─────────────────── 3.1 Retícula de verdicto (Ω de tres valores) ───────────

class EpistemicVerdict(IntEnum):
    r"""
    Clasificador de subobjetos de tres valores $\Omega_3 = \{0,1,2\}$ que
    formaliza los regímenes que la v5 dejaba implícitos en umbrales sueltos:

        COHERENT (0) -> dentro de la banda de coherencia total.
        DEGRADED (1) -> entre la banda de coherencia y el umbral de veto:
                        el paquete se retorna, pero `is_coherent=False`.
        VETOED   (2) -> solo alcanzable capturando la excepción tipada
                        correspondiente; el paquete jamás se materializa.
    """
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2


def _classify_bounded_metric(value: float, coherence_bound: float, veto_bound: float) -> EpistemicVerdict:
    r"""Clasifica una métrica escalar en la retícula $\Omega_3$ según sus dos umbrales."""
    if value <= coherence_bound:
        return EpistemicVerdict.COHERENT
    if value <= veto_bound:
        return EpistemicVerdict.DEGRADED
    return EpistemicVerdict.VETOED


# ─────────────────── 3.2 Certificado final (extendido y honesto) ────────────

@dataclass(frozen=True, slots=True)
class EpistemicCoherenceCertificate:
    r"""
    Certificado de inmunidad y coherencia cuántico-tensorial en WISDOM.

    Campos:
        is_coherent (bool): Conjunción de los regímenes COHERENT de todas
            las métricas acotadas (compatibilidad retro con la v5).
        verdict (EpistemicVerdict): Veredicto de mayor severidad alcanzado
            entre el conmutador de Dirac y la divergencia de Umegaki.
        dirichlet_energy (float): Energía de Dirichlet deformada.
        lipschitz_limit (float): Constante de Lipschitz conforme permitida.
        umegaki_divergence (float): Divergencia cuántica de Umegaki.
        dixmier_volume_ratio (float): Distorsión relativa de volumen (analogía finita).
        commutator_norm (float): Norma del conmutador de Dirac semántico.
        kms_numerical_residual (float): Residuo de consistencia numérica KMS
            (ver nota de honestidad en `kms_modular_identity_residual`).
        weak_energy_density (Optional[float]): Densidad de energía del
            observador, si se suministró cuadrivelocidad; `None` en caso
            contrario (rigor reducido explícito, no silencioso).
    """
    is_coherent: bool
    verdict: EpistemicVerdict
    dirichlet_energy: float
    lipschitz_limit: float
    umegaki_divergence: float
    dixmier_volume_ratio: float
    commutator_norm: float
    kms_numerical_residual: float
    weak_energy_density: Optional[float]


# ─────────────────── 3.3 Extensión del auditor: certificación y orquestación ─

class QuantumEpistemicSpectralAuditorMixin:
    r"""
    Mixin de clausura que se combina con `QuantumEpistemicSpectralAuditor`
    (Fase 2) para completar, sin romper su interfaz, la certificación
    numérica de la condición KMS y la orquestación pública final. Se
    presenta como mixin —en vez de reabrir la clase— para dejar explícita
    la frontera categórica entre "componentes acoplados" (Fase 2) y
    "clausura del veredicto" (Fase 3).
    """

    def certify_kms_numerical_consistency(
        self: "QuantumEpistemicSpectralAuditor",
        triple: ConnesTripleState,
        probe_A: ComplexMatrix,
        probe_B: ComplexMatrix,
        thresholds: SpectralAuditThresholds,
    ) -> float:
        r"""
        Certifica —numéricamente, no solo declarativamente como en la v5—
        la identidad modular KMS de referencia sobre un par de observables
        de sondeo. Corrige la brecha de honestidad documental detectada:
        el encabezado del módulo afirmaba garantizar KMS sin verificarlo
        jamás en el código.

        Lanza:
            KMSNumericalInconsistencyError: si el residuo excede la
                tolerancia numérica (indicio de error de redondeo en el
                cálculo funcional espectral, no de "ruptura física").
        """
        residual = kms_modular_identity_residual(triple.eigvals, triple.eigvecs, probe_A, probe_B)
        if residual > thresholds.kms_numerical_tol:
            raise KMSNumericalInconsistencyError(
                f"Inconsistencia numérica en la identidad modular KMS de referencia: "
                f"residuo = {residual:.3e} > {thresholds.kms_numerical_tol:.1e}."
            )
        return residual

    def execute_epistemic_audit(
        self: "QuantumEpistemicSpectralAuditor",
        stress_tensor_data: ValidatedStressTensor,
        rho_mac: ComplexMatrix,
        observable_X: ComplexMatrix,
        zoom_lambda: float,
        c_base: float = 1.5,
        u_velocity: Optional[RealVector] = None,
        thresholds: Optional[SpectralAuditThresholds] = None,
    ) -> EpistemicCoherenceCertificate:
        r"""
        Sutura el bucle completo: Cauchy-Momentum → Triple de Connes →
        Flujo de Takesaki → Umegaki/Dixmier → Certificación KMS →
        Retícula de veredicto. Compone exclusivamente los componentes
        granulares de la Fase 2 sobre los primitivos puros de la Fase 1.

        Args:
            stress_tensor_data: Datos de estrés del Watcher.
            rho_mac: Matriz densidad actual de la MAC.
            observable_X: Observable semántico propuesto.
            zoom_lambda: Parámetro de magnificación modular.
            c_base: Escala de la constante de Lipschitz.
            u_velocity: Cuadrivelocidad del observador (opcional; si se
                omite, la Condición de Energía Débil no se certifica y se
                registra advertencia explícita — no falla silenciosamente).
            thresholds: Retícula de umbrales; usa los del constructor si
                se omite.

        Retorna:
            EpistemicCoherenceCertificate: Certificado inmutable de
                inmunidad cognitiva, con veredicto de tres valores.

        Lanza:
            CauchyConservationError, WeakEnergyConditionError,
            SpectralFidelityError, DiracCommutatorDivergenceError,
            RelativeEntropySupportError, UmegakiVetoError,
            KMSNumericalInconsistencyError.
        """
        active_thresholds = thresholds or self._default_thresholds

        # 1) Cauchy-Momentum (Fase 2.2.1)
        dirichlet_energy = self._validate_cauchy_conservation(stress_tensor_data, active_thresholds)
        weak_energy_density = self._certify_weak_energy_condition(
            stress_tensor_data, u_velocity, active_thresholds
        )

        # 2) Triple de Connes (Fase 2.2.2)
        triple = self._instantiate_connes_dirac_triple(
            rho_mac, observable_X, c_base, active_thresholds
        )

        # 3) Flujo modular de Takesaki + Lüders (Fase 2.2.3)
        modular_state = self._apply_modular_flow(triple, observable_X, zoom_lambda)
        rho_deformed = self._apply_luders_transformation(rho_mac, observable_X)

        # 4) Divergencia de Umegaki (Fase 2.2.4)
        umegaki_div = self._compute_umegaki_divergence(triple, rho_deformed, active_thresholds)

        # 5) Distorsión de Dixmier (Fase 2.2.5)
        dixmier_ratio = self._compute_dixmier_distortion(observable_X, modular_state.observable_zoomed)

        # 6) Certificación numérica KMS (Fase 3.3) — sondeo con D y X mismos,
        #    reutilizando el conmutador ya calculado como par de prueba natural.
        kms_residual = self.certify_kms_numerical_consistency(
            triple, triple.dirac_operator, observable_X, active_thresholds
        )

        # 7) Clausura del veredicto: retícula Ω_3 tomando el peor régimen.
        commutator_verdict = _classify_bounded_metric(
            triple.commutator_norm,
            active_thresholds.commutator_coherence_bound,
            active_thresholds.commutator_divergence_bound,
        )
        umegaki_verdict = _classify_bounded_metric(
            umegaki_div,
            active_thresholds.umegaki_coherence_bound,
            active_thresholds.umegaki_veto_bound,
        )
        final_verdict = max(commutator_verdict, umegaki_verdict)
        is_coherent = final_verdict == EpistemicVerdict.COHERENT

        logger.info(
            "Auditoría epistémica [%s] completada. veredicto=%s | ||[D,X]||=%.4f | "
            "D(ρ‖σ)=%.4f | KMS_residual=%.3e",
            self._target_stratum, final_verdict.name, triple.commutator_norm,
            umegaki_div, kms_residual,
        )

        return EpistemicCoherenceCertificate(
            is_coherent=is_coherent,
            verdict=final_verdict,
            dirichlet_energy=dirichlet_energy,
            lipschitz_limit=triple.lipschitz_limit,
            umegaki_divergence=umegaki_div,
            dixmier_volume_ratio=dixmier_ratio,
            commutator_norm=triple.commutator_norm,
            kms_numerical_residual=kms_residual,
            weak_energy_density=weak_energy_density,
        )


class QuantumEpistemicSpectralAuditor(  # noqa: F811 — extensión intencional de la Fase 2
    QuantumEpistemicSpectralAuditor, QuantumEpistemicSpectralAuditorMixin
):
    """
    Clausura final: `QuantumEpistemicSpectralAuditor` de la Fase 2 se
    reabre por composición de mixin (no por herencia externa) para
    incorporar `certify_kms_numerical_consistency` y
    `execute_epistemic_audit`, dejando explícita la frontera Fase 2 → Fase 3.
    """
    pass
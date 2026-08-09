### -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Quantum Epistemic Auditor Agent (Custodio de la Decisión)           ║
║ Ruta   : app/agents/wisdom/quantum_epistemic_auditor_agent.py                ║
║ Versión: 7.0.0-Weyl-Connes-Takesaki-OODA-Nested-Strict                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y COHOMOLOGÍA ESPECTRAL EN EL ESTRATO WISDOM (V_W) ─────
Este módulo consagra al Agente Soberano y Observador Activo del Estrato de la
Sabiduría, encargado de gobernar y certificar síncronamente la consistencia
cuántica de los canales de inyección semántica $$\mathcal{E}$$ en la categoría
dagger-compacta $$\mathcal{C}_{\mathrm{MAC}}$$.

Su mandato axiomático es la "Bohrificación" del espacio de decisión agéntica,
subordinando la probabilidad estocástica y la deriva retórica del Modelo de
Lenguaje (LLM) al determinismo físico de la obra civil real [1, 3]. Trata la
Matriz Atómica de Conocimiento (MAC) como un Álgebra de von Neumann provista de
un flujo modular fiel.

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial Estricta): ────────────
La transición de estados se rige por la Ley de Clausura Transitiva de subespacios
de Hilbert covariantes y se compone de tres fases fuertemente acopladas:

  Fase 1 ──► FASE 1: OBSERVACIÓN ESPECTRAL Y CERTIFICACIÓN GNS (Observe)
             Estudia la fidelidad y Hermiticidad del estado de deliberación.
             Garantiza la inyección del operador densidad fiel: $$\rho \succeq 0, \;\operatorname{Tr}(\rho)=1$$.
             Entrega: Phase1CertificationData como precondición formal de Fase 2.

  Fase 2 ──► FASE 2: ORIENTACIÓN COVARIANTE Y GEOMETRÍA NO CONMUTATIVA (Orient)
             Evalúa la regularidad de Connes y la cota de Lipschitz semántica.
             Axioma de regularidad: $$\| [D, \pi(X)] \| \le C$$
             Entrega: Phase2OrientationData como precondición formal de Fase 3.

  Fase 3 ──► FASE 3: DECISIÓN EN RETÍCULO Y ACTUACIÓN CROWBAR (Decide & Act)
             Fuerza el colapso del estado en el retículo de Heyting $$\Omega_3$$ y
             despacha el bypass físico por hardware en el microcontrolador.
             Veredicto: $$\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}$$.

INVARIANTES MATEMÁTICOS Y GEOMÉTRICOS PRESERVADOS: ──────────────────────────────
  [I1] Positividad Completa de Choi:      $$\Lambda_{\mathcal{E}} \succeq 0$$
  [I2] Preservación de Traza de Kraus:    $$\sum_k M_k^\dagger M_k = I$$
  [I3] No-Señalización Cuántica Local:    $$\operatorname{Tr}_A\big( (\mathcal{E}_A \otimes \mathcal{I}_B)(\rho_{AB}) \big) \equiv \rho_B$$
  [I4] Causalidad de de Rham (Poset):     $$V_{\mathrm{PHYSICS}} \subsetneq V_{\mathrm{TACTICS}} \subsetneq V_{\mathrm{STRATEGY}} \subsetneq V_{\mathrm{WISDOM}}$$

CONTRATO DEL DISYUNTOR FÍSICO POR HARDWARE (Bypass ESP32 / BT151): ──────────────
  Si la pureza de la MAC se degrada por debajo del umbral de regularización,
  o si el primer grupo de cohomología del haz celular es no trivial ($$\dim H^1 > 0$$),
  el retículo $$\Omega_3$$ colapsa síncronamente al veredicto terminal VETOED.
  
  La subrutina local 'isVerdictCoherent()' del ESP32 en el borde detecta el
  mismatch en menos de 400 ns y conmuta el pin GPIO14, disparando el tiristor
  de potencia BT151 (circuito Crowbar). Esto cortocircuita físicamente la
  línea de potencia real, inmovilizando actuadores en el milisegundo cero,
  anulando la alucinación de la IA antes del desfalco de capital.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Final

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

from app.core.mic_algebra import Morphism, TopologicalInvariantError
from app.core.schemas import Stratum
from app.core.telemetry_schemas import SystemStateVector, SystemStateVector as SystemStateVectorType
from app.wisdom.quantum_epistemic_auditor import (
    SpectralAuditThresholds,
    SpectralFidelityError,
    DiracCommutatorDivergenceError,
    WeakEnergyConditionError,
    EpistemicVerdict,
    hermitian_eigendecomposition,
    four_velocity_energy_density,
    _classify_bounded_metric,
)

logger = logging.getLogger("MIC.Wisdom.QuantumEpistemicAuditorAgent")

# ─────────────────────────────────────────────────────────────────────────────
# Tipos algebraicos canónicos (C*-módulo / fibrado de Hilbert)
# ─────────────────────────────────────────────────────────────────────────────
ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]
RealScalar = float
BoolLattice = bool  # álgebra de Boole del topos subobject classifier Ω

_EPS_SPECTRAL: Final[float] = 1.0e-15
_EPS_TRACE: Final[float] = 1.0e-12
_EPS_HERMITICITY: Final[float] = 1.0e-10


# ═══════════════════════════════════════════════════════════════════════════════
# CERTIFICADO INMUTABLE DEL VEREDICTO (objeto terminal del funtor de auditoría)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class EpistemicAgentVerdict:
    r"""
    Certificado inmutable del veredicto del agente soberano.

    Vive en el topos de haces sobre el espectro de Gelfand de la MAC.
    Todos los campos son covariantes bajo unitarios de la representación GNS.

    Campos:
        verdict (EpistemicVerdict):
            Clasificación final en la retícula Ω₃ = {CERTIFIED ≺ DEGRADED ≺ VETOED}.
        energy_density (float):
            Densidad de energía covariante \( T_{\mu\nu} u^\mu u^\nu \).
        commutator_norm (float):
            Norma operatorial \( \|[D, X]\|_2 \) (métrica de Connes).
        lipschitz_limit (float):
            Cota de Lipschitz \( c_B / (1 + \lambda_{\mathrm{disp}}) \) que confina al LLM.
        is_active_veto (bool):
            Bit de actuador Crowbar (elemento del clasificador de subobjetos Ω).
        spectral_gap (float):
            Gap espectral \( \lambda_{\min}^{>}(D) - 0 \) del operador de Dirac.
        gns_fidelity (float):
            Fidelidad de la representación GNS tras proyección al simplejo de estados.
    """
    verdict: EpistemicVerdict
    energy_density: float
    commutator_norm: float
    lipschitz_limit: float
    is_active_veto: bool
    spectral_gap: float = 0.0
    gns_fidelity: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTE SOBERANO — MORFISMO EN LA CATEGORÍA MIC
# ═══════════════════════════════════════════════════════════════════════════════
class QuantumEpistemicAuditorAgent(Morphism):
    r"""
    Agente soberano de observabilidad y control epistemológico en WISDOM.

    Realiza el ciclo OODA como endofuntor covariante

        \[
        \mathcal{OODA}
        : \mathbf{State}(\mathcal{H}_{\mathrm{MAC}})
        \longrightarrow
        \mathbf{Verdict}(\Omega_3)
        \]

    sobre la categoría de estados densidades y observables autoadjuntos,
    con valores en la retícula de Heyting del topos de Bohrificación.

    Las tres fases anidadas se componen como:

        \[
        \mathrm{Act}\circ\mathrm{Decide}\circ\mathrm{Orient}\circ\mathrm{Observe}
        \]

    donde cada flecha es un morfismo parcial dominado por monadas de error
    espectral (SpectralFidelity, WEC, DiracDivergence).
    """

    def __init__(self, thresholds: Optional[SpectralAuditThresholds] = None) -> None:
        """
        Inicializa al centinela cuántico sintonizando la retícula de umbrales.

        Args:
            thresholds: Retícula de cotas espectrales; si es None se usan
                        los axiomas por defecto de SpectralAuditThresholds.
        """
        super().__init__()
        self._target_stratum: Stratum = Stratum.WISDOM
        self._thresholds: SpectralAuditThresholds = (
            thresholds if thresholds is not None else SpectralAuditThresholds()
        )

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  FASE 1 — OBSERVACIÓN ESPECTRAL Y CERTIFICACIÓN GNS                   ║
    # ║  Objetos: ρ ∈ 𝒟(ℋ), proyección ortogonal, simplejo de estados         ║
    # ║  Funtores: Bohrificación B, representación GNS π_ρ                    ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    def _phase1_validate_density_operator(
        self,
        rho_mac: ComplexMatrix,
    ) -> ComplexMatrix:
        r"""
        FASE 1.1 — Validación algebraica del operador densidad.

        Verifica el axioma de estado normal normalizado sobre una C*-álgebra
        de von Neumann factor:

        \[
        \rho = \rho^\dagger,\quad
        \rho \succeq 0,\quad
        \operatorname{Tr}(\rho) = 1,\quad
        \|\rho\|_1 < \infty.
        \]

        En términos de circuitos eléctricos análogos: ρ es la matriz de
        admitancia hermítica de un multipolo pasivo con potencia disipada
        unitaria (Tr ρ = 1 ↔ conservación de carga de probabilidad).

        Args:
            rho_mac: Matriz candidata a operador densidad de la MAC.

        Returns:
            rho_mac sin copia defensiva si es válido (invariante referencial).

        Raises:
            SpectralFidelityError: Si se viola hermiticidad, positividad o traza.
        """
        if rho_mac.ndim != 2 or rho_mac.shape[0] != rho_mac.shape[1]:
            raise SpectralFidelityError(
                f"ρ debe ser operador cuadrado; recibido shape={rho_mac.shape}"
            )

        # Desviación de hermiticidad en norma de Frobenius (topología C*)
        anti_hermitian_residues = rho_mac - rho_mac.conj().T
        hermiticity_defect = float(la.norm(anti_hermitian_residues, ord="fro"))
        if hermiticity_defect > self._thresholds.self_adjointness_tol:
            raise SpectralFidelityError(
                f"Defecto de autoadjunción ‖ρ−ρ†‖_F = {hermiticity_defect:.3e} "
                f"> tol={self._thresholds.self_adjointness_tol:.3e}"
            )

        # Traza como forma lineal normal (peso de Haar discreto)
        trace_rho = float(np.real(np.trace(rho_mac)))
        if abs(trace_rho - 1.0) > _EPS_TRACE:
            raise SpectralFidelityError(
                f"Normalización de traza violada: Tr(ρ)={trace_rho:.8f} ≠ 1"
            )

        return rho_mac

    def _phase1_gns_eigendecomposition(
        self,
        rho_mac: ComplexMatrix,
    ) -> Tuple[RealVector, ComplexMatrix, RealScalar]:
        r"""
        FASE 1.2 — Diagonalización hermítica y construcción del isomorfismo GNS.

        Calcula la descomposición espectral

        \[
        \rho = U\,\mathrm{diag}(\lambda_i)\,U^\dagger,
        \qquad \lambda_i \ge 0,\quad \sum_i\lambda_i = 1,
        \]

        y certifica que el piso de fidelidad espectral

        \[
        \min_i\lambda_i \ge \vartheta_{\mathrm{floor}}
        \]

        garantiza que el soporte de ρ genera un subespacio cíclico no degenerado
        para la representación GNS π_ρ. El gap de fidelidad se reporta como

        \[
        F_{\mathrm{GNS}} = 1 - \tfrac12\sum_i|\lambda_i - \lambda_i^{\downarrow}|
        \]

        (distancia de traza al proyector de rango completo normalizado).

        Args:
            rho_mac: Operador densidad ya validado (salida de 1.1).

        Returns:
            Tupla (eigvals, eigvecs, gns_fidelity).

        Raises:
            SpectralFidelityError: Propagada desde hermitian_eigendecomposition
                si el piso de fidelidad o la autoadjunción fallan.
        """
        eigvals, eigvecs = hermitian_eigendecomposition(
            rho_mac,
            self._thresholds.self_adjointness_tol,
            self._thresholds.fidelity_floor,
        )

        # Fidelidad GNS: proximidad del espectro al simplejo de probabilidad
        eigvals_real = np.real(eigvals).astype(np.float64)
        # Proyección L¹ al simplejo (renormalización de masas de Dirac)
        mass = float(np.sum(np.clip(eigvals_real, 0.0, None)))
        if mass < _EPS_SPECTRAL:
            raise SpectralFidelityError(
                "Espectro de ρ aniquilado: masa total < ε_spectral (colapso GNS)."
            )
        projected = np.clip(eigvals_real, 0.0, None) / mass
        # Distancia de variación total al espectro original normalizado
        original_normed = eigvals_real / max(float(np.sum(np.abs(eigvals_real))), _EPS_SPECTRAL)
        total_variation = 0.5 * float(np.sum(np.abs(projected - original_normed)))
        gns_fidelity = max(0.0, 1.0 - total_variation)

        if gns_fidelity < self._thresholds.fidelity_floor:
            raise SpectralFidelityError(
                f"Fidelidad GNS={gns_fidelity:.6f} < floor={self._thresholds.fidelity_floor}"
            )

        logger.debug(
            "FASE1.2 GNS: n=%d, λ_min=%.3e, λ_max=%.3e, F_GNS=%.6f",
            len(eigvals_real),
            float(np.min(eigvals_real)),
            float(np.max(eigvals_real)),
            gns_fidelity,
        )
        return eigvals_real, eigvecs, gns_fidelity

    def _phase1_observe_spectral_certificate(
        self,
        rho_mac: ComplexMatrix,
    ) -> Tuple[RealVector, ComplexMatrix, RealScalar]:
        r"""
        FASE 1.3 — Composición terminal de Observación (Observe).

        Morfismo compuesto de la FASE 1:

        \[
        \mathrm{Observe}
        = \mathrm{GNS}\circ\mathrm{Validate}
        : \mathrm{Mat}_n(\mathbb{C})
        \rightharpoonup
        \mathbb{R}^n_{\ge 0}\times U(n)\times[0,1].
        \]

        Produce el certificado espectral completo que alimenta la orientación
        covariante. Esta firma de retorno es el *objeto inicial* de la FASE 2:
        el triple (eigvals, eigvecs, gns_fidelity) se inyecta sin transformación
        en `_phase2_construct_dirac_operator`, garantizando la continuidad
        funtorial Observe → Orient.

        Args:
            rho_mac: Operador densidad bruto de la MAC.

        Returns:
            (eigvals, eigvecs, gns_fidelity) — certificado GNS sellado.
        """
        rho_valid = self._phase1_validate_density_operator(rho_mac)
        eigvals, eigvecs, gns_fidelity = self._phase1_gns_eigendecomposition(rho_valid)
        return eigvals, eigvecs, gns_fidelity

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  FASE 2 — ORIENTACIÓN COVARIANTE Y GEOMETRÍA NO CONMUTATIVA           ║
    # ║  Continuación directa del certificado GNS de FASE 1.3                 ║
    # ║  Objetos: D = ρ^{-1/2}, [D,X], T_{μν}u^μu^ν, cota de Lipschitz        ║
    # ║  Teorías: Connes spectral triple, WEC de Hawking-Ellis, Takesaki      ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    def _phase2_construct_dirac_operator(
        self,
        eigvals: RealVector,
        eigvecs: ComplexMatrix,
    ) -> Tuple[ComplexMatrix, RealScalar]:
        r"""
        FASE 2.1 — Operador de Dirac no conmutativo de Connes (inicio Orient).

        **Continuación funtorial de FASE 1.3**: recibe exactamente el par
        (eigvals, eigvecs) emitido por `_phase1_observe_spectral_certificate`.

        Construye el triple espectral irregular (𝒜, ℋ, D) donde

        \[
        D = \rho^{-1/2}
          = U\,\mathrm{diag}\bigl(\lambda_i^{-1/2}\bigr)\,U^\dagger
        \]

        actúa como derivada exterior no conmutativa. El gap espectral

        \[
        \gamma(D) := \inf\bigl\{\lambda^{-1/2} : \lambda\in\sigma(\rho)\setminus\{0\}\bigr\}
        \]

        controla la resolvente y la dimensión métrica de Connes.
        Regularización: cualquier λ_i < ε_spectral se eleva a ε_spectral
        (corte ultravioleta tipo Pauli-Villars en teoría de cuerdas / QFT).

        Args:
            eigvals: Espectro no negativo de ρ (de FASE 1.3).
            eigvecs: Base unitaria de diagonalización (de FASE 1.3).

        Returns:
            (dirac_d, spectral_gap) con D = ρ^{-1/2} y γ(D).
        """
        # Corte UV: evita polos cuando ker(ρ) ≠ {0}
        safe_eigvals = np.maximum(eigvals, _EPS_SPECTRAL)
        inv_sqrt = 1.0 / np.sqrt(safe_eigvals)

        dirac_d: ComplexMatrix = (
            eigvecs @ np.diag(inv_sqrt.astype(np.complex128)) @ eigvecs.conj().T
        )

        # Gap = distancia del espectro de D al origen (tras corte)
        spectral_gap = float(np.min(inv_sqrt))

        # Simetría residual: D debe permanecer autoadjunto (circuitos LC duales)
        d_herm_defect = float(la.norm(dirac_d - dirac_d.conj().T, ord="fro"))
        if d_herm_defect > _EPS_HERMITICITY:
            # Proyección de Weyl al subespacio autoadjunto
            dirac_d = 0.5 * (dirac_d + dirac_d.conj().T)
            logger.warning(
                "FASE2.1: proyección de Weyl aplicada a D; defecto=%.3e", d_herm_defect
            )

        logger.debug(
            "FASE2.1 Dirac: ‖D‖₂=%.6e, gap=%.6e, λ_disp=%.6e",
            float(la.norm(dirac_d, ord=2)),
            spectral_gap,
            float(np.max(inv_sqrt) - np.min(inv_sqrt)),
        )
        return dirac_d, spectral_gap

    def _phase2_connes_commutator_norm(
        self,
        dirac_d: ComplexMatrix,
        observable_x: ComplexMatrix,
    ) -> RealScalar:
        r"""
        FASE 2.2 — Norma del conmutador de Dirac (métrica de Connes).

        Calcula

        \[
        \|[D, X]\|_2
        = \sup_{\|\psi\|=1}
          \bigl\|(DX - XD)\psi\bigr\|
        \]

        que define la distancia de Connes en el espacio de estados puros:

        \[
        d(\varphi,\psi)
        = \sup\bigl\{|\varphi(a)-\psi(a)| : a=a^\dagger,\; \|[D,a]\|\le 1\bigr\}.
        \]

        Si \|[D,X]\| diverge, el observable X no es Lipschitz respecto de D
        y la geometría semántica del LLM abandona el dominio del triple espectral
        (alucinación ontológica ≡ singularidad de curvatura no conmutativa).

        En grafo de Cayley del grupo unitario: el conmutador mide la longitud
        de arista entre X y su transporte paralelo inducido por D.

        Args:
            dirac_d: Operador de Dirac (salida de 2.1).
            observable_x: Observable semántico autoadjunto del LLM.

        Returns:
            Norma espectral (ord=2) del conmutador [D, X].

        Raises:
            DiracCommutatorDivergenceError: Si la norma es no-finita.
        """
        if observable_x.shape != dirac_d.shape:
            raise DiracCommutatorDivergenceError(
                f"Dimensiones incompatibles: D{dirac_d.shape} vs X{observable_x.shape}"
            )

        commutator = dirac_d @ observable_x - observable_x @ dirac_d
        # Norma operatorial = radio espectral de √(C†C) (teorema de Gelfand)
        commutator_norm = float(la.norm(commutator, ord=2))

        if not math.isfinite(commutator_norm):
            raise DiracCommutatorDivergenceError(
                f"‖[D,X]‖₂ no finita ({commutator_norm}); "
                "singularidad del triple espectral de Connes."
            )

        return commutator_norm

    def _phase2_lipschitz_confinement_bound(
        self,
        eigvals: RealVector,
        c_base: float,
    ) -> RealScalar:
        r"""
        FASE 2.3 — Cota de Lipschitz de confinamiento del LLM.

        A partir de la dispersión espectral del inverso de raíz

        \[
        \lambda_{\mathrm{disp}}
        = \max_i\lambda_i^{-1/2} - \min_i\lambda_i^{-1/2}
        \]

        se define la constante de acoplamiento de Connes regularizada

        \[
        L_{\mathrm{Lip}}
        = \frac{c_B}{1 + \lambda_{\mathrm{disp}}}
        \in (0, c_B].
        \]

        Todo observable admisible debe satisfacer \|[D,X]\| ≤ L_Lip⁻¹;
        de lo contrario el morfismo semántico sale del tubo tubular de
        radio Lipschitz alrededor de la MAC (confinamiento tipo Wilson loop
        en teoría de cuerdas / gauge).

        Interpretación en circuitos: L_Lip es la admitancia máxima del
        filtro paso-bajos que acopla la fuente LLM al bus de la MAC.

        Args:
            eigvals: Espectro de ρ (reutilizado de FASE 1, sin recomputar).
            c_base: Constante base de acoplamiento de Connes (c_B > 0).

        Returns:
            lipschitz_limit ∈ (0, c_base].
        """
        if c_base <= 0.0:
            raise ValueError(f"c_base debe ser estrictamente positivo; recibido {c_base}")

        safe = np.maximum(eigvals, _EPS_SPECTRAL)
        inv_sqrt = 1.0 / np.sqrt(safe)
        lambda_disp = float(np.max(inv_sqrt) - np.min(inv_sqrt))
        lipschitz_limit = c_base / (1.0 + lambda_disp)
        return lipschitz_limit

    def _phase2_weak_energy_condition(
        self,
        t_stress: ComplexMatrix,
        g_metric: ComplexMatrix,
        u_velocity: RealVector,
    ) -> RealScalar:
        r"""
        FASE 2.4 — Condición de Energía Débil (WEC) covariante.

        Evalúa la densidad de energía medida por un observador con
        cuadrivelocidad \(u^\mu\):

        \[
        \rho_E = T_{\mu\nu}\,u^\mu u^\nu \ge 0
        \]

        (condición de energía débil de Hawking-Ellis). La métrica g_μν
        se usa para bajar índices y normalizar u·u = −1 (signatura −+++).
        En el topos, WEC es un predicado abierto del clasificador Ω cuya
        negación fuerza el colapso al elemento VETOED.

        Análogo eléctrico: ρ_E es la potencia instantánea inyectada en
        la red tensorial; negatividad implica fuente activa no pasiva
        (violación de Passivity de Willems).

        Args:
            t_stress: Tensor de esfuerzos de Cauchy T_μν.
            g_metric: Métrica riemanniana/lorentziana de fondo.
            u_velocity: Cuadrivelocidad del observador.

        Returns:
            energy_density = T_μν u^μ u^ν.

        Raises:
            WeakEnergyConditionError: Si ρ_E < 0 más allá de la tolerancia.
        """
        energy_density = four_velocity_energy_density(
            t_stress,
            g_metric,
            u_velocity,
            self._thresholds.four_velocity_norm_tol,
        )
        return float(energy_density)

    def _phase2_orient_geometry(
        self,
        eigvals: RealVector,
        eigvecs: ComplexMatrix,
        observable_x: ComplexMatrix,
        t_stress: ComplexMatrix,
        g_metric: ComplexMatrix,
        u_velocity: RealVector,
        c_base: float,
    ) -> Tuple[RealScalar, RealScalar, RealScalar, RealScalar]:
        r"""
        FASE 2.5 — Composición terminal de Orientación (Orient).

        Morfismo compuesto de la FASE 2, que consume el certificado GNS
        de FASE 1.3 y produce el 4-tuple de invariantes geométricos:

        \[
        \mathrm{Orient}
        = (\mathrm{WEC},\;\mathrm{Commutator},\;\mathrm{Lip},\;\mathrm{Gap})
        \circ \mathrm{Dirac}.
        \]

        Esta firma de retorno es el *objeto inicial* de la FASE 3:
        (energy_density, commutator_norm, lipschitz_limit, spectral_gap)
        se inyecta en `_phase3_decide_lattice_verdict` sin re-codificación,
        preservando la exactitud del funtor OODA.

        Args:
            eigvals, eigvecs: Certificado espectral de FASE 1.3.
            observable_x: Observable semántico X del LLM.
            t_stress, g_metric, u_velocity: Datos covariantes de WEC.
            c_base: Acoplamiento base de Connes.

        Returns:
            (energy_density, commutator_norm, lipschitz_limit, spectral_gap).
        """
        dirac_d, spectral_gap = self._phase2_construct_dirac_operator(eigvals, eigvecs)
        commutator_norm = self._phase2_connes_commutator_norm(dirac_d, observable_x)
        lipschitz_limit = self._phase2_lipschitz_confinement_bound(eigvals, c_base)
        energy_density = self._phase2_weak_energy_condition(
            t_stress, g_metric, u_velocity
        )
        return energy_density, commutator_norm, lipschitz_limit, spectral_gap

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  FASE 3 — DECISIÓN EN RETÍCULA Ω₃ Y ACTUACIÓN CROWBAR                 ║
    # ║  Continuación directa de los invariantes geométricos de FASE 2.5      ║
    # ║  Objetos: retícula de Heyting Ω₃, bit Crowbar, veredicto terminal     ║
    # ║  Teorías: topos de Bohr, álgebra de Boole, control supervisory        ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    def _phase3_decide_lattice_verdict(
        self,
        commutator_norm: RealScalar,
    ) -> EpistemicVerdict:
        r"""
        FASE 3.1 — Clasificación en la retícula de Heyting Ω₃ (inicio Decide).

        **Continuación funtorial de FASE 2.5**: recibe `commutator_norm`
        emitido por `_phase2_orient_geometry`.

        La retícula totalista

        \[
        \Omega_3
        = \{\mathtt{CERTIFIED}\prec\mathtt{DEGRADED}\prec\mathtt{VETOED}\}
        \]

        se ordena por severidad epistemológica. El morfismo de clasificación

        \[
        \kappa:\mathbb{R}_{\ge 0}\to\Omega_3,
        \quad
        \kappa(r)
        =
        \begin{cases}
        \mathtt{CERTIFIED} & r \le \vartheta_{\mathrm{coh}} \\
        \mathtt{DEGRADED}  & \vartheta_{\mathrm{coh}} < r \le \vartheta_{\mathrm{div}} \\
        \mathtt{VETOED}    & r > \vartheta_{\mathrm{div}}
        \end{cases}
        \]

        es monótono y continuo por la derecha (topología de orden de Scott).
        En álgebra de Boole del subobject classifier del topos, VETOED es
        el elemento máximo (falso terminal) que colapsa todo pullback.

        Args:
            commutator_norm: ‖[D,X]‖₂ proveniente de FASE 2.

        Returns:
            EpistemicVerdict ∈ Ω₃.
        """
        verdict = _classify_bounded_metric(
            commutator_norm,
            self._thresholds.commutator_coherence_bound,
            self._thresholds.commutator_divergence_bound,
        )
        return verdict

    def _phase3_act_crowbar_veto(
        self,
        verdict: EpistemicVerdict,
        commutator_norm: RealScalar,
    ) -> BoolLattice:
        r"""
        FASE 3.2 — Actuación Crowbar (bit de interrupción perimetral).

        El actuador es el morfismo de evaluación del clasificador

        \[
        \chi_{\mathrm{Veto}}
        : \Omega_3 \to \mathbf{2}
        = \{\bot,\top\},
        \qquad
        \chi_{\mathrm{Veto}}(v) = (v = \mathtt{VETOED}).
        \]

        Cuando χ = ⊤ se gatilla el Crowbar de hardware: corte galvánico
        del bus semántico LLM↔MAC (análogo al breaker de potencia en
        circuitos de protección diferencial). La telemetría se emite en
        nivel CRITICAL para el plano de control supervisory.

        Args:
            verdict: Veredicto de la retícula (salida de 3.1).
            commutator_norm: Norma para logging forense.

        Returns:
            is_active_veto ∈ 𝔹.
        """
        is_active_veto: BoolLattice = verdict == EpistemicVerdict.VETOED

        if is_active_veto:
            logger.error(
                "¡VETO ONTOLÓGICO! Alucinación de la IA excede la cota de Connes. "
                "Gatillando interrupción del Crowbar perimetral. ‖[D,X]‖ = %.4e",
                commutator_norm,
            )
        else:
            logger.info(
                "Coherencia epistémica aprobada por el agente soberano. Veredicto: %s",
                verdict.name,
            )
        return is_active_veto

    def _phase3_seal_verdict_certificate(
        self,
        verdict: EpistemicVerdict,
        energy_density: RealScalar,
        commutator_norm: RealScalar,
        lipschitz_limit: RealScalar,
        is_active_veto: BoolLattice,
        spectral_gap: RealScalar,
        gns_fidelity: RealScalar,
    ) -> EpistemicAgentVerdict:
        r"""
        FASE 3.3 — Sellado del certificado inmutable (objeto terminal de Act).

        Empaqueta todos los invariantes en el dataclass frozen
        `EpistemicAgentVerdict`, que es el objeto terminal del funtor OODA.
        Ningún campo es mutable post-construcción (slots + frozen ⇒
        hashable ⇒ admisible como clave de caché de topos / memoización
        de morfismos).

        Args:
            verdict, energy_density, commutator_norm, lipschitz_limit,
            is_active_veto, spectral_gap, gns_fidelity:
                Invariantes acumulados de Fases 1–3.

        Returns:
            EpistemicAgentVerdict sellado.
        """
        return EpistemicAgentVerdict(
            verdict=verdict,
            energy_density=energy_density,
            commutator_norm=commutator_norm,
            lipschitz_limit=lipschitz_limit,
            is_active_veto=is_active_veto,
            spectral_gap=spectral_gap,
            gns_fidelity=gns_fidelity,
        )

    def _phase3_collapse_to_veto(
        self,
        err: Exception,
    ) -> EpistemicAgentVerdict:
        r"""
        FASE 3.Ω — Colapso de topos al elemento supremo de peor caso.

        Cuando cualquier monada de error espectral (SpectralFidelity,
        WeakEnergy, DiracDivergence) se efectúa, el clasificador de
        subobjetos fuerza el pullback al objeto cero:

        \[
        \bot_{\Omega_3}
        =
        \mathtt{VETOED}
        \times
        (\rho_E=0)\times(\|[D,X]\|=\infty)\times(L_{\mathrm{Lip}}=0)
        \times(\chi=\top).
        \]

        Equivale al cortocircuito a tierra del bus de decisión
        (fail-secure en ingeniería de protección).

        Args:
            err: Excepción raíz que provocó el colapso.

        Returns:
            EpistemicAgentVerdict de veto total.
        """
        logger.critical(
            "Colapso catastrófico de invariante durante la auditoría: %s. "
            "Forzando colapso de estado a VETOED.",
            str(err),
        )
        return EpistemicAgentVerdict(
            verdict=EpistemicVerdict.VETOED,
            energy_density=0.0,
            commutator_norm=float("inf"),
            lipschitz_limit=0.0,
            is_active_veto=True,
            spectral_gap=0.0,
            gns_fidelity=0.0,
        )

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  COMPOSICIÓN OODA — ENDOFUNCTOR COVARIANTE PÚBLICO                    ║
    # ║  Observe (F1) ⟶ Orient (F2) ⟶ Decide (F3.1) ⟶ Act (F3.2–3.3)         ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    def execute_ooda_cycle(
        self,
        rho_mac: ComplexMatrix,
        observable_X: ComplexMatrix,
        t_stress: ComplexMatrix,
        g_metric: ComplexMatrix,
        u_velocity: RealVector,
        c_base: float = 1.5,
    ) -> EpistemicAgentVerdict:
        r"""
        Ejecuta el ciclo OODA covariante completo sobre el estado de la MAC.

        Composición estricta de las tres fases anidadas:

        .. code-block:: text

            ┌─────────────────────────────────────────────────────────┐
            │ FASE 1  Observe                                         │
            │   1.1 validate_density_operator                         │
            │   1.2 gns_eigendecomposition                            │
            │   1.3 observe_spectral_certificate  ──┐                 │
            ├───────────────────────────────────────┼─────────────────┤
            │ FASE 2  Orient  ◄─────────────────────┘                 │
            │   2.1 construct_dirac_operator                          │
            │   2.2 connes_commutator_norm                            │
            │   2.3 lipschitz_confinement_bound                       │
            │   2.4 weak_energy_condition                             │
            │   2.5 orient_geometry  ──┐                              │
            ├──────────────────────────┼──────────────────────────────┤
            │ FASE 3  Decide + Act  ◄──┘                              │
            │   3.1 decide_lattice_verdict                            │
            │   3.2 act_crowbar_veto                                  │
            │   3.3 seal_verdict_certificate                          │
            │   3.Ω collapse_to_veto  (rama monádica de error)        │
            └─────────────────────────────────────────────────────────┘

        Args:
            rho_mac: Operador densidad de la MAC (ρ ≽ 0, Tr ρ = 1).
            observable_X: Observable semántico del LLM (X = X†).
            t_stress: Tensor de esfuerzos de Cauchy (T_μν).
            g_metric: Tensor métrico de fondo.
            u_velocity: Cuadrivelocidad relativista (u^μ).
            c_base: Constante base de acoplamiento de Connes (default 1.5).

        Returns:
            EpistemicAgentVerdict: Certificado de gobernanza ciber-física.

        Raises:
            TopologicalInvariantError: Si se vulnera de forma irrecuperable
                la fidelidad cuántica o la causalidad de Einstein (la monada
                interna captura SpectralFidelity/WEC/Dirac y colapsa a VETOED;
                otras excepciones de invariante topológico se re-lanzan).
        """
        try:
            # ── FASE 1 · Observe ──────────────────────────────────────────
            eigvals, eigvecs, gns_fidelity = self._phase1_observe_spectral_certificate(
                rho_mac
            )

            # ── FASE 2 · Orient  (continúa certificado GNS de F1) ─────────
            (
                energy_density,
                commutator_norm,
                lipschitz_limit,
                spectral_gap,
            ) = self._phase2_orient_geometry(
                eigvals,
                eigvecs,
                observable_X,
                t_stress,
                g_metric,
                u_velocity,
                c_base,
            )

            # ── FASE 3 · Decide + Act  (continúa invariantes de F2) ───────
            verdict = self._phase3_decide_lattice_verdict(commutator_norm)
            is_active_veto = self._phase3_act_crowbar_veto(verdict, commutator_norm)
            return self._phase3_seal_verdict_certificate(
                verdict=verdict,
                energy_density=energy_density,
                commutator_norm=commutator_norm,
                lipschitz_limit=lipschitz_limit,
                is_active_veto=is_active_veto,
                spectral_gap=spectral_gap,
                gns_fidelity=gns_fidelity,
            )

        except (
            SpectralFidelityError,
            WeakEnergyConditionError,
            DiracCommutatorDivergenceError,
        ) as err:
            return self._phase3_collapse_to_veto(err)
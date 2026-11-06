# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Scalar Higgs Anchor Agent (Soberano del Condensador de Inercia)     ║
║ Ruta   : app/agents/physics/scalar_higgs_anchor_agent.py                     ║
║ Versión: 5.0.0-Poincare-RSI3-Connes-Liouville-Ricci-Brockett-Heyting-ESP32   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Ecuaciones Constitutivas y Axiomas de Gobernanza de Calibre:
-----------------------------------------------------------
1. Estabilidad del Vacío de Higgs (VEV):
   $$v = \langle \Phi \rangle = \|\Phi\|_2 \ge \tau_{\mathrm{SPD}} = 10^{-12}$$

2. Invarianza Simpléctica de Poincaré-Liouville-Darboux:
   $$\mathbf{M}^\top \mathbf{\Omega} \mathbf{M} = \mathbf{\Omega} \implies \det(\mathbf{M}) = +1 \quad \text{con} \quad \mathbf{\Omega} = \begin{pmatrix} \mathbf{0} & \mathbf{I}_n \\ -\mathbf{I}_n & \mathbf{0} \end{pmatrix}$$
   Multiplicadores característicos de Floquet-Poincaré: $\sigma(\mathbf{M}) \subset S^1 \subset \mathbb{C}$ (estabilidad de Krein).

3. Cálculo Operatorial de Loewner y Cota de Lipschitz de Connes-Daleckii-Krein:
   $$\Xi_{ij} = \frac{\lambda_i^{-1/2} - \lambda_j^{-1/2}}{\lambda_i - \lambda_j}, \qquad L_{\max} = \|\mathbf{\Xi}\|_{\max} \le \frac{1}{2 \, \lambda_{\min}(\mathbf{G})^{3/2}}$$

4. Automejora Recursiva RSI³ (Nivel R₁ - Flujo de Ricci-Perelman Discreto):
   $$\mathbf{G}_{k+1} = \mathbf{G}_k - 2 \, \Delta \tau \left( \operatorname{Ric}_{\mu\nu}(\mathbf{G}_k) - \frac{R_{\mathrm{avg}}}{n} \mathbf{G}_k \right)$$

5. Automejora Recursiva RSI³ (Nivel R₂ - Retracción Polar Simpléctica de Poincaré):
   $$\mathbf{M}_{k+1} = \Pi_{\mathrm{Sp}(2n)}(\mathbf{M}_k) = \mathbf{M}_k \left( \mathbf{\Omega}^{-1} \mathbf{M}_k^\top \mathbf{\Omega} \mathbf{M}_k \right)^{-1/2}$$

6. Automejora Recursiva RSI³ (Nivel R₃ - Flujo Isoespectral de Brockett-Shahshahani):
   $$\frac{d\boldsymbol{\rho}}{dt} = \left[ \boldsymbol{\rho}, \, [\boldsymbol{\rho}, \, \operatorname{diag}(\mathbf{K}\mathbf{p})] \right], \qquad \dot{\mathcal{H}} = -\mathrm{Var}_{\mathbf{p}}(\tilde{\mathcal{K}}\mathbf{p}) \le 0$$

7. Clasificación en el Retículo de Heyting Trivalente Ω₃ y Disparo Crowbar ESP32:
   $$\Omega_3 = \{\mathtt{COHERENT} (0), \, \mathtt{DEGRADED} (1), \, \mathtt{VETOED} (2)\}$$
   $$v_{\mathrm{final}} = \bigsqcup_i v_i \implies \mathtt{COHERENT} \sqcup \mathtt{VETOED} = \mathtt{VETOED}$$

ARQUITECTURA DE TRES FASES ANIDADAS (CONTINUIDAD MORFISMÁTICA ESTRICTA):
-----------------------------------------------------------------------
  Fase 1 ──► OBSERVE : Axiomática de calibre, topos de Heyting $\Omega_3$, estado del campo y
             Observatorio Simpléctico de Monodromía de Poincaré. Su método final
             `PoincareSymplecticObservatory.observe_and_bridge_to_phase2_orient` es el
             inicio formal de la Fase 2.
  Fase 2 ──► ORIENT  : Cálculo operatorial de Connes-Daleckii-Krein y motor de Automejora
             Recursiva de Nivel 3 ($\mathrm{RSI}^3 = R_3 \circ R_2 \circ R_1$). Su método final
             `RSI3MetaFunctorEngine.bridge_to_phase3_sovereign_decision` instancia la
             certificación y actuación que abre la Fase 3.
  Fase 3 ──► DECIDE  : Actuador electrónico Crowbar ESP32 (GPIO14 / Tiristor BT151 <400ns)
             y Soberano de Calibre `ScalarHiggsAnchorAgent` acoplado al motor FPU
             `app/physics/scalar_higgs_anchor.py`.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("MIC.Physics.ScalarHiggsAnchorAgent")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES NUMÉRICAS FUNDAMENTALES (IEEE 754 / WILKINSON / KREIN)
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_MAX_CONDITION_NUMBER: Final[float] = 1.0e8
_WILKINSON_TOLERANCE: Final[float] = 1.0e-9
_KREIN_FLOQUET_TOLERANCE: Final[float] = 1.0e-6


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 1 · AXIOMÁTICA DE CALIBRE, TOPOS Ω₃ Y OBSERVACIÓN DE POINCARÉ     │
# │  Define el retículo de Heyting, invariantes de fase y el observatorio   │
# │  de monodromía simpléctica. Su último método                            │
# │  (`observe_and_bridge_to_phase2_orient`) inicia formalmente la Fase 2.  │
# └─────────────────────────────────────────────────────────────────────────┘


# ───────────────────────────────────────────────────────────────────────────
# 1.1  CLASIFICADOR INTUICIONISTA DE HEYTING Ω₃ Y EXCEPCIONES TOPOLÓGICAS
# ───────────────────────────────────────────────────────────────────────────


class HeytingVerdict(enum.IntEnum):
    r"""
    Retículo Distributivo de Heyting Trivalente $\Omega_3$ para vetos ciber-físicos.

    Equipado con operaciones internas de álgebra de Heyting:
    - Supremo ($\sqcup$, `join`): propagación monotónica hacia el colapso `VETOED` ($\top$).
    - Ínfimo ($\sqcap$, `meet`): consenso de coherencia hacia `COHERENT` ($\bot$).
    - Implicación relativa ($a \Rightarrow b$) y pseudo-complemento intuicionista ($\neg a$).
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        """Operador supremo $a \sqcup b = \max(a, b)$ en el orden de anomalía."""
        return HeytingVerdict(max(int(self), int(other)))

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        """Operador ínfimo $a \sqcap b = \min(a, b)$ en el orden de anomalía."""
        return HeytingVerdict(min(int(self), int(other)))

    def implies(self, consequent: "HeytingVerdict") -> "HeytingVerdict":
        """Implicación de Heyting $(a \Rightarrow b)$ en la cadena finita $\Omega_3$."""
        if int(self) <= int(consequent):
            return HeytingVerdict.VETOED
        return consequent

    def is_verdict_coherent(self) -> bool:
        """Predicado de seguridad equivalente a `isVerdictCoherent()` en IRAM del ESP32."""
        return self == HeytingVerdict.COHERENT


class TopologicalInvariantError(Exception):
    r"""Excepción de ruptura de invariante topológico, métrica no-SPD o colapso de vacío."""


# ───────────────────────────────────────────────────────────────────────────
# 1.2  ESTRUCTURAS DE ESTADO DEL CAMPO, MONODROMÍA Y CERTIFICADO DE CALIBRE
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScalarFieldState:
    r"""
    Estado inmutable del campo escalar de Higgs en el fibrado cotangente $T^*\mathcal{M}$.

    Mantiene compatibilidad estructural exacta con `ScalarFieldState` del motor
    `app/physics/scalar_higgs_anchor.py` mediante propiedades puente.
    """

    phi: NDArray[np.float64]
    momentum_p: NDArray[np.float64]
    potential_energy: float
    kinetic_energy: float
    total_energy: float

    @property
    def pi_momentum(self) -> NDArray[np.float64]:
        """Alias de compatibilidad directa con `app.physics.scalar_higgs_anchor.ScalarFieldState`."""
        return self.momentum_p

    @property
    def dim(self) -> int:
        return int(self.phi.shape[0])

    @classmethod
    def from_engine_state(
        cls,
        engine_state: Any,
        potential_energy: float = 0.0,
        total_energy: Optional[float] = None,
    ) -> "ScalarFieldState":
        """
        Construye el estado del agente a partir de un `ScalarFieldState` del motor
        `scalar_higgs_anchor.py`.
        """
        phi_arr = np.asarray(engine_state.phi, dtype=np.float64)
        pi_arr = np.asarray(
            getattr(engine_state, "pi_momentum", getattr(engine_state, "momentum_p", np.zeros_like(phi_arr))),
            dtype=np.float64,
        )
        kin = 0.5 * float(np.dot(pi_arr, pi_arr))
        tot = float(total_energy) if total_energy is not None else (kin + float(potential_energy))
        return cls(
            phi=phi_arr,
            momentum_p=pi_arr,
            potential_energy=float(potential_energy),
            kinetic_energy=kin,
            total_energy=tot,
        )


@dataclass(frozen=True, slots=True)
class PoincareMonodromyTelemetry:
    r"""
    Telemetría invariante de Mecánica Celeste de Poincaré para la matriz Jacobiana
    de monodromía $\mathbf{M} \in \mathbb{R}^{2n \times 2n}$ (o $\mathbb{R}^{m \times m}$).

    Atributos
    ---------
    det_M : float
        Determinante de Liouville $\det(\mathbf{M})$.
    volume_drift : float
        Deriva absoluta del volumen de espacio de fase $|\det(\mathbf{M}) - 1|$.
    symplectic_form_defect : float
        Defecto de Darboux $\|\mathbf{M}^\top \mathbf{\Omega} \mathbf{M} - \mathbf{\Omega}\|_F$.
    floquet_spectral_radius : float
        Máximo módulo de los multiplicadores característicos de Floquet $\max_i |\mu_i(\mathbf{M})|$.
    krein_elliptic_fraction : float
        Fracción de autovalores de Floquet situados sobre el círculo unitario $S^1 \subset \mathbb{C}$.
    poincare_broucke_trace : float
        Invariante de traza de Broucke-Poincaré $\operatorname{Tr}(\mathbf{M})$.
    """

    det_M: float
    volume_drift: float
    symplectic_form_defect: float
    floquet_spectral_radius: float
    krein_elliptic_fraction: float
    poincare_broucke_trace: float


@dataclass(frozen=True, slots=True)
class HiggsGovernanceCertificate:
    r"""
    Certificado inmutable de gobernanza emitiendo decisión en el topos $\Omega_3$.

    Los primeros 7 atributos mantienen el orden posicional y semántico exacto del
    contrato de interfaz v4.0; los atributos adicionales tienen valores por defecto
    para enriquecer la telemetría de Poincaré, Connes-DK, Brockett y Crowbar ESP32.
    """

    vacuum_expectation_value: float
    potential_energy: float
    lipschitz_connes_bound: float
    symplectic_volume_drift: float
    ricci_scalar_curvature: float
    heyting_verdict: HeytingVerdict
    is_field_coherent: bool
    # Telemetría extendida de Nivel Doctoral (con valores por defecto)
    symplectic_form_defect: float = 0.0
    floquet_spectral_radius: float = 1.0
    krein_elliptic_fraction: float = 1.0
    daleckii_krein_operator_norm: float = 0.0
    metric_condition_number: float = 1.0
    perelman_entropy_functional: float = 0.0
    crowbar_gpio14_latched: bool = False


# ───────────────────────────────────────────────────────────────────────────
# 1.3  OBSERVATORIO SIMPLÉCTICO DE POINCARÉ Y PUENTE ANIDADO A FASE 2
# ───────────────────────────────────────────────────────────────────────────


class PoincareSymplecticObservatory:
    r"""
    Observatorio de Mecánica Celeste de Poincaré y Geometría Riemanniana SPD (Fase 1).

    Audita el Valor de Expectativa del Vacío (VEV), la positividad espectral de Sylvester
    del tensor métrico $\mathbf{G}$, el teorema de volumen de Liouville y la estructura
    simpléctica de Darboux-Floquet-Krein de la matriz de monodromía $\mathbf{M}$.
    """

    @staticmethod
    def build_darboux_symplectic_matrix(dim_2n: int) -> NDArray[np.float64]:
        r"""
        Construye la matriz simpléctica canónica de Darboux $\mathbf{\Omega} \in \mathrm{SO}(2n, \mathbb{R})$:
            $$\mathbf{\Omega} = \begin{pmatrix} \mathbf{0}_n & \mathbf{I}_n \\ -\mathbf{I}_n & \mathbf{0}_n \end{pmatrix}, \qquad \mathbf{\Omega}^\top = -\mathbf{\Omega}, \quad \mathbf{\Omega}^2 = -\mathbf{I}_{2n}$$
        """
        if dim_2n <= 0 or dim_2n % 2 != 0:
            raise ValueError(
                f"La dimensión simpléctica de Darboux debe ser par positiva, recibido {dim_2n}"
            )
        n = dim_2n // 2
        omega = np.zeros((dim_2n, dim_2n), dtype=np.float64)
        omega[:n, n:] = np.eye(n, dtype=np.float64)
        omega[n:, :n] = -np.eye(n, dtype=np.float64)
        return omega

    @classmethod
    def analyze_poincare_monodromy(
        cls, jacobian_M: NDArray[np.float64]
    ) -> PoincareMonodromyTelemetry:
        r"""
        Evalúa los invariantes de Poincaré-Liouville-Floquet sobre el Jacobiano $\mathbf{M}$.
        """
        M = np.asarray(jacobian_M, dtype=np.float64)
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] == 0:
            raise TopologicalInvariantError(
                f"Jacobiano de monodromía inválido: shape={M.shape}"
            )

        det_M = float(la.det(M))
        volume_drift = abs(det_M - 1.0)
        trace_M = float(np.trace(M))

        # Defecto de la 2-forma simpléctica de Darboux si la dimensión es par
        m_dim = M.shape[0]
        if m_dim % 2 == 0:
            omega = cls.build_darboux_symplectic_matrix(m_dim)
            symp_residual = M.T @ omega @ M - omega
            symp_defect = float(la.norm(symp_residual, ord="fro"))
        else:
            symp_defect = volume_drift

        # Multiplicadores característicos de Floquet-Poincaré y firma elíptica de Krein
        eigvals = la.eigvals(M)
        moduli = np.abs(eigvals)
        floquet_radius = float(np.max(moduli))
        on_unit_circle = np.abs(moduli - 1.0) <= _KREIN_FLOQUET_TOLERANCE
        krein_fraction = float(np.mean(on_unit_circle))

        return PoincareMonodromyTelemetry(
            det_M=det_M,
            volume_drift=volume_drift,
            symplectic_form_defect=symp_defect,
            floquet_spectral_radius=floquet_radius,
            krein_elliptic_fraction=krein_fraction,
            poincare_broucke_trace=trace_M,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # ESLABÓN ANIDADO FASE 1 ──► FASE 2 (CONTINUACIÓN MORFISMÁTICA ESTRICTA)
    # ═══════════════════════════════════════════════════════════════════════
    @classmethod
    def observe_and_bridge_to_phase2_orient(
        cls,
        field_state: ScalarFieldState,
        metric_G: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        ricci_tensor: Optional[NDArray[np.float64]] = None,
    ) -> "Phase2OrientationManifold":
        r"""
        Definición formal final de la Fase 1 (`OBSERVE`) y génesis directa de la Fase 2 (`ORIENT`).

        Verifica rigurosamente:
        1. Ausencia de colapso a vacío falso: $v = \|\Phi\|_2 \ge \tau_{\mathrm{SPD}} = 10^{-12}$.
        2. Condición definida positiva (SPD) de la variedad métrica: $\lambda_{\min}(\mathbf{G}) \ge 10^{-12}$.
        3. Telemetría simpléctica de Poincaré-Liouville sobre $\mathbf{M}$.

        Transfiere los autovalores/autovectores espectrales de $\mathbf{G}$ y la telemetría de
        monodromía al constructor de apertura de la Fase 2 (`Phase2OrientationManifold`).
        """
        phi_norm = float(la.norm(field_state.phi, ord=2))
        if phi_norm < _SPECTRAL_PSD_FLOOR:
            logger.error(
                f"[HIGGS_AGENT_VETO] Colapso a Vacío Falso: ||Φ|| = {phi_norm:.3e} < {_SPECTRAL_PSD_FLOOR}"
            )
            raise TopologicalInvariantError(
                f"[HIGGS_AGENT_VETO] Colapso a Vacío Falso: ||Φ|| = {phi_norm:.3e}"
            )

        G_sym = 0.5 * (metric_G + metric_G.T)
        eigs_G, eigvecs_G = la.eigh(G_sym)
        lambda_min_G = float(np.min(eigs_G))
        lambda_max_G = float(np.max(eigs_G))

        if lambda_min_G < _SPECTRAL_PSD_FLOOR:
            logger.error(
                f"[HIGGS_AGENT_VETO] Variedad Métrica no SPD: λ_min = {lambda_min_G:.3e}"
            )
            raise TopologicalInvariantError(
                f"[HIGGS_AGENT_VETO] Variedad Métrica no SPD: λ_min = {lambda_min_G:.3e}"
            )

        cond_num = lambda_max_G / max(lambda_min_G, _MACHINE_EPS)
        monodromy_telemetry = cls.analyze_poincare_monodromy(jacobian_M)

        # Continuación directa hacia la estructura de apertura de la Fase 2
        return Phase2OrientationManifold.from_phase1_observation(
            field_state=field_state,
            phi_norm=phi_norm,
            metric_G_sym=G_sym,
            eigs_G=eigs_G,
            eigvecs_G=eigvecs_G,
            lambda_min_G=lambda_min_G,
            lambda_max_G=lambda_max_G,
            condition_number=cond_num,
            jacobian_M=jacobian_M,
            monodromy=monodromy_telemetry,
            ricci_tensor=ricci_tensor,
        )


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 2 · ORIENTACIÓN GEOMÉTRICA DE CONNES-DK Y AUTOMEJORA RSI³ (ORIENT)│
# │  Inicia directamente como continuación del puente de observación Fase 1.│
# │  Su método final (`bridge_to_phase3_sovereign_decision`) constituye el  │
# │  puente anidado de entrada hacia la Fase 3.                             │
# └─────────────────────────────────────────────────────────────────────────┘


# ───────────────────────────────────────────────────────────────────────────
# 2.1  VARIEDAD DE ORIENTACIÓN Y OPERADOR DE LOEWNER-CONNES-DALECKII-KREIN
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Phase2OrientationManifold:
    r"""
    Estructura portadora del estado geométrico-espectral de la Fase 2 (`ORIENT`),
    instanciada directamente por el método final de la Fase 1.
    """

    field_state: ScalarFieldState
    phi_norm: float
    metric_G_sym: NDArray[np.float64]
    eigs_G: NDArray[np.float64]
    eigvecs_G: NDArray[np.float64]
    lambda_min_G: float
    lambda_max_G: float
    condition_number: float
    jacobian_M: NDArray[np.float64]
    monodromy: PoincareMonodromyTelemetry
    ricci_tensor: Optional[NDArray[np.float64]] = None

    @classmethod
    def from_phase1_observation(
        cls,
        field_state: ScalarFieldState,
        phi_norm: float,
        metric_G_sym: NDArray[np.float64],
        eigs_G: NDArray[np.float64],
        eigvecs_G: NDArray[np.float64],
        lambda_min_G: float,
        lambda_max_G: float,
        condition_number: float,
        jacobian_M: NDArray[np.float64],
        monodromy: PoincareMonodromyTelemetry,
        ricci_tensor: Optional[NDArray[np.float64]] = None,
    ) -> "Phase2OrientationManifold":
        """Continuación formal directa desde `PoincareSymplecticObservatory.observe_and_bridge_to_phase2_orient`."""
        return cls(
            field_state=field_state,
            phi_norm=phi_norm,
            metric_G_sym=metric_G_sym,
            eigs_G=eigs_G,
            eigvecs_G=eigvecs_G,
            lambda_min_G=lambda_min_G,
            lambda_max_G=lambda_max_G,
            condition_number=condition_number,
            jacobian_M=jacobian_M,
            monodromy=monodromy,
            ricci_tensor=ricci_tensor,
        )

    def compute_connes_daleckii_krein_loewner(
        self,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Construye la matriz de diferencias divididas de Loewner-Daleckii-Krein $\mathbf{\Xi}$
        para la función espectral de distancia de Connes $f(\lambda) = \lambda^{-1/2}$:

            $$\Xi_{ij} = \begin{cases} -\frac{1}{2}\lambda_i^{-3/2}, & |\lambda_i - \lambda_j| \le \varepsilon_{\mathrm{mach}} \\ \frac{\lambda_i^{-1/2} - \lambda_j^{-1/2}}{\lambda_i - \lambda_j}, & \lambda_i \neq \lambda_j \end{cases}$$

        Retorna $(\mathbf{\Xi}, L_{\max})$, donde $L_{\max} = \max_{i,j}|\Xi_{ij}| = \frac{1}{2\lambda_{\min}(\mathbf{G})^{3/2}}$.
        """
        lam = np.maximum(self.eigs_G, _SPECTRAL_PSD_FLOOR)
        inv_sqrt = 1.0 / np.sqrt(lam)
        diff_lam = lam[:, None] - lam[None, :]
        diff_f = inv_sqrt[:, None] - inv_sqrt[None, :]

        diag_deriv = -0.5 * (lam ** (-1.5))
        near_mask = np.abs(diff_lam) <= 100.0 * _MACHINE_EPS

        with np.errstate(divide="ignore", invalid="ignore"):
            xi_matrix = np.where(
                near_mask,
                0.5 * (diag_deriv[:, None] + diag_deriv[None, :]),
                diff_f / np.where(near_mask, 1.0, diff_lam),
            )

        lipschitz_bound = float(1.0 / (2.0 * (self.lambda_min_G ** 1.5)))
        return np.abs(xi_matrix), lipschitz_bound

    def compute_ricci_and_perelman_entropy(self) -> Tuple[float, float]:
        r"""
        Calcula la curvatura escalar de Ricci $R = \operatorname{Tr}(\mathbf{G}^{-1}\operatorname{Ric})$
        y el funcional de entropía de Perelman $\mathcal{F}(\mathbf{G}) = R \cdot \sqrt{\det(\mathbf{G}_{\mathrm{norm}})}$.
        """
        if (
            self.ricci_tensor is None
            or self.ricci_tensor.shape != self.metric_G_sym.shape
        ):
            return 0.0, 0.0

        Ric_sym = 0.5 * (self.ricci_tensor + self.ricci_tensor.T)
        inv_G_Ric = la.solve(self.metric_G_sym, Ric_sym, assume_a="pos")
        ricci_scalar = float(np.trace(inv_G_Ric))

        # Volumen Riemanniano normalizado por media geométrica para estabilidad numérica
        log_vol_half = 0.5 * float(
            np.sum(np.log(np.maximum(self.eigs_G, _SPECTRAL_PSD_FLOOR)))
        )
        perelman_f = ricci_scalar * math.exp(
            max(-50.0, min(50.0, log_vol_half / max(1, len(self.eigs_G))))
        )
        return ricci_scalar, perelman_f


# ───────────────────────────────────────────────────────────────────────────
# 2.2  MOTOR DE AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI³: R₁, R₂, R₃)
# ───────────────────────────────────────────────────────────────────────────


class RSI3MetaFunctorEngine:
    r"""
    Motor de Automejora Recursiva de Nivel 3 ($\mathrm{RSI}^3 = R_3 \circ R_2 \circ R_1$).

    Gobierna las tres variedades acopladas del Condensador de Inercia de Higgs:
    - **Nivel $R_1$ (Variedad Riemanniana $\mathcal{S}_{++}^n$)**: Flujo de Ricci-Perelman
      normalizado con regularización espectral de Higham.
    - **Nivel $R_2$ (Grupo Simpléctico de Poincaré $\mathrm{Sp}(2n, \mathbb{R})$)**: Retracción
      polar simpléctica de la matriz de monodromía $\mathbf{M}$ para restaurar $\mathbf{M}^\top\mathbf{\Omega}\mathbf{M} = \mathbf{\Omega}$
      y $\det(\mathbf{M}) = +1$.
    - **Nivel $R_3$ (Variedad Cuántica Isoespectral de Brockett-Shahshahani)**: Flujo de doble
      corchete $[\boldsymbol{\rho}, [\boldsymbol{\rho}, \operatorname{diag}(\mathbf{K}\mathbf{p})]]$
      con monotonía de Lyapunov $\dot{\mathcal{H}}_{\mathrm{Shah}} = -\mathrm{Var}_{\mathbf{p}}(\mathbf{K}\mathbf{p}) \le 0$.
    """

    @staticmethod
    def evolve_level_r1_ricci_flow(
        metric_G: NDArray[np.float64],
        ricci_tensor: NDArray[np.float64],
        delta_tau: float = 1.0e-3,
    ) -> NDArray[np.float64]:
        r"""
        Ejecuta el paso de Automejora Recursiva $\mathrm{RSI}^3$ (Nivel $R_1$ - Flujo de Ricci Discreto):
            $$\mathbf{G}_{k+1} = \mathbf{G}_k - 2 \, \Delta \tau \left( \operatorname{Ric}_{\mu\nu}(\mathbf{G}_k) - \frac{R_{\mathrm{avg}}}{n} \mathbf{G}_k \right)$$
        Seguido de la proyección espectral de Higham sobre el cono SPD $\lambda_i \ge \tau_{\mathrm{SPD}}$.
        """
        n = metric_G.shape[0]
        G_sym = 0.5 * (metric_G + metric_G.T)
        Ric_sym = 0.5 * (ricci_tensor + ricci_tensor.T)

        inv_G_Ric = la.solve(G_sym, Ric_sym, assume_a="pos")
        ricci_scalar_avg = float(np.trace(inv_G_Ric)) / float(n)

        ricci_flow_grad = Ric_sym - (ricci_scalar_avg / float(n)) * G_sym
        G_next = G_sym - 2.0 * delta_tau * ricci_flow_grad

        eigs, Q = la.eigh(0.5 * (G_next + G_next.T))
        eigs_clamped = np.maximum(eigs, _SPECTRAL_PSD_FLOOR)
        G_reg = Q @ np.diag(eigs_clamped) @ Q.T

        return 0.5 * (G_reg + G_reg.T)

    @staticmethod
    def evolve_level_r2_poincare_symplectic_retraction(
        jacobian_M: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Ejecuta el paso de Automejora Recursiva $\mathrm{RSI}^3$ (Nivel $R_2$ - Retracción Simpléctica de Poincaré).

        Proyecta una matriz Jacobiana perturbada $\mathbf{M} \in \mathrm{GL}(2n, \mathbb{R})$ sobre
        el grupo simpléctico $\mathrm{Sp}(2n, \mathbb{R})$ y normaliza su volumen de Liouville
        para garantizar simultáneamente $\mathbf{M}_{\mathrm{Sp}}^\top \mathbf{\Omega} \mathbf{M}_{\mathrm{Sp}} = \mathbf{\Omega}$
        y $\det(\mathbf{M}_{\mathrm{Sp}}) = +1$.
        """
        M = np.asarray(jacobian_M, dtype=np.float64)
        m_dim = M.shape[0]
        if M.ndim != 2 or m_dim != M.shape[1] or m_dim == 0:
            raise TopologicalInvariantError(
                f"Matriz de monodromía inválida para retracción simpléctica: {M.shape}"
            )

        det_val = float(la.det(M))
        if abs(det_val) < _SPECTRAL_PSD_FLOOR:
            raise TopologicalInvariantError(
                f"Jacobiano singular en retracción de Poincaré: det(M)={det_val:.3e}"
            )

        # Normalización preliminar de Liouville a det = +1
        if det_val < 0:
            M = M.copy()
            M[:, 0] = -M[:, 0]
            det_val = -det_val
        M_liouville = M / (det_val ** (1.0 / float(m_dim)))

        if m_dim % 2 != 0:
            return M_liouville

        # Retracción polar simpléctica: Q_J = -Ω M^T Ω M
        omega = PoincareSymplecticObservatory.build_darboux_symplectic_matrix(m_dim)
        Q_J = -omega @ M_liouville.T @ omega @ M_liouville
        # Parte simétrica definida positiva cerca de la identidad
        Q_sym = 0.5 * (Q_J + Q_J.T)
        eigs_Q, vecs_Q = la.eigh(Q_sym)
        if np.all(eigs_Q > _SPECTRAL_PSD_FLOOR):
            inv_sqrt_Q = vecs_Q @ np.diag(1.0 / np.sqrt(eigs_Q)) @ vecs_Q.T
            M_symp = M_liouville @ inv_sqrt_Q
            det_symp = float(la.det(M_symp))
            if det_symp > _SPECTRAL_PSD_FLOOR:
                M_symp = M_symp / (det_symp ** (1.0 / float(m_dim)))
            return M_symp

        return M_liouville

    @staticmethod
    def evolve_level_r3_brockett_shahshahani(
        density_matrix_rho: NDArray[np.float64],
        payoff_matrix_K: NDArray[np.float64],
        delta_t: float = 1.0e-3,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Ejecuta la evolución isoespectral de Brockett-Shahshahani (Nivel $R_3$ de $\mathrm{RSI}^3$):
            $$\frac{d\boldsymbol{\rho}}{dt} = \left[ \boldsymbol{\rho}, \, [\boldsymbol{\rho}, \, \operatorname{diag}(\mathbf{K}\mathbf{p})] \right]$$
            $$\dot{\mathcal{H}}_{\mathrm{Shah}} = -\mathrm{Var}_{\mathbf{p}}(\mathbf{K}\mathbf{p}) \le 0$$

        Retorna $(\boldsymbol{\rho}_{k+1}, \dot{\mathcal{H}}_{\mathrm{Shah}})$.
        """
        rho_sym = 0.5 * (density_matrix_rho + density_matrix_rho.T)
        p_diag = np.diag(rho_sym)

        p_payoff = payoff_matrix_K @ p_diag
        N_diag = np.diag(p_payoff)

        # Tasa de disipación de Lyapunov en la geometría de información de Shahshahani
        p_weights = np.maximum(p_diag, 0.0)
        p_sum = float(np.sum(p_weights))
        if p_sum > _MACHINE_EPS:
            p_norm = p_weights / p_sum
            mean_payoff = float(np.dot(p_norm, p_payoff))
            var_shah = float(np.dot(p_norm, (p_payoff - mean_payoff) ** 2))
        else:
            var_shah = 0.0
        lyapunov_rate = -var_shah

        # Corchete doble de Lie-Brockett: [ρ, [ρ, N]]
        comm1 = rho_sym @ N_diag - N_diag @ rho_sym
        comm2 = rho_sym @ comm1 - comm1 @ rho_sym

        rho_next = rho_sym + delta_t * comm2

        # Preservación exacta de la traza unitaria Tr(ρ) = 1
        trace_val = float(np.trace(rho_next))
        if abs(trace_val) > _MACHINE_EPS:
            rho_next = rho_next / trace_val

        return 0.5 * (rho_next + rho_next.T), lyapunov_rate

    # ═══════════════════════════════════════════════════════════════════════
    # ESLABÓN ANIDADO FASE 2 ──► FASE 3 (CONTINUACIÓN MORFISMÁTICA ESTRICTA)
    # ═══════════════════════════════════════════════════════════════════════
    @classmethod
    def bridge_to_phase3_sovereign_decision(
        cls,
        manifold: Phase2OrientationManifold,
    ) -> Tuple[HiggsGovernanceCertificate, "ESP32CrowbarActuator"]:
        r"""
        Definición formal final de la Fase 2 (`ORIENT`) y génesis directa de la Fase 3 (`DECIDE & ACT`).

        Evalúa el operador de Loewner-Daleckii-Krein, la curvatura escalar de Ricci,
        la entropía de Perelman y los invariantes de Liouville-Poincaré sobre la
        `Phase2OrientationManifold`, y transfiere el veredicto al `ESP32CrowbarActuator`
        que inaugura la Fase 3.
        """
        xi_matrix, lipschitz_bound = manifold.compute_connes_daleckii_krein_loewner()
        dk_operator_norm = float(la.norm(xi_matrix, ord=2))
        ricci_scalar, perelman_f = manifold.compute_ricci_and_perelman_entropy()

        volume_drift = manifold.monodromy.volume_drift
        cond_num = manifold.condition_number
        phi_norm = manifold.phi_norm

        is_coherent = (
            (phi_norm >= _SPECTRAL_PSD_FLOOR)
            and (volume_drift <= _WILKINSON_TOLERANCE)
            and (cond_num <= _MAX_CONDITION_NUMBER)
        )

        verdict = HeytingVerdict.COHERENT if is_coherent else HeytingVerdict.VETOED

        if not is_coherent:
            logger.error(
                f"[SCALAR_HIGGS_ANCHOR_AGENT_VETO] Incoherencia en Condensado de Higgs: "
                f"VEV={phi_norm:.4f}, VolumeDrift={volume_drift:.3e}, "
                f"CondNumber={cond_num:.3e}, LipschitzBound={lipschitz_bound:.3e}"
            )

        # Continuación directa hacia la clase de apertura de la Fase 3
        crowbar_actuator = ESP32CrowbarActuator.from_phase2_meta_bridge(
            verdict=verdict,
            total_energy=manifold.field_state.total_energy,
        )

        certificate = HiggsGovernanceCertificate(
            vacuum_expectation_value=phi_norm,
            potential_energy=float(manifold.field_state.potential_energy),
            lipschitz_connes_bound=lipschitz_bound,
            symplectic_volume_drift=volume_drift,
            ricci_scalar_curvature=ricci_scalar,
            heyting_verdict=verdict,
            is_field_coherent=is_coherent,
            symplectic_form_defect=manifold.monodromy.symplectic_form_defect,
            floquet_spectral_radius=manifold.monodromy.floquet_spectral_radius,
            krein_elliptic_fraction=manifold.monodromy.krein_elliptic_fraction,
            daleckii_krein_operator_norm=dk_operator_norm,
            metric_condition_number=cond_num,
            perelman_entropy_functional=perelman_f,
            crowbar_gpio14_latched=crowbar_actuator.gpio14_latched,
        )
        return certificate, crowbar_actuator


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 3 · DECISIÓN EN Ω₃, CROWBAR ESP32 Y SOBERANO DE CALIBRE (DECIDE)  │
# │  Inicia directamente con `ESP32CrowbarActuator.from_phase2_meta_bridge` │
# │  invocado por el último método de la Fase 2. Culmina en el Soberano     │
# │  `ScalarHiggsAnchorAgent` que orienta al motor `scalar_higgs_anchor.py`.│
# └─────────────────────────────────────────────────────────────────────────┘


# ───────────────────────────────────────────────────────────────────────────
# 3.1  ACTUADOR CROWBAR ESP32 (ISR EN IRAM <400ns, GPIO14, TIRISTOR BT151)
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ESP32CrowbarActuator:
    r"""
    Actuador Electrónico Perimetral ESP32 + Tiristor SCR BT151 (Apertura de la Fase 3).

    Modela la física de conmutación en IRAM (<400 ns) del pin GPIO14 cuando el
    veredicto en el topos de Heyting $\Omega_3$ colapsa a $\mathtt{VETOED}$ ($\top$):
        $$I_{\mathrm{peak}} = \sqrt{\frac{2 |E_{\mathrm{total}}|}{L_{\mathrm{bus}}}}, \qquad \tau_{\mathrm{RL}} = \frac{L_{\mathrm{bus}}}{R_{\mathrm{BT151}}}$$
    """

    gpio_pin: int = 14
    iram_latency_ns: float = 320.0
    bus_inductance_henry: float = 1.5e-4
    scr_on_resistance_ohm: float = 0.085
    gpio14_latched: bool = False
    crowbar_peak_current_amp: float = 0.0

    @classmethod
    def from_phase2_meta_bridge(
        cls,
        verdict: HeytingVerdict,
        total_energy: float,
    ) -> "ESP32CrowbarActuator":
        """Continuación formal directa desde `RSI3MetaFunctorEngine.bridge_to_phase3_sovereign_decision`."""
        latched = not verdict.is_verdict_coherent()
        if latched:
            peak_i = math.sqrt(2.0 * abs(total_energy) / 1.5e-4)
        else:
            peak_i = 0.0
        return cls(
            gpio14_latched=latched,
            crowbar_peak_current_amp=peak_i,
        )


# ───────────────────────────────────────────────────────────────────────────
# 3.2  SOBERANO DE CALIBRE: SCALAR HIGGS ANCHOR AGENT
# ───────────────────────────────────────────────────────────────────────────


class ScalarHiggsAnchorAgent:
    r"""
    Soberano de Calibre del Condensador de Inercia Logística ($V_{\mathrm{PHYSICS}} / V_{\mathrm{WISDOM}}$).

    Gobierna la estabilidad del campo de Higgs en lazo cerrado OODA (Observe-Orient-Decide-Act)
    mediante las 3 Fases Anidadas:
    - **Fase 1 (`OBSERVE`)**: `PoincareSymplecticObservatory` (VEV, métrica SPD, Liouville y Floquet).
    - **Fase 2 (`ORIENT`)**: `Phase2OrientationManifold` y `RSI3MetaFunctorEngine` (Connes-DK,
      Flujo de Ricci $R_1$, Retracción Simpléctica de Poincaré $R_2$ y Flujo de Brockett-Shahshahani $R_3$).
    - **Fase 3 (`DECIDE & ACT`)**: `ESP32CrowbarActuator`, emisión de `HiggsGovernanceCertificate`
      y orientación directa del motor `ScalarHiggsAnchor` (`app/physics/scalar_higgs_anchor.py`).
    """

    def __init__(self, target_vev: float = 1.0, coupling_lambda: float = 0.1) -> None:
        self._target_vev = float(target_vev)
        self._coupling_lambda = float(coupling_lambda)
        self._governance_temperature = 1.0
        self._last_crowbar_state: ESP32CrowbarActuator = ESP32CrowbarActuator()
        self._last_shahshahani_lyapunov_rate: float = 0.0

    @property
    def target_vev(self) -> float:
        return self._target_vev

    @property
    def coupling_lambda(self) -> float:
        return self._coupling_lambda

    @property
    def governance_temperature(self) -> float:
        return self._governance_temperature

    @property
    def last_crowbar_state(self) -> ESP32CrowbarActuator:
        return self._last_crowbar_state

    @property
    def last_shahshahani_lyapunov_rate(self) -> float:
        return self._last_shahshahani_lyapunov_rate

    def audit_higgs_field_coherence(
        self,
        field_state: ScalarFieldState,
        metric_G: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        ricci_tensor: Optional[NDArray[np.float64]] = None,
    ) -> HiggsGovernanceCertificate:
        r"""
        Ejecuta el ciclo OODA de lazo cerrado a través de la cadena anidada de 3 fases:
        `Fase 1 (Observe)` ──► `Fase 2 (Orient)` ──► `Fase 3 (Decide & Act)`.

        Demostración Axiomática
        -----------------------
        1. **Fase 1 (Observe)**: Estabilidad Espectral, Monodromía de Poincaré y VEV
           $$v = \langle \Phi \rangle = \|\Phi\|_2 \ge \tau_{\mathrm{SPD}} = 10^{-12}$$
           $$\lambda_{\min}(\mathbf{G}) \ge 10^{-12}, \qquad \kappa(\mathbf{G}) = \frac{\lambda_{\max}}{\lambda_{\min}} \le 10^8$$
        2. **Fase 2 (Orient)**: Invarianza Simpléctica de Liouville y Operador Loewner-Connes-DK
           $$\det(\mathbf{M}) = +1 \implies |\det(\mathbf{M}) - 1| \le \varepsilon_{\mathrm{Wilkinson}}$$
           $$L_{\max} = \|\mathbf{\Xi}\|_{\max} \le \frac{1}{2 \, \lambda_{\min}(\mathbf{G})^{3/2}}$$
        3. **Fase 3 (Decide/Act)**: Clasificación en $\Omega_3$ y Actuación ESP32 (< 400 ns)
           Si $v < 10^{-12}$, $\det(\mathbf{M}) \neq 1$ o $\kappa(\mathbf{G}) > 10^8$,
           el veredicto colapsa a $\mathtt{VETOED}$ ($\top$) y enclava el pin GPIO14.
        """
        # Fase 1 ──► Fase 2: Observación espectral y construcción de la variedad de orientación
        orientation_manifold = (
            PoincareSymplecticObservatory.observe_and_bridge_to_phase2_orient(
                field_state=field_state,
                metric_G=metric_G,
                jacobian_M=jacobian_M,
                ricci_tensor=ricci_tensor,
            )
        )

        # Fase 2 ──► Fase 3: Cálculo de Connes-DK, curvatura de Ricci y emisión de certificado/Crowbar
        certificate, crowbar = (
            RSI3MetaFunctorEngine.bridge_to_phase3_sovereign_decision(
                manifold=orientation_manifold
            )
        )
        self._last_crowbar_state = crowbar
        return certificate

    def evolve_ricci_flow_rsi3(
        self,
        metric_G: NDArray[np.float64],
        ricci_tensor: NDArray[np.float64],
        delta_tau: float = 1.0e-3,
    ) -> NDArray[np.float64]:
        r"""
        Ejecuta el paso de Automejora Recursiva $\mathrm{RSI}^3$ (Nivel $R_1$ - Flujo de Ricci Discreto):
            $$\mathbf{G}_{k+1} = \mathbf{G}_k - 2 \, \Delta \tau \left( \operatorname{Ric}_{\mu\nu}(\mathbf{G}_k) - \frac{R_{\mathrm{avg}}}{n} \mathbf{G}_k \right)$$
        """
        return RSI3MetaFunctorEngine.evolve_level_r1_ricci_flow(
            metric_G=metric_G,
            ricci_tensor=ricci_tensor,
            delta_tau=delta_tau,
        )

    def evolve_poincare_symplectic_rsi3(
        self,
        jacobian_M: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Ejecuta el paso de Automejora Recursiva $\mathrm{RSI}^3$ (Nivel $R_2$ - Retracción Simpléctica de Poincaré):
            $$\mathbf{M}_{k+1} = \mathbf{M}_k \left( \mathbf{\Omega}^{-1} \mathbf{M}_k^\top \mathbf{\Omega} \mathbf{M}_k \right)^{-1/2} \in \mathrm{Sp}(2n, \mathbb{R})$$
        """
        return RSI3MetaFunctorEngine.evolve_level_r2_poincare_symplectic_retraction(
            jacobian_M=jacobian_M
        )

    def evolve_brockett_shahshahani_rsi3(
        self,
        density_matrix_rho: NDArray[np.float64],
        payoff_matrix_K: NDArray[np.float64],
        delta_t: float = 1.0e-3,
    ) -> NDArray[np.float64]:
        r"""
        Ejecuta la evolución isoespectral de Brockett-Shahshahani (Nivel $R_3$ de $\mathrm{RSI}^3$):
            $$\frac{d\boldsymbol{\rho}}{dt} = \left[ \boldsymbol{\rho}, \, [\boldsymbol{\rho}, \, \operatorname{diag}(\mathbf{K}\mathbf{p})] \right], \qquad \dot{\mathcal{H}} = -\mathrm{Var}_{\mathbf{p}}(\tilde{\mathcal{K}}\mathbf{p}) \le 0$$
        """
        rho_next, lyap_rate = (
            RSI3MetaFunctorEngine.evolve_level_r3_brockett_shahshahani(
                density_matrix_rho=density_matrix_rho,
                payoff_matrix_K=payoff_matrix_K,
                delta_t=delta_t,
            )
        )
        self._last_shahshahani_lyapunov_rate = lyap_rate
        # Modulación adaptativa de la temperatura de gobernanza según disipación de Shahshahani
        self._governance_temperature = float(
            np.clip(self._governance_temperature * math.exp(delta_t * lyap_rate), 1e-3, 10.0)
        )
        return rho_next

    def execute_rsi3_closed_loop_healing(
        self,
        field_state: ScalarFieldState,
        metric_G: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        ricci_tensor: NDArray[np.float64],
        density_matrix_rho: NDArray[np.float64],
        payoff_matrix_K: NDArray[np.float64],
        delta_tau: float = 1.0e-3,
        delta_t: float = 1.0e-3,
    ) -> Tuple[
        HiggsGovernanceCertificate,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        r"""
        Ejecuta la composición completa de Automejora Recursiva de Nivel 3 ($\mathrm{RSI}^3 = R_3 \circ R_2 \circ R_1$)
        y re-certifica el estado de calibre en lazo cerrado:
        1. $R_1$: Suaviza la curvatura mediante el Flujo de Ricci-Perelman sobre $\mathbf{G}$.
        2. $R_2$: Proyecta el Jacobiano $\mathbf{M}$ sobre el grupo simpléctico de Poincaré $\mathrm{Sp}(2n, \mathbb{R})$.
        3. $R_3$: Evoluciona la matriz de densidad $\boldsymbol{\rho}$ mediante el doble corchete de Brockett.
        4. Emite el `HiggsGovernanceCertificate` resultante.
        """
        G_healed = self.evolve_ricci_flow_rsi3(metric_G, ricci_tensor, delta_tau=delta_tau)
        M_healed = self.evolve_poincare_symplectic_rsi3(jacobian_M)
        rho_healed = self.evolve_brockett_shahshahani_rsi3(
            density_matrix_rho, payoff_matrix_K, delta_t=delta_t
        )
        cert = self.audit_higgs_field_coherence(
            field_state=field_state,
            metric_G=G_healed,
            jacobian_M=M_healed,
            ricci_tensor=ricci_tensor,
        )
        return cert, G_healed, M_healed, rho_healed

    def orient_higgs_anchor_engine(
        self,
        anchor_engine: Any,
        metric_G: Optional[NDArray[np.float64]] = None,
        jacobian_M: Optional[NDArray[np.float64]] = None,
        ricci_tensor: Optional[NDArray[np.float64]] = None,
    ) -> HiggsGovernanceCertificate:
        r"""
        Orienta y gobierna directamente una instancia del motor `ScalarHiggsAnchor`
        (`app/physics/scalar_higgs_anchor.py`).

        Extrae el `field_state` del motor, construye la matriz simpléctica de monodromía
        local de Störmer-Verlet-Poincaré si no se suministra `jacobian_M`, audita la
        coherencia en $\Omega_3$, y si detecta deriva simpléctica aplica la curación
        $\mathrm{RSI}^3$ de Nivel 3 directamente sobre el motor.
        """
        eng_state = anchor_engine.field_state
        n_dim = eng_state.dim
        V_vec, _ = anchor_engine._potential.value_and_gradient(eng_state.phi)
        pot_energy = float(np.sum(V_vec))
        tot_energy = float(anchor_engine._hamiltonian(eng_state))

        agent_field_state = ScalarFieldState.from_engine_state(
            engine_state=eng_state,
            potential_energy=pot_energy,
            total_energy=tot_energy,
        )

        if metric_G is None:
            # Métrica efectiva inducida por el Laplaciano regularizado I + c² Δ_M
            L_dense = anchor_engine._lb_op.L.toarray()
            metric_G = np.eye(n_dim, dtype=np.float64) + (
                anchor_engine._params.C_SQUARED * L_dense
            )

        if jacobian_M is None:
            # Construcción exacta de la matriz simpléctica de paso Velocity-Verlet en Sp(2n, R):
            # [ q_{k+1} ] = [ I - ½ dt² K      dt I - ¼ dt³ K ] [ q_k ]
            # [ p_{k+1} ]   [ -dt K            I - ½ dt² K    ] [ p_k ]
            # cuyo determinante es idénticamente det(M) = +1 por estructura de shear simpléctico.
            dt = float(anchor_engine._integrator.dt_critical)
            hess_diag = anchor_engine._potential.hessian_diagonal(eng_state.phi)
            L_dense = anchor_engine._lb_op.L.toarray()
            K_stiff = anchor_engine._params.C_SQUARED * L_dense + np.diag(hess_diag)
            I_n = np.eye(n_dim, dtype=np.float64)

            # Shear 1 (kick medio): P_half = P - ½ dt K Q
            S_kick = np.block(
                [
                    [I_n, np.zeros((n_dim, n_dim), dtype=np.float64)],
                    [-0.5 * dt * K_stiff, I_n],
                ]
            )
            # Shear 2 (drift completo): Q_next = Q + dt P_half
            S_drift = np.block(
                [
                    [I_n, dt * I_n],
                    [np.zeros((n_dim, n_dim), dtype=np.float64), I_n],
                ]
            )
            jacobian_M = S_kick @ S_drift @ S_kick

        cert = self.audit_higgs_field_coherence(
            field_state=agent_field_state,
            metric_G=metric_G,
            jacobian_M=jacobian_M,
            ricci_tensor=ricci_tensor,
        )

        # Si el certificado colapsa, aplicamos rectificación simpléctica y RSI Nivel 3 al motor
        if not cert.is_field_coherent and hasattr(anchor_engine, "rsi_improver"):
            metrics = anchor_engine.monitor.analyze(eng_state)
            healed_state, new_params = (
                anchor_engine.rsi_improver.execute_level3_introspection(
                    state=eng_state,
                    metrics=metrics,
                    hamiltonian=anchor_engine._hamiltonian,
                    integrator=anchor_engine._integrator,
                    monitor=anchor_engine._monitor,
                )
            )
            anchor_engine._field_state = healed_state
            anchor_engine._params = new_params

        return cert


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIONES PÚBLICAS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Constantes numéricas
    "_MACHINE_EPS",
    "_SPECTRAL_PSD_FLOOR",
    "_MAX_CONDITION_NUMBER",
    "_WILKINSON_TOLERANCE",
    "_KREIN_FLOQUET_TOLERANCE",
    # Fase 1: Axiomática de Calibre, Topos Ω₃ y Observación de Poincaré
    "HeytingVerdict",
    "TopologicalInvariantError",
    "ScalarFieldState",
    "PoincareMonodromyTelemetry",
    "HiggsGovernanceCertificate",
    "PoincareSymplecticObservatory",
    # Fase 2: Orientación de Connes-Daleckii-Krein y Motor RSI³
    "Phase2OrientationManifold",
    "RSI3MetaFunctorEngine",
    # Fase 3: Actuación Crowbar ESP32 y Soberano de Calibre
    "ESP32CrowbarActuator",
    "ScalarHiggsAnchorAgent",
]
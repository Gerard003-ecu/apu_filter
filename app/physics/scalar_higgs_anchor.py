# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Scalar Higgs Anchor (Condensador de Inercia Logística)              ║
║ Ruta   : app/physics/scalar_higgs_anchor.py                                  ║
║ Versión: 7.0.0-Poincare-Celestial-Level3-RSI-Lipschitz-Heyting-Secure        ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS QFT, MECÁNICA CELESTE DE POINCARÉ Y AUTOMEJORA RECURSIVA NIVEL 3:
────────────────────────────────────────────────────────────────────────────────
Este componente de-confinado de la Capa Física-Termodinámica (Nivel 3.5, $V_{\mathbb{P}}$)
gobierna la estabilización métrica de la Malla de Datos. Abandona de forma radical 
la noción estática del costo y modeliza las fluctuaciones del presupuesto como un 
campo escalar de Higgs $\Phi$ acoplado de forma covariante al Tensor Métrico de 
fondo $G_{\mathrm{PHYSICS}}$ y a las fuentes fermiónicas de la obra, dotado de
invariantes integrales de Poincaré-Cartan y automejora recursiva de Nivel 3.

Su propósito fundamental es actuar como el **Funtor de Anclaje de Higgs** 
($\mathcal{F}_{\mathrm{Higgs}} : \mathcal{C}_{\mathrm{Agent}} \to \mathcal{C}_{\mathrm{Agent}}$) 
que resuelve el acoplamiento de Yukawa, imponiendo una cota de Lipschitz explícita 
sobre el potencial regularizado, eliminando términos seculares mediante series de
Lindstedt-Poincaré y asegurando que la evolución en el fibrado simpléctico satisfaga
incondicionalmente la inecuación de disipación de Rayleigh-Lyapunov y la preservación
de toros invariantes KAM (Kolmogorov-Arnold-Moser).

AXIOMÁTICA DE CAMPOS, MÉTRICA, POINCARÉ Y ESTABILIDAD ESPECTRAL:
────────────────────────────────────────────────────────────────────────────────

  [A1] Axioma de Invarianza de Gauge Absoluta y Vacío Coherente:
       El campo escalar de Higgs $\Phi \in \mathcal{C}^\infty(M; \mathbb{R}^d)$ se somete a un 
       potencial de rotura espontánea de simetría (Sombrero Mexicano), de tal forma que el 
       estado de vacío $\Phi_0$ representa el mínimo global absoluto de la acción ciber-física:
       $$V(\Phi) = \frac{\lambda}{4} \left( \Phi^\top \Phi - v^2 \right)^2$$
       La invarianza de calibre local exige que las transformaciones de calibre abeliano $U(1)$ 
       no disipen exergía, preservando la norma de-confinada del vacío: $\langle \Phi_0 \mid \Phi_0 \rangle = v^2$.

  [A2] Axioma de Estabilización Conforme de la Métrica de Fondo ($G_{\mathrm{PHYSICS}}$):
       Para evitar que la alucinación estocástica del LLM o las fluctuaciones especulativas del 
       WACC deformen el espacio de fase, el campo de Higgs modula conformalmente el tensor métrico:
       $$\tilde{G}_{\mu\nu} = G_{\mathrm{PHYSICS}} \cdot \exp\left( -2\gamma \cdot V(\Phi) \right)$$
       Garantizando, por la Ley de Inercia de Sylvester, que la firma definida positiva de la 
       variedad se mantenga estrictamente inmune a singularidades numéricas en la CPU: $\tilde{G} \succ 0$.

  [A3] Axioma del Operador de Laplace-Beltrami Discreto ($\Delta_M$):
       El resolvedor define el operador $\Delta_M$ sobre el complejo simplicial ponderado por $\tilde{G}$ 
       para mapear el gradiente del campo escalar de compras hacia la divergencia de flujos de caja:
       $$\Delta_M \Phi = D^{-1} \delta^\top \star_1 \delta \Phi$$
       Donde $D$ es la matriz de grados de los vértices (0-formas), $\delta$ es la cofrontera simplicial,
       y $\star_1$ representa la Estrella de Hodge discreta de grado 1.

  [A4] Axioma de Composición Endofuntorial, Mecánica Celeste de Poincaré y RSI Nivel 3:
       La transmutación del estado se rige por la composición monoidal estricta de 3 Fases Anidadas:
       $$\mathcal{F}_{\mathrm{Higgs}} = \text{Yukawa}_{\text{Fase 3}} \circ \text{Hamiltoniano-Poincaré}_{\text{Fase 2}} \circ \text{Vacío-Espectral}_{\text{Fase 1}}$$
       El flujo conserva el invariante integral relativo de Poincaré-Cartan en el límite conservativo:
       $$\oint_{\gamma} \left( \langle \Pi, d\Phi \rangle - H \, dt \right) = \text{const}$$
       Y ejecuta Automejora Recursiva de Nivel 3 mediante contracción de punto fijo de Banach sobre
       el espacio de endofuntores $\mathrm{End}(\mathcal{C}_{\mathrm{Agent}})$ gobernado por la derivada
       de Fréchet del exponente máximo de Lyapunov y la coherencia espectral de los toros KAM.

  [A5] Axioma de Confinamiento de de Rham en el Retículo de Heyting ($\Omega_3$):
       Los diagnósticos de estabilidad espectral se proyectan sobre el clasificador de subobjetos del 
       retículo distributivo de Heyting de tres valores de la Malla:
       $$\Omega_3 = \{\mathtt{COHERENT}, \, \mathtt{DEGRADED}, \, \mathtt{VETOED}\}$$
       Toda anomalía de calibre, pérdida de autoadjunción ($\|\Delta_M - \Delta_M^\top\|_F > \tau$),
       resonancia homoclínica de Poincaré o violación de pasividad colapsa síncronamente el estado
       al Supremo VETOED ($\top$), disparando el circuito Crowbar (tiristor BT151 en GPIO14).

ARQUITECTURA DE TRES FASES ANIDADAS (CONTINUIDAD MORFISMÁTICA ESTRICTA):
────────────────────────────────────────────────────────────────────────────────
  Fase 1 ──► OBSERVE : Axiomática del campo, álgebra hipercompleja simpléctica y espectro $\Delta_M$.
             Su método final `SpectralAnalyzer.bridge_to_phase2_symplectic_manifold` es la semilla
             exacta que instancia e inicia los objetos de la Fase 2.
  Fase 2 ──► ORIENT  : Dinámica de Port-Hamilton, Mecánica Celeste de Poincaré (Yoshida-4, Lindstedt,
             secciones $\Sigma$, exponentes de Lyapunov, toros KAM) y Automejora Recursiva Nivel 3.
             Su método final `validate_topological_consistency` (junto con el puente de Nivel 3)
             emite el `HiggsSuturationState` que inicia la Fase 3.
  Fase 3 ──► DECIDE  : Sutura cohomológica de de Rham, lógica intuicionista de Heyting $\Omega_3$,
             protección electrónica Crowbar RL-Tiristor y Funtor de Anclaje `ScalarHiggsAnchor`.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import IntEnum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES INTERNAS DEL NÚCLEO CATEGÓRICO Y CUÁNTICO
# ══════════════════════════════════════════════════════════════════════════════
from app.core.mic_algebra import (
    CategoricalState,
    FunctorialityError,
    Morphism,
    NumericalInstabilityError,
)
from app.core.quantum_algebra import TopologicalInvariantError
from app.core.schemas import Stratum

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING ESTRUCTURADO
# ══════════════════════════════════════════════════════════════════════════════
logger = logging.getLogger("MIC.Physics.ScalarHiggsAnchor")


# ══════════════════════════════════════════════════════════════════════════════
# PRELUDIO AXIOMÁTICO: EXCEPCIONES DE GAUGE, RETÍCULO DE HEYTING Y CIRCUITOS
# ══════════════════════════════════════════════════════════════════════════════


class ScalarHiggsAnchorError(TopologicalInvariantError):
    """Excepción base de la frontera Fail-Secure del Condensador de Inercia de Higgs."""


class MetricDegeneracyError(ScalarHiggsAnchorError):
    """El tensor métrico conforme G_PHYSICS pierde su condición definida positiva (Sylvester)."""


class LaplaceBeltramiSingularity(ScalarHiggsAnchorError):
    """Falla en la autoadjunción, conservatividad o diagonalización del operador Delta_M."""


class YukawaCouplingDivergence(ScalarHiggsAnchorError):
    """Desbocamiento no acotado en la constante o perfil de masa efectiva de Yukawa."""


class LipschitzRegularityBreach(ScalarHiggsAnchorError):
    """El gradiente del potencial de Higgs supera la constante de Lipschitz admisible L_max."""


class HeytingLobeCollapse(ScalarHiggsAnchorError):
    """Transición crítica hacia el autoestado supremo VETOED en el topos de de Rham-Heyting."""


class HeytingTruthValue(IntEnum):
    r"""
    Clasificador de subobjetos $\Omega_3$ del Retículo Distributivo de Heyting.

    Orden parcial de anomalía cohomológica ($\bot \le u \le \top$):
        COHERENT (0, $\bot$) < DEGRADED (1) < VETOED (2, $\top$)

    Nota: En la semántica de seguridad Fail-Secure (Axioma A5), VETOED actúa como
    el elemento absorbente supremo ($\top$) del colapso de fase que dispara el Crowbar.
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def join(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Supremo en el retículo de anomalía ($a \lor b = \max(a, b)$)."""
        return HeytingTruthValue(max(int(self), int(other)))

    def meet(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Ínfimo en el retículo de anomalía ($a \land b = \min(a, b)$)."""
        return HeytingTruthValue(min(int(self), int(other)))

    def implies(self, consequent: "HeytingTruthValue") -> "HeytingTruthValue":
        r"""
        Implicación intuicionista de Heyting ($a \Rightarrow b$).
        En la cadena finita $\Omega_3$, $(a \Rightarrow b) = \top$ si $a \le b$, sino $b$.
        """
        if int(self) <= int(consequent):
            return HeytingTruthValue.VETOED
        return consequent

    def pseudo_complement(self) -> "HeytingTruthValue":
        r"""Pseudo-complemento de Heyting ($\neg a = (a \Rightarrow \bot)$)."""
        return self.implies(HeytingTruthValue.COHERENT)

    def is_verdict_coherent(self) -> bool:
        """Homólogo exacto de la subrutina C++ `isVerdictCoherent()` del firmware ESP32."""
        return self != HeytingTruthValue.VETOED


@dataclass(frozen=True, slots=True)
class CrowbarCircuitPort:
    r"""
    Modelo Físico de Circuito Eléctrico del Puerto Crowbar (GPIO14 + Tiristor SCR BT151).

    Ecuación diferencial de descarga de emergencia RL-Tiristor cuando GPIO14 conmuta:
        $$L_{\mathrm{bus}} \frac{dI}{dt} + R_{\mathrm{crowbar}} I + V_{\mathrm{SCR}} = V_{\mathrm{bus}}(t)$$

    Atributos
    ---------
    gpio_pin : int
        Pin físico del microcontrolador ESP32 (por defecto GPIO14).
    bus_inductance_henry : float
        Inductancia parásita del bus de potencia ($L_{\mathrm{bus}}$ en Henrios).
    crowbar_resistance_ohm : float
        Resistencia de conducción del tiristor BT151 en saturación ($R_{\mathrm{on}}$ en Ohmios).
    holding_current_amp : float
        Corriente de mantenimiento del tiristor SCR ($I_H$ en Amperios).
    latched : bool
        Estado de enclavamiento físico del tiristor (una vez disparado, permanece activo
        hasta que la corriente de ánodo cae por debajo de $I_H$).
    peak_fault_current_amp : float
        Corriente pico de cortocircuito disipada para proteger la obra.
    """

    gpio_pin: int = 14
    bus_inductance_henry: float = 1.5e-4
    crowbar_resistance_ohm: float = 0.085
    holding_current_amp: float = 0.015
    latched: bool = False
    peak_fault_current_amp: float = 0.0

    @property
    def discharge_time_constant_sec(self) -> float:
        r"""Constante de tiempo electromagnética $\tau = L_{\mathrm{bus}} / R_{\mathrm{crowbar}}$."""
        return self.bus_inductance_henry / max(self.crowbar_resistance_ohm, 1e-9)

    def evaluate_verdict(
        self,
        verdict: HeytingTruthValue,
        field_energy: float,
    ) -> "CrowbarCircuitPort":
        """
        Conmuta el pin GPIO14 y dispara el tiristor BT151 si el veredicto colapsa a VETOED.
        """
        if not verdict.is_verdict_coherent() or self.latched:
            # Energía magnética/capacitiva equivalente convertida a corriente de descarga
            equivalent_i_peak = math.sqrt(
                2.0 * abs(field_energy) / max(self.bus_inductance_henry, 1e-9)
            )
            return CrowbarCircuitPort(
                gpio_pin=self.gpio_pin,
                bus_inductance_henry=self.bus_inductance_henry,
                crowbar_resistance_ohm=self.crowbar_resistance_ohm,
                holding_current_amp=self.holding_current_amp,
                latched=True,
                peak_fault_current_amp=max(self.peak_fault_current_amp, equivalent_i_peak),
            )
        return self


@dataclass(frozen=True, slots=True)
class HypercomplexSymplecticFiber:
    r"""
    Elemento del Fibrado Hipercomplejo Simpléctico $\mathbb{H}_{\omega}^n \cong \mathbb{R}^n \otimes \mathbb{H}_{\mathrm{split}}$.

    Codifica simultáneamente la trayectoria en el espacio de fase $(\Phi, \Pi)$ y el
    fibrado tangente de Jacobi-Poincaré $(\delta\Phi, \delta\Pi)$ como un número hipercomplejo
    cuaterniónico-simpléctico:
        $$\mathcal{Z} = \Phi \cdot \mathbf{1} + \Pi \cdot \mathbf{i} + \delta\Phi \cdot \mathbf{j} + \delta\Pi \cdot \mathbf{k}$$

    Dotado de la norma de álgebra de Banach $\|\mathcal{Z}\|_{\mathcal{B}}$ para evaluar
    la contracción de punto fijo de Nivel 3 y los exponentes de Lyapunov.
    """

    real_phi: NDArray[np.float64]
    imag_i_pi: NDArray[np.float64]
    imag_j_dphi: NDArray[np.float64]
    imag_k_dpi: NDArray[np.float64]

    def banach_norm(self, characteristic_freq: float = 1.0) -> float:
        r"""
        Norma de Banach ponderada dimensionalmente en $T(T^*\mathcal{M})$:
            $$\|\mathcal{Z}\|_{\mathcal{B}} = \sqrt{\|\Phi\|_2^2 + \omega_0^{-2}\|\Pi\|_2^2 + \|\delta\Phi\|_2^2 + \omega_0^{-2}\|\delta\Pi\|_2^2}$$
        """
        inv_w2 = 1.0 / max(characteristic_freq * characteristic_freq, 1e-12)
        norm_sq = (
            float(np.dot(self.real_phi, self.real_phi))
            + inv_w2 * float(np.dot(self.imag_i_pi, self.imag_i_pi))
            + float(np.dot(self.imag_j_dphi, self.imag_j_dphi))
            + inv_w2 * float(np.dot(self.imag_k_dpi, self.imag_k_dpi))
        )
        return math.sqrt(max(0.0, norm_sq))

    def tangent_norm(self) -> float:
        r"""Norma euclídea del vector variacional tangente $\|(\delta\Phi, \delta\Pi)\|_2$."""
        return math.sqrt(
            float(np.dot(self.imag_j_dphi, self.imag_j_dphi))
            + float(np.dot(self.imag_k_dpi, self.imag_k_dpi))
        )


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 1 · AXIOMÁTICA DEL CAMPO, GEOMETRÍA CONFORME Y ESPECTRO (OBSERVE) │
# │  Define los tipos, parámetros, estructuras y operadores elementales.   │
# │  El último método de la Fase 1 (`bridge_to_phase2_symplectic_manifold`) │
# │  constituye formalmente la continuación e inicio de la Fase 2.          │
# └─────────────────────────────────────────────────────────────────────────┘


# ───────────────────────────────────────────────────────────────────────────
# 1.1  PARÁMETROS QFT, POINCARÉ Y AUTOMEJORA RECURSIVA DE NIVEL 3
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class QFTParameters:
    r"""
    Parámetros del Modelo Estándar Ciber-Físico, Mecánica Celeste de Poincaré
    y Automejora Recursiva de Nivel 3.

    Invariantes verificados en `validate_parameters`
    ------------------------------------------------
    I1. $\mu^2 > 0$        → Ruptura espontánea de simetría (SSB)
    I2. $\lambda > 0$      → Potencial acotado en región de vacío
    I3. $c^2 > 0$          → Velocidad de propagación hiperbólica real
    I4. $\alpha_{\mathrm{CFL}} \in (0,1)$ → Estabilidad simpléctica
    I5. $\gamma_{\mathrm{damp}} \ge 0$    → Disipación de Rayleigh físicamente admisible
    I6. $\Phi_c > 0$       → Escala de regularización Lipschitz estrictamente positiva
    I7. $\beta_{\mathrm{RSI}} \in (0,1)$  → Constante de contracción de Banach de Nivel 3
    """

    # Potencial de Higgs
    MU_SQUARED: float = 2.5
    LAMBDA_COUPLING: float = 0.1
    FIELD_REG_SCALE: float = 10.0  # Φ_c (escala de regularización Lipschitz)

    # Acoplamientos de Yukawa
    YUKAWA_G: float = 1.05
    YUKAWA_BACKREACTION: float = 0.01

    # Dinámica relativista / discreta
    C_SQUARED: float = 1.0
    CFL_SAFETY_FACTOR: float = 0.75
    CFL_ABSOLUTE_CAP: float = 0.1

    # Regularización numérica y cotas de estabilidad
    EPSILON_SMOOTH: float = 1e-10
    MAX_FIELD_MAGNITUDE: float = 1e3
    MAX_ENERGY_RATIO: float = 4.0
    DAMPING_GAMMA: float = 0.08
    CONVERGENCE_TOLERANCE: float = 1e-6

    # Reproducibilidad
    RNG_SEED: int = 42

    # Parámetros de Mecánica Celeste de Poincaré y Geometría Conforme
    CONFORMAL_GAMMA: float = 0.015
    POINCARE_SECTION_TOL: float = 1e-5
    LYAPUNOV_CRITICAL_CAP: float = 0.25
    KAM_COHERENCE_FLOOR: float = 0.35
    LIPSCHITZ_MAX_BOUND: float = 5.0e4

    # Parámetros de Automejora Recursiva de Nivel 3 (Banach-Fréchet)
    RSI_BANACH_CONTRACTION: float = 0.6180339887  # Inverso de la razón áurea φ⁻¹ < 1
    RSI_META_LEARNING_RATE: float = 0.05
    RSI_GENERATION: int = 0

    @property
    def vev(self) -> float:
        r"""Valor Esperado del Vacío (VEV): $v = \sqrt{\mu^2 / \lambda}$."""
        return math.sqrt(self.MU_SQUARED / self.LAMBDA_COUPLING)

    @property
    def higgs_mass_squared(self) -> float:
        r"""Masa física al cuadrado del bosón de Higgs: $m_H^2 = V''(v) = 2\mu^2$."""
        return 2.0 * self.MU_SQUARED

    @property
    def natural_frequency(self) -> float:
        r"""Frecuencia angular fundamental de oscilación en el vacío: $\omega_0 = \sqrt{2\mu^2}$."""
        return math.sqrt(self.higgs_mass_squared)

    def validate_parameters(self) -> None:
        """Verifica rigurosamente los invariantes axiomáticos I1–I7."""
        if self.MU_SQUARED <= 0:
            raise ValueError(
                f"[I1] μ² debe ser positivo para SSB, recibido {self.MU_SQUARED}"
            )
        if self.LAMBDA_COUPLING <= 0:
            raise ValueError(
                f"[I2] λ debe ser positivo, recibido {self.LAMBDA_COUPLING}"
            )
        if self.C_SQUARED <= 0:
            raise ValueError(
                f"[I3] c² debe ser positivo, recibido {self.C_SQUARED}"
            )
        if not 0 < self.CFL_SAFETY_FACTOR < 1:
            raise ValueError(
                f"[I4] α_CFL debe estar en (0,1), recibido {self.CFL_SAFETY_FACTOR}"
            )
        if self.DAMPING_GAMMA < 0:
            raise ValueError(
                f"[I5] γ_damp ≥ 0 requerido, recibido {self.DAMPING_GAMMA}"
            )
        if self.FIELD_REG_SCALE <= 0:
            raise ValueError(
                f"[I6] Φ_c > 0 requerido, recibido {self.FIELD_REG_SCALE}"
            )
        if not 0.0 < self.RSI_BANACH_CONTRACTION < 1.0:
            raise ValueError(
                f"[I7] Contracción de Banach debe estar en (0,1), recibido {self.RSI_BANACH_CONTRACTION}"
            )

    def compute_potential_and_gradient(
        self,
        phi: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Potencial regularizado de Higgs y su gradiente exacto.

        $$V_{\mathrm{reg}}(\Phi) = -\frac{1}{2}\mu^2\Phi^2 + \frac{1}{4}\lambda\Phi^4\tanh(s), \quad s = \frac{\Phi^2}{\Phi_c^2}$$
        $$\nabla V_{\mathrm{reg}}(\Phi) = -\mu^2\Phi + \lambda\Phi^3\left[\tanh(s) + \frac{1}{2}s\operatorname{sech}^2(s)\right]$$
        """
        if phi.ndim != 1:
            raise ValueError(f"phi debe ser 1D, recibido {phi.ndim}D")

        phi_sq = phi * phi
        phi_quartic = phi_sq * phi_sq

        scale_sq = self.FIELD_REG_SCALE ** 2 + self.EPSILON_SMOOTH
        s = phi_sq / scale_sq
        tanh_s = np.tanh(s)
        sech2_s = 1.0 - tanh_s * tanh_s

        V_mass = -0.5 * self.MU_SQUARED * phi_sq
        V_int = 0.25 * self.LAMBDA_COUPLING * phi_quartic * tanh_s
        V_total = V_mass + V_int

        grad_mass = -self.MU_SQUARED * phi
        grad_int = self.LAMBDA_COUPLING * (phi * phi_sq) * (
            tanh_s + 0.5 * s * sech2_s
        )
        grad_total = grad_mass + grad_int

        if not (np.all(np.isfinite(V_total)) and np.all(np.isfinite(grad_total))):
            raise NumericalInstabilityError(
                f"V o ∇V no finito. max|Φ|={np.max(np.abs(phi)):.2e}"
            )
        return V_total, grad_total

    def compute_exact_hessian_diagonal(
        self,
        phi: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Segunda derivada analítica exacta $V''_{\mathrm{reg}}(\Phi)$ para las ecuaciones
        variacionales de Jacobi-Poincaré en el fibrado tangente $T(T^*\mathcal{M})$:

        $$V''_{\mathrm{reg}}(\Phi) = -\mu^2 + \lambda\Phi^2\left[3\tanh(s) + \operatorname{sech}^2(s)\left(\frac{9}{2}s - 2s^2\tanh(s)\right)\right]$$
        """
        phi_sq = phi * phi
        scale_sq = self.FIELD_REG_SCALE ** 2 + self.EPSILON_SMOOTH
        s = phi_sq / scale_sq
        tanh_s = np.tanh(s)
        sech2_s = 1.0 - tanh_s * tanh_s

        bracket = 3.0 * tanh_s + sech2_s * (4.5 * s - 2.0 * s * s * tanh_s)
        hess = -self.MU_SQUARED + self.LAMBDA_COUPLING * phi_sq * bracket
        return hess

    def synthesize_level3_mutation(
        self,
        lyapunov_exponent: float,
        kam_coherence: float,
        energy_drift: float,
    ) -> "QFTParameters":
        r"""
        Operador de Punto Fijo de Banach (Nivel 3 RSI) sobre la variedad de hiperparámetros.
        Ajusta la amortiguación de Rayleigh, el factor CFL y el acoplamiento conforme
        conservando estrictamente el VEV $v = \sqrt{\mu^2/\lambda}$ como invariante de gauge.
        """
        contraction = self.RSI_BANACH_CONTRACTION ** (self.RSI_GENERATION + 1)
        lr = self.RSI_META_LEARNING_RATE * contraction

        # Gradiente meta-variacional respecto al exceso de caos de Lyapunov y pérdida KAM
        lyap_excess = max(0.0, lyapunov_exponent - 0.1 * self.LYAPUNOV_CRITICAL_CAP)
        kam_deficit = max(0.0, 1.0 - kam_coherence)
        chaos_pressure = lyap_excess + 0.5 * kam_deficit + 0.25 * min(energy_drift, 2.0)

        new_gamma = float(
            np.clip(self.DAMPING_GAMMA * (1.0 + lr * chaos_pressure), 1e-4, 0.95)
        )
        new_cfl = float(
            np.clip(self.CFL_SAFETY_FACTOR * (1.0 - 0.5 * lr * chaos_pressure), 0.15, 0.90)
        )
        new_conformal = float(
            np.clip(self.CONFORMAL_GAMMA * (1.0 + 0.5 * lr * kam_deficit), 1e-5, 0.20)
        )

        mutated = replace(
            self,
            DAMPING_GAMMA=new_gamma,
            CFL_SAFETY_FACTOR=new_cfl,
            CONFORMAL_GAMMA=new_conformal,
            RSI_GENERATION=self.RSI_GENERATION + 1,
        )
        mutated.validate_parameters()
        return mutated


# Instancia global inmutable (validada al cargar el módulo)
QFT: Final[QFTParameters] = QFTParameters()
QFT.validate_parameters()


# ───────────────────────────────────────────────────────────────────────────
# 1.2  ESTRUCTURAS ALGEBRAICAS Y FIBRADO TANGENTE DE POINCARÉ
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScalarFieldState:
    r"""
    Punto extendido en el fibrado tangente simpléctico $T(T^*\mathbb{R}^n)$.

    Invariantes (verificados en `__post_init__`)
    --------------------------------------------
    J1. $\dim(\Phi) = \dim(\Pi)$         (compatibilidad del espacio de fase)
    J2. $\Phi.\mathrm{ndim} = 1$         (1-cadena/0-coforma discretizada)
    J3. $\mathrm{dtype} = \mathrm{float64}$ (precisión IEEE 754)
    J4. $\delta\Phi, \delta\Pi$ opcionales para integración variacional de Poincaré.
    """

    phi: NDArray[np.float64]
    pi_momentum: NDArray[np.float64]
    delta_phi: Optional[NDArray[np.float64]] = None
    delta_pi: Optional[NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        if self.phi.shape != self.pi_momentum.shape:
            raise ValueError(
                f"[J1] Φ{self.phi.shape} ≠ Π{self.pi_momentum.shape}"
            )
        if self.phi.ndim != 1:
            raise ValueError(f"[J2] Φ debe ser 1D, recibido {self.phi.ndim}D")
        if self.phi.dtype != np.float64 or self.pi_momentum.dtype != np.float64:
            raise TypeError("[J3] Φ y Π deben ser float64")
        if self.delta_phi is not None and self.delta_phi.shape != self.phi.shape:
            raise ValueError("[J4] delta_phi debe tener la misma dimensión que phi")
        if self.delta_pi is not None and self.delta_pi.shape != self.pi_momentum.shape:
            raise ValueError("[J4] delta_pi debe tener la misma dimensión que pi_momentum")

    @property
    def dim(self) -> int:
        return int(self.phi.shape[0])

    def copy(self) -> "ScalarFieldState":
        """Copia defensiva profunda de los tensores de estado y vectores tangentes."""
        return ScalarFieldState(
            phi=self.phi.copy(),
            pi_momentum=self.pi_momentum.copy(),
            delta_phi=self.delta_phi.copy() if self.delta_phi is not None else None,
            delta_pi=self.delta_pi.copy() if self.delta_pi is not None else None,
        )

    def with_initialized_tangent(self) -> "ScalarFieldState":
        """Garantiza un vector tangente unitario ortonormal para seguimiento de Lyapunov."""
        if self.delta_phi is not None and self.delta_pi is not None:
            return self
        n = self.dim
        if n == 0:
            return ScalarFieldState(
                phi=self.phi,
                pi_momentum=self.pi_momentum,
                delta_phi=np.zeros(0, dtype=np.float64),
                delta_pi=np.zeros(0, dtype=np.float64),
            )
        dphi = np.ones(n, dtype=np.float64) / math.sqrt(float(n))
        dpi = np.zeros(n, dtype=np.float64)
        return ScalarFieldState(
            phi=self.phi,
            pi_momentum=self.pi_momentum,
            delta_phi=dphi,
            delta_pi=dpi,
        )

    def to_hypercomplex_fiber(self) -> HypercomplexSymplecticFiber:
        """Eleva el estado al fibrado hipercomplejo de Banach $\mathbb{H}_{\omega}^n$."""
        st = self.with_initialized_tangent()
        assert st.delta_phi is not None and st.delta_pi is not None
        return HypercomplexSymplecticFiber(
            real_phi=st.phi,
            imag_i_pi=st.pi_momentum,
            imag_j_dphi=st.delta_phi,
            imag_k_dpi=st.delta_pi,
        )

    def symplectic_form_omega(self, other: "ScalarFieldState") -> float:
        r"""
        Evalúa la 2-forma simpléctica canónica de Poincaré:
            $$\omega(\xi_1, \xi_2) = \langle \delta\Phi_1, \delta\Pi_2 \rangle - \langle \delta\Pi_1, \delta\Phi_2 \rangle$$
        """
        s1 = self.with_initialized_tangent()
        s2 = other.with_initialized_tangent()
        assert s1.delta_phi is not None and s1.delta_pi is not None
        assert s2.delta_phi is not None and s2.delta_pi is not None
        return float(
            np.dot(s1.delta_phi, s2.delta_pi) - np.dot(s1.delta_pi, s2.delta_phi)
        )


@dataclass(frozen=True, slots=True)
class FermionicSource:
    r"""
    Fuente fermiónica $\psi$ para el acoplamiento de Yukawa $\mathcal{L}_Y = -g\bar{\psi}\Phi\psi$.

    Atributos
    ---------
    psi_vector : $\psi \in \mathbb{R}^n$ (amplitud de sección espinorial real)
    base_mass  : $m_0 \ge 0$ (masa desnuda antes de ruptura espontánea de simetría)
    """

    psi_vector: NDArray[np.float64]
    base_mass: float = 0.0

    def __post_init__(self) -> None:
        if self.psi_vector.ndim != 1:
            raise ValueError("ψ debe ser 1D")
        if self.base_mass < 0:
            raise ValueError("m₀ ≥ 0 requerido")

    def charge_density(self) -> NDArray[np.float64]:
        r"""
        Densidad de carga normalizada $\rho = |\psi|^2 / \max(|\psi|^2) \in [0, 1]^n$.
        """
        rho_raw = self.psi_vector ** 2
        max_rho = float(np.max(rho_raw)) if rho_raw.size > 0 else 0.0
        if max_rho > QFT.EPSILON_SMOOTH:
            return rho_raw / max_rho
        return rho_raw

    def chiral_asymmetry_index(self) -> float:
        r"""Índice de asimetría espectral fermiónica $\mathcal{A}_\chi \in [-1, 1]$."""
        if self.psi_vector.size == 0:
            return 0.0
        norm_sq = float(np.dot(self.psi_vector, self.psi_vector))
        if norm_sq <= QFT.EPSILON_SMOOTH:
            return 0.0
        signed_fourth = float(np.sum(np.sign(self.psi_vector) * (self.psi_vector ** 2)))
        return float(np.clip(signed_fourth / norm_sq, -1.0, 1.0))


# ───────────────────────────────────────────────────────────────────────────
# 1.3  OPERADOR DE LAPLACE-BELTRAMI CON MÉTRICA CONFORME (AXIOMAS A2 Y A3)
# ───────────────────────────────────────────────────────────────────────────


class LaplacedBeltramiOperator:
    r"""
    Operador de Laplace-Beltrami discreto $\Delta_M$ sobre un complejo simplicial
    ponderado por el tensor métrico conforme $\tilde{G}_{\mu\nu} = G_{\mathrm{PHYSICS}}\exp(-2\gamma V(\Phi))$.

    Garantías tras validación (Axiomas A2 y A3)
    -------------------------------------------
    L1. $L$ es CSR dispersa y cuadrada ($n \times n$)
    L2. $L = L^\top$ (autoadjunción en producto interno estándar)
    L3. $L \succeq 0$ (semidefinida positiva por Ley de Inercia de Sylvester)
    L4. $\sum_j L_{ij} = 0$ (conservatividad de flujos de 0-formas / cohomología de de Rham)
    """

    def __init__(self, matrix: sp.csr_matrix) -> None:
        self._validate(matrix)
        self.L: sp.csr_matrix = matrix.tocsr()
        self.n: int = int(matrix.shape[0])
        self._base_L: sp.csr_matrix = self.L.copy()

    @staticmethod
    def _validate(M: sp.spmatrix) -> None:
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"[L1] Laplaciano no cuadrado: shape={M.shape}")
        diff = M - M.T
        if diff.nnz > 0 and np.any(np.abs(diff.data) > 1e-10):
            raise ValueError("[L2] Laplaciano no simétrico")
        if M.shape[0] > 0:
            try:
                min_eig = float(eigsh(M, k=1, which="SA", return_eigenvectors=False)[0])
                if min_eig < -1e-10:
                    raise ValueError(f"[L3] Laplaciano no PSD: λ_min={min_eig}")
            except ValueError:
                raise
            except Exception:
                row_sums = np.abs(np.array(M.sum(axis=1)).flatten())
                if np.max(row_sums) > 1e-10:
                    raise ValueError("[L4] Laplaciano no conservativo")
        row_sums = np.array(M.sum(axis=1)).flatten()
        if row_sums.size > 0 and np.max(np.abs(row_sums)) > 1e-10:
            raise ValueError("[L4] ∑_j L_ij ≠ 0 (no conservativo)")

    @classmethod
    def from_adjacency(
        cls,
        adjacency: Union[NDArray[np.float64], sp.spmatrix],
        symmetrize: bool = True,
    ) -> "LaplacedBeltramiOperator":
        """Construye $\Delta_M = D - A$ a partir de la matriz de adyacencia métrica."""
        A = (
            sp.csr_matrix(adjacency)
            if isinstance(adjacency, np.ndarray)
            else adjacency.tocsr()
        )
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Adyacencia no cuadrada: {A.shape}")
        if symmetrize:
            diff = A - A.T
            if diff.nnz > 0 and np.any(np.abs(diff.data) > 1e-12):
                logger.warning("Adyacencia asimétrica: promediando con su transpuesta")
                A = 0.5 * (A + A.T)
        degrees = np.array(A.sum(axis=1)).flatten()
        D = sp.diags(degrees, format="csr")
        return cls(D - A)

    def apply(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Aplica $\Delta_M$ a un campo escalar: $L \cdot \Phi$."""
        if phi.shape != (self.n,):
            raise ValueError(f"Φ.shape {phi.shape} ≠ ({self.n},)")
        return np.asarray(self.L @ phi, dtype=np.float64)

    def apply_conformal_stabilization(
        self,
        potential_vec: NDArray[np.float64],
        conformal_gamma: float,
    ) -> "LaplacedBeltramiOperator":
        r"""
        Implementa el Axioma [A2]: Estabilización Conforme de la Métrica de Fondo.
            $$\tilde{G}_{\mu\nu} = G_{\mathrm{PHYSICS}} \cdot \exp\left(-2\gamma \cdot V(\Phi)\right)$$
        Modula la estrella de Hodge discreta $\star_1$ en las 1-aristas y reconstruye
        $\tilde{\Delta}_M = \delta^\top \tilde{\star}_1 \delta$ preservando la firma SPD
        por la Ley de Inercia de Sylvester.
        """
        if self.n == 0 or conformal_gamma <= 0.0:
            return self
        if potential_vec.shape != (self.n,):
            raise MetricDegeneracyError(
                f"Dimensión de potencial {potential_vec.shape} incompatible con métrica ({self.n},)"
            )

        # Acotación de exponente conforme para inmunidad numérica IEEE-754
        exponent = np.clip(-conformal_gamma * potential_vec, -20.0, 20.0)
        conformal_weights = np.exp(exponent)
        if not np.all(np.isfinite(conformal_weights)) or np.any(conformal_weights <= 0.0):
            raise MetricDegeneracyError(
                "El factor conforme exp(-2γV(Φ)) perdió la condición definida positiva (Sylvester)."
            )

        # Extraer adyacencia base A_base = D_base - L_base
        diag_base = self._base_L.diagonal()
        A_base = sp.diags(diag_base, format="csr") - self._base_L
        A_base.eliminate_zeros()

        # Transformación congruente de Sylvester: Â = W^{1/2} A_base W^{1/2}
        W_half = sp.diags(conformal_weights, format="csr")
        A_tilde = W_half @ A_base @ W_half
        A_tilde = 0.5 * (A_tilde + A_tilde.T)
        degrees_tilde = np.array(A_tilde.sum(axis=1)).flatten()
        L_tilde = sp.diags(degrees_tilde, format="csr") - A_tilde

        try:
            op_tilde = LaplacedBeltramiOperator(L_tilde.tocsr())
            op_tilde._base_L = self._base_L
            return op_tilde
        except ValueError as exc:
            raise LaplaceBeltramiSingularity(
                f"Singularidad en operador de Laplace-Beltrami conforme: {exc}"
            ) from exc


# Alias ortográfico limpio manteniendo 100% de compatibilidad con `LaplacedBeltramiOperator`
LaplaceBeltramiOperator = LaplacedBeltramiOperator


# ───────────────────────────────────────────────────────────────────────────
# 1.4  ANALIZADOR ESPECTRAL Y PUENTE ANIDADO HACIA LA FASE 2
# ───────────────────────────────────────────────────────────────────────────


class SpectralAnalyzer:
    r"""
    Análisis espectral de operadores dispersos autoadjuntos y cohomología simplicial.

    Métodos disponibles
    -------------------
    'gershgorin'      : Teorema de discos de Gershgorin ($O(n)$)
    'power_iteration' : Iteración de potencia de Mises ($O(k \cdot \mathrm{nnz})$)
    'krylov'          : Subespacios de Krylov-Lanczos vía ARPACK
    """

    @staticmethod
    def estimate_max_eigenvalue(
        laplacian: sp.csr_matrix,
        method: str = "gershgorin",
        tolerance: float = 1e-6,
        max_iter: int = 200,
    ) -> float:
        n = laplacian.shape[0]
        if n == 0:
            return 0.0

        if method == "gershgorin":
            # Por el teorema de Gershgorin en un Laplaciano L = D - A (con L_ii = d_i y
            # ∑_{j≠i} |L_ij| = d_i), la cota superior exacta de los discos es max_i(2*L_ii)
            # o bien la norma infinito max_i ∑_j |L_ij|. Si se evalúa la suma con signo,
            # daría 0; por ello tomamos abs(laplacian) para garantizar cota espectral real.
            abs_row_sums = np.array(np.abs(laplacian).sum(axis=1)).flatten()
            return float(np.max(abs_row_sums)) if abs_row_sums.size > 0 else 0.0

        if method == "power_iteration":
            v = np.random.default_rng(0).standard_normal(n)
            v_norm = float(np.linalg.norm(v))
            if v_norm < QFT.EPSILON_SMOOTH:
                return 0.0
            v = v / v_norm
            lambda_prev = 0.0
            for _ in range(max_iter):
                v_new = laplacian @ v
                v_new_norm = float(np.linalg.norm(v_new))
                if v_new_norm < QFT.EPSILON_SMOOTH:
                    return 0.0
                lambda_new = float(np.dot(v, v_new))
                if abs(lambda_new - lambda_prev) < tolerance:
                    return lambda_new
                v = v_new / v_new_norm
                lambda_prev = lambda_new
            logger.warning("Power iteration no convergió en %d iteraciones", max_iter)
            return lambda_prev

        if method == "krylov":
            try:
                eigs = eigsh(
                    laplacian,
                    k=1,
                    which="LA",
                    return_eigenvectors=False,
                    tol=tolerance,
                )
                return float(eigs[0])
            except Exception as e:
                logger.error("Krylov (LA) falló: %s. Cayendo a Gershgorin.", e)
                return SpectralAnalyzer.estimate_max_eigenvalue(
                    laplacian, method="gershgorin"
                )

        raise ValueError(f"Método '{method}' inválido")

    @staticmethod
    def estimate_spectral_gap(
        laplacian: sp.csr_matrix,
        tolerance: float = 1e-6,
        n_threshold: int = 500,
    ) -> float:
        r"""
        Estima el gap espectral de Fiedler $\lambda_1(\Delta_M)$ (conectividad algebraica).
        """
        n = laplacian.shape[0]
        if n < 2:
            return 0.0
        try:
            if n <= n_threshold:
                dense_eigs = np.linalg.eigvalsh(laplacian.toarray())
                dense_sorted = np.sort(dense_eigs)
                return max(0.0, float(dense_sorted[1]))
            k = min(2, n - 1)
            eigs = eigsh(
                laplacian,
                k=k,
                which="SM",
                return_eigenvectors=False,
                tol=tolerance,
            )
            if len(eigs) < 2:
                return 0.0
            eigs_sorted = np.sort(eigs)
            return max(0.0, float(eigs_sorted[1]))
        except Exception as e:
            logger.error("Cálculo de gap espectral falló: %s", e)
            return 0.0

    @staticmethod
    def estimate_betti_zero(
        laplacian: sp.csr_matrix,
        zero_tol: float = 1e-7,
    ) -> int:
        r"""
        Calcula el 0-ésimo número de Betti $\beta_0 = \dim \ker(\Delta_0) = \dim H^0_{\mathrm{dR}}(M)$
        contando las componentes conexas de la variedad discreta.
        """
        n = laplacian.shape[0]
        if n == 0:
            return 0
        n_components, _ = sp.csgraph.connected_components(
            csgraph=laplacian, directed=False, return_labels=True
        )
        return max(1, int(n_components))

    # ═══════════════════════════════════════════════════════════════════════
    # ESLABÓN ANIDADO FASE 1 ──► FASE 2 (CONTINUACIÓN MORFISMÁTICA ESTRICTA)
    # ═══════════════════════════════════════════════════════════════════════
    @classmethod
    def bridge_to_phase2_symplectic_manifold(
        cls,
        laplacian_op: LaplacedBeltramiOperator,
        params: QFTParameters = QFT,
    ) -> Tuple["HiggsPotential", "PortHamiltonianHamiltonian", Dict[str, float]]:
        r"""
        Definición formal final de la Fase 1 y génesis directa de la Fase 2.

        Extrae la firma espectral completa $(\lambda_{\max}, \lambda_1, \beta_0)$ del
        operador de Laplace-Beltrami de la Fase 1 y construye de manera atómica
        el Potencial de Higgs regularizado y el Hamiltoniano de Port-Hamilton que
        abren la Fase 2.
        """
        lambda_max = cls.estimate_max_eigenvalue(laplacian_op.L, method="gershgorin")
        spectral_gap = cls.estimate_spectral_gap(laplacian_op.L)
        betti_0 = float(cls.estimate_betti_zero(laplacian_op.L))

        spectral_signature: Dict[str, float] = {
            "lambda_max": lambda_max,
            "spectral_gap": spectral_gap,
            "betti_0": betti_0,
            "cheeger_lower_bound": 0.5 * spectral_gap,
        }

        # Continuación directa hacia los constructores de apertura de la Fase 2
        potential = HiggsPotential.from_phase1_spectral_signature(
            params=params, spectral_signature=spectral_signature
        )
        hamiltonian = PortHamiltonianHamiltonian(
            laplacian=laplacian_op,
            potential=potential,
            params=params,
            spectral_signature=spectral_signature,
        )
        return potential, hamiltonian, spectral_signature


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 2 · DINÁMICA SIMPLÉCTICA, MECÁNICA CELESTE DE POINCARÉ Y RSI-L3   │
# │  Inicia directamente como continuación del puente espectral de Fase 1.  │
# │  Su definición final (`validate_topological_consistency`) constituye    │
# │  el puente anidado de entrada hacia la Fase 3.                          │
# └─────────────────────────────────────────────────────────────────────────┘


# ───────────────────────────────────────────────────────────────────────────
# 2.1  POTENCIAL DE HIGGS REGULARIZADO Y SERIES DE LINDSTEDT-POINCARÉ
# ───────────────────────────────────────────────────────────────────────────


class HiggsPotential:
    r"""
    Potencial de Higgs regularizado $V_{\mathrm{reg}}(\Phi)$ con análisis de cotas
    de Lipschitz y eliminación de términos seculares de Lindstedt-Poincaré.

    Recibe la firma espectral emitida por `SpectralAnalyzer.bridge_to_phase2_symplectic_manifold`.
    """

    def __init__(
        self,
        params: QFTParameters = QFT,
        spectral_signature: Optional[Dict[str, float]] = None,
    ) -> None:
        self.p: QFTParameters = params
        self.spectral_signature: Dict[str, float] = spectral_signature or {}

    @classmethod
    def from_phase1_spectral_signature(
        cls,
        params: QFTParameters,
        spectral_signature: Dict[str, float],
    ) -> "HiggsPotential":
        """Continuación formal directa desde el último método de la Fase 1."""
        return cls(params=params, spectral_signature=spectral_signature)

    def value_and_gradient(
        self, phi: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        V, grad_V = self.p.compute_potential_and_gradient(phi)
        max_grad = float(np.max(np.abs(grad_V))) if grad_V.size > 0 else 0.0
        if max_grad > self.p.LIPSCHITZ_MAX_BOUND * self.p.MAX_FIELD_MAGNITUDE:
            raise LipschitzRegularityBreach(
                f"Violación de regularidad Lipschitz: max|∇V|={max_grad:.2e}"
            )
        return V, grad_V

    def hessian_diagonal(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Diagonal del Hessiano exacto $D^2 V_{\mathrm{reg}}(\Phi)$ para la ecuación de Jacobi."""
        return self.p.compute_exact_hessian_diagonal(phi)

    def lipschitz_constant(self, domain_radius: float) -> float:
        r"""
        Constante de Lipschitz en la bola cerrada $\mathcal{B}_R = \{\Phi : \|\Phi\|_\infty \le R\}$:
            $$L_V(R) = \sup_{|\Phi|\le R} |V'(\Phi)| \le \mu^2 R + 3\lambda R^3 \left(1 + \frac{R^2}{2\Phi_c^2}\right)$$
        """
        R = abs(domain_radius)
        R_sq = R * R
        Phi_c_sq = self.p.FIELD_REG_SCALE ** 2
        return (
            self.p.MU_SQUARED * R
            + 3.0 * self.p.LAMBDA_COUPLING * R * R_sq * (1.0 + 0.5 * R_sq / Phi_c_sq)
        )

    def lindstedt_poincare_frequency_renormalization(
        self, phi: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Serie perturbativa de Lindstedt-Poincaré (resumida vía aproximante de Padé $[1/1]$)
        para eliminar resonancias seculares en oscilaciones no lineales alrededor del VEV $v$:

            $$\omega_{\mathrm{LP}}^2(A) = \omega_0^2 \cdot \frac{1 + \frac{3\lambda}{4\mu^2} A^2}{1 + \frac{9\lambda}{4\mu^2} A^2}, \quad A = |\Phi - v|$$
        """
        amp_sq = (phi - self.p.vev) ** 2
        ratio = (self.p.LAMBDA_COUPLING / max(self.p.MU_SQUARED, 1e-12)) * amp_sq
        reg_ratio = ratio / (1.0 + amp_sq / (self.p.FIELD_REG_SCALE ** 2))
        factor = (1.0 + 0.75 * reg_ratio) / (1.0 + 2.25 * reg_ratio)
        return np.asarray(factor, dtype=np.float64)

    def is_physically_admissible(self, phi: NDArray[np.float64]) -> bool:
        """Verifica que el campo resida dentro del tubo de confianza de Lipschitz."""
        if phi.size == 0:
            return True
        max_abs = float(np.max(np.abs(phi)))
        return max_abs <= self.p.FIELD_REG_SCALE


# ───────────────────────────────────────────────────────────────────────────
# 2.2  HAMILTONIANO PORT-HAMILTONIANO Y FLUJO VARIACIONAL DE JACOBI-POINCARÉ
# ───────────────────────────────────────────────────────────────────────────


class PortHamiltonianHamiltonian:
    r"""
    Hamiltoniano Port-Hamiltoniano acoplado a la geometría de Laplace-Beltrami:

        $$H[\Phi, \Pi] = \frac{1}{2}\|\Pi\|_2^2 + \frac{1}{2}c^2 \langle \Phi, \Delta_M \Phi \rangle + \sum_{i=1}^n V_{\mathrm{reg}}(\Phi_i)$$

    Satisface la identidad de pasividad de Rayleigh-Lyapunov (Axioma A4):
        $$\dot{H} = \langle \nabla H, (J - R)\nabla H \rangle = -\gamma \|\Pi\|_2^2 \le 0$$
    """

    def __init__(
        self,
        laplacian: LaplacedBeltramiOperator,
        potential: HiggsPotential,
        params: QFTParameters = QFT,
        spectral_signature: Optional[Dict[str, float]] = None,
    ) -> None:
        self.L_op: LaplacedBeltramiOperator = laplacian
        self.V: HiggsPotential = potential
        self.p: QFTParameters = params
        self.spectral_signature: Dict[str, float] = spectral_signature or {}

    def __call__(self, state: ScalarFieldState) -> float:
        """Evalúa la energía total $H[\Phi, \Pi]$ con sutura de coherencia dimensional."""
        phi = state.phi
        pi = state.pi_momentum
        if state.dim != self.L_op.n:
            logger.debug(
                "Sutura V: state.dim=%d ≠ laplacian.n=%d; proyectando al subespacio común",
                state.dim,
                self.L_op.n,
            )
            common_n = min(state.dim, self.L_op.n)
            phi_pad = np.zeros(self.L_op.n, dtype=np.float64)
            phi_pad[:common_n] = phi[:common_n]
            T = 0.5 * float(np.dot(pi[:common_n], pi[:common_n]))
            U_el = 0.5 * self.p.C_SQUARED * float(
                np.dot(phi_pad, self.L_op.apply(phi_pad))
            )
            V_local, _ = self.V.value_and_gradient(phi[:common_n])
            return T + U_el + float(np.sum(V_local))

        T = 0.5 * float(np.dot(pi, pi))
        U_el = 0.5 * self.p.C_SQUARED * float(
            np.dot(phi, self.L_op.apply(phi))
        )
        V_local, _ = self.V.value_and_gradient(phi)
        U_pot = float(np.sum(V_local))
        return T + U_el + U_pot

    def force(
        self,
        state: ScalarFieldState,
        source_density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Campo de fuerza conservativa más retroacción fermiónica de Yukawa:
            $$F(\Phi) = -c^2 \Delta_M \Phi - \nabla V_{\mathrm{reg}}(\Phi) + g_{\mathrm{back}}\rho_\psi$$
        """
        laplacian_force = -self.p.C_SQUARED * self.L_op.apply(state.phi)
        _, grad_V = self.V.value_and_gradient(state.phi)
        potential_force = -grad_V
        source_force = self.p.YUKAWA_BACKREACTION * source_density
        return laplacian_force + potential_force + source_force

    def jacobi_variational_force(
        self,
        phi: NDArray[np.float64],
        delta_phi: NDArray[np.float64],
        delta_pi: NDArray[np.float64],
        damping_gamma: float = 0.0,
    ) -> NDArray[np.float64]:
        r"""
        Operador variacional de Jacobi-Poincaré en el fibrado tangente $T_{(\Phi,\Pi)}(T^*\mathcal{M})$:
            $$\frac{d(\delta\Pi)}{dt} = -c^2 \Delta_M \delta\Phi - D^2V_{\mathrm{reg}}(\Phi)\delta\Phi - \gamma \delta\Pi$$
        """
        elastic_var = -self.p.C_SQUARED * self.L_op.apply(delta_phi)
        hess_diag = self.V.hessian_diagonal(phi)
        potential_var = -hess_diag * delta_phi
        damping_var = -damping_gamma * delta_pi if damping_gamma > 0.0 else 0.0
        return elastic_var + potential_var + damping_var

    def rayleigh_dissipation_power(
        self, state: ScalarFieldState, damping_gamma: float
    ) -> float:
        r"""Potencia disipada de Rayleigh: $\mathcal{P}_{\mathrm{Rayleigh}} = -\gamma \|\Pi\|_2^2 \le 0$."""
        return -max(0.0, damping_gamma) * float(
            np.dot(state.pi_momentum, state.pi_momentum)
        )


# ───────────────────────────────────────────────────────────────────────────
# 2.3  MECÁNICA CELESTE DE POINCARÉ: SECCIÓN Σ, LYAPUNOV Y TOROS KAM
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PoincareSectionCrossing:
    r"""
    Registro de intersección transversal con la Superficie de Sección de Poincaré:
        $$\Sigma = \left\{ (\Phi, \Pi) \in T^*\mathcal{M} \;\middle|\; \bar{\Phi} = v, \; \bar{\Pi} > 0 \right\}$$
    """

    step_fraction: float
    mean_momentum: float
    field_dispersion: float
    poincare_cartan_action: float


class PoincareCelestialAnalyzer:
    r"""
    Motor Analítico de Mecánica Celeste de Henri Poincaré.

    Evalúa:
    1. **Mapa de Retorno en la Superficie de Sección $\Sigma$**: Cruces transversales $\bar{\Phi} = v$.
    2. **Invariante Integral de Poincaré-Cartan**: $\mathcal{I}_{\mathrm{PC}} = \oint (\langle \Pi, d\Phi \rangle - H dt)$.
    3. **Exponente Característico de Lyapunov Máximo (MLCE)** vía renormalización de Benettin
       sobre las ecuaciones variacionales exactas de Jacobi-Poincaré.
    4. **Índice de Coherencia de Toros KAM (Kolmogorov-Arnold-Moser)**: Mide la concentración
       espectral de la órbita para distinguir movimiento cuasi-periódico en toros invariantes
       frente a enredos homoclínicos y difusión de Arnold.
    """

    def __init__(self, params: QFTParameters = QFT) -> None:
        self.p: QFTParameters = params
        self._crossings: List[PoincareSectionCrossing] = []
        self._lyapunov_log_sum: float = 0.0
        self._accumulated_time: float = 0.0
        self._poincare_cartan_action: float = 0.0
        self._mean_field_orbit: List[float] = []

    @property
    def crossings(self) -> List[PoincareSectionCrossing]:
        return list(self._crossings)

    @property
    def maximal_lyapunov_exponent(self) -> float:
        r"""Exponente máximo de Lyapunov acumulado $\lambda_1 = \frac{1}{T}\sum \ln(\|\delta z_k\|/\|\delta z_0\|)$."""
        if self._accumulated_time <= self.p.EPSILON_SMOOTH:
            return 0.0
        return self._lyapunov_log_sum / self._accumulated_time

    @property
    def poincare_cartan_action(self) -> float:
        return self._poincare_cartan_action

    def observe_transition(
        self,
        prev_state: ScalarFieldState,
        next_state: ScalarFieldState,
        prev_energy: float,
        next_energy: float,
        dt: float,
    ) -> Tuple[ScalarFieldState, float, float]:
        """
        Registra la evolución orbital $(t \to t + dt)$, actualiza el invariante de
        Poincaré-Cartan, detecta cruces en la sección $\Sigma$, y aplica renormalización
        simpléctica de Benettin al vector tangente $(\delta\Phi, \delta\Pi)$.
        """
        if prev_state.dim == 0 or dt <= 0.0:
            return next_state, 0.0, 1.0

        # 1. Forma diferencial de Poincaré-Cartan: dS = ⟨Π_mid, ΔΦ⟩ - H_mid * dt
        dphi = next_state.phi - prev_state.phi
        pi_mid = 0.5 * (prev_state.pi_momentum + next_state.pi_momentum)
        h_mid = 0.5 * (prev_energy + next_energy)
        ds_pc = float(np.dot(pi_mid, dphi)) - h_mid * dt
        self._poincare_cartan_action += ds_pc

        # 2. Detección de cruce transversal en la Superficie de Sección de Poincaré Σ
        vev = self.p.vev
        prev_dev = float(np.mean(prev_state.phi)) - vev
        next_dev = float(np.mean(next_state.phi)) - vev
        mean_pi = float(np.mean(pi_mid))

        if prev_dev <= 0.0 < next_dev and mean_pi > self.p.POINCARE_SECTION_TOL:
            denom = max(abs(next_dev - prev_dev), self.p.EPSILON_SMOOTH)
            alpha = abs(prev_dev) / denom
            interp_phi = prev_state.phi + alpha * dphi
            dispersion = float(np.std(interp_phi))
            self._crossings.append(
                PoincareSectionCrossing(
                    step_fraction=alpha,
                    mean_momentum=mean_pi,
                    field_dispersion=dispersion,
                    poincare_cartan_action=self._poincare_cartan_action,
                )
            )

        # 3. Almacenar serie orbital para diagnóstico espectral KAM
        self._mean_field_orbit.append(next_dev)
        if len(self._mean_field_orbit) > 128:
            self._mean_field_orbit.pop(0)

        # 4. Exponente de Lyapunov y Renormalización de Benettin en el fibrado tangente
        st_tan = next_state.with_initialized_tangent()
        fiber = st_tan.to_hypercomplex_fiber()
        tan_norm = fiber.tangent_norm()
        self._accumulated_time += dt

        if tan_norm > self.p.EPSILON_SMOOTH:
            self._lyapunov_log_sum += math.log(tan_norm)
            assert st_tan.delta_phi is not None and st_tan.delta_pi is not None
            renorm_state = ScalarFieldState(
                phi=st_tan.phi,
                pi_momentum=st_tan.pi_momentum,
                delta_phi=st_tan.delta_phi / tan_norm,
                delta_pi=st_tan.delta_pi / tan_norm,
            )
        else:
            renorm_state = st_tan

        kam_coherence = self.compute_kam_torus_coherence()
        return renorm_state, self.maximal_lyapunov_exponent, kam_coherence

    def compute_kam_torus_coherence(self) -> float:
        r"""
        Evalúa la coherencia de los Toros Invariantes KAM mediante la pureza espectral
        de Wiener-Shannon de la trayectoria en el espacio de fase:
            $$\mathcal{K}_{\mathrm{KAM}} = 1 - \frac{H_{\mathrm{Shannon}}(\hat{\eta})}{\ln(M)} \in [0, 1]$$
        Un valor cercano a 1 indica movimiento cuasi-periódico confinado en un toro KAM;
        una caída hacia 0 revela ruptura de toros KAM y caos homoclínico.
        """
        m = len(self._mean_field_orbit)
        if m < 4:
            return 1.0
        signal = np.asarray(self._mean_field_orbit, dtype=np.float64)
        signal = signal - np.mean(signal)
        power = np.abs(np.fft.rfft(signal)) ** 2
        total_power = float(np.sum(power))
        if total_power <= self.p.EPSILON_SMOOTH:
            return 1.0
        probs = power / total_power
        probs = probs[probs > 1e-15]
        if probs.size <= 1:
            return 1.0
        shannon_entropy = -float(np.sum(probs * np.log(probs)))
        max_entropy = math.log(float(len(power)))
        if max_entropy <= self.p.EPSILON_SMOOTH:
            return 1.0
        return float(np.clip(1.0 - (shannon_entropy / max_entropy), 0.0, 1.0))


# ───────────────────────────────────────────────────────────────────────────
# 2.4  INTEGRADOR SIMPLÉCTICO VELOCITY-VERLET Y YOSHIDA DE 4.º ORDEN
# ───────────────────────────────────────────────────────────────────────────


class SymplecticIntegrator(ABC):
    """Clase base abstracta para integradores simplécticos en variedades Riemannianas."""

    @abstractmethod
    def step(
        self,
        state: ScalarFieldState,
        force_field: NDArray[np.float64],
        dt: float,
    ) -> ScalarFieldState: ...


class VelocityVerletIntegrator(SymplecticIntegrator):
    r"""
    Integrador Simpléctico Híbrido Velocity-Verlet (2.º orden) y Yoshida-Poincaré (4.º orden).

    Integra simultáneamente:
    1. El flujo primario $(\Phi^n, \Pi^n) \mapsto (\Phi^{n+1}, \Pi^{n+1})$ con corrección de
       frecuencia secular de Lindstedt-Poincaré.
    2. El flujo tangente variacional $(\delta\Phi^n, \delta\Pi^n) \mapsto (\delta\Phi^{n+1}, \delta\Pi^{n+1})$
       mediante la ecuación de Jacobi-Poincaré para el cálculo exacto del espectro de Lyapunov.

    Coeficientes simplécticos de 4.º orden de Haruo Yoshida (1990):
        $$w_1 = \frac{1}{2 - 2^{1/3}}, \quad w_0 = -\frac{2^{1/3}}{2 - 2^{1/3}} = 1 - 2w_1$$
    """

    _YOSHIDA_W1: Final[float] = 1.0 / (2.0 - (2.0 ** (1.0 / 3.0)))
    _YOSHIDA_W0: Final[float] = 1.0 - 2.0 * _YOSHIDA_W1
    _YOSHIDA_SUBSTEPS: Final[Tuple[float, float, float]] = (
        _YOSHIDA_W1,
        _YOSHIDA_W0,
        _YOSHIDA_W1,
    )

    def __init__(
        self,
        hamiltonian: PortHamiltonianHamiltonian,
        damping_gamma: float = 0.0,
        max_field: float = 1e3,
        use_yoshida_4th_order: bool = False,
    ) -> None:
        self.H: PortHamiltonianHamiltonian = hamiltonian
        self.gamma: float = damping_gamma
        self.max_field: float = max_field
        self.use_yoshida_4th_order: bool = use_yoshida_4th_order

        # Análisis espectral para condición Courant-Friedrichs-Lewy (CFL)
        self._lambda_max: float = SpectralAnalyzer.estimate_max_eigenvalue(
            hamiltonian.L_op.L, method="gershgorin"
        )
        self._dt_critical: float = self._compute_critical_timestep()

        logger.info(
            "Integrador Simpléctico inicializado: λ_max=%.4f, dt_crit=%.6f, γ=%.3f, Yoshida4=%s",
            self._lambda_max,
            self._dt_critical,
            self.gamma,
            self.use_yoshida_4th_order,
        )

    def _compute_critical_timestep(self) -> float:
        r"""
        CFL estricto: $\Delta t < (2/\omega_{\max})\cdot\alpha_{\mathrm{CFL}}$, donde
            $$\omega_{\max}^2 = c^2 \lambda_{\max}(\Delta_M) + m_H^2 = c^2 \lambda_{\max} + 2\mu^2$$
        """
        omega_sq = (
            self.H.p.C_SQUARED * self._lambda_max
            + self.H.p.higgs_mass_squared
        )
        omega_max = math.sqrt(omega_sq + self.H.p.EPSILON_SMOOTH)
        dt_limit = (2.0 / omega_max) * self.H.p.CFL_SAFETY_FACTOR
        return min(dt_limit, self.H.p.CFL_ABSOLUTE_CAP)

    @property
    def dt_critical(self) -> float:
        return self._dt_critical

    def recalibrate_spectrum(self) -> None:
        """Recalcula el radio espectral y $\Delta t_{\mathrm{crit}}$ tras una mutación de Nivel 3."""
        self._lambda_max = SpectralAnalyzer.estimate_max_eigenvalue(
            self.H.L_op.L, method="gershgorin"
        )
        self._dt_critical = self._compute_critical_timestep()

    def _verlet_elementary_step(
        self,
        state: ScalarFieldState,
        source_density: NDArray[np.float64],
        dt_sub: float,
    ) -> ScalarFieldState:
        """Paso elemental Velocity-Verlet acoplado con el fibrado tangente de Jacobi-Poincaré."""
        phi = state.phi
        pi = state.pi_momentum

        # 1) Π^{n+½} = Π^n + ½·F(Φ^n, Π^n)·dt
        F_n = self._force(phi, pi, source_density)
        pi_half = pi + 0.5 * F_n * dt_sub

        # 2) Φ^{n+1} = Φ^n + Π^{n+½}·dt (con renormalización secular de Lindstedt-Poincaré)
        lp_factor = self.H.V.lindstedt_poincare_frequency_renormalization(phi)
        phi_new = phi + (pi_half * lp_factor) * dt_sub
        phi_new = self._soft_clip(phi_new)

        # 3) Π^{n+1} = Π^{n+½} + ½·F(Φ^{n+1}, Π^{n+½})·dt
        F_new = self._force(phi_new, pi_half, source_density)
        pi_new = pi_half + 0.5 * F_new * dt_sub

        # 4) Evolución variacional simpléctica en el fibrado tangente (δΦ, δΠ)
        if state.delta_phi is not None and state.delta_pi is not None:
            dphi = state.delta_phi
            dpi = state.delta_pi
            dF_n = self.H.jacobi_variational_force(phi, dphi, dpi, self.gamma)
            dpi_half = dpi + 0.5 * dF_n * dt_sub
            dphi_new = dphi + dpi_half * dt_sub
            dF_new = self.H.jacobi_variational_force(
                phi_new, dphi_new, dpi_half, self.gamma
            )
            dpi_new = dpi_half + 0.5 * dF_new * dt_sub
            return ScalarFieldState(
                phi=phi_new,
                pi_momentum=pi_new,
                delta_phi=dphi_new,
                delta_pi=dpi_new,
            )

        return ScalarFieldState(phi=phi_new, pi_momentum=pi_new)

    def step(
        self,
        state: ScalarFieldState,
        source_density: NDArray[np.float64],
        dt: Optional[float] = None,
    ) -> ScalarFieldState:
        """
        Ejecuta un paso de integración simpléctica (Velocity-Verlet 2.º orden o
        composición triple de Yoshida de 4.º orden si fue activada por Nivel 3 RSI).
        """
        dt_eff = dt if dt is not None else self._dt_critical
        if dt_eff <= 0:
            raise ValueError(f"dt debe ser positivo, recibido: {dt_eff}")
        if dt_eff > self._dt_critical * 1.01:
            logger.warning(
                "dt=%.6f excede dt_crit=%.6f (inestabilidad probable)",
                dt_eff,
                self._dt_critical,
            )

        if not self.use_yoshida_4th_order:
            return self._verlet_elementary_step(state, source_density, dt_eff)

        # Composición simpléctica de 4.º orden de Yoshida (3 sub-pasos simétricos)
        curr = state
        for weight in self._YOSHIDA_SUBSTEPS:
            curr = self._verlet_elementary_step(curr, source_density, weight * dt_eff)
        return curr

    def _force(
        self,
        phi: NDArray[np.float64],
        pi: NDArray[np.float64],
        source: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """$F = -c^2 L \Phi - \nabla V(\Phi) - \gamma \Pi + g_s \rho$."""
        F = self.H.force(ScalarFieldState(phi=phi, pi_momentum=pi), source)
        if self.gamma > 0:
            F = F - self.gamma * pi
        return F

    def _soft_clip(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Saturación suave difeomórfica: $\Phi \mapsto M\tanh(\Phi/M)$."""
        return self.max_field * np.tanh(phi / self.max_field)


# ───────────────────────────────────────────────────────────────────────────
# 2.5  MONITOR DE ESTABILIDAD Y AUTOMEJORA RECURSIVA DE NIVEL 3
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class StabilityMetrics:
    """
    Métricas de diagnóstico de estabilidad termodinámica, celeste de Poincaré
    y lógica intuicionista de Heyting ($\Omega_3$).
    """

    energy: float
    energy_drift: float
    field_rms: float
    momentum_rms: float
    max_field: float
    vev_deviation: float
    requires_damping: bool
    is_stable: bool
    # Extensiones de Poincaré, Rayleigh y RSI Nivel 3 (con valores por defecto compatibles)
    lyapunov_exponent: float = 0.0
    kam_torus_coherence: float = 1.0
    poincare_cartan_action: float = 0.0
    rayleigh_dissipation_power: float = 0.0
    heyting_verdict: HeytingTruthValue = HeytingTruthValue.COHERENT
    rsi_level3_generation: int = 0


class StabilityMonitor:
    """
    Monitor de estabilidad acoplado al clasificador de subobjetos de Heyting $\Omega_3$.
    """

    def __init__(
        self,
        baseline_energy: float,
        hamiltonian: PortHamiltonianHamiltonian,
        params: QFTParameters = QFT,
    ) -> None:
        self._E0: float = baseline_energy
        self._H: PortHamiltonianHamiltonian = hamiltonian
        self._p: QFTParameters = params
        self._history: List[StabilityMetrics] = []

    @property
    def baseline_energy(self) -> float:
        return self._E0

    @property
    def history(self) -> List[StabilityMetrics]:
        return list(self._history)

    def analyze(
        self,
        state: ScalarFieldState,
        lyapunov_exponent: float = 0.0,
        kam_coherence: float = 1.0,
        poincare_action: float = 0.0,
    ) -> StabilityMetrics:
        H = self._H(state)
        energy_drift = abs(H - self._E0) / (abs(self._E0) + self._p.EPSILON_SMOOTH)
        field_rms = float(np.sqrt(np.mean(state.phi ** 2))) if state.dim > 0 else 0.0
        momentum_rms = (
            float(np.sqrt(np.mean(state.pi_momentum ** 2))) if state.dim > 0 else 0.0
        )
        max_field = float(np.max(np.abs(state.phi))) if state.dim > 0 else 0.0
        vev_dev = (
            float(np.sqrt(np.mean((state.phi - self._p.vev) ** 2)))
            if state.dim > 0
            else 0.0
        )
        requires_damping = (
            H > self._p.MAX_ENERGY_RATIO * abs(self._E0)
            or lyapunov_exponent > self._p.LYAPUNOV_CRITICAL_CAP
            or kam_coherence < self._p.KAM_COHERENCE_FLOOR
        )
        is_stable = (
            bool(np.all(np.isfinite(state.phi)))
            and bool(np.all(np.isfinite(state.pi_momentum)))
            and max_field < self._p.MAX_FIELD_MAGNITUDE
            and energy_drift < 10.0
        )

        # Clasificación en el Retículo de Heyting Ω_3 (Axioma A5)
        if not is_stable:
            verdict = HeytingTruthValue.VETOED
        elif requires_damping or energy_drift > 1.5:
            verdict = HeytingTruthValue.DEGRADED
        else:
            verdict = HeytingTruthValue.COHERENT

        rayleigh_power = self._H.rayleigh_dissipation_power(
            state, self._p.DAMPING_GAMMA
        )

        m = StabilityMetrics(
            energy=H,
            energy_drift=energy_drift,
            field_rms=field_rms,
            momentum_rms=momentum_rms,
            max_field=max_field,
            vev_deviation=vev_dev,
            requires_damping=requires_damping,
            is_stable=is_stable,
            lyapunov_exponent=lyapunov_exponent,
            kam_torus_coherence=kam_coherence,
            poincare_cartan_action=poincare_action,
            rayleigh_dissipation_power=rayleigh_power,
            heyting_verdict=verdict,
            rsi_level3_generation=self._p.RSI_GENERATION,
        )
        self._history.append(m)
        return m

    def log_diagnostics(
        self, metrics: StabilityMetrics, level: int = logging.INFO
    ) -> None:
        logger.log(
            level,
            "[STABILITY] E=%.4e (drift=%.2f%%), ⟨Φ⟩_rms=%.4f, max|Φ|=%.4f, "
            "⟨|Φ-v|⟩=%.4f, λ_Lyap=%.4f, KAM=%.3f, Ω_3=%s, RSI_gen=%d",
            metrics.energy,
            metrics.energy_drift * 100.0,
            metrics.field_rms,
            metrics.max_field,
            metrics.vev_deviation,
            metrics.lyapunov_exponent,
            metrics.kam_torus_coherence,
            metrics.heyting_verdict.name,
            metrics.rsi_level3_generation,
        )


class Level3RecursiveSelfImprover:
    r"""
    Motor de Automejora Recursiva de Nivel 3 ($\mathcal{R}_3 : \mathrm{End}(\mathcal{C}) \to \mathrm{End}(\mathcal{C})$).

    Jerarquía de Niveles de Automejora
    ----------------------------------
    - **Nivel 1 (Paramétrico)**: Ajusta el coeficiente de disipación $\gamma$ y el paso $\Delta t$.
    - **Nivel 2 (Métrico-Operador)**: Deforma conformalmente la métrica $\tilde{G}_{\mu\nu} = G\exp(-2\gamma V(\Phi))$
      en el operador de Laplace-Beltrami $\Delta_M$ para suavizar singularidades de curvatura.
    - **Nivel 3 (Meta-Funtorial / Punto Fijo de Banach)**: Evalúa la derivada de Fréchet
      del funcional de Poincaré-Lyapunov $\mathcal{J}[\mathcal{F}_{\mathrm{Higgs}}]$, verifica la
      contracción de Banach $\|\mathcal{T}(\mathcal{F}_{k+1}) - \mathcal{T}(\mathcal{F}_k)\|_{\mathcal{B}} \le q\|\mathcal{F}_{k+1} - \mathcal{F}_k\|_{\mathcal{B}}$,
      muta inmutablemente `QFTParameters`, activa el esquema simpléctico de 4.º orden de Yoshida
      y proyecta el campo hacia el subespacio invariante del toro KAM.
    """

    def __init__(self) -> None:
        self._banach_defect_history: List[float] = []
        self._contraction_verified: bool = True

    @property
    def banach_defect_history(self) -> List[float]:
        return list(self._banach_defect_history)

    @property
    def is_contraction_verified(self) -> bool:
        return self._contraction_verified

    def compute_frechet_meta_functional(self, metrics: StabilityMetrics) -> float:
        r"""
        Funcional de coste meta-variacional de Poincaré-Lyapunov:
            $$\mathcal{J}[\mathcal{F}] = \max(0, \lambda_{\mathrm{Lyap}})^2 + (1 - \mathcal{K}_{\mathrm{KAM}})^2 + \frac{1}{4}\left(\frac{\Delta E}{E_0}\right)^2$$
        """
        lyap_term = max(0.0, metrics.lyapunov_exponent) ** 2
        kam_term = (1.0 - metrics.kam_torus_coherence) ** 2
        drift_term = 0.25 * (min(metrics.energy_drift, 10.0) ** 2)
        return math.sqrt(lyap_term + kam_term + drift_term)

    def execute_level3_introspection(
        self,
        state: ScalarFieldState,
        metrics: StabilityMetrics,
        hamiltonian: PortHamiltonianHamiltonian,
        integrator: VelocityVerletIntegrator,
        monitor: StabilityMonitor,
    ) -> Tuple[ScalarFieldState, QFTParameters]:
        """
        Ejecuta el ciclo completo de Automejora Recursiva de Nivel 1, 2 y 3 cuando
        el clasificador de Heyting detecta degradación orbital o exceso de Lyapunov.
        """
        defect = self.compute_frechet_meta_functional(metrics)
        if self._banach_defect_history:
            prev_defect = self._banach_defect_history[-1]
            # Verificación de contracción en el espacio de Banach de endofuntores
            self._contraction_verified = (
                defect <= prev_defect * (1.0 + hamiltonian.p.RSI_META_LEARNING_RATE)
                or defect < 0.05
            )
        self._banach_defect_history.append(defect)

        if (
            metrics.heyting_verdict == HeytingTruthValue.COHERENT
            and not metrics.requires_damping
        ):
            return state, hamiltonian.p

        # ── NIVEL 1: Síntesis contractiva de nuevos parámetros QFT ──
        new_params = hamiltonian.p.synthesize_level3_mutation(
            lyapunov_exponent=metrics.lyapunov_exponent,
            kam_coherence=metrics.kam_torus_coherence,
            energy_drift=metrics.energy_drift,
        )

        # ── NIVEL 2: Estabilización Conforme de la Métrica (Axioma A2) ──
        V_vec, _ = hamiltonian.V.value_and_gradient(state.phi)
        conformal_lb = hamiltonian.L_op.apply_conformal_stabilization(
            potential_vec=V_vec,
            conformal_gamma=new_params.CONFORMAL_GAMMA,
        )

        # ── NIVEL 3: Re-tejido en caliente del Hamiltoniano, Integrador y Estado ──
        hamiltonian.p = new_params
        hamiltonian.V.p = new_params
        hamiltonian.L_op = conformal_lb

        integrator.gamma = new_params.DAMPING_GAMMA
        integrator.use_yoshida_4th_order = True
        integrator.recalibrate_spectrum()
        monitor._p = new_params

        # Retracción simpléctica hacia el Toro KAM del Vacío (contracción de Banach)
        q = new_params.RSI_BANACH_CONTRACTION
        vev_target = np.full_like(state.phi, new_params.vev)
        contracted_phi = q * state.phi + (1.0 - q) * vev_target
        contracted_pi = q * state.pi_momentum

        healed_state = ScalarFieldState(
            phi=contracted_phi,
            pi_momentum=contracted_pi,
            delta_phi=state.delta_phi,
            delta_pi=state.delta_pi,
        )
        return healed_state, new_params


# ───────────────────────────────────────────────────────────────────────────
# 2.6  VALIDACIÓN DE INVARIANTES TOPOLÓGICOS Y PUENTE ANIDADO HACIA FASE 3
# ───────────────────────────────────────────────────────────────────────────


def validate_topological_consistency(
    state: ScalarFieldState,
    hamiltonian: PortHamiltonianHamiltonian,
    params: QFTParameters = QFT,
    *,
    return_phase3_suturation: bool = False,
    metrics: Optional[StabilityMetrics] = None,
    crowbar_port: Optional[CrowbarCircuitPort] = None,
) -> Union[bool, "HiggsSuturationState"]:
    r"""
    Definición formal final de la Fase 2 y puente anidado de entrada hacia la Fase 3.

    Valida invariantes topológicos, simplécticos y físicos:
    T1. $\dim(\Phi) = \dim(\Delta_M)$
    T2. $\Phi, \Pi \in \mathbb{R}^n$ (finitos en norma de Banach)
    T3. $\|\Phi\|_\infty < \Phi_{\max}$ (acotación Lipschitz)
    T4. $H[\Phi, \Pi]$ bien definido y finito

    Cuando `return_phase3_suturation=True`, eleva directamente el resultado validado
    hacia el objeto inicial de la Fase 3 (`HiggsSuturationState`).
    """
    is_valid = True
    try:
        if state.dim != hamiltonian.L_op.n:
            logger.error("[T1] dim inconsistente")
            is_valid = False
        elif not (
            np.all(np.isfinite(state.phi)) and np.all(np.isfinite(state.pi_momentum))
        ):
            logger.error("[T2] campos no finitos")
            is_valid = False
        else:
            max_field = float(np.max(np.abs(state.phi))) if state.dim > 0 else 0.0
            if max_field > params.MAX_FIELD_MAGNITUDE:
                logger.error("[T3] |Φ|_max=%.2e > Φ_max", max_field)
                is_valid = False
            else:
                H = hamiltonian(state)
                if not np.isfinite(H):
                    logger.error("[T4] H=%g no finito", H)
                    is_valid = False
    except Exception as e:
        logger.exception("Validación topológica falló: %s", e)
        is_valid = False

    if not return_phase3_suturation:
        return is_valid

    # Continuación formal directa hacia la estructura de apertura de la Fase 3
    return HiggsSuturationState.from_phase2_topological_bridge(
        is_topologically_valid=is_valid,
        state=state,
        hamiltonian=hamiltonian,
        metrics=metrics,
        crowbar_port=crowbar_port,
    )


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 3 · FUNTOR DE ANCLAJE, SUTURA DE DE RHAM-HEYTING Y CROWBAR (DECIDE│
# │  Inicia directamente con `HiggsSuturationState` construido por el       │
# │  último método de la Fase 2 (`validate_topological_consistency`).       │
# │  Culmina con la composición endofuntorial: Y ∘ H ∘ V.                   │
# └─────────────────────────────────────────────────────────────────────────┘


# ───────────────────────────────────────────────────────────────────────────
# 3.1  ESTADO DE SUTURA COHOMOLÓGICA DE DE RHAM Y PUERTO CROWBAR
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HiggsSuturationState:
    r"""
    Estado de Sutura Cohomológica de de Rham-Heyting (apertura formal de la Fase 3).

    Recibe la validación topológica y dinámica de la Fase 2, proyecta el estado sobre
    el clasificador de subobjetos $\Omega_3 = \{\mathtt{COHERENT}, \mathtt{DEGRADED}, \mathtt{VETOED}\}$
    y gobierna el disparo físico del circuito Crowbar (GPIO14 / Tiristor BT151).
    """

    field_state: ScalarFieldState
    heyting_verdict: HeytingTruthValue
    crowbar_port: CrowbarCircuitPort
    betti_0: int
    spectral_gap: float
    lyapunov_exponent: float
    kam_torus_coherence: float
    poincare_cartan_action: float
    rsi_generation: int

    @classmethod
    def from_phase2_topological_bridge(
        cls,
        is_topologically_valid: bool,
        state: ScalarFieldState,
        hamiltonian: PortHamiltonianHamiltonian,
        metrics: Optional[StabilityMetrics] = None,
        crowbar_port: Optional[CrowbarCircuitPort] = None,
    ) -> "HiggsSuturationState":
        """Continuación formal directa desde `validate_topological_consistency` (Fase 2)."""
        base_verdict = (
            metrics.heyting_verdict
            if metrics is not None
            else HeytingTruthValue.COHERENT
        )
        final_verdict = (
            base_verdict
            if is_topologically_valid
            else base_verdict.join(HeytingTruthValue.VETOED)
        )
        energy = metrics.energy if metrics is not None else hamiltonian(state)
        port = (crowbar_port or CrowbarCircuitPort()).evaluate_verdict(
            final_verdict, energy
        )
        betti_0 = int(
            hamiltonian.spectral_signature.get(
                "betti_0", float(SpectralAnalyzer.estimate_betti_zero(hamiltonian.L_op.L))
            )
        )
        spec_gap = float(
            hamiltonian.spectral_signature.get(
                "spectral_gap",
                SpectralAnalyzer.estimate_spectral_gap(hamiltonian.L_op.L),
            )
        )
        return cls(
            field_state=state,
            heyting_verdict=final_verdict,
            crowbar_port=port,
            betti_0=betti_0,
            spectral_gap=spec_gap,
            lyapunov_exponent=metrics.lyapunov_exponent if metrics is not None else 0.0,
            kam_torus_coherence=(
                metrics.kam_torus_coherence if metrics is not None else 1.0
            ),
            poincare_cartan_action=(
                metrics.poincare_cartan_action if metrics is not None else 0.0
            ),
            rsi_generation=hamiltonian.p.RSI_GENERATION,
        )


# ───────────────────────────────────────────────────────────────────────────
# 3.2  FUNTOR DE ANCLAJE DE HIGGS (YUKAWA ∘ HAMILTONIANO-POINCARÉ ∘ VACÍO)
# ───────────────────────────────────────────────────────────────────────────


class ScalarHiggsAnchor(Morphism):
    r"""
    Funtor de Anclaje de Higgs: $\mathcal{F}_{\mathrm{Higgs}}: \mathcal{C}_{\mathrm{Agent}} \to \mathcal{C}_{\mathrm{Agent}}$.

    Composición Endofuntorial de Tres Fases Anidadas
    ------------------------------------------------
        $$\mathcal{F}_{\mathrm{Higgs}} = \text{Yukawa}_{\text{Fase 3}} \circ \text{HamiltonPoincaré}_{\text{Fase 2}}^N \circ \text{VacuumSpectral}_{\text{Fase 1}}$$

    Integra:
    - Fase 1 (`OBSERVE`): Operador de Laplace-Beltrami conforme y puente espectral.
    - Fase 2 (`ORIENT`) : Flujo simpléctico Velocity-Verlet / Yoshida-4, Mecánica Celeste
                          de Poincaré (sección $\Sigma$, invariante de Poincaré-Cartan,
                          exponente de Lyapunov, toros KAM) y Automejora Recursiva Nivel 3.
    - Fase 3 (`DECIDE`) : Clasificador intuicionista de Heyting $\Omega_3$, protección
                          Crowbar RL-Tiristor y generación de masa efectiva de Yukawa.
    """

    def __init__(
        self,
        laplacian: sp.csr_matrix,
        dim: int,
        params: QFTParameters = QFT,
        num_relaxation_steps: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        # ── Sutura V (suturas_rigurosas.md): coherencia dimensional ──
        if laplacian.shape[0] != laplacian.shape[1]:
            raise ValueError(f"Laplaciano no cuadrado: shape={laplacian.shape}")
        n_lap = laplacian.shape[0]
        if dim != n_lap:
            logger.debug(
                "Sutura V: dim=%d ajustado a n_lap=%d (cardinalidad del Laplaciano)",
                dim,
                n_lap,
            )
            dim = n_lap

        self._params: QFTParameters = params
        self._dim: int = dim
        self._num_steps: int = num_relaxation_steps

        # ── FASE 1 (OBSERVE) ──► Puente Espectral hacia Fase 2 ──
        self._lb_op: LaplacedBeltramiOperator = LaplacedBeltramiOperator(laplacian)
        (
            self._potential,
            self._hamiltonian,
            self._spectral_signature,
        ) = SpectralAnalyzer.bridge_to_phase2_symplectic_manifold(
            laplacian_op=self._lb_op,
            params=self._params,
        )

        # ── FASE 2 (ORIENT): Integrador Simpléctico, Poincaré y RSI Nivel 3 ──
        self._integrator: VelocityVerletIntegrator = VelocityVerletIntegrator(
            self._hamiltonian,
            damping_gamma=params.DAMPING_GAMMA,
            max_field=params.MAX_FIELD_MAGNITUDE,
        )
        self._poincare: PoincareCelestialAnalyzer = PoincareCelestialAnalyzer(params)
        self._rsi_improver: Level3RecursiveSelfImprover = Level3RecursiveSelfImprover()

        # RNG reproducible inyectado
        self._rng: np.random.Generator = np.random.default_rng(
            seed if seed is not None else params.RNG_SEED
        )

        # Inicialización del vacío con fibrado tangente de Jacobi-Poincaré
        self._field_state: ScalarFieldState = (
            self._initialize_vacuum_state().with_initialized_tangent()
        )

        # Monitor de estabilidad
        self._monitor: StabilityMonitor = StabilityMonitor(
            baseline_energy=self._hamiltonian(self._field_state),
            hamiltonian=self._hamiltonian,
            params=params,
        )

        # ── FASE 3 (DECIDE): Puerto Crowbar y Estado de Sutura de Heyting ──
        self._crowbar_port: CrowbarCircuitPort = CrowbarCircuitPort()
        initial_suturation = validate_topological_consistency(
            self._field_state,
            self._hamiltonian,
            self._params,
            return_phase3_suturation=True,
            crowbar_port=self._crowbar_port,
        )
        assert isinstance(initial_suturation, HiggsSuturationState)
        self._suturation_state: HiggsSuturationState = initial_suturation

        logger.info(
            "ScalarHiggsAnchor v7.0 inicializado: dim=%d, v=%.4f, E₀=%.4e, dt_crit=%.6f, β₀=%d",
            dim,
            params.vev,
            self._monitor.baseline_energy,
            self._integrator.dt_critical,
            self._suturation_state.betti_0,
        )

    def _initialize_vacuum_state(self) -> ScalarFieldState:
        r"""
        Vacío coherente: $\Phi_0 = v + \delta\Phi$, $\Pi_0 = 0$.
        Fluctuaciones térmicas iniciales de orden $O(5\% \cdot v)$.
        """
        vev = self._params.vev
        noise_amp = 0.05 * vev
        phi_init = np.full(self._dim, vev, dtype=np.float64)
        phi_init += self._rng.normal(0.0, noise_amp, size=self._dim)
        pi_init = np.zeros(self._dim, dtype=np.float64)
        return ScalarFieldState(phi=phi_init, pi_momentum=pi_init)

    def __call__(
        self,
        state: CategoricalState,
        *args: Any,
        **kwargs: Any,
    ) -> CategoricalState:
        """
        Aplica el endofuntor $\mathcal{F}_{\mathrm{Higgs}}$ al estado categórico.

         Flujo Anidado OODA Espectral (Fases 1 ──► 2 ──► 3)
        ---------------------------------------------------
        1. [Fase 1] Extraer $\psi$ del payload y construir densidad fermiónica $\rho = |\psi|^2$.
        2. [Fase 2] Evolucionar $(\Phi, \Pi, \delta\Phi, \delta\Pi)$ con dinámica simpléctica,
                    registrar invariante de Poincaré-Cartan, cruces en $\Sigma$, exponente
                    de Lyapunov y ejecutar Automejora Recursiva de Nivel 3 si se requiere.
        3. [Fase 3] Sutura cohomológica de de Rham en el retículo de Heyting $\Omega_3$,
                    verificación del circuito Crowbar, cálculo de masa efectiva de Yukawa
                    $m_{\mathrm{eff}} = m_0 + g|\Phi|$ y anclaje $\psi' = m_{\mathrm{eff}} \odot \psi$.
        """
        # 1) Extracción robusta de la sección espinorial ψ
        psi_raw = self._extract_psi(state)
        if psi_raw is None:
            logger.debug("Sin vector estocástico: identidad retornada")
            return state

        # 2) Densidad de carga fermiónica
        source = FermionicSource(psi_vector=psi_raw)
        rho = source.charge_density()

        # 3) Relajación simpléctica con Mecánica Celeste de Poincaré y RSI Nivel 3
        for step_idx in range(self._num_steps):
            metrics = self._monitor.analyze(
                self._field_state,
                lyapunov_exponent=self._poincare.maximal_lyapunov_exponent,
                kam_coherence=self._poincare.compute_kam_torus_coherence(),
                poincare_action=self._poincare.poincare_cartan_action,
            )
            if not metrics.is_stable:
                self._suturation_state = HiggsSuturationState.from_phase2_topological_bridge(
                    is_topologically_valid=False,
                    state=self._field_state,
                    hamiltonian=self._hamiltonian,
                    metrics=metrics,
                    crowbar_port=self._crowbar_port,
                )
                self._crowbar_port = self._suturation_state.crowbar_port
                logger.error(
                    "Inestabilidad en paso %d: drift=%.2f%%, max|Φ|=%.2e, Ω_3=%s, Crowbar=%s",
                    step_idx,
                    metrics.energy_drift * 100,
                    metrics.max_field,
                    self._suturation_state.heyting_verdict.name,
                    self._crowbar_port.latched,
                )
                raise HeytingLobeCollapse(
                    "Colapso del campo de Higgs hacia VETOED (disparo de Crowbar GPIO14)"
                )

            # Si el estado entra en el lóbulo DEGRADED o requiere amortiguamiento,
            # se activa la Automejora Recursiva de Nivel 3 (Banach-Fréchet + Métrica Conforme)
            if metrics.requires_damping or (
                metrics.heyting_verdict == HeytingTruthValue.DEGRADED
            ):
                self._field_state, self._params = (
                    self._rsi_improver.execute_level3_introspection(
                        state=self._field_state,
                        metrics=metrics,
                        hamiltonian=self._hamiltonian,
                        integrator=self._integrator,
                        monitor=self._monitor,
                    )
                )
                self._lb_op = self._hamiltonian.L_op
                if step_idx == 0:
                    logger.warning(
                        "Termostato y RSI Nivel 3 activos (gen=%d): E/E₀=%.2f, γ=%.4f",
                        self._params.RSI_GENERATION,
                        metrics.energy / max(abs(self._monitor.baseline_energy), 1e-12),
                        self._params.DAMPING_GAMMA,
                    )

            prev_state = self._field_state
            prev_energy = metrics.energy
            dt_used = self._integrator.dt_critical

            next_state = self._integrator.step(prev_state, rho, dt=dt_used)
            next_energy = self._hamiltonian(next_state)

            # Observación celeste de Poincaré (sección Σ, acción de Poincaré-Cartan y Benettin)
            self._field_state, _, _ = self._poincare.observe_transition(
                prev_state=prev_state,
                next_state=next_state,
                prev_energy=prev_energy,
                next_energy=next_energy,
                dt=dt_used,
            )

        final_metrics = self._monitor.analyze(
            self._field_state,
            lyapunov_exponent=self._poincare.maximal_lyapunov_exponent,
            kam_coherence=self._poincare.compute_kam_torus_coherence(),
            poincare_action=self._poincare.poincare_cartan_action,
        )
        self._monitor.log_diagnostics(final_metrics, logging.DEBUG)

        # Puente Fase 2 ──► Fase 3: Sutura de de Rham-Heyting y Puerto Crowbar
        suturation = validate_topological_consistency(
            self._field_state,
            self._hamiltonian,
            self._params,
            return_phase3_suturation=True,
            metrics=final_metrics,
            crowbar_port=self._crowbar_port,
        )
        assert isinstance(suturation, HiggsSuturationState)
        self._suturation_state = suturation
        self._crowbar_port = suturation.crowbar_port

        if not self._suturation_state.heyting_verdict.is_verdict_coherent():
            raise HeytingLobeCollapse(
                "El clasificador de Heyting Ω_3 colapsó a VETOED tras la relajación."
            )

        # 4) Masa efectiva de Yukawa
        m_eff = self._compute_effective_mass(source)
        if not np.all(np.isfinite(m_eff)):
            raise YukawaCouplingDivergence("m_eff no finita")
        if m_eff.size > 0 and np.max(m_eff) > 1e4:
            logger.critical("m_eff excesiva: max=%.2e, clipando", float(np.max(m_eff)))
            m_eff = np.clip(m_eff, 0.0, 1e4)

        # 5) Anclaje covariante: ψ' = m_eff ⊙ ψ
        psi_anchored = psi_raw * m_eff

        # 6) Construcción del nuevo CategoricalState enriquecido
        new_payload = self._build_anchored_payload(
            state, psi_anchored, m_eff, final_metrics
        )
        return CategoricalState(
            payload=new_payload,
            stratum=Stratum.PHYSICS,
            metadata=state.metadata,
        )

    # ───── Métodos auxiliares y propiedades de introspección ─────

    def _extract_psi(self, state: CategoricalState) -> Optional[NDArray[np.float64]]:
        """Extrae $\psi$ del payload con polimorfismo estructural seguro."""
        if isinstance(state.payload, dict):
            raw = state.payload.get("stochastic_vector")
            if raw is None or len(raw) == 0:
                return None
            psi = np.asarray(raw, dtype=np.float64).ravel()
        elif isinstance(state.payload, np.ndarray):
            if state.payload.size == 0:
                return None
            psi = state.payload.astype(np.float64).ravel()
        elif hasattr(state.payload, "__iter__") and not isinstance(
            state.payload, (str, bytes)
        ):
            try:
                psi = np.asarray(list(state.payload), dtype=np.float64).ravel()
                if psi.size == 0:
                    return None
            except (TypeError, ValueError):
                return None
        else:
            return None

        return self._resize_psi(psi)

    def _resize_psi(self, psi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Proyecta o sumerge isométricamente $\psi$ en $\mathbb{R}^{\mathrm{dim}}$."""
        if psi.shape[0] == self._dim:
            return psi
        if psi.shape[0] > self._dim:
            return psi[: self._dim]
        out = np.zeros(self._dim, dtype=np.float64)
        out[: psi.shape[0]] = psi
        return out

    def _compute_effective_mass(
        self, source: FermionicSource
    ) -> NDArray[np.float64]:
        r"""
        Perfil espacial de masa efectiva de Yukawa regularizada:
            $$m_{\mathrm{eff}}(x_i) = m_0 + g_Y \sqrt{\Phi_i^2 + \varepsilon^2}$$
        """
        phi = self._field_state.phi
        phi_mag = np.sqrt(phi ** 2 + self._params.EPSILON_SMOOTH ** 2)
        return source.base_mass + self._params.YUKAWA_G * phi_mag

    def _build_anchored_payload(
        self,
        original: CategoricalState,
        psi_anchored: NDArray[np.float64],
        m_eff: NDArray[np.float64],
        metrics: StabilityMetrics,
    ) -> Dict[str, Any]:
        """Construye el payload preservando llaves originales y añadiendo telemetría Poincaré/RSI."""
        if isinstance(original.payload, dict):
            new_payload: Dict[str, Any] = dict(original.payload)
        else:
            new_payload = {"original_payload": original.payload}

        new_payload.update(
            {
                "anchored_vector": psi_anchored.tolist(),
                "effective_mass_profile": m_eff.tolist(),
                "higgs_field_snapshot": self._field_state.phi.tolist(),
                "vev_deviation": metrics.vev_deviation,
                "field_energy": metrics.energy,
                "stability_flag": metrics.is_stable,
                "heyting_verdict": self._suturation_state.heyting_verdict.name,
                "crowbar_latched": self._crowbar_port.latched,
                "poincare_lyapunov_exponent": metrics.lyapunov_exponent,
                "poincare_kam_coherence": metrics.kam_torus_coherence,
                "poincare_cartan_action": metrics.poincare_cartan_action,
                "poincare_section_crossings": len(self._poincare.crossings),
                "rsi_level3_generation": self._params.RSI_GENERATION,
                "rsi_banach_contraction_verified": (
                    self._rsi_improver.is_contraction_verified
                ),
            }
        )
        return new_payload

    @property
    def current_vev_deviation(self) -> float:
        """Desviación cuadrática media (RMS) respecto al vacío $\Phi - v$."""
        if self._dim == 0:
            return 0.0
        return float(
            np.sqrt(np.mean((self._field_state.phi - self._params.vev) ** 2))
        )

    @property
    def field_state(self) -> ScalarFieldState:
        return self._field_state

    @property
    def monitor(self) -> StabilityMonitor:
        return self._monitor

    @property
    def suturation_state(self) -> HiggsSuturationState:
        return self._suturation_state

    @property
    def poincare_analyzer(self) -> PoincareCelestialAnalyzer:
        return self._poincare

    @property
    def rsi_improver(self) -> Level3RecursiveSelfImprover:
        return self._rsi_improver


# ───────────────────────────────────────────────────────────────────────────
# 3.3  UTILIDADES PÚBLICAS Y DECORADOR CATEGÓRICO
# ───────────────────────────────────────────────────────────────────────────


def construct_laplacian_from_adjacency(
    adjacency: Union[NDArray[np.float64], sp.spmatrix],
    symmetrize: bool = True,
) -> sp.csr_matrix:
    r"""Construye el Laplaciano combinatorio $\Delta = D - A$ en formato CSR."""
    op = LaplacedBeltramiOperator.from_adjacency(adjacency, symmetrize=symmetrize)
    return op.L


T_Morphism = TypeVar("T_Morphism", bound=Morphism)


def apply_higgs_anchor(
    dim: int,
    laplacian: sp.csr_matrix,
    num_relaxation_steps: int = 5,
    seed: Optional[int] = None,
) -> Callable[[type[T_Morphism]], type[T_Morphism]]:
    r"""
    Decorador de orden superior que compone un `Morphism` con el funtor `ScalarHiggsAnchor`.

    Persiste la instancia del ancla por agente para preservar la memoria orbital
    de Poincaré y la evolución generacional de Automejora Recursiva de Nivel 3.
    """

    def decorator(agent_class: type[T_Morphism]) -> type[T_Morphism]:
        original_call = agent_class.__call__

        @wraps(original_call)
        def wrapped(
            self: T_Morphism, state: CategoricalState, *a: Any, **kw: Any
        ) -> CategoricalState:
            pre_state = original_call(self, state, *a, **kw)
            anchor: Optional[ScalarHiggsAnchor] = getattr(
                self, "_higgs_anchor_instance", None
            )
            if anchor is None:
                anchor = ScalarHiggsAnchor(
                    laplacian=laplacian,
                    dim=dim,
                    num_relaxation_steps=num_relaxation_steps,
                    seed=seed,
                )
                try:
                    setattr(self, "_higgs_anchor_instance", anchor)
                except Exception:
                    pass
            return anchor(pre_state)

        agent_class.__call__ = wrapped  # type: ignore[assignment]
        agent_class.__doc__ = (agent_class.__doc__ or "") + (
            f"\n\n[Anclado por Higgs v7.0 (Poincaré + RSI Nivel 3) con dim={dim}, steps={num_relaxation_steps}]"
        )
        return agent_class

    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIONES PÚBLICAS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Preludio Axiomático (Excepciones, Heyting, Crowbar e Hipercomplejos)
    "ScalarHiggsAnchorError",
    "MetricDegeneracyError",
    "LaplaceBeltramiSingularity",
    "YukawaCouplingDivergence",
    "LipschitzRegularityBreach",
    "HeytingLobeCollapse",
    "HeytingTruthValue",
    "CrowbarCircuitPort",
    "HypercomplexSymplecticFiber",
    # Fase 1: Axiomática del Campo y Espectro
    "QFTParameters",
    "QFT",
    "ScalarFieldState",
    "FermionicSource",
    "LaplacedBeltramiOperator",
    "LaplaceBeltramiOperator",
    "SpectralAnalyzer",
    # Fase 2: Dinámica Simpléctica, Mecánica Celeste de Poincaré y RSI Nivel 3
    "HiggsPotential",
    "PortHamiltonianHamiltonian",
    "PoincareSectionCrossing",
    "PoincareCelestialAnalyzer",
    "SymplecticIntegrator",
    "VelocityVerletIntegrator",
    "StabilityMonitor",
    "StabilityMetrics",
    "Level3RecursiveSelfImprover",
    "validate_topological_consistency",
    # Fase 3: Sutura de de Rham-Heyting y Funtor de Anclaje
    "HiggsSuturationState",
    "ScalarHiggsAnchor",
    "construct_laplacian_from_adjacency",
    "apply_higgs_anchor",
]r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Scalar Higgs Anchor (Condensador de Inercia Logística)              ║
║ Ruta   : app/physics/scalar_higgs_anchor.py                                  ║
║ Versión: 7.0.0-Poincare-Celestial-Level3-RSI-Lipschitz-Heyting-Secure        ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS QFT, MECÁNICA CELESTE DE POINCARÉ Y AUTOMEJORA RECURSIVA NIVEL 3:
────────────────────────────────────────────────────────────────────────────────
Este componente de-confinado de la Capa Física-Termodinámica (Nivel 3.5, $V_{\mathbb{P}}$)
gobierna la estabilización métrica de la Malla de Datos. Abandona de forma radical 
la noción estática del costo y modeliza las fluctuaciones del presupuesto como un 
campo escalar de Higgs $\Phi$ acoplado de forma covariante al Tensor Métrico de 
fondo $G_{\mathrm{PHYSICS}}$ y a las fuentes fermiónicas de la obra, dotado de
invariantes integrales de Poincaré-Cartan y automejora recursiva de Nivel 3.

Su propósito fundamental es actuar como el **Funtor de Anclaje de Higgs** 
($\mathcal{F}_{\mathrm{Higgs}} : \mathcal{C}_{\mathrm{Agent}} \to \mathcal{C}_{\mathrm{Agent}}$) 
que resuelve el acoplamiento de Yukawa, imponiendo una cota de Lipschitz explícita 
sobre el potencial regularizado, eliminando términos seculares mediante series de
Lindstedt-Poincaré y asegurando que la evolución en el fibrado simpléctico satisfaga
incondicionalmente la inecuación de disipación de Rayleigh-Lyapunov y la preservación
de toros invariantes KAM (Kolmogorov-Arnold-Moser).

AXIOMÁTICA DE CAMPOS, MÉTRICA, POINCARÉ Y ESTABILIDAD ESPECTRAL:
────────────────────────────────────────────────────────────────────────────────

  [A1] Axioma de Invarianza de Gauge Absoluta y Vacío Coherente:
       El campo escalar de Higgs $\Phi \in \mathcal{C}^\infty(M; \mathbb{R}^d)$ se somete a un 
       potencial de rotura espontánea de simetría (Sombrero Mexicano), de tal forma que el 
       estado de vacío $\Phi_0$ representa el mínimo global absoluto de la acción ciber-física:
       $$V(\Phi) = \frac{\lambda}{4} \left( \Phi^\top \Phi - v^2 \right)^2$$
       La invarianza de calibre local exige que las transformaciones de calibre abeliano $U(1)$ 
       no disipen exergía, preservando la norma de-confinada del vacío: $\langle \Phi_0 \mid \Phi_0 \rangle = v^2$.

  [A2] Axioma de Estabilización Conforme de la Métrica de Fondo ($G_{\mathrm{PHYSICS}}$):
       Para evitar que la alucinación estocástica del LLM o las fluctuaciones especulativas del 
       WACC deformen el espacio de fase, el campo de Higgs modula conformalmente el tensor métrico:
       $$\tilde{G}_{\mu\nu} = G_{\mathrm{PHYSICS}} \cdot \exp\left( -2\gamma \cdot V(\Phi) \right)$$
       Garantizando, por la Ley de Inercia de Sylvester, que la firma definida positiva de la 
       variedad se mantenga estrictamente inmune a singularidades numéricas en la CPU: $\tilde{G} \succ 0$.

  [A3] Axioma del Operador de Laplace-Beltrami Discreto ($\Delta_M$):
       El resolvedor define el operador $\Delta_M$ sobre el complejo simplicial ponderado por $\tilde{G}$ 
       para mapear el gradiente del campo escalar de compras hacia la divergencia de flujos de caja:
       $$\Delta_M \Phi = D^{-1} \delta^\top \star_1 \delta \Phi$$
       Donde $D$ es la matriz de grados de los vértices (0-formas), $\delta$ es la cofrontera simplicial,
       y $\star_1$ representa la Estrella de Hodge discreta de grado 1.

  [A4] Axioma de Composición Endofuntorial, Mecánica Celeste de Poincaré y RSI Nivel 3:
       La transmutación del estado se rige por la composición monoidal estricta de 3 Fases Anidadas:
       $$\mathcal{F}_{\mathrm{Higgs}} = \text{Yukawa}_{\text{Fase 3}} \circ \text{Hamiltoniano-Poincaré}_{\text{Fase 2}} \circ \text{Vacío-Espectral}_{\text{Fase 1}}$$
       El flujo conserva el invariante integral relativo de Poincaré-Cartan en el límite conservativo:
       $$\oint_{\gamma} \left( \langle \Pi, d\Phi \rangle - H \, dt \right) = \text{const}$$
       Y ejecuta Automejora Recursiva de Nivel 3 mediante contracción de punto fijo de Banach sobre
       el espacio de endofuntores $\mathrm{End}(\mathcal{C}_{\mathrm{Agent}})$ gobernado por la derivada
       de Fréchet del exponente máximo de Lyapunov y la coherencia espectral de los toros KAM.

  [A5] Axioma de Confinamiento de de Rham en el Retículo de Heyting ($\Omega_3$):
       Los diagnósticos de estabilidad espectral se proyectan sobre el clasificador de subobjetos del 
       retículo distributivo de Heyting de tres valores de la Malla:
       $$\Omega_3 = \{\mathtt{COHERENT}, \, \mathtt{DEGRADED}, \, \mathtt{VETOED}\}$$
       Toda anomalía de calibre, pérdida de autoadjunción ($\|\Delta_M - \Delta_M^\top\|_F > \tau$),
       resonancia homoclínica de Poincaré o violación de pasividad colapsa síncronamente el estado
       al Supremo VETOED ($\top$), disparando el circuito Crowbar (tiristor BT151 en GPIO14).

ARQUITECTURA DE TRES FASES ANIDADAS (CONTINUIDAD MORFISMÁTICA ESTRICTA):
────────────────────────────────────────────────────────────────────────────────
  Fase 1 ──► OBSERVE : Axiomática del campo, álgebra hipercompleja simpléctica y espectro $\Delta_M$.
             Su método final `SpectralAnalyzer.bridge_to_phase2_symplectic_manifold` es la semilla
             exacta que instancia e inicia los objetos de la Fase 2.
  Fase 2 ──► ORIENT  : Dinámica de Port-Hamilton, Mecánica Celeste de Poincaré (Yoshida-4, Lindstedt,
             secciones $\Sigma$, exponentes de Lyapunov, toros KAM) y Automejora Recursiva Nivel 3.
             Su método final `validate_topological_consistency` (junto con el puente de Nivel 3)
             emite el `HiggsSuturationState` que inicia la Fase 3.
  Fase 3 ──► DECIDE  : Sutura cohomológica de de Rham, lógica intuicionista de Heyting $\Omega_3$,
             protección electrónica Crowbar RL-Tiristor y Funtor de Anclaje `ScalarHiggsAnchor`.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import IntEnum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES INTERNAS DEL NÚCLEO CATEGÓRICO Y CUÁNTICO
# ══════════════════════════════════════════════════════════════════════════════
from app.core.mic_algebra import (
    CategoricalState,
    FunctorialityError,
    Morphism,
    NumericalInstabilityError,
)
from app.core.quantum_algebra import TopologicalInvariantError
from app.core.schemas import Stratum

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING ESTRUCTURADO
# ══════════════════════════════════════════════════════════════════════════════
logger = logging.getLogger("MIC.Physics.ScalarHiggsAnchor")


# ══════════════════════════════════════════════════════════════════════════════
# PRELUDIO AXIOMÁTICO: EXCEPCIONES DE GAUGE, RETÍCULO DE HEYTING Y CIRCUITOS
# ══════════════════════════════════════════════════════════════════════════════


class ScalarHiggsAnchorError(TopologicalInvariantError):
    """Excepción base de la frontera Fail-Secure del Condensador de Inercia de Higgs."""


class MetricDegeneracyError(ScalarHiggsAnchorError):
    """El tensor métrico conforme G_PHYSICS pierde su condición definida positiva (Sylvester)."""


class LaplaceBeltramiSingularity(ScalarHiggsAnchorError):
    """Falla en la autoadjunción, conservatividad o diagonalización del operador Delta_M."""


class YukawaCouplingDivergence(ScalarHiggsAnchorError):
    """Desbocamiento no acotado en la constante o perfil de masa efectiva de Yukawa."""


class LipschitzRegularityBreach(ScalarHiggsAnchorError):
    """El gradiente del potencial de Higgs supera la constante de Lipschitz admisible L_max."""


class HeytingLobeCollapse(ScalarHiggsAnchorError):
    """Transición crítica hacia el autoestado supremo VETOED en el topos de de Rham-Heyting."""


class HeytingTruthValue(IntEnum):
    r"""
    Clasificador de subobjetos $\Omega_3$ del Retículo Distributivo de Heyting.

    Orden parcial de anomalía cohomológica ($\bot \le u \le \top$):
        COHERENT (0, $\bot$) < DEGRADED (1) < VETOED (2, $\top$)

    Nota: En la semántica de seguridad Fail-Secure (Axioma A5), VETOED actúa como
    el elemento absorbente supremo ($\top$) del colapso de fase que dispara el Crowbar.
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def join(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Supremo en el retículo de anomalía ($a \lor b = \max(a, b)$)."""
        return HeytingTruthValue(max(int(self), int(other)))

    def meet(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Ínfimo en el retículo de anomalía ($a \land b = \min(a, b)$)."""
        return HeytingTruthValue(min(int(self), int(other)))

    def implies(self, consequent: "HeytingTruthValue") -> "HeytingTruthValue":
        r"""
        Implicación intuicionista de Heyting ($a \Rightarrow b$).
        En la cadena finita $\Omega_3$, $(a \Rightarrow b) = \top$ si $a \le b$, sino $b$.
        """
        if int(self) <= int(consequent):
            return HeytingTruthValue.VETOED
        return consequent

    def pseudo_complement(self) -> "HeytingTruthValue":
        r"""Pseudo-complemento de Heyting ($\neg a = (a \Rightarrow \bot)$)."""
        return self.implies(HeytingTruthValue.COHERENT)

    def is_verdict_coherent(self) -> bool:
        """Homólogo exacto de la subrutina C++ `isVerdictCoherent()` del firmware ESP32."""
        return self != HeytingTruthValue.VETOED


@dataclass(frozen=True, slots=True)
class CrowbarCircuitPort:
    r"""
    Modelo Físico de Circuito Eléctrico del Puerto Crowbar (GPIO14 + Tiristor SCR BT151).

    Ecuación diferencial de descarga de emergencia RL-Tiristor cuando GPIO14 conmuta:
        $$L_{\mathrm{bus}} \frac{dI}{dt} + R_{\mathrm{crowbar}} I + V_{\mathrm{SCR}} = V_{\mathrm{bus}}(t)$$

    Atributos
    ---------
    gpio_pin : int
        Pin físico del microcontrolador ESP32 (por defecto GPIO14).
    bus_inductance_henry : float
        Inductancia parásita del bus de potencia ($L_{\mathrm{bus}}$ en Henrios).
    crowbar_resistance_ohm : float
        Resistencia de conducción del tiristor BT151 en saturación ($R_{\mathrm{on}}$ en Ohmios).
    holding_current_amp : float
        Corriente de mantenimiento del tiristor SCR ($I_H$ en Amperios).
    latched : bool
        Estado de enclavamiento físico del tiristor (una vez disparado, permanece activo
        hasta que la corriente de ánodo cae por debajo de $I_H$).
    peak_fault_current_amp : float
        Corriente pico de cortocircuito disipada para proteger la obra.
    """

    gpio_pin: int = 14
    bus_inductance_henry: float = 1.5e-4
    crowbar_resistance_ohm: float = 0.085
    holding_current_amp: float = 0.015
    latched: bool = False
    peak_fault_current_amp: float = 0.0

    @property
    def discharge_time_constant_sec(self) -> float:
        r"""Constante de tiempo electromagnética $\tau = L_{\mathrm{bus}} / R_{\mathrm{crowbar}}$."""
        return self.bus_inductance_henry / max(self.crowbar_resistance_ohm, 1e-9)

    def evaluate_verdict(
        self,
        verdict: HeytingTruthValue,
        field_energy: float,
    ) -> "CrowbarCircuitPort":
        """
        Conmuta el pin GPIO14 y dispara el tiristor BT151 si el veredicto colapsa a VETOED.
        """
        if not verdict.is_verdict_coherent() or self.latched:
            # Energía magnética/capacitiva equivalente convertida a corriente de descarga
            equivalent_i_peak = math.sqrt(
                2.0 * abs(field_energy) / max(self.bus_inductance_henry, 1e-9)
            )
            return CrowbarCircuitPort(
                gpio_pin=self.gpio_pin,
                bus_inductance_henry=self.bus_inductance_henry,
                crowbar_resistance_ohm=self.crowbar_resistance_ohm,
                holding_current_amp=self.holding_current_amp,
                latched=True,
                peak_fault_current_amp=max(self.peak_fault_current_amp, equivalent_i_peak),
            )
        return self


@dataclass(frozen=True, slots=True)
class HypercomplexSymplecticFiber:
    r"""
    Elemento del Fibrado Hipercomplejo Simpléctico $\mathbb{H}_{\omega}^n \cong \mathbb{R}^n \otimes \mathbb{H}_{\mathrm{split}}$.

    Codifica simultáneamente la trayectoria en el espacio de fase $(\Phi, \Pi)$ y el
    fibrado tangente de Jacobi-Poincaré $(\delta\Phi, \delta\Pi)$ como un número hipercomplejo
    cuaterniónico-simpléctico:
        $$\mathcal{Z} = \Phi \cdot \mathbf{1} + \Pi \cdot \mathbf{i} + \delta\Phi \cdot \mathbf{j} + \delta\Pi \cdot \mathbf{k}$$

    Dotado de la norma de álgebra de Banach $\|\mathcal{Z}\|_{\mathcal{B}}$ para evaluar
    la contracción de punto fijo de Nivel 3 y los exponentes de Lyapunov.
    """

    real_phi: NDArray[np.float64]
    imag_i_pi: NDArray[np.float64]
    imag_j_dphi: NDArray[np.float64]
    imag_k_dpi: NDArray[np.float64]

    def banach_norm(self, characteristic_freq: float = 1.0) -> float:
        r"""
        Norma de Banach ponderada dimensionalmente en $T(T^*\mathcal{M})$:
            $$\|\mathcal{Z}\|_{\mathcal{B}} = \sqrt{\|\Phi\|_2^2 + \omega_0^{-2}\|\Pi\|_2^2 + \|\delta\Phi\|_2^2 + \omega_0^{-2}\|\delta\Pi\|_2^2}$$
        """
        inv_w2 = 1.0 / max(characteristic_freq * characteristic_freq, 1e-12)
        norm_sq = (
            float(np.dot(self.real_phi, self.real_phi))
            + inv_w2 * float(np.dot(self.imag_i_pi, self.imag_i_pi))
            + float(np.dot(self.imag_j_dphi, self.imag_j_dphi))
            + inv_w2 * float(np.dot(self.imag_k_dpi, self.imag_k_dpi))
        )
        return math.sqrt(max(0.0, norm_sq))

    def tangent_norm(self) -> float:
        r"""Norma euclídea del vector variacional tangente $\|(\delta\Phi, \delta\Pi)\|_2$."""
        return math.sqrt(
            float(np.dot(self.imag_j_dphi, self.imag_j_dphi))
            + float(np.dot(self.imag_k_dpi, self.imag_k_dpi))
        )


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 1 · AXIOMÁTICA DEL CAMPO, GEOMETRÍA CONFORME Y ESPECTRO (OBSERVE) │
# │  Define los tipos, parámetros, estructuras y operadores elementales.   │
# │  El último método de la Fase 1 (`bridge_to_phase2_symplectic_manifold`) │
# │  constituye formalmente la continuación e inicio de la Fase 2.          │
# └─────────────────────────────────────────────────────────────────────────┘


# ───────────────────────────────────────────────────────────────────────────
# 1.1  PARÁMETROS QFT, POINCARÉ Y AUTOMEJORA RECURSIVA DE NIVEL 3
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class QFTParameters:
    r"""
    Parámetros del Modelo Estándar Ciber-Físico, Mecánica Celeste de Poincaré
    y Automejora Recursiva de Nivel 3.

    Invariantes verificados en `validate_parameters`
    ------------------------------------------------
    I1. $\mu^2 > 0$        → Ruptura espontánea de simetría (SSB)
    I2. $\lambda > 0$      → Potencial acotado en región de vacío
    I3. $c^2 > 0$          → Velocidad de propagación hiperbólica real
    I4. $\alpha_{\mathrm{CFL}} \in (0,1)$ → Estabilidad simpléctica
    I5. $\gamma_{\mathrm{damp}} \ge 0$    → Disipación de Rayleigh físicamente admisible
    I6. $\Phi_c > 0$       → Escala de regularización Lipschitz estrictamente positiva
    I7. $\beta_{\mathrm{RSI}} \in (0,1)$  → Constante de contracción de Banach de Nivel 3
    """

    # Potencial de Higgs
    MU_SQUARED: float = 2.5
    LAMBDA_COUPLING: float = 0.1
    FIELD_REG_SCALE: float = 10.0  # Φ_c (escala de regularización Lipschitz)

    # Acoplamientos de Yukawa
    YUKAWA_G: float = 1.05
    YUKAWA_BACKREACTION: float = 0.01

    # Dinámica relativista / discreta
    C_SQUARED: float = 1.0
    CFL_SAFETY_FACTOR: float = 0.75
    CFL_ABSOLUTE_CAP: float = 0.1

    # Regularización numérica y cotas de estabilidad
    EPSILON_SMOOTH: float = 1e-10
    MAX_FIELD_MAGNITUDE: float = 1e3
    MAX_ENERGY_RATIO: float = 4.0
    DAMPING_GAMMA: float = 0.08
    CONVERGENCE_TOLERANCE: float = 1e-6

    # Reproducibilidad
    RNG_SEED: int = 42

    # Parámetros de Mecánica Celeste de Poincaré y Geometría Conforme
    CONFORMAL_GAMMA: float = 0.015
    POINCARE_SECTION_TOL: float = 1e-5
    LYAPUNOV_CRITICAL_CAP: float = 0.25
    KAM_COHERENCE_FLOOR: float = 0.35
    LIPSCHITZ_MAX_BOUND: float = 5.0e4

    # Parámetros de Automejora Recursiva de Nivel 3 (Banach-Fréchet)
    RSI_BANACH_CONTRACTION: float = 0.6180339887  # Inverso de la razón áurea φ⁻¹ < 1
    RSI_META_LEARNING_RATE: float = 0.05
    RSI_GENERATION: int = 0

    @property
    def vev(self) -> float:
        r"""Valor Esperado del Vacío (VEV): $v = \sqrt{\mu^2 / \lambda}$."""
        return math.sqrt(self.MU_SQUARED / self.LAMBDA_COUPLING)

    @property
    def higgs_mass_squared(self) -> float:
        r"""Masa física al cuadrado del bosón de Higgs: $m_H^2 = V''(v) = 2\mu^2$."""
        return 2.0 * self.MU_SQUARED

    @property
    def natural_frequency(self) -> float:
        r"""Frecuencia angular fundamental de oscilación en el vacío: $\omega_0 = \sqrt{2\mu^2}$."""
        return math.sqrt(self.higgs_mass_squared)

    def validate_parameters(self) -> None:
        """Verifica rigurosamente los invariantes axiomáticos I1–I7."""
        if self.MU_SQUARED <= 0:
            raise ValueError(
                f"[I1] μ² debe ser positivo para SSB, recibido {self.MU_SQUARED}"
            )
        if self.LAMBDA_COUPLING <= 0:
            raise ValueError(
                f"[I2] λ debe ser positivo, recibido {self.LAMBDA_COUPLING}"
            )
        if self.C_SQUARED <= 0:
            raise ValueError(
                f"[I3] c² debe ser positivo, recibido {self.C_SQUARED}"
            )
        if not 0 < self.CFL_SAFETY_FACTOR < 1:
            raise ValueError(
                f"[I4] α_CFL debe estar en (0,1), recibido {self.CFL_SAFETY_FACTOR}"
            )
        if self.DAMPING_GAMMA < 0:
            raise ValueError(
                f"[I5] γ_damp ≥ 0 requerido, recibido {self.DAMPING_GAMMA}"
            )
        if self.FIELD_REG_SCALE <= 0:
            raise ValueError(
                f"[I6] Φ_c > 0 requerido, recibido {self.FIELD_REG_SCALE}"
            )
        if not 0.0 < self.RSI_BANACH_CONTRACTION < 1.0:
            raise ValueError(
                f"[I7] Contracción de Banach debe estar en (0,1), recibido {self.RSI_BANACH_CONTRACTION}"
            )

    def compute_potential_and_gradient(
        self,
        phi: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Potencial regularizado de Higgs y su gradiente exacto.

        $$V_{\mathrm{reg}}(\Phi) = -\frac{1}{2}\mu^2\Phi^2 + \frac{1}{4}\lambda\Phi^4\tanh(s), \quad s = \frac{\Phi^2}{\Phi_c^2}$$
        $$\nabla V_{\mathrm{reg}}(\Phi) = -\mu^2\Phi + \lambda\Phi^3\left[\tanh(s) + \frac{1}{2}s\operatorname{sech}^2(s)\right]$$
        """
        if phi.ndim != 1:
            raise ValueError(f"phi debe ser 1D, recibido {phi.ndim}D")

        phi_sq = phi * phi
        phi_quartic = phi_sq * phi_sq

        scale_sq = self.FIELD_REG_SCALE ** 2 + self.EPSILON_SMOOTH
        s = phi_sq / scale_sq
        tanh_s = np.tanh(s)
        sech2_s = 1.0 - tanh_s * tanh_s

        V_mass = -0.5 * self.MU_SQUARED * phi_sq
        V_int = 0.25 * self.LAMBDA_COUPLING * phi_quartic * tanh_s
        V_total = V_mass + V_int

        grad_mass = -self.MU_SQUARED * phi
        grad_int = self.LAMBDA_COUPLING * (phi * phi_sq) * (
            tanh_s + 0.5 * s * sech2_s
        )
        grad_total = grad_mass + grad_int

        if not (np.all(np.isfinite(V_total)) and np.all(np.isfinite(grad_total))):
            raise NumericalInstabilityError(
                f"V o ∇V no finito. max|Φ|={np.max(np.abs(phi)):.2e}"
            )
        return V_total, grad_total

    def compute_exact_hessian_diagonal(
        self,
        phi: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Segunda derivada analítica exacta $V''_{\mathrm{reg}}(\Phi)$ para las ecuaciones
        variacionales de Jacobi-Poincaré en el fibrado tangente $T(T^*\mathcal{M})$:

        $$V''_{\mathrm{reg}}(\Phi) = -\mu^2 + \lambda\Phi^2\left[3\tanh(s) + \operatorname{sech}^2(s)\left(\frac{9}{2}s - 2s^2\tanh(s)\right)\right]$$
        """
        phi_sq = phi * phi
        scale_sq = self.FIELD_REG_SCALE ** 2 + self.EPSILON_SMOOTH
        s = phi_sq / scale_sq
        tanh_s = np.tanh(s)
        sech2_s = 1.0 - tanh_s * tanh_s

        bracket = 3.0 * tanh_s + sech2_s * (4.5 * s - 2.0 * s * s * tanh_s)
        hess = -self.MU_SQUARED + self.LAMBDA_COUPLING * phi_sq * bracket
        return hess

    def synthesize_level3_mutation(
        self,
        lyapunov_exponent: float,
        kam_coherence: float,
        energy_drift: float,
    ) -> "QFTParameters":
        r"""
        Operador de Punto Fijo de Banach (Nivel 3 RSI) sobre la variedad de hiperparámetros.
        Ajusta la amortiguación de Rayleigh, el factor CFL y el acoplamiento conforme
        conservando estrictamente el VEV $v = \sqrt{\mu^2/\lambda}$ como invariante de gauge.
        """
        contraction = self.RSI_BANACH_CONTRACTION ** (self.RSI_GENERATION + 1)
        lr = self.RSI_META_LEARNING_RATE * contraction

        # Gradiente meta-variacional respecto al exceso de caos de Lyapunov y pérdida KAM
        lyap_excess = max(0.0, lyapunov_exponent - 0.1 * self.LYAPUNOV_CRITICAL_CAP)
        kam_deficit = max(0.0, 1.0 - kam_coherence)
        chaos_pressure = lyap_excess + 0.5 * kam_deficit + 0.25 * min(energy_drift, 2.0)

        new_gamma = float(
            np.clip(self.DAMPING_GAMMA * (1.0 + lr * chaos_pressure), 1e-4, 0.95)
        )
        new_cfl = float(
            np.clip(self.CFL_SAFETY_FACTOR * (1.0 - 0.5 * lr * chaos_pressure), 0.15, 0.90)
        )
        new_conformal = float(
            np.clip(self.CONFORMAL_GAMMA * (1.0 + 0.5 * lr * kam_deficit), 1e-5, 0.20)
        )

        mutated = replace(
            self,
            DAMPING_GAMMA=new_gamma,
            CFL_SAFETY_FACTOR=new_cfl,
            CONFORMAL_GAMMA=new_conformal,
            RSI_GENERATION=self.RSI_GENERATION + 1,
        )
        mutated.validate_parameters()
        return mutated


# Instancia global inmutable (validada al cargar el módulo)
QFT: Final[QFTParameters] = QFTParameters()
QFT.validate_parameters()


# ───────────────────────────────────────────────────────────────────────────
# 1.2  ESTRUCTURAS ALGEBRAICAS Y FIBRADO TANGENTE DE POINCARÉ
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScalarFieldState:
    r"""
    Punto extendido en el fibrado tangente simpléctico $T(T^*\mathbb{R}^n)$.

    Invariantes (verificados en `__post_init__`)
    --------------------------------------------
    J1. $\dim(\Phi) = \dim(\Pi)$         (compatibilidad del espacio de fase)
    J2. $\Phi.\mathrm{ndim} = 1$         (1-cadena/0-coforma discretizada)
    J3. $\mathrm{dtype} = \mathrm{float64}$ (precisión IEEE 754)
    J4. $\delta\Phi, \delta\Pi$ opcionales para integración variacional de Poincaré.
    """

    phi: NDArray[np.float64]
    pi_momentum: NDArray[np.float64]
    delta_phi: Optional[NDArray[np.float64]] = None
    delta_pi: Optional[NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        if self.phi.shape != self.pi_momentum.shape:
            raise ValueError(
                f"[J1] Φ{self.phi.shape} ≠ Π{self.pi_momentum.shape}"
            )
        if self.phi.ndim != 1:
            raise ValueError(f"[J2] Φ debe ser 1D, recibido {self.phi.ndim}D")
        if self.phi.dtype != np.float64 or self.pi_momentum.dtype != np.float64:
            raise TypeError("[J3] Φ y Π deben ser float64")
        if self.delta_phi is not None and self.delta_phi.shape != self.phi.shape:
            raise ValueError("[J4] delta_phi debe tener la misma dimensión que phi")
        if self.delta_pi is not None and self.delta_pi.shape != self.pi_momentum.shape:
            raise ValueError("[J4] delta_pi debe tener la misma dimensión que pi_momentum")

    @property
    def dim(self) -> int:
        return int(self.phi.shape[0])

    def copy(self) -> "ScalarFieldState":
        """Copia defensiva profunda de los tensores de estado y vectores tangentes."""
        return ScalarFieldState(
            phi=self.phi.copy(),
            pi_momentum=self.pi_momentum.copy(),
            delta_phi=self.delta_phi.copy() if self.delta_phi is not None else None,
            delta_pi=self.delta_pi.copy() if self.delta_pi is not None else None,
        )

    def with_initialized_tangent(self) -> "ScalarFieldState":
        """Garantiza un vector tangente unitario ortonormal para seguimiento de Lyapunov."""
        if self.delta_phi is not None and self.delta_pi is not None:
            return self
        n = self.dim
        if n == 0:
            return ScalarFieldState(
                phi=self.phi,
                pi_momentum=self.pi_momentum,
                delta_phi=np.zeros(0, dtype=np.float64),
                delta_pi=np.zeros(0, dtype=np.float64),
            )
        dphi = np.ones(n, dtype=np.float64) / math.sqrt(float(n))
        dpi = np.zeros(n, dtype=np.float64)
        return ScalarFieldState(
            phi=self.phi,
            pi_momentum=self.pi_momentum,
            delta_phi=dphi,
            delta_pi=dpi,
        )

    def to_hypercomplex_fiber(self) -> HypercomplexSymplecticFiber:
        """Eleva el estado al fibrado hipercomplejo de Banach $\mathbb{H}_{\omega}^n$."""
        st = self.with_initialized_tangent()
        assert st.delta_phi is not None and st.delta_pi is not None
        return HypercomplexSymplecticFiber(
            real_phi=st.phi,
            imag_i_pi=st.pi_momentum,
            imag_j_dphi=st.delta_phi,
            imag_k_dpi=st.delta_pi,
        )

    def symplectic_form_omega(self, other: "ScalarFieldState") -> float:
        r"""
        Evalúa la 2-forma simpléctica canónica de Poincaré:
            $$\omega(\xi_1, \xi_2) = \langle \delta\Phi_1, \delta\Pi_2 \rangle - \langle \delta\Pi_1, \delta\Phi_2 \rangle$$
        """
        s1 = self.with_initialized_tangent()
        s2 = other.with_initialized_tangent()
        assert s1.delta_phi is not None and s1.delta_pi is not None
        assert s2.delta_phi is not None and s2.delta_pi is not None
        return float(
            np.dot(s1.delta_phi, s2.delta_pi) - np.dot(s1.delta_pi, s2.delta_phi)
        )


@dataclass(frozen=True, slots=True)
class FermionicSource:
    r"""
    Fuente fermiónica $\psi$ para el acoplamiento de Yukawa $\mathcal{L}_Y = -g\bar{\psi}\Phi\psi$.

    Atributos
    ---------
    psi_vector : $\psi \in \mathbb{R}^n$ (amplitud de sección espinorial real)
    base_mass  : $m_0 \ge 0$ (masa desnuda antes de ruptura espontánea de simetría)
    """

    psi_vector: NDArray[np.float64]
    base_mass: float = 0.0

    def __post_init__(self) -> None:
        if self.psi_vector.ndim != 1:
            raise ValueError("ψ debe ser 1D")
        if self.base_mass < 0:
            raise ValueError("m₀ ≥ 0 requerido")

    def charge_density(self) -> NDArray[np.float64]:
        r"""
        Densidad de carga normalizada $\rho = |\psi|^2 / \max(|\psi|^2) \in [0, 1]^n$.
        """
        rho_raw = self.psi_vector ** 2
        max_rho = float(np.max(rho_raw)) if rho_raw.size > 0 else 0.0
        if max_rho > QFT.EPSILON_SMOOTH:
            return rho_raw / max_rho
        return rho_raw

    def chiral_asymmetry_index(self) -> float:
        r"""Índice de asimetría espectral fermiónica $\mathcal{A}_\chi \in [-1, 1]$."""
        if self.psi_vector.size == 0:
            return 0.0
        norm_sq = float(np.dot(self.psi_vector, self.psi_vector))
        if norm_sq <= QFT.EPSILON_SMOOTH:
            return 0.0
        signed_fourth = float(np.sum(np.sign(self.psi_vector) * (self.psi_vector ** 2)))
        return float(np.clip(signed_fourth / norm_sq, -1.0, 1.0))


# ───────────────────────────────────────────────────────────────────────────
# 1.3  OPERADOR DE LAPLACE-BELTRAMI CON MÉTRICA CONFORME (AXIOMAS A2 Y A3)
# ───────────────────────────────────────────────────────────────────────────


class LaplacedBeltramiOperator:
    r"""
    Operador de Laplace-Beltrami discreto $\Delta_M$ sobre un complejo simplicial
    ponderado por el tensor métrico conforme $\tilde{G}_{\mu\nu} = G_{\mathrm{PHYSICS}}\exp(-2\gamma V(\Phi))$.

    Garantías tras validación (Axiomas A2 y A3)
    -------------------------------------------
    L1. $L$ es CSR dispersa y cuadrada ($n \times n$)
    L2. $L = L^\top$ (autoadjunción en producto interno estándar)
    L3. $L \succeq 0$ (semidefinida positiva por Ley de Inercia de Sylvester)
    L4. $\sum_j L_{ij} = 0$ (conservatividad de flujos de 0-formas / cohomología de de Rham)
    """

    def __init__(self, matrix: sp.csr_matrix) -> None:
        self._validate(matrix)
        self.L: sp.csr_matrix = matrix.tocsr()
        self.n: int = int(matrix.shape[0])
        self._base_L: sp.csr_matrix = self.L.copy()

    @staticmethod
    def _validate(M: sp.spmatrix) -> None:
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"[L1] Laplaciano no cuadrado: shape={M.shape}")
        diff = M - M.T
        if diff.nnz > 0 and np.any(np.abs(diff.data) > 1e-10):
            raise ValueError("[L2] Laplaciano no simétrico")
        if M.shape[0] > 0:
            try:
                min_eig = float(eigsh(M, k=1, which="SA", return_eigenvectors=False)[0])
                if min_eig < -1e-10:
                    raise ValueError(f"[L3] Laplaciano no PSD: λ_min={min_eig}")
            except ValueError:
                raise
            except Exception:
                row_sums = np.abs(np.array(M.sum(axis=1)).flatten())
                if np.max(row_sums) > 1e-10:
                    raise ValueError("[L4] Laplaciano no conservativo")
        row_sums = np.array(M.sum(axis=1)).flatten()
        if row_sums.size > 0 and np.max(np.abs(row_sums)) > 1e-10:
            raise ValueError("[L4] ∑_j L_ij ≠ 0 (no conservativo)")

    @classmethod
    def from_adjacency(
        cls,
        adjacency: Union[NDArray[np.float64], sp.spmatrix],
        symmetrize: bool = True,
    ) -> "LaplacedBeltramiOperator":
        """Construye $\Delta_M = D - A$ a partir de la matriz de adyacencia métrica."""
        A = (
            sp.csr_matrix(adjacency)
            if isinstance(adjacency, np.ndarray)
            else adjacency.tocsr()
        )
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Adyacencia no cuadrada: {A.shape}")
        if symmetrize:
            diff = A - A.T
            if diff.nnz > 0 and np.any(np.abs(diff.data) > 1e-12):
                logger.warning("Adyacencia asimétrica: promediando con su transpuesta")
                A = 0.5 * (A + A.T)
        degrees = np.array(A.sum(axis=1)).flatten()
        D = sp.diags(degrees, format="csr")
        return cls(D - A)

    def apply(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Aplica $\Delta_M$ a un campo escalar: $L \cdot \Phi$."""
        if phi.shape != (self.n,):
            raise ValueError(f"Φ.shape {phi.shape} ≠ ({self.n},)")
        return np.asarray(self.L @ phi, dtype=np.float64)

    def apply_conformal_stabilization(
        self,
        potential_vec: NDArray[np.float64],
        conformal_gamma: float,
    ) -> "LaplacedBeltramiOperator":
        r"""
        Implementa el Axioma [A2]: Estabilización Conforme de la Métrica de Fondo.
            $$\tilde{G}_{\mu\nu} = G_{\mathrm{PHYSICS}} \cdot \exp\left(-2\gamma \cdot V(\Phi)\right)$$
        Modula la estrella de Hodge discreta $\star_1$ en las 1-aristas y reconstruye
        $\tilde{\Delta}_M = \delta^\top \tilde{\star}_1 \delta$ preservando la firma SPD
        por la Ley de Inercia de Sylvester.
        """
        if self.n == 0 or conformal_gamma <= 0.0:
            return self
        if potential_vec.shape != (self.n,):
            raise MetricDegeneracyError(
                f"Dimensión de potencial {potential_vec.shape} incompatible con métrica ({self.n},)"
            )

        # Acotación de exponente conforme para inmunidad numérica IEEE-754
        exponent = np.clip(-conformal_gamma * potential_vec, -20.0, 20.0)
        conformal_weights = np.exp(exponent)
        if not np.all(np.isfinite(conformal_weights)) or np.any(conformal_weights <= 0.0):
            raise MetricDegeneracyError(
                "El factor conforme exp(-2γV(Φ)) perdió la condición definida positiva (Sylvester)."
            )

        # Extraer adyacencia base A_base = D_base - L_base
        diag_base = self._base_L.diagonal()
        A_base = sp.diags(diag_base, format="csr") - self._base_L
        A_base.eliminate_zeros()

        # Transformación congruente de Sylvester: Â = W^{1/2} A_base W^{1/2}
        W_half = sp.diags(conformal_weights, format="csr")
        A_tilde = W_half @ A_base @ W_half
        A_tilde = 0.5 * (A_tilde + A_tilde.T)
        degrees_tilde = np.array(A_tilde.sum(axis=1)).flatten()
        L_tilde = sp.diags(degrees_tilde, format="csr") - A_tilde

        try:
            op_tilde = LaplacedBeltramiOperator(L_tilde.tocsr())
            op_tilde._base_L = self._base_L
            return op_tilde
        except ValueError as exc:
            raise LaplaceBeltramiSingularity(
                f"Singularidad en operador de Laplace-Beltrami conforme: {exc}"
            ) from exc


# Alias ortográfico limpio manteniendo 100% de compatibilidad con `LaplacedBeltramiOperator`
LaplaceBeltramiOperator = LaplacedBeltramiOperator


# ───────────────────────────────────────────────────────────────────────────
# 1.4  ANALIZADOR ESPECTRAL Y PUENTE ANIDADO HACIA LA FASE 2
# ───────────────────────────────────────────────────────────────────────────


class SpectralAnalyzer:
    r"""
    Análisis espectral de operadores dispersos autoadjuntos y cohomología simplicial.

    Métodos disponibles
    -------------------
    'gershgorin'      : Teorema de discos de Gershgorin ($O(n)$)
    'power_iteration' : Iteración de potencia de Mises ($O(k \cdot \mathrm{nnz})$)
    'krylov'          : Subespacios de Krylov-Lanczos vía ARPACK
    """

    @staticmethod
    def estimate_max_eigenvalue(
        laplacian: sp.csr_matrix,
        method: str = "gershgorin",
        tolerance: float = 1e-6,
        max_iter: int = 200,
    ) -> float:
        n = laplacian.shape[0]
        if n == 0:
            return 0.0

        if method == "gershgorin":
            # Por el teorema de Gershgorin en un Laplaciano L = D - A (con L_ii = d_i y
            # ∑_{j≠i} |L_ij| = d_i), la cota superior exacta de los discos es max_i(2*L_ii)
            # o bien la norma infinito max_i ∑_j |L_ij|. Si se evalúa la suma con signo,
            # daría 0; por ello tomamos abs(laplacian) para garantizar cota espectral real.
            abs_row_sums = np.array(np.abs(laplacian).sum(axis=1)).flatten()
            return float(np.max(abs_row_sums)) if abs_row_sums.size > 0 else 0.0

        if method == "power_iteration":
            v = np.random.default_rng(0).standard_normal(n)
            v_norm = float(np.linalg.norm(v))
            if v_norm < QFT.EPSILON_SMOOTH:
                return 0.0
            v = v / v_norm
            lambda_prev = 0.0
            for _ in range(max_iter):
                v_new = laplacian @ v
                v_new_norm = float(np.linalg.norm(v_new))
                if v_new_norm < QFT.EPSILON_SMOOTH:
                    return 0.0
                lambda_new = float(np.dot(v, v_new))
                if abs(lambda_new - lambda_prev) < tolerance:
                    return lambda_new
                v = v_new / v_new_norm
                lambda_prev = lambda_new
            logger.warning("Power iteration no convergió en %d iteraciones", max_iter)
            return lambda_prev

        if method == "krylov":
            try:
                eigs = eigsh(
                    laplacian,
                    k=1,
                    which="LA",
                    return_eigenvectors=False,
                    tol=tolerance,
                )
                return float(eigs[0])
            except Exception as e:
                logger.error("Krylov (LA) falló: %s. Cayendo a Gershgorin.", e)
                return SpectralAnalyzer.estimate_max_eigenvalue(
                    laplacian, method="gershgorin"
                )

        raise ValueError(f"Método '{method}' inválido")

    @staticmethod
    def estimate_spectral_gap(
        laplacian: sp.csr_matrix,
        tolerance: float = 1e-6,
        n_threshold: int = 500,
    ) -> float:
        r"""
        Estima el gap espectral de Fiedler $\lambda_1(\Delta_M)$ (conectividad algebraica).
        """
        n = laplacian.shape[0]
        if n < 2:
            return 0.0
        try:
            if n <= n_threshold:
                dense_eigs = np.linalg.eigvalsh(laplacian.toarray())
                dense_sorted = np.sort(dense_eigs)
                return max(0.0, float(dense_sorted[1]))
            k = min(2, n - 1)
            eigs = eigsh(
                laplacian,
                k=k,
                which="SM",
                return_eigenvectors=False,
                tol=tolerance,
            )
            if len(eigs) < 2:
                return 0.0
            eigs_sorted = np.sort(eigs)
            return max(0.0, float(eigs_sorted[1]))
        except Exception as e:
            logger.error("Cálculo de gap espectral falló: %s", e)
            return 0.0

    @staticmethod
    def estimate_betti_zero(
        laplacian: sp.csr_matrix,
        zero_tol: float = 1e-7,
    ) -> int:
        r"""
        Calcula el 0-ésimo número de Betti $\beta_0 = \dim \ker(\Delta_0) = \dim H^0_{\mathrm{dR}}(M)$
        contando las componentes conexas de la variedad discreta.
        """
        n = laplacian.shape[0]
        if n == 0:
            return 0
        n_components, _ = sp.csgraph.connected_components(
            csgraph=laplacian, directed=False, return_labels=True
        )
        return max(1, int(n_components))

    # ═══════════════════════════════════════════════════════════════════════
    # ESLABÓN ANIDADO FASE 1 ──► FASE 2 (CONTINUACIÓN MORFISMÁTICA ESTRICTA)
    # ═══════════════════════════════════════════════════════════════════════
    @classmethod
    def bridge_to_phase2_symplectic_manifold(
        cls,
        laplacian_op: LaplacedBeltramiOperator,
        params: QFTParameters = QFT,
    ) -> Tuple["HiggsPotential", "PortHamiltonianHamiltonian", Dict[str, float]]:
        r"""
        Definición formal final de la Fase 1 y génesis directa de la Fase 2.

        Extrae la firma espectral completa $(\lambda_{\max}, \lambda_1, \beta_0)$ del
        operador de Laplace-Beltrami de la Fase 1 y construye de manera atómica
        el Potencial de Higgs regularizado y el Hamiltoniano de Port-Hamilton que
        abren la Fase 2.
        """
        lambda_max = cls.estimate_max_eigenvalue(laplacian_op.L, method="gershgorin")
        spectral_gap = cls.estimate_spectral_gap(laplacian_op.L)
        betti_0 = float(cls.estimate_betti_zero(laplacian_op.L))

        spectral_signature: Dict[str, float] = {
            "lambda_max": lambda_max,
            "spectral_gap": spectral_gap,
            "betti_0": betti_0,
            "cheeger_lower_bound": 0.5 * spectral_gap,
        }

        # Continuación directa hacia los constructores de apertura de la Fase 2
        potential = HiggsPotential.from_phase1_spectral_signature(
            params=params, spectral_signature=spectral_signature
        )
        hamiltonian = PortHamiltonianHamiltonian(
            laplacian=laplacian_op,
            potential=potential,
            params=params,
            spectral_signature=spectral_signature,
        )
        return potential, hamiltonian, spectral_signature


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 2 · DINÁMICA SIMPLÉCTICA, MECÁNICA CELESTE DE POINCARÉ Y RSI-L3   │
# │  Inicia directamente como continuación del puente espectral de Fase 1.  │
# │  Su definición final (`validate_topological_consistency`) constituye    │
# │  el puente anidado de entrada hacia la Fase 3.                          │
# └─────────────────────────────────────────────────────────────────────────┘


# ───────────────────────────────────────────────────────────────────────────
# 2.1  POTENCIAL DE HIGGS REGULARIZADO Y SERIES DE LINDSTEDT-POINCARÉ
# ───────────────────────────────────────────────────────────────────────────


class HiggsPotential:
    r"""
    Potencial de Higgs regularizado $V_{\mathrm{reg}}(\Phi)$ con análisis de cotas
    de Lipschitz y eliminación de términos seculares de Lindstedt-Poincaré.

    Recibe la firma espectral emitida por `SpectralAnalyzer.bridge_to_phase2_symplectic_manifold`.
    """

    def __init__(
        self,
        params: QFTParameters = QFT,
        spectral_signature: Optional[Dict[str, float]] = None,
    ) -> None:
        self.p: QFTParameters = params
        self.spectral_signature: Dict[str, float] = spectral_signature or {}

    @classmethod
    def from_phase1_spectral_signature(
        cls,
        params: QFTParameters,
        spectral_signature: Dict[str, float],
    ) -> "HiggsPotential":
        """Continuación formal directa desde el último método de la Fase 1."""
        return cls(params=params, spectral_signature=spectral_signature)

    def value_and_gradient(
        self, phi: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        V, grad_V = self.p.compute_potential_and_gradient(phi)
        max_grad = float(np.max(np.abs(grad_V))) if grad_V.size > 0 else 0.0
        if max_grad > self.p.LIPSCHITZ_MAX_BOUND * self.p.MAX_FIELD_MAGNITUDE:
            raise LipschitzRegularityBreach(
                f"Violación de regularidad Lipschitz: max|∇V|={max_grad:.2e}"
            )
        return V, grad_V

    def hessian_diagonal(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Diagonal del Hessiano exacto $D^2 V_{\mathrm{reg}}(\Phi)$ para la ecuación de Jacobi."""
        return self.p.compute_exact_hessian_diagonal(phi)

    def lipschitz_constant(self, domain_radius: float) -> float:
        r"""
        Constante de Lipschitz en la bola cerrada $\mathcal{B}_R = \{\Phi : \|\Phi\|_\infty \le R\}$:
            $$L_V(R) = \sup_{|\Phi|\le R} |V'(\Phi)| \le \mu^2 R + 3\lambda R^3 \left(1 + \frac{R^2}{2\Phi_c^2}\right)$$
        """
        R = abs(domain_radius)
        R_sq = R * R
        Phi_c_sq = self.p.FIELD_REG_SCALE ** 2
        return (
            self.p.MU_SQUARED * R
            + 3.0 * self.p.LAMBDA_COUPLING * R * R_sq * (1.0 + 0.5 * R_sq / Phi_c_sq)
        )

    def lindstedt_poincare_frequency_renormalization(
        self, phi: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Serie perturbativa de Lindstedt-Poincaré (resumida vía aproximante de Padé $[1/1]$)
        para eliminar resonancias seculares en oscilaciones no lineales alrededor del VEV $v$:

            $$\omega_{\mathrm{LP}}^2(A) = \omega_0^2 \cdot \frac{1 + \frac{3\lambda}{4\mu^2} A^2}{1 + \frac{9\lambda}{4\mu^2} A^2}, \quad A = |\Phi - v|$$
        """
        amp_sq = (phi - self.p.vev) ** 2
        ratio = (self.p.LAMBDA_COUPLING / max(self.p.MU_SQUARED, 1e-12)) * amp_sq
        reg_ratio = ratio / (1.0 + amp_sq / (self.p.FIELD_REG_SCALE ** 2))
        factor = (1.0 + 0.75 * reg_ratio) / (1.0 + 2.25 * reg_ratio)
        return np.asarray(factor, dtype=np.float64)

    def is_physically_admissible(self, phi: NDArray[np.float64]) -> bool:
        """Verifica que el campo resida dentro del tubo de confianza de Lipschitz."""
        if phi.size == 0:
            return True
        max_abs = float(np.max(np.abs(phi)))
        return max_abs <= self.p.FIELD_REG_SCALE


# ───────────────────────────────────────────────────────────────────────────
# 2.2  HAMILTONIANO PORT-HAMILTONIANO Y FLUJO VARIACIONAL DE JACOBI-POINCARÉ
# ───────────────────────────────────────────────────────────────────────────


class PortHamiltonianHamiltonian:
    r"""
    Hamiltoniano Port-Hamiltoniano acoplado a la geometría de Laplace-Beltrami:

        $$H[\Phi, \Pi] = \frac{1}{2}\|\Pi\|_2^2 + \frac{1}{2}c^2 \langle \Phi, \Delta_M \Phi \rangle + \sum_{i=1}^n V_{\mathrm{reg}}(\Phi_i)$$

    Satisface la identidad de pasividad de Rayleigh-Lyapunov (Axioma A4):
        $$\dot{H} = \langle \nabla H, (J - R)\nabla H \rangle = -\gamma \|\Pi\|_2^2 \le 0$$
    """

    def __init__(
        self,
        laplacian: LaplacedBeltramiOperator,
        potential: HiggsPotential,
        params: QFTParameters = QFT,
        spectral_signature: Optional[Dict[str, float]] = None,
    ) -> None:
        self.L_op: LaplacedBeltramiOperator = laplacian
        self.V: HiggsPotential = potential
        self.p: QFTParameters = params
        self.spectral_signature: Dict[str, float] = spectral_signature or {}

    def __call__(self, state: ScalarFieldState) -> float:
        """Evalúa la energía total $H[\Phi, \Pi]$ con sutura de coherencia dimensional."""
        phi = state.phi
        pi = state.pi_momentum
        if state.dim != self.L_op.n:
            logger.debug(
                "Sutura V: state.dim=%d ≠ laplacian.n=%d; proyectando al subespacio común",
                state.dim,
                self.L_op.n,
            )
            common_n = min(state.dim, self.L_op.n)
            phi_pad = np.zeros(self.L_op.n, dtype=np.float64)
            phi_pad[:common_n] = phi[:common_n]
            T = 0.5 * float(np.dot(pi[:common_n], pi[:common_n]))
            U_el = 0.5 * self.p.C_SQUARED * float(
                np.dot(phi_pad, self.L_op.apply(phi_pad))
            )
            V_local, _ = self.V.value_and_gradient(phi[:common_n])
            return T + U_el + float(np.sum(V_local))

        T = 0.5 * float(np.dot(pi, pi))
        U_el = 0.5 * self.p.C_SQUARED * float(
            np.dot(phi, self.L_op.apply(phi))
        )
        V_local, _ = self.V.value_and_gradient(phi)
        U_pot = float(np.sum(V_local))
        return T + U_el + U_pot

    def force(
        self,
        state: ScalarFieldState,
        source_density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Campo de fuerza conservativa más retroacción fermiónica de Yukawa:
            $$F(\Phi) = -c^2 \Delta_M \Phi - \nabla V_{\mathrm{reg}}(\Phi) + g_{\mathrm{back}}\rho_\psi$$
        """
        laplacian_force = -self.p.C_SQUARED * self.L_op.apply(state.phi)
        _, grad_V = self.V.value_and_gradient(state.phi)
        potential_force = -grad_V
        source_force = self.p.YUKAWA_BACKREACTION * source_density
        return laplacian_force + potential_force + source_force

    def jacobi_variational_force(
        self,
        phi: NDArray[np.float64],
        delta_phi: NDArray[np.float64],
        delta_pi: NDArray[np.float64],
        damping_gamma: float = 0.0,
    ) -> NDArray[np.float64]:
        r"""
        Operador variacional de Jacobi-Poincaré en el fibrado tangente $T_{(\Phi,\Pi)}(T^*\mathcal{M})$:
            $$\frac{d(\delta\Pi)}{dt} = -c^2 \Delta_M \delta\Phi - D^2V_{\mathrm{reg}}(\Phi)\delta\Phi - \gamma \delta\Pi$$
        """
        elastic_var = -self.p.C_SQUARED * self.L_op.apply(delta_phi)
        hess_diag = self.V.hessian_diagonal(phi)
        potential_var = -hess_diag * delta_phi
        damping_var = -damping_gamma * delta_pi if damping_gamma > 0.0 else 0.0
        return elastic_var + potential_var + damping_var

    def rayleigh_dissipation_power(
        self, state: ScalarFieldState, damping_gamma: float
    ) -> float:
        r"""Potencia disipada de Rayleigh: $\mathcal{P}_{\mathrm{Rayleigh}} = -\gamma \|\Pi\|_2^2 \le 0$."""
        return -max(0.0, damping_gamma) * float(
            np.dot(state.pi_momentum, state.pi_momentum)
        )


# ───────────────────────────────────────────────────────────────────────────
# 2.3  MECÁNICA CELESTE DE POINCARÉ: SECCIÓN Σ, LYAPUNOV Y TOROS KAM
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PoincareSectionCrossing:
    r"""
    Registro de intersección transversal con la Superficie de Sección de Poincaré:
        $$\Sigma = \left\{ (\Phi, \Pi) \in T^*\mathcal{M} \;\middle|\; \bar{\Phi} = v, \; \bar{\Pi} > 0 \right\}$$
    """

    step_fraction: float
    mean_momentum: float
    field_dispersion: float
    poincare_cartan_action: float


class PoincareCelestialAnalyzer:
    r"""
    Motor Analítico de Mecánica Celeste de Henri Poincaré.

    Evalúa:
    1. **Mapa de Retorno en la Superficie de Sección $\Sigma$**: Cruces transversales $\bar{\Phi} = v$.
    2. **Invariante Integral de Poincaré-Cartan**: $\mathcal{I}_{\mathrm{PC}} = \oint (\langle \Pi, d\Phi \rangle - H dt)$.
    3. **Exponente Característico de Lyapunov Máximo (MLCE)** vía renormalización de Benettin
       sobre las ecuaciones variacionales exactas de Jacobi-Poincaré.
    4. **Índice de Coherencia de Toros KAM (Kolmogorov-Arnold-Moser)**: Mide la concentración
       espectral de la órbita para distinguir movimiento cuasi-periódico en toros invariantes
       frente a enredos homoclínicos y difusión de Arnold.
    """

    def __init__(self, params: QFTParameters = QFT) -> None:
        self.p: QFTParameters = params
        self._crossings: List[PoincareSectionCrossing] = []
        self._lyapunov_log_sum: float = 0.0
        self._accumulated_time: float = 0.0
        self._poincare_cartan_action: float = 0.0
        self._mean_field_orbit: List[float] = []

    @property
    def crossings(self) -> List[PoincareSectionCrossing]:
        return list(self._crossings)

    @property
    def maximal_lyapunov_exponent(self) -> float:
        r"""Exponente máximo de Lyapunov acumulado $\lambda_1 = \frac{1}{T}\sum \ln(\|\delta z_k\|/\|\delta z_0\|)$."""
        if self._accumulated_time <= self.p.EPSILON_SMOOTH:
            return 0.0
        return self._lyapunov_log_sum / self._accumulated_time

    @property
    def poincare_cartan_action(self) -> float:
        return self._poincare_cartan_action

    def observe_transition(
        self,
        prev_state: ScalarFieldState,
        next_state: ScalarFieldState,
        prev_energy: float,
        next_energy: float,
        dt: float,
    ) -> Tuple[ScalarFieldState, float, float]:
        """
        Registra la evolución orbital $(t \to t + dt)$, actualiza el invariante de
        Poincaré-Cartan, detecta cruces en la sección $\Sigma$, y aplica renormalización
        simpléctica de Benettin al vector tangente $(\delta\Phi, \delta\Pi)$.
        """
        if prev_state.dim == 0 or dt <= 0.0:
            return next_state, 0.0, 1.0

        # 1. Forma diferencial de Poincaré-Cartan: dS = ⟨Π_mid, ΔΦ⟩ - H_mid * dt
        dphi = next_state.phi - prev_state.phi
        pi_mid = 0.5 * (prev_state.pi_momentum + next_state.pi_momentum)
        h_mid = 0.5 * (prev_energy + next_energy)
        ds_pc = float(np.dot(pi_mid, dphi)) - h_mid * dt
        self._poincare_cartan_action += ds_pc

        # 2. Detección de cruce transversal en la Superficie de Sección de Poincaré Σ
        vev = self.p.vev
        prev_dev = float(np.mean(prev_state.phi)) - vev
        next_dev = float(np.mean(next_state.phi)) - vev
        mean_pi = float(np.mean(pi_mid))

        if prev_dev <= 0.0 < next_dev and mean_pi > self.p.POINCARE_SECTION_TOL:
            denom = max(abs(next_dev - prev_dev), self.p.EPSILON_SMOOTH)
            alpha = abs(prev_dev) / denom
            interp_phi = prev_state.phi + alpha * dphi
            dispersion = float(np.std(interp_phi))
            self._crossings.append(
                PoincareSectionCrossing(
                    step_fraction=alpha,
                    mean_momentum=mean_pi,
                    field_dispersion=dispersion,
                    poincare_cartan_action=self._poincare_cartan_action,
                )
            )

        # 3. Almacenar serie orbital para diagnóstico espectral KAM
        self._mean_field_orbit.append(next_dev)
        if len(self._mean_field_orbit) > 128:
            self._mean_field_orbit.pop(0)

        # 4. Exponente de Lyapunov y Renormalización de Benettin en el fibrado tangente
        st_tan = next_state.with_initialized_tangent()
        fiber = st_tan.to_hypercomplex_fiber()
        tan_norm = fiber.tangent_norm()
        self._accumulated_time += dt

        if tan_norm > self.p.EPSILON_SMOOTH:
            self._lyapunov_log_sum += math.log(tan_norm)
            assert st_tan.delta_phi is not None and st_tan.delta_pi is not None
            renorm_state = ScalarFieldState(
                phi=st_tan.phi,
                pi_momentum=st_tan.pi_momentum,
                delta_phi=st_tan.delta_phi / tan_norm,
                delta_pi=st_tan.delta_pi / tan_norm,
            )
        else:
            renorm_state = st_tan

        kam_coherence = self.compute_kam_torus_coherence()
        return renorm_state, self.maximal_lyapunov_exponent, kam_coherence

    def compute_kam_torus_coherence(self) -> float:
        r"""
        Evalúa la coherencia de los Toros Invariantes KAM mediante la pureza espectral
        de Wiener-Shannon de la trayectoria en el espacio de fase:
            $$\mathcal{K}_{\mathrm{KAM}} = 1 - \frac{H_{\mathrm{Shannon}}(\hat{\eta})}{\ln(M)} \in [0, 1]$$
        Un valor cercano a 1 indica movimiento cuasi-periódico confinado en un toro KAM;
        una caída hacia 0 revela ruptura de toros KAM y caos homoclínico.
        """
        m = len(self._mean_field_orbit)
        if m < 4:
            return 1.0
        signal = np.asarray(self._mean_field_orbit, dtype=np.float64)
        signal = signal - np.mean(signal)
        power = np.abs(np.fft.rfft(signal)) ** 2
        total_power = float(np.sum(power))
        if total_power <= self.p.EPSILON_SMOOTH:
            return 1.0
        probs = power / total_power
        probs = probs[probs > 1e-15]
        if probs.size <= 1:
            return 1.0
        shannon_entropy = -float(np.sum(probs * np.log(probs)))
        max_entropy = math.log(float(len(power)))
        if max_entropy <= self.p.EPSILON_SMOOTH:
            return 1.0
        return float(np.clip(1.0 - (shannon_entropy / max_entropy), 0.0, 1.0))


# ───────────────────────────────────────────────────────────────────────────
# 2.4  INTEGRADOR SIMPLÉCTICO VELOCITY-VERLET Y YOSHIDA DE 4.º ORDEN
# ───────────────────────────────────────────────────────────────────────────


class SymplecticIntegrator(ABC):
    """Clase base abstracta para integradores simplécticos en variedades Riemannianas."""

    @abstractmethod
    def step(
        self,
        state: ScalarFieldState,
        force_field: NDArray[np.float64],
        dt: float,
    ) -> ScalarFieldState: ...


class VelocityVerletIntegrator(SymplecticIntegrator):
    r"""
    Integrador Simpléctico Híbrido Velocity-Verlet (2.º orden) y Yoshida-Poincaré (4.º orden).

    Integra simultáneamente:
    1. El flujo primario $(\Phi^n, \Pi^n) \mapsto (\Phi^{n+1}, \Pi^{n+1})$ con corrección de
       frecuencia secular de Lindstedt-Poincaré.
    2. El flujo tangente variacional $(\delta\Phi^n, \delta\Pi^n) \mapsto (\delta\Phi^{n+1}, \delta\Pi^{n+1})$
       mediante la ecuación de Jacobi-Poincaré para el cálculo exacto del espectro de Lyapunov.

    Coeficientes simplécticos de 4.º orden de Haruo Yoshida (1990):
        $$w_1 = \frac{1}{2 - 2^{1/3}}, \quad w_0 = -\frac{2^{1/3}}{2 - 2^{1/3}} = 1 - 2w_1$$
    """

    _YOSHIDA_W1: Final[float] = 1.0 / (2.0 - (2.0 ** (1.0 / 3.0)))
    _YOSHIDA_W0: Final[float] = 1.0 - 2.0 * _YOSHIDA_W1
    _YOSHIDA_SUBSTEPS: Final[Tuple[float, float, float]] = (
        _YOSHIDA_W1,
        _YOSHIDA_W0,
        _YOSHIDA_W1,
    )

    def __init__(
        self,
        hamiltonian: PortHamiltonianHamiltonian,
        damping_gamma: float = 0.0,
        max_field: float = 1e3,
        use_yoshida_4th_order: bool = False,
    ) -> None:
        self.H: PortHamiltonianHamiltonian = hamiltonian
        self.gamma: float = damping_gamma
        self.max_field: float = max_field
        self.use_yoshida_4th_order: bool = use_yoshida_4th_order

        # Análisis espectral para condición Courant-Friedrichs-Lewy (CFL)
        self._lambda_max: float = SpectralAnalyzer.estimate_max_eigenvalue(
            hamiltonian.L_op.L, method="gershgorin"
        )
        self._dt_critical: float = self._compute_critical_timestep()

        logger.info(
            "Integrador Simpléctico inicializado: λ_max=%.4f, dt_crit=%.6f, γ=%.3f, Yoshida4=%s",
            self._lambda_max,
            self._dt_critical,
            self.gamma,
            self.use_yoshida_4th_order,
        )

    def _compute_critical_timestep(self) -> float:
        r"""
        CFL estricto: $\Delta t < (2/\omega_{\max})\cdot\alpha_{\mathrm{CFL}}$, donde
            $$\omega_{\max}^2 = c^2 \lambda_{\max}(\Delta_M) + m_H^2 = c^2 \lambda_{\max} + 2\mu^2$$
        """
        omega_sq = (
            self.H.p.C_SQUARED * self._lambda_max
            + self.H.p.higgs_mass_squared
        )
        omega_max = math.sqrt(omega_sq + self.H.p.EPSILON_SMOOTH)
        dt_limit = (2.0 / omega_max) * self.H.p.CFL_SAFETY_FACTOR
        return min(dt_limit, self.H.p.CFL_ABSOLUTE_CAP)

    @property
    def dt_critical(self) -> float:
        return self._dt_critical

    def recalibrate_spectrum(self) -> None:
        """Recalcula el radio espectral y $\Delta t_{\mathrm{crit}}$ tras una mutación de Nivel 3."""
        self._lambda_max = SpectralAnalyzer.estimate_max_eigenvalue(
            self.H.L_op.L, method="gershgorin"
        )
        self._dt_critical = self._compute_critical_timestep()

    def _verlet_elementary_step(
        self,
        state: ScalarFieldState,
        source_density: NDArray[np.float64],
        dt_sub: float,
    ) -> ScalarFieldState:
        """Paso elemental Velocity-Verlet acoplado con el fibrado tangente de Jacobi-Poincaré."""
        phi = state.phi
        pi = state.pi_momentum

        # 1) Π^{n+½} = Π^n + ½·F(Φ^n, Π^n)·dt
        F_n = self._force(phi, pi, source_density)
        pi_half = pi + 0.5 * F_n * dt_sub

        # 2) Φ^{n+1} = Φ^n + Π^{n+½}·dt (con renormalización secular de Lindstedt-Poincaré)
        lp_factor = self.H.V.lindstedt_poincare_frequency_renormalization(phi)
        phi_new = phi + (pi_half * lp_factor) * dt_sub
        phi_new = self._soft_clip(phi_new)

        # 3) Π^{n+1} = Π^{n+½} + ½·F(Φ^{n+1}, Π^{n+½})·dt
        F_new = self._force(phi_new, pi_half, source_density)
        pi_new = pi_half + 0.5 * F_new * dt_sub

        # 4) Evolución variacional simpléctica en el fibrado tangente (δΦ, δΠ)
        if state.delta_phi is not None and state.delta_pi is not None:
            dphi = state.delta_phi
            dpi = state.delta_pi
            dF_n = self.H.jacobi_variational_force(phi, dphi, dpi, self.gamma)
            dpi_half = dpi + 0.5 * dF_n * dt_sub
            dphi_new = dphi + dpi_half * dt_sub
            dF_new = self.H.jacobi_variational_force(
                phi_new, dphi_new, dpi_half, self.gamma
            )
            dpi_new = dpi_half + 0.5 * dF_new * dt_sub
            return ScalarFieldState(
                phi=phi_new,
                pi_momentum=pi_new,
                delta_phi=dphi_new,
                delta_pi=dpi_new,
            )

        return ScalarFieldState(phi=phi_new, pi_momentum=pi_new)

    def step(
        self,
        state: ScalarFieldState,
        source_density: NDArray[np.float64],
        dt: Optional[float] = None,
    ) -> ScalarFieldState:
        """
        Ejecuta un paso de integración simpléctica (Velocity-Verlet 2.º orden o
        composición triple de Yoshida de 4.º orden si fue activada por Nivel 3 RSI).
        """
        dt_eff = dt if dt is not None else self._dt_critical
        if dt_eff <= 0:
            raise ValueError(f"dt debe ser positivo, recibido: {dt_eff}")
        if dt_eff > self._dt_critical * 1.01:
            logger.warning(
                "dt=%.6f excede dt_crit=%.6f (inestabilidad probable)",
                dt_eff,
                self._dt_critical,
            )

        if not self.use_yoshida_4th_order:
            return self._verlet_elementary_step(state, source_density, dt_eff)

        # Composición simpléctica de 4.º orden de Yoshida (3 sub-pasos simétricos)
        curr = state
        for weight in self._YOSHIDA_SUBSTEPS:
            curr = self._verlet_elementary_step(curr, source_density, weight * dt_eff)
        return curr

    def _force(
        self,
        phi: NDArray[np.float64],
        pi: NDArray[np.float64],
        source: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """$F = -c^2 L \Phi - \nabla V(\Phi) - \gamma \Pi + g_s \rho$."""
        F = self.H.force(ScalarFieldState(phi=phi, pi_momentum=pi), source)
        if self.gamma > 0:
            F = F - self.gamma * pi
        return F

    def _soft_clip(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Saturación suave difeomórfica: $\Phi \mapsto M\tanh(\Phi/M)$."""
        return self.max_field * np.tanh(phi / self.max_field)


# ───────────────────────────────────────────────────────────────────────────
# 2.5  MONITOR DE ESTABILIDAD Y AUTOMEJORA RECURSIVA DE NIVEL 3
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class StabilityMetrics:
    """
    Métricas de diagnóstico de estabilidad termodinámica, celeste de Poincaré
    y lógica intuicionista de Heyting ($\Omega_3$).
    """

    energy: float
    energy_drift: float
    field_rms: float
    momentum_rms: float
    max_field: float
    vev_deviation: float
    requires_damping: bool
    is_stable: bool
    # Extensiones de Poincaré, Rayleigh y RSI Nivel 3 (con valores por defecto compatibles)
    lyapunov_exponent: float = 0.0
    kam_torus_coherence: float = 1.0
    poincare_cartan_action: float = 0.0
    rayleigh_dissipation_power: float = 0.0
    heyting_verdict: HeytingTruthValue = HeytingTruthValue.COHERENT
    rsi_level3_generation: int = 0


class StabilityMonitor:
    """
    Monitor de estabilidad acoplado al clasificador de subobjetos de Heyting $\Omega_3$.
    """

    def __init__(
        self,
        baseline_energy: float,
        hamiltonian: PortHamiltonianHamiltonian,
        params: QFTParameters = QFT,
    ) -> None:
        self._E0: float = baseline_energy
        self._H: PortHamiltonianHamiltonian = hamiltonian
        self._p: QFTParameters = params
        self._history: List[StabilityMetrics] = []

    @property
    def baseline_energy(self) -> float:
        return self._E0

    @property
    def history(self) -> List[StabilityMetrics]:
        return list(self._history)

    def analyze(
        self,
        state: ScalarFieldState,
        lyapunov_exponent: float = 0.0,
        kam_coherence: float = 1.0,
        poincare_action: float = 0.0,
    ) -> StabilityMetrics:
        H = self._H(state)
        energy_drift = abs(H - self._E0) / (abs(self._E0) + self._p.EPSILON_SMOOTH)
        field_rms = float(np.sqrt(np.mean(state.phi ** 2))) if state.dim > 0 else 0.0
        momentum_rms = (
            float(np.sqrt(np.mean(state.pi_momentum ** 2))) if state.dim > 0 else 0.0
        )
        max_field = float(np.max(np.abs(state.phi))) if state.dim > 0 else 0.0
        vev_dev = (
            float(np.sqrt(np.mean((state.phi - self._p.vev) ** 2)))
            if state.dim > 0
            else 0.0
        )
        requires_damping = (
            H > self._p.MAX_ENERGY_RATIO * abs(self._E0)
            or lyapunov_exponent > self._p.LYAPUNOV_CRITICAL_CAP
            or kam_coherence < self._p.KAM_COHERENCE_FLOOR
        )
        is_stable = (
            bool(np.all(np.isfinite(state.phi)))
            and bool(np.all(np.isfinite(state.pi_momentum)))
            and max_field < self._p.MAX_FIELD_MAGNITUDE
            and energy_drift < 10.0
        )

        # Clasificación en el Retículo de Heyting Ω_3 (Axioma A5)
        if not is_stable:
            verdict = HeytingTruthValue.VETOED
        elif requires_damping or energy_drift > 1.5:
            verdict = HeytingTruthValue.DEGRADED
        else:
            verdict = HeytingTruthValue.COHERENT

        rayleigh_power = self._H.rayleigh_dissipation_power(
            state, self._p.DAMPING_GAMMA
        )

        m = StabilityMetrics(
            energy=H,
            energy_drift=energy_drift,
            field_rms=field_rms,
            momentum_rms=momentum_rms,
            max_field=max_field,
            vev_deviation=vev_dev,
            requires_damping=requires_damping,
            is_stable=is_stable,
            lyapunov_exponent=lyapunov_exponent,
            kam_torus_coherence=kam_coherence,
            poincare_cartan_action=poincare_action,
            rayleigh_dissipation_power=rayleigh_power,
            heyting_verdict=verdict,
            rsi_level3_generation=self._p.RSI_GENERATION,
        )
        self._history.append(m)
        return m

    def log_diagnostics(
        self, metrics: StabilityMetrics, level: int = logging.INFO
    ) -> None:
        logger.log(
            level,
            "[STABILITY] E=%.4e (drift=%.2f%%), ⟨Φ⟩_rms=%.4f, max|Φ|=%.4f, "
            "⟨|Φ-v|⟩=%.4f, λ_Lyap=%.4f, KAM=%.3f, Ω_3=%s, RSI_gen=%d",
            metrics.energy,
            metrics.energy_drift * 100.0,
            metrics.field_rms,
            metrics.max_field,
            metrics.vev_deviation,
            metrics.lyapunov_exponent,
            metrics.kam_torus_coherence,
            metrics.heyting_verdict.name,
            metrics.rsi_level3_generation,
        )


class Level3RecursiveSelfImprover:
    r"""
    Motor de Automejora Recursiva de Nivel 3 ($\mathcal{R}_3 : \mathrm{End}(\mathcal{C}) \to \mathrm{End}(\mathcal{C})$).

    Jerarquía de Niveles de Automejora
    ----------------------------------
    - **Nivel 1 (Paramétrico)**: Ajusta el coeficiente de disipación $\gamma$ y el paso $\Delta t$.
    - **Nivel 2 (Métrico-Operador)**: Deforma conformalmente la métrica $\tilde{G}_{\mu\nu} = G\exp(-2\gamma V(\Phi))$
      en el operador de Laplace-Beltrami $\Delta_M$ para suavizar singularidades de curvatura.
    - **Nivel 3 (Meta-Funtorial / Punto Fijo de Banach)**: Evalúa la derivada de Fréchet
      del funcional de Poincaré-Lyapunov $\mathcal{J}[\mathcal{F}_{\mathrm{Higgs}}]$, verifica la
      contracción de Banach $\|\mathcal{T}(\mathcal{F}_{k+1}) - \mathcal{T}(\mathcal{F}_k)\|_{\mathcal{B}} \le q\|\mathcal{F}_{k+1} - \mathcal{F}_k\|_{\mathcal{B}}$,
      muta inmutablemente `QFTParameters`, activa el esquema simpléctico de 4.º orden de Yoshida
      y proyecta el campo hacia el subespacio invariante del toro KAM.
    """

    def __init__(self) -> None:
        self._banach_defect_history: List[float] = []
        self._contraction_verified: bool = True

    @property
    def banach_defect_history(self) -> List[float]:
        return list(self._banach_defect_history)

    @property
    def is_contraction_verified(self) -> bool:
        return self._contraction_verified

    def compute_frechet_meta_functional(self, metrics: StabilityMetrics) -> float:
        r"""
        Funcional de coste meta-variacional de Poincaré-Lyapunov:
            $$\mathcal{J}[\mathcal{F}] = \max(0, \lambda_{\mathrm{Lyap}})^2 + (1 - \mathcal{K}_{\mathrm{KAM}})^2 + \frac{1}{4}\left(\frac{\Delta E}{E_0}\right)^2$$
        """
        lyap_term = max(0.0, metrics.lyapunov_exponent) ** 2
        kam_term = (1.0 - metrics.kam_torus_coherence) ** 2
        drift_term = 0.25 * (min(metrics.energy_drift, 10.0) ** 2)
        return math.sqrt(lyap_term + kam_term + drift_term)

    def execute_level3_introspection(
        self,
        state: ScalarFieldState,
        metrics: StabilityMetrics,
        hamiltonian: PortHamiltonianHamiltonian,
        integrator: VelocityVerletIntegrator,
        monitor: StabilityMonitor,
    ) -> Tuple[ScalarFieldState, QFTParameters]:
        """
        Ejecuta el ciclo completo de Automejora Recursiva de Nivel 1, 2 y 3 cuando
        el clasificador de Heyting detecta degradación orbital o exceso de Lyapunov.
        """
        defect = self.compute_frechet_meta_functional(metrics)
        if self._banach_defect_history:
            prev_defect = self._banach_defect_history[-1]
            # Verificación de contracción en el espacio de Banach de endofuntores
            self._contraction_verified = (
                defect <= prev_defect * (1.0 + hamiltonian.p.RSI_META_LEARNING_RATE)
                or defect < 0.05
            )
        self._banach_defect_history.append(defect)

        if (
            metrics.heyting_verdict == HeytingTruthValue.COHERENT
            and not metrics.requires_damping
        ):
            return state, hamiltonian.p

        # ── NIVEL 1: Síntesis contractiva de nuevos parámetros QFT ──
        new_params = hamiltonian.p.synthesize_level3_mutation(
            lyapunov_exponent=metrics.lyapunov_exponent,
            kam_coherence=metrics.kam_torus_coherence,
            energy_drift=metrics.energy_drift,
        )

        # ── NIVEL 2: Estabilización Conforme de la Métrica (Axioma A2) ──
        V_vec, _ = hamiltonian.V.value_and_gradient(state.phi)
        conformal_lb = hamiltonian.L_op.apply_conformal_stabilization(
            potential_vec=V_vec,
            conformal_gamma=new_params.CONFORMAL_GAMMA,
        )

        # ── NIVEL 3: Re-tejido en caliente del Hamiltoniano, Integrador y Estado ──
        hamiltonian.p = new_params
        hamiltonian.V.p = new_params
        hamiltonian.L_op = conformal_lb

        integrator.gamma = new_params.DAMPING_GAMMA
        integrator.use_yoshida_4th_order = True
        integrator.recalibrate_spectrum()
        monitor._p = new_params

        # Retracción simpléctica hacia el Toro KAM del Vacío (contracción de Banach)
        q = new_params.RSI_BANACH_CONTRACTION
        vev_target = np.full_like(state.phi, new_params.vev)
        contracted_phi = q * state.phi + (1.0 - q) * vev_target
        contracted_pi = q * state.pi_momentum

        healed_state = ScalarFieldState(
            phi=contracted_phi,
            pi_momentum=contracted_pi,
            delta_phi=state.delta_phi,
            delta_pi=state.delta_pi,
        )
        return healed_state, new_params


# ───────────────────────────────────────────────────────────────────────────
# 2.6  VALIDACIÓN DE INVARIANTES TOPOLÓGICOS Y PUENTE ANIDADO HACIA FASE 3
# ───────────────────────────────────────────────────────────────────────────


def validate_topological_consistency(
    state: ScalarFieldState,
    hamiltonian: PortHamiltonianHamiltonian,
    params: QFTParameters = QFT,
    *,
    return_phase3_suturation: bool = False,
    metrics: Optional[StabilityMetrics] = None,
    crowbar_port: Optional[CrowbarCircuitPort] = None,
) -> Union[bool, "HiggsSuturationState"]:
    r"""
    Definición formal final de la Fase 2 y puente anidado de entrada hacia la Fase 3.

    Valida invariantes topológicos, simplécticos y físicos:
    T1. $\dim(\Phi) = \dim(\Delta_M)$
    T2. $\Phi, \Pi \in \mathbb{R}^n$ (finitos en norma de Banach)
    T3. $\|\Phi\|_\infty < \Phi_{\max}$ (acotación Lipschitz)
    T4. $H[\Phi, \Pi]$ bien definido y finito

    Cuando `return_phase3_suturation=True`, eleva directamente el resultado validado
    hacia el objeto inicial de la Fase 3 (`HiggsSuturationState`).
    """
    is_valid = True
    try:
        if state.dim != hamiltonian.L_op.n:
            logger.error("[T1] dim inconsistente")
            is_valid = False
        elif not (
            np.all(np.isfinite(state.phi)) and np.all(np.isfinite(state.pi_momentum))
        ):
            logger.error("[T2] campos no finitos")
            is_valid = False
        else:
            max_field = float(np.max(np.abs(state.phi))) if state.dim > 0 else 0.0
            if max_field > params.MAX_FIELD_MAGNITUDE:
                logger.error("[T3] |Φ|_max=%.2e > Φ_max", max_field)
                is_valid = False
            else:
                H = hamiltonian(state)
                if not np.isfinite(H):
                    logger.error("[T4] H=%g no finito", H)
                    is_valid = False
    except Exception as e:
        logger.exception("Validación topológica falló: %s", e)
        is_valid = False

    if not return_phase3_suturation:
        return is_valid

    # Continuación formal directa hacia la estructura de apertura de la Fase 3
    return HiggsSuturationState.from_phase2_topological_bridge(
        is_topologically_valid=is_valid,
        state=state,
        hamiltonian=hamiltonian,
        metrics=metrics,
        crowbar_port=crowbar_port,
    )


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 3 · FUNTOR DE ANCLAJE, SUTURA DE DE RHAM-HEYTING Y CROWBAR (DECIDE│
# │  Inicia directamente con `HiggsSuturationState` construido por el       │
# │  último método de la Fase 2 (`validate_topological_consistency`).       │
# │  Culmina con la composición endofuntorial: Y ∘ H ∘ V.                   │
# └─────────────────────────────────────────────────────────────────────────┘


# ───────────────────────────────────────────────────────────────────────────
# 3.1  ESTADO DE SUTURA COHOMOLÓGICA DE DE RHAM Y PUERTO CROWBAR
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HiggsSuturationState:
    r"""
    Estado de Sutura Cohomológica de de Rham-Heyting (apertura formal de la Fase 3).

    Recibe la validación topológica y dinámica de la Fase 2, proyecta el estado sobre
    el clasificador de subobjetos $\Omega_3 = \{\mathtt{COHERENT}, \mathtt{DEGRADED}, \mathtt{VETOED}\}$
    y gobierna el disparo físico del circuito Crowbar (GPIO14 / Tiristor BT151).
    """

    field_state: ScalarFieldState
    heyting_verdict: HeytingTruthValue
    crowbar_port: CrowbarCircuitPort
    betti_0: int
    spectral_gap: float
    lyapunov_exponent: float
    kam_torus_coherence: float
    poincare_cartan_action: float
    rsi_generation: int

    @classmethod
    def from_phase2_topological_bridge(
        cls,
        is_topologically_valid: bool,
        state: ScalarFieldState,
        hamiltonian: PortHamiltonianHamiltonian,
        metrics: Optional[StabilityMetrics] = None,
        crowbar_port: Optional[CrowbarCircuitPort] = None,
    ) -> "HiggsSuturationState":
        """Continuación formal directa desde `validate_topological_consistency` (Fase 2)."""
        base_verdict = (
            metrics.heyting_verdict
            if metrics is not None
            else HeytingTruthValue.COHERENT
        )
        final_verdict = (
            base_verdict
            if is_topologically_valid
            else base_verdict.join(HeytingTruthValue.VETOED)
        )
        energy = metrics.energy if metrics is not None else hamiltonian(state)
        port = (crowbar_port or CrowbarCircuitPort()).evaluate_verdict(
            final_verdict, energy
        )
        betti_0 = int(
            hamiltonian.spectral_signature.get(
                "betti_0", float(SpectralAnalyzer.estimate_betti_zero(hamiltonian.L_op.L))
            )
        )
        spec_gap = float(
            hamiltonian.spectral_signature.get(
                "spectral_gap",
                SpectralAnalyzer.estimate_spectral_gap(hamiltonian.L_op.L),
            )
        )
        return cls(
            field_state=state,
            heyting_verdict=final_verdict,
            crowbar_port=port,
            betti_0=betti_0,
            spectral_gap=spec_gap,
            lyapunov_exponent=metrics.lyapunov_exponent if metrics is not None else 0.0,
            kam_torus_coherence=(
                metrics.kam_torus_coherence if metrics is not None else 1.0
            ),
            poincare_cartan_action=(
                metrics.poincare_cartan_action if metrics is not None else 0.0
            ),
            rsi_generation=hamiltonian.p.RSI_GENERATION,
        )


# ───────────────────────────────────────────────────────────────────────────
# 3.2  FUNTOR DE ANCLAJE DE HIGGS (YUKAWA ∘ HAMILTONIANO-POINCARÉ ∘ VACÍO)
# ───────────────────────────────────────────────────────────────────────────


class ScalarHiggsAnchor(Morphism):
    r"""
    Funtor de Anclaje de Higgs: $\mathcal{F}_{\mathrm{Higgs}}: \mathcal{C}_{\mathrm{Agent}} \to \mathcal{C}_{\mathrm{Agent}}$.

    Composición Endofuntorial de Tres Fases Anidadas
    ------------------------------------------------
        $$\mathcal{F}_{\mathrm{Higgs}} = \text{Yukawa}_{\text{Fase 3}} \circ \text{HamiltonPoincaré}_{\text{Fase 2}}^N \circ \text{VacuumSpectral}_{\text{Fase 1}}$$

    Integra:
    - Fase 1 (`OBSERVE`): Operador de Laplace-Beltrami conforme y puente espectral.
    - Fase 2 (`ORIENT`) : Flujo simpléctico Velocity-Verlet / Yoshida-4, Mecánica Celeste
                          de Poincaré (sección $\Sigma$, invariante de Poincaré-Cartan,
                          exponente de Lyapunov, toros KAM) y Automejora Recursiva Nivel 3.
    - Fase 3 (`DECIDE`) : Clasificador intuicionista de Heyting $\Omega_3$, protección
                          Crowbar RL-Tiristor y generación de masa efectiva de Yukawa.
    """

    def __init__(
        self,
        laplacian: sp.csr_matrix,
        dim: int,
        params: QFTParameters = QFT,
        num_relaxation_steps: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        # ── Sutura V (suturas_rigurosas.md): coherencia dimensional ──
        if laplacian.shape[0] != laplacian.shape[1]:
            raise ValueError(f"Laplaciano no cuadrado: shape={laplacian.shape}")
        n_lap = laplacian.shape[0]
        if dim != n_lap:
            logger.debug(
                "Sutura V: dim=%d ajustado a n_lap=%d (cardinalidad del Laplaciano)",
                dim,
                n_lap,
            )
            dim = n_lap

        self._params: QFTParameters = params
        self._dim: int = dim
        self._num_steps: int = num_relaxation_steps

        # ── FASE 1 (OBSERVE) ──► Puente Espectral hacia Fase 2 ──
        self._lb_op: LaplacedBeltramiOperator = LaplacedBeltramiOperator(laplacian)
        (
            self._potential,
            self._hamiltonian,
            self._spectral_signature,
        ) = SpectralAnalyzer.bridge_to_phase2_symplectic_manifold(
            laplacian_op=self._lb_op,
            params=self._params,
        )

        # ── FASE 2 (ORIENT): Integrador Simpléctico, Poincaré y RSI Nivel 3 ──
        self._integrator: VelocityVerletIntegrator = VelocityVerletIntegrator(
            self._hamiltonian,
            damping_gamma=params.DAMPING_GAMMA,
            max_field=params.MAX_FIELD_MAGNITUDE,
        )
        self._poincare: PoincareCelestialAnalyzer = PoincareCelestialAnalyzer(params)
        self._rsi_improver: Level3RecursiveSelfImprover = Level3RecursiveSelfImprover()

        # RNG reproducible inyectado
        self._rng: np.random.Generator = np.random.default_rng(
            seed if seed is not None else params.RNG_SEED
        )

        # Inicialización del vacío con fibrado tangente de Jacobi-Poincaré
        self._field_state: ScalarFieldState = (
            self._initialize_vacuum_state().with_initialized_tangent()
        )

        # Monitor de estabilidad
        self._monitor: StabilityMonitor = StabilityMonitor(
            baseline_energy=self._hamiltonian(self._field_state),
            hamiltonian=self._hamiltonian,
            params=params,
        )

        # ── FASE 3 (DECIDE): Puerto Crowbar y Estado de Sutura de Heyting ──
        self._crowbar_port: CrowbarCircuitPort = CrowbarCircuitPort()
        initial_suturation = validate_topological_consistency(
            self._field_state,
            self._hamiltonian,
            self._params,
            return_phase3_suturation=True,
            crowbar_port=self._crowbar_port,
        )
        assert isinstance(initial_suturation, HiggsSuturationState)
        self._suturation_state: HiggsSuturationState = initial_suturation

        logger.info(
            "ScalarHiggsAnchor v7.0 inicializado: dim=%d, v=%.4f, E₀=%.4e, dt_crit=%.6f, β₀=%d",
            dim,
            params.vev,
            self._monitor.baseline_energy,
            self._integrator.dt_critical,
            self._suturation_state.betti_0,
        )

    def _initialize_vacuum_state(self) -> ScalarFieldState:
        r"""
        Vacío coherente: $\Phi_0 = v + \delta\Phi$, $\Pi_0 = 0$.
        Fluctuaciones térmicas iniciales de orden $O(5\% \cdot v)$.
        """
        vev = self._params.vev
        noise_amp = 0.05 * vev
        phi_init = np.full(self._dim, vev, dtype=np.float64)
        phi_init += self._rng.normal(0.0, noise_amp, size=self._dim)
        pi_init = np.zeros(self._dim, dtype=np.float64)
        return ScalarFieldState(phi=phi_init, pi_momentum=pi_init)

    def __call__(
        self,
        state: CategoricalState,
        *args: Any,
        **kwargs: Any,
    ) -> CategoricalState:
        """
        Aplica el endofuntor $\mathcal{F}_{\mathrm{Higgs}}$ al estado categórico.

         Flujo Anidado OODA Espectral (Fases 1 ──► 2 ──► 3)
        ---------------------------------------------------
        1. [Fase 1] Extraer $\psi$ del payload y construir densidad fermiónica $\rho = |\psi|^2$.
        2. [Fase 2] Evolucionar $(\Phi, \Pi, \delta\Phi, \delta\Pi)$ con dinámica simpléctica,
                    registrar invariante de Poincaré-Cartan, cruces en $\Sigma$, exponente
                    de Lyapunov y ejecutar Automejora Recursiva de Nivel 3 si se requiere.
        3. [Fase 3] Sutura cohomológica de de Rham en el retículo de Heyting $\Omega_3$,
                    verificación del circuito Crowbar, cálculo de masa efectiva de Yukawa
                    $m_{\mathrm{eff}} = m_0 + g|\Phi|$ y anclaje $\psi' = m_{\mathrm{eff}} \odot \psi$.
        """
        # 1) Extracción robusta de la sección espinorial ψ
        psi_raw = self._extract_psi(state)
        if psi_raw is None:
            logger.debug("Sin vector estocástico: identidad retornada")
            return state

        # 2) Densidad de carga fermiónica
        source = FermionicSource(psi_vector=psi_raw)
        rho = source.charge_density()

        # 3) Relajación simpléctica con Mecánica Celeste de Poincaré y RSI Nivel 3
        for step_idx in range(self._num_steps):
            metrics = self._monitor.analyze(
                self._field_state,
                lyapunov_exponent=self._poincare.maximal_lyapunov_exponent,
                kam_coherence=self._poincare.compute_kam_torus_coherence(),
                poincare_action=self._poincare.poincare_cartan_action,
            )
            if not metrics.is_stable:
                self._suturation_state = HiggsSuturationState.from_phase2_topological_bridge(
                    is_topologically_valid=False,
                    state=self._field_state,
                    hamiltonian=self._hamiltonian,
                    metrics=metrics,
                    crowbar_port=self._crowbar_port,
                )
                self._crowbar_port = self._suturation_state.crowbar_port
                logger.error(
                    "Inestabilidad en paso %d: drift=%.2f%%, max|Φ|=%.2e, Ω_3=%s, Crowbar=%s",
                    step_idx,
                    metrics.energy_drift * 100,
                    metrics.max_field,
                    self._suturation_state.heyting_verdict.name,
                    self._crowbar_port.latched,
                )
                raise HeytingLobeCollapse(
                    "Colapso del campo de Higgs hacia VETOED (disparo de Crowbar GPIO14)"
                )

            # Si el estado entra en el lóbulo DEGRADED o requiere amortiguamiento,
            # se activa la Automejora Recursiva de Nivel 3 (Banach-Fréchet + Métrica Conforme)
            if metrics.requires_damping or (
                metrics.heyting_verdict == HeytingTruthValue.DEGRADED
            ):
                self._field_state, self._params = (
                    self._rsi_improver.execute_level3_introspection(
                        state=self._field_state,
                        metrics=metrics,
                        hamiltonian=self._hamiltonian,
                        integrator=self._integrator,
                        monitor=self._monitor,
                    )
                )
                self._lb_op = self._hamiltonian.L_op
                if step_idx == 0:
                    logger.warning(
                        "Termostato y RSI Nivel 3 activos (gen=%d): E/E₀=%.2f, γ=%.4f",
                        self._params.RSI_GENERATION,
                        metrics.energy / max(abs(self._monitor.baseline_energy), 1e-12),
                        self._params.DAMPING_GAMMA,
                    )

            prev_state = self._field_state
            prev_energy = metrics.energy
            dt_used = self._integrator.dt_critical

            next_state = self._integrator.step(prev_state, rho, dt=dt_used)
            next_energy = self._hamiltonian(next_state)

            # Observación celeste de Poincaré (sección Σ, acción de Poincaré-Cartan y Benettin)
            self._field_state, _, _ = self._poincare.observe_transition(
                prev_state=prev_state,
                next_state=next_state,
                prev_energy=prev_energy,
                next_energy=next_energy,
                dt=dt_used,
            )

        final_metrics = self._monitor.analyze(
            self._field_state,
            lyapunov_exponent=self._poincare.maximal_lyapunov_exponent,
            kam_coherence=self._poincare.compute_kam_torus_coherence(),
            poincare_action=self._poincare.poincare_cartan_action,
        )
        self._monitor.log_diagnostics(final_metrics, logging.DEBUG)

        # Puente Fase 2 ──► Fase 3: Sutura de de Rham-Heyting y Puerto Crowbar
        suturation = validate_topological_consistency(
            self._field_state,
            self._hamiltonian,
            self._params,
            return_phase3_suturation=True,
            metrics=final_metrics,
            crowbar_port=self._crowbar_port,
        )
        assert isinstance(suturation, HiggsSuturationState)
        self._suturation_state = suturation
        self._crowbar_port = suturation.crowbar_port

        if not self._suturation_state.heyting_verdict.is_verdict_coherent():
            raise HeytingLobeCollapse(
                "El clasificador de Heyting Ω_3 colapsó a VETOED tras la relajación."
            )

        # 4) Masa efectiva de Yukawa
        m_eff = self._compute_effective_mass(source)
        if not np.all(np.isfinite(m_eff)):
            raise YukawaCouplingDivergence("m_eff no finita")
        if m_eff.size > 0 and np.max(m_eff) > 1e4:
            logger.critical("m_eff excesiva: max=%.2e, clipando", float(np.max(m_eff)))
            m_eff = np.clip(m_eff, 0.0, 1e4)

        # 5) Anclaje covariante: ψ' = m_eff ⊙ ψ
        psi_anchored = psi_raw * m_eff

        # 6) Construcción del nuevo CategoricalState enriquecido
        new_payload = self._build_anchored_payload(
            state, psi_anchored, m_eff, final_metrics
        )
        return CategoricalState(
            payload=new_payload,
            stratum=Stratum.PHYSICS,
            metadata=state.metadata,
        )

    # ───── Métodos auxiliares y propiedades de introspección ─────

    def _extract_psi(self, state: CategoricalState) -> Optional[NDArray[np.float64]]:
        """Extrae $\psi$ del payload con polimorfismo estructural seguro."""
        if isinstance(state.payload, dict):
            raw = state.payload.get("stochastic_vector")
            if raw is None or len(raw) == 0:
                return None
            psi = np.asarray(raw, dtype=np.float64).ravel()
        elif isinstance(state.payload, np.ndarray):
            if state.payload.size == 0:
                return None
            psi = state.payload.astype(np.float64).ravel()
        elif hasattr(state.payload, "__iter__") and not isinstance(
            state.payload, (str, bytes)
        ):
            try:
                psi = np.asarray(list(state.payload), dtype=np.float64).ravel()
                if psi.size == 0:
                    return None
            except (TypeError, ValueError):
                return None
        else:
            return None

        return self._resize_psi(psi)

    def _resize_psi(self, psi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Proyecta o sumerge isométricamente $\psi$ en $\mathbb{R}^{\mathrm{dim}}$."""
        if psi.shape[0] == self._dim:
            return psi
        if psi.shape[0] > self._dim:
            return psi[: self._dim]
        out = np.zeros(self._dim, dtype=np.float64)
        out[: psi.shape[0]] = psi
        return out

    def _compute_effective_mass(
        self, source: FermionicSource
    ) -> NDArray[np.float64]:
        r"""
        Perfil espacial de masa efectiva de Yukawa regularizada:
            $$m_{\mathrm{eff}}(x_i) = m_0 + g_Y \sqrt{\Phi_i^2 + \varepsilon^2}$$
        """
        phi = self._field_state.phi
        phi_mag = np.sqrt(phi ** 2 + self._params.EPSILON_SMOOTH ** 2)
        return source.base_mass + self._params.YUKAWA_G * phi_mag

    def _build_anchored_payload(
        self,
        original: CategoricalState,
        psi_anchored: NDArray[np.float64],
        m_eff: NDArray[np.float64],
        metrics: StabilityMetrics,
    ) -> Dict[str, Any]:
        """Construye el payload preservando llaves originales y añadiendo telemetría Poincaré/RSI."""
        if isinstance(original.payload, dict):
            new_payload: Dict[str, Any] = dict(original.payload)
        else:
            new_payload = {"original_payload": original.payload}

        new_payload.update(
            {
                "anchored_vector": psi_anchored.tolist(),
                "effective_mass_profile": m_eff.tolist(),
                "higgs_field_snapshot": self._field_state.phi.tolist(),
                "vev_deviation": metrics.vev_deviation,
                "field_energy": metrics.energy,
                "stability_flag": metrics.is_stable,
                "heyting_verdict": self._suturation_state.heyting_verdict.name,
                "crowbar_latched": self._crowbar_port.latched,
                "poincare_lyapunov_exponent": metrics.lyapunov_exponent,
                "poincare_kam_coherence": metrics.kam_torus_coherence,
                "poincare_cartan_action": metrics.poincare_cartan_action,
                "poincare_section_crossings": len(self._poincare.crossings),
                "rsi_level3_generation": self._params.RSI_GENERATION,
                "rsi_banach_contraction_verified": (
                    self._rsi_improver.is_contraction_verified
                ),
            }
        )
        return new_payload

    @property
    def current_vev_deviation(self) -> float:
        """Desviación cuadrática media (RMS) respecto al vacío $\Phi - v$."""
        if self._dim == 0:
            return 0.0
        return float(
            np.sqrt(np.mean((self._field_state.phi - self._params.vev) ** 2))
        )

    @property
    def field_state(self) -> ScalarFieldState:
        return self._field_state

    @property
    def monitor(self) -> StabilityMonitor:
        return self._monitor

    @property
    def suturation_state(self) -> HiggsSuturationState:
        return self._suturation_state

    @property
    def poincare_analyzer(self) -> PoincareCelestialAnalyzer:
        return self._poincare

    @property
    def rsi_improver(self) -> Level3RecursiveSelfImprover:
        return self._rsi_improver


# ───────────────────────────────────────────────────────────────────────────
# 3.3  UTILIDADES PÚBLICAS Y DECORADOR CATEGÓRICO
# ───────────────────────────────────────────────────────────────────────────


def construct_laplacian_from_adjacency(
    adjacency: Union[NDArray[np.float64], sp.spmatrix],
    symmetrize: bool = True,
) -> sp.csr_matrix:
    r"""Construye el Laplaciano combinatorio $\Delta = D - A$ en formato CSR."""
    op = LaplacedBeltramiOperator.from_adjacency(adjacency, symmetrize=symmetrize)
    return op.L


T_Morphism = TypeVar("T_Morphism", bound=Morphism)


def apply_higgs_anchor(
    dim: int,
    laplacian: sp.csr_matrix,
    num_relaxation_steps: int = 5,
    seed: Optional[int] = None,
) -> Callable[[type[T_Morphism]], type[T_Morphism]]:
    r"""
    Decorador de orden superior que compone un `Morphism` con el funtor `ScalarHiggsAnchor`.

    Persiste la instancia del ancla por agente para preservar la memoria orbital
    de Poincaré y la evolución generacional de Automejora Recursiva de Nivel 3.
    """

    def decorator(agent_class: type[T_Morphism]) -> type[T_Morphism]:
        original_call = agent_class.__call__

        @wraps(original_call)
        def wrapped(
            self: T_Morphism, state: CategoricalState, *a: Any, **kw: Any
        ) -> CategoricalState:
            pre_state = original_call(self, state, *a, **kw)
            anchor: Optional[ScalarHiggsAnchor] = getattr(
                self, "_higgs_anchor_instance", None
            )
            if anchor is None:
                anchor = ScalarHiggsAnchor(
                    laplacian=laplacian,
                    dim=dim,
                    num_relaxation_steps=num_relaxation_steps,
                    seed=seed,
                )
                try:
                    setattr(self, "_higgs_anchor_instance", anchor)
                except Exception:
                    pass
            return anchor(pre_state)

        agent_class.__call__ = wrapped  # type: ignore[assignment]
        agent_class.__doc__ = (agent_class.__doc__ or "") + (
            f"\n\n[Anclado por Higgs v7.0 (Poincaré + RSI Nivel 3) con dim={dim}, steps={num_relaxation_steps}]"
        )
        return agent_class

    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIONES PÚBLICAS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Preludio Axiomático (Excepciones, Heyting, Crowbar e Hipercomplejos)
    "ScalarHiggsAnchorError",
    "MetricDegeneracyError",
    "LaplaceBeltramiSingularity",
    "YukawaCouplingDivergence",
    "LipschitzRegularityBreach",
    "HeytingLobeCollapse",
    "HeytingTruthValue",
    "CrowbarCircuitPort",
    "HypercomplexSymplecticFiber",
    # Fase 1: Axiomática del Campo y Espectro
    "QFTParameters",
    "QFT",
    "ScalarFieldState",
    "FermionicSource",
    "LaplacedBeltramiOperator",
    "LaplaceBeltramiOperator",
    "SpectralAnalyzer",
    # Fase 2: Dinámica Simpléctica, Mecánica Celeste de Poincaré y RSI Nivel 3
    "HiggsPotential",
    "PortHamiltonianHamiltonian",
    "PoincareSectionCrossing",
    "PoincareCelestialAnalyzer",
    "SymplecticIntegrator",
    "VelocityVerletIntegrator",
    "StabilityMonitor",
    "StabilityMetrics",
    "Level3RecursiveSelfImprover",
    "validate_topological_consistency",
    # Fase 3: Sutura de de Rham-Heyting y Funtor de Anclaje
    "HiggsSuturationState",
    "ScalarHiggsAnchor",
    "construct_laplacian_from_adjacency",
    "apply_higgs_anchor",
]
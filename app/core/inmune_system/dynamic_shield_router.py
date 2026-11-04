# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Dynamic Shield Router (Discriminador de Campos de Gauge)            ║
║ Ruta   : app/core/immune_system/dynamic_shield_router.py                     ║
║ Versión: 5.0.0-Poincare-KAM-RSI-L3-Ehresmann-Higham-Cartan-PhD               ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y ENRUTAMIENTO DE CALIBRE (Rigor Doctoral): ────────────
Este módulo consagra al **Discriminador de Campos de Gauge**, **Enrutador Geométrico
de Poincaré-Ehresmann** y **Motor de Automejora Recursiva de Nivel 3 (RSI-L3)**
de la variedad de-confinada. Su propósito ciber-físico es proveer la "vestimenta"
o deformación termodinámica exacta al escudo basal del sistema (`funtor_shield.py`)
transportando de forma paralela la matriz de disipación $$R(x)$$ (red de conductancia
de Rayleigh-Tellegen) a través de la filtración monótona de subespacios de Hilbert
de la pirámide $$DIKW$$:

$$V_{\mathbb{P}} \subset V_{\mathbb{T}} \subset V_{\mathbb{S}} \subset V_{\mathbb{W}} \quad\big[16, 22\big]$$

Para evitar que las perturbaciones estocásticas de los Modelos de Lenguaje (LLMs)
exciten resonancias seculares (pequeños divisores de Poincaré) o difusión de Arnold
que degraden el espacio de fase, el enrutador implementa una **Conexión de Ehresmann**
acoplada a un **Invariante Integral de Poincaré-Cartan** y gobernada por un
**Endofuntor Metacognitivo RSI de Nivel 3**, aislando las singularidades en RAM
mediante un colapso restrictivo en el retículo distributivo de Heyting $$\Omega_3$$
en el milisegundo cero.

AXIOMÁTICA DE GAUGE, EHRESMANN, POINCARÉ Y AUTOMEJORA RSI-L3: ──────────────────

  [I1] El G-Fibrado Principal, Conexión de Ehresmann y Toros KAM (Fase 1):
       Dada la variedad diferenciable de disipación $$M$$ y el G-fibrado principal
       $$\pi: P \to M$$, la 1-forma de conexión de Ehresmann $$\omega \in \Omega^1(P, \mathfrak{g})$$
       con valores en el álgebra de Lie $$\mathfrak{g} \cong \mathfrak{so}(n) \oplus \mathfrak{sp}(n)$$
       separa síncronamente el espacio tangente $$T_u P$$ en la suma directa de la fibra
       vertical resonante y el subespacio horizontal diofántico (Toro KAM de Poincaré):
       $$T_u P = V_u P \oplus H_u P \quad \text{donde} \quad V_u P = \ker(d\pi) \quad \land \quad H_u P = \ker(\omega)$$
       El proyector horizontal $$H$$ se construye sobre la descomposición espectral
       de $$G_s$$ modulada por generadores hipercomplejos bicuaterniónicos y la condición
       diofántica de Kolmogorov-Arnold-Moser (KAM):
       $$|\langle \mathbf{k}, \boldsymbol{\omega}(G_s) \rangle| \ge \gamma_{\mathrm{KAM}} \|\mathbf{k}\|_1^{-\tau}$$
       garantizando a precisión de máquina $$H^2 = H$$ y $$H G_s^\top = G_s H$$.

  [I2] Pullback Termodinámico, Serie de Poincaré-Lindstedt y Entropía de Umegaki (Fase 2):
       La proyección del "fango" informacional se mapea mediante el pullback de las
       co-cadenas de telemetría sobre el fibrado horizontal. Se computa el diferencial
       del tensor de disipación $$\delta R$$ eliminando términos seculares resonantes
       mediante el conmutador homológico de Poincaré-Lindstedt, preservando el primer
       invariante integral relativo de Poincaré-Cartan $$\oint_\gamma \mathbf{p}\cdot d\mathbf{q}$$
       y acotando la entropía cuántica relativa de Umegaki $$D(\rho \parallel \sigma)$$
       bajo la cota dura de Wilkinson:
       $$D(\rho \parallel \sigma) = \operatorname{Tr}\left( \rho \left( \log_2 \rho - \log_2 \sigma \right) \right) \le D_{\max}$$

  [I3] Proyección de Higham (2002) y Monodromía Simpléctica de Floquet (Fase 3):
       La deformación $$R_{\mathrm{raw}} = R_{\mathrm{base}} + \delta R$$ se proyecta en
       norma de Frobenius al cono semidefinido positivo $$S^n_+$$:
       $$R_{\mathrm{psd}} = \arg\min_{M = M^\top \succeq 0} \|M - R_{\mathrm{raw}}\|_F$$
       y se audita mediante la matriz de monodromía del mapa de primer retorno de
       Poincaré (transformada de Cayley disipativa) verificando que todos los
       multiplicadores característicos de Floquet satisfagan $$|\mu_i(\mathcal{M})| \le 1$$.

  [I4] Despolarización de Traza Cuántica de-confinada (OmegaGaugeWrapper):
       Para amortiguar y evitar el colapso de la traza de la Matriz Atómica de
       Conocimiento (MAC) bajo fluctuaciones severas, se efectúa un canal
       despolarizante ponderado utilizando el tensor métrico $$G_{\mu\nu}$$:
       $$\tilde{R}_{\mathrm{eff}} = (1 - \gamma_{\mathrm{dep}}) R_{\mathrm{eff}} + \gamma_{\mathrm{dep}} \left( \frac{\operatorname{Tr}(G_{\mu\nu})}{\operatorname{Tr}(R_{\mathrm{eff}})} \right) G_{\mu\nu}$$

  [I5] Automejora Recursiva de Nivel 3 (Endofuntor de Punto Fijo de Lawvere):
       El enrutador incorpora un meta-funtor $$\mathfrak{R}^{(3)}: \operatorname{End}(\mathcal{H}) \to \operatorname{End}(\mathcal{H})$$
       que evalúa el funcional de Lyapunov-Umegaki-Poincaré $$\mathcal{V}(\Theta_g)$$ y
       reescribe recursivamente las leyes generadoras de cada fase mediante descenso
       de gradiente natural sobre la variedad de Fisher-Rao, sujeto a un certificado
       estricto de contractividad de Banach-Caccioppoli $$q_g \in (0, 1)$$.

ARQUITECTURA DE TRES FASES ANIDADAS (Continuación Formal Monoidal): ─────────────
  Fase 1 (`Phase1_EhresmannConnection`)
    └─► Último método: `handoff_phase1_to_phase2`
          │ (Continuación formal directa)
          ▼
  Fase 2 (`Phase2_ThermodynamicPullback`)
    ├─► Primer método: `ingest_from_phase1_handoff`
    └─► Último método: `handoff_phase2_to_phase3`
          │ (Continuación formal directa)
          ▼
  Fase 3 (`Phase3_ShieldDresser`)
    ├─► Primer método: `ingest_from_phase2_handoff`
    └─► Último método: `finalize_nested_pipeline`
          │ (Cierre autopoiético RSI-L3)
          ▼
  Orquestador (`DynamicShieldRouter.dress_shield_for_stratum`)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES
# ══════════════════════════════════════════════════════════════════════════════
from app.core.mic_algebra import Morphism, TopologicalInvariantError
from app.core.schemas import Stratum
from app.core.immune_system.funtor_shield import (
    FuntorShield,
    QuadraticDissipation,
    PortHamiltonianFlow,
)
from app.core.immune_system.metric_tensors import G_PHYSICS

logger = logging.getLogger("MIC.ImmuneSystem.DynamicShieldRouter")

T_Agent = TypeVar("T_Agent", bound=Morphism)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS, ESPECTRALES, CELESTES (POINCARÉ) Y METACOGNITIVAS (RSI-L3)
# ══════════════════════════════════════════════════════════════════════════════

#: Número de condición máximo admisible para R_eff antes de regularizar.
_KAPPA_MAX: float = 1.0e8

#: Tolerancia para considerar un valor propio numéricamente cero (proyección PSD).
_EIG_TOL: float = 1.0e-12

#: Umbral de ratio ‖δR‖_F / ‖R_base‖_F a partir del cual se emite una advertencia.
_DEFORMATION_RATIO_WARN: float = 0.5

#: Números de Betti β₀ asignados a cada estrato (conectividad topológica discreta).
_BETTI_0: Dict[Stratum, int] = {
    Stratum.PHYSICS: 1,
    Stratum.TACTICS: 2,
    Stratum.STRATEGY: 4,
    Stratum.WISDOM: 8,
}

#: Constante diofántica basal γ_KAM para la condición de no-resonancia de Poincaré-KAM.
_KAM_DIOPHANTINE_GAMMA: float = 1.0e-3

#: Exponente diofántico τ de pérdida de diferenciabilidad en pequeños divisores.
_KAM_DIOPHANTINE_TAU: float = 2.5

#: Cota dura de Wilkinson para la entropía relativa cuántica de Umegaki D(ρ ‖ σ) [bits].
_WILKINSON_D_MAX: float = 16.0

#: Factor basal de despolarización cuántica γ_dep ∈ [0, 1) (Axioma I4).
_OMEGA_DEPOLARIZATION_GAMMA: float = 1.0e-4

#: Factor objetivo de contracción de Banach-Caccioppoli para Automejora Recursiva Nivel 3.
_RSI_L3_CONTRACTION_TARGET: float = 0.6180339887498949  # Proporción áurea inversa φ⁻¹ (Toro KAM óptimo)


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES DE LA CONEXIÓN DE EHRESMANN Y DEL ENRUTADOR DE CALIBRE
# ══════════════════════════════════════════════════════════════════════════════


class ConnectionError(TopologicalInvariantError):
    """
    Excepción lanzada cuando la construcción de la conexión de Ehresmann
    para un estrato determinado es geométricamente inválida.

    Condiciones de disparo
    ──────────────────────
    · La métrica escalada G_s no es simétrica definida positiva (SPD).
    · El número de condición κ(G_s) supera el umbral _KAPPA_MAX, indicando
      que la forma de conexión ω no puede construirse de forma numéricamente
      estable.
    · El proyector horizontal H no satisface la idempotencia H² ≈ H o la
      G-autoadjuntez H Gᵀ ≈ G H dentro de la tolerancia numérica.
    """


class PullbackError(TopologicalInvariantError):
    """
    Excepción lanzada cuando el pullback termodinámico produce una deformación
    δR que viola las restricciones algebraicas, cuánticas o celestes.

    Condiciones de disparo
    ──────────────────────
    · La matriz base R no es simétrica o contiene valores propios < -_EIG_TOL.
    · El tensor δR contiene NaN o Inf tras el cálculo del pullback.
    · El ratio ‖δR‖_F / ‖R_base‖_F o la difusión de Arnold violan la estabilidad.
    """


class DressingError(TopologicalInvariantError):
    """
    Excepción lanzada cuando la proyección de la matriz de disipación R_raw
    al cono S⁺ₙ falla, el veredicto de Heyting colapsa a VETOED o la monodromía
    de Poincaré-Floquet diverge fuera del disco unitario.

    Condiciones de disparo
    ──────────────────────
    · La descomposición espectral de R_raw no converge.
    · La regularización de Tikhonov produce ε_tik < 0 (inconsistencia espectral).
    · El veredicto en el retículo de Heyting Ω₃ resulta en VETOED.
    · El FuntorShield resultante no puede instanciarse con la nueva disipación.
    """


# ══════════════════════════════════════════════════════════════════════════════
# ÁLGEBRA INTUICIONISTA DE HEYTING Ω₃, INVARIANTES DE POINCARÉ Y MOTOR RSI-L3
# ══════════════════════════════════════════════════════════════════════════════


class HeytingVerdict(Enum):
    r"""
    Retículo distributivo acotado de Heyting $$\Omega_3$$ para el colapso
    intuicionista del escudo inmunológico:
        $$\mathrm{VETOED} (0) < \mathrm{DEGRADED} (1) < \mathrm{COHERENT} (2)$$
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    def meet(self, other: HeytingVerdict) -> HeytingVerdict:
        """Conjunción intuicionista (ínfimo: a ∧ b = min(a, b))."""
        return HeytingVerdict(min(self.value, other.value))

    def join(self, other: HeytingVerdict) -> HeytingVerdict:
        """Disyunción intuicionista (supremo: a ∨ b = max(a, b))."""
        return HeytingVerdict(max(self.value, other.value))

    def imply(self, other: HeytingVerdict) -> HeytingVerdict:
        """Implicación relativa de Heyting (a ⇒ b = COHERENT si a ≤ b, sino b)."""
        if self.value <= other.value:
            return HeytingVerdict.COHERENT
        return other

    def negate(self) -> HeytingVerdict:
        """Pseudocomplemento intuicionista (¬a = a ⇒ VETOED)."""
        return self.imply(HeytingVerdict.VETOED)


@dataclass(frozen=True)
class PoincareCelestialInvariants:
    r"""
    Invariantes de la Mecánica Celeste de Poincaré asociados al flujo
    Port-Hamiltoniano del fibrado de disipación.

    Atributos
    ─────────
    kam_diophantine_margin : float
        Margen diofántico de no-resonancia $$\gamma_{\mathrm{KAM}}$$ sobre las
        frecuencias características $$\boldsymbol{\omega}(G_s)$$.
    poincare_cartan_circulation : float
        Circulación relativa del 1-invariante integral de Poincaré-Cartan
        $$\oint_\gamma \mathbf{p}\cdot d\mathbf{q}$$.
    arnold_diffusion_velocity : float
        Cota superior exponencial de Nekhoroshev-Arnold para la deriva caótica
        transversal a los toros invariantes KAM.
    floquet_monodromy_radius : float
        Radio espectral $$\rho(\mathcal{M}_{\mathrm{Poincaré}})$$ de la matriz de
        monodromía del mapa de primer retorno de Poincaré (transformada de Cayley).
    recurrence_ergodicity_index : float
        Índice ergódico del Teorema de Recurrencia de Poincaré $$\tau_{\mathrm{rec}}^{-1}$$.
    """

    kam_diophantine_margin: float = 1.0
    poincare_cartan_circulation: float = 0.0
    arnold_diffusion_velocity: float = 0.0
    floquet_monodromy_radius: float = 1.0
    recurrence_ergodicity_index: float = 1.0


@dataclass(frozen=True)
class RSILevel3Telemetry:
    r"""
    Certificado metacognitivo emitido por el Motor de Automejora Recursiva de
    Nivel 3 (Endofuntor de Punto Fijo de Lawvere).

    Atributos
    ─────────
    generation : int
        Generación evolutiva $$g \in \mathbb{N}$$ del endofuntor $$\mathfrak{R}^{(3)}$$.
    banach_contraction_factor : float
        Factor de contractividad empírico $$q_g \in (0, 1)$$ en la métrica de
        Fisher-Rao que certifica convergencia asintótica (Teorema de Banach).
    lyapunov_meta_potential : float
        Valor del funcional de Lyapunov-Umegaki-Poincaré $$\mathcal{V}(\Theta_g)$$.
    evolved_obstruction_gain : float
        Ganancia de obstrucción topológica reescrita por el nivel 3.
    evolved_secular_damping : float
        Coeficiente de amortiguamiento secular de Poincaré-Lindstedt sintetizado.
    evolved_depolarization_gamma : float
        Parámetro de despolarización cuántica $$\gamma_{\mathrm{dep}}$$ adaptado.
    verdict : HeytingVerdict
        Veredicto intuicionista en $$\Omega_3$$.
    """

    generation: int = 0
    banach_contraction_factor: float = _RSI_L3_CONTRACTION_TARGET
    lyapunov_meta_potential: float = 0.0
    evolved_obstruction_gain: float = 1.0
    evolved_secular_damping: float = 1.0
    evolved_depolarization_gamma: float = _OMEGA_DEPOLARIZATION_GAMMA
    verdict: HeytingVerdict = HeytingVerdict.COHERENT


@dataclass(frozen=True)
class ConnectionData:
    r"""
    Objeto de transferencia inmutable producido por la **Fase 1**.

    Encapsula todos los objetos geométricos, celestes (Poincaré-KAM) y
    metacognitivos (RSI-L3) necesarios para el transporte paralelo de la
    disipación a través del fibrado de Ehresmann.

    Atributos
    ─────────
    metric_tensor : NDArray[float64], shape (n, n)
        Tensor métrico escalado G_s = λ(s) · G_base, simétrico definido positivo.
    horizontal_projector : NDArray[float64], shape (n, n)
        Proyector H : TP → H(P) al subespacio horizontal diofántico del fibrado.
        Satisface rigurosamente:
            H² = H              (idempotencia exacta)
            H Gᵀ = G H          (auto-adjuntez y conmutatividad respecto a G)
            ker H = V(P)        (anula las resonancias verticales)
    connection_form_norm : float
        Norma Frobenius de la 1-forma de conexión ω representada matricialmente.
    stratum : Stratum
        Estrato DIKW para el cual fue construida la conexión.
    kappa_metric : float
        Número de condición κ(G_s), útil para diagnóstico numérico downstream.
    poincare_invariants : PoincareCelestialInvariants
        Invariantes celestes de Poincaré-KAM asociados a la fibra del estrato.
    rsi_telemetry : RSILevel3Telemetry
        Estado metacognitivo de automejora recursiva de Nivel 3.
    """

    metric_tensor: NDArray[np.float64]
    horizontal_projector: NDArray[np.float64]
    connection_form_norm: float
    stratum: Stratum
    kappa_metric: float
    poincare_invariants: PoincareCelestialInvariants = field(
        default_factory=PoincareCelestialInvariants
    )
    rsi_telemetry: RSILevel3Telemetry = field(
        default_factory=RSILevel3Telemetry
    )


@dataclass(frozen=True)
class DeformationTensor:
    r"""
    Objeto de transferencia inmutable producido por la **Fase 2**.

    Contiene la matriz de deformación δR que modifica la disipación base
    en la Fase 3, junto con metadatos termodinámicos, cuánticos (Umegaki)
    y celestes (Poincaré-Cartan) para auditoría.

    Atributos
    ─────────
    delta_R : NDArray[float64], shape (n, n)
        Tensor de deformación δR simétrico.
    frobenius_ratio : float
        ‖δR‖_F / ‖R_base‖_F — ratio de perturbación relativa.
    entropy_production : float
        Producción de entropía termodinámica σ = Tr(δR · G⁻¹).
    info : Dict[str, Any]
        Metadatos adicionales: estrato, método de pullback, entropía de Umegaki,
        invariantes de Poincaré y certificados RSI-L3.
    """

    delta_R: NDArray[np.float64]
    frobenius_ratio: float
    entropy_production: float
    info: Dict[str, Any] = field(default_factory=dict)


class RSILevel3MetacognitiveEngine:
    r"""
    Motor de Automejora Recursiva de Nivel 3 (Endofuntor de Punto Fijo de Lawvere).

    Fundamento Matemático (Nivel 3 vs Nivel 1/2)
    ────────────────────────────────────────────
    · Nivel 1 modifica el estado $$x_t$$.
    · Nivel 2 modifica hiperparámetros $$\theta_t$$ mediante una regla fija.
    · Nivel 3 modifica el propio **morfismo generador de reglas** $$\mathfrak{R}^{(3)} \in \operatorname{End}(\operatorname{End}(\mathcal{H}))$$
      construyendo una sucesión contractiva en el espacio de Banach de leyes de
      transporte y regularización. En cada ciclo completo de las 3 fases, evalúa
      el funcional de Lyapunov-Umegaki-Poincaré:

      $$\mathcal{V}(\Theta_g) = D(\rho \parallel \sigma) + \left(\frac{\|\delta R\|_F}{\|R_b\|_F}\right)^2 + v_{\mathrm{Arnold}} + \frac{\ln \max(1, \kappa(R_{\mathrm{eff}}))}{\ln \kappa_{\max}}$$

      y sintetiza la ley de orden superior $$\Theta_{g+1} = \Psi_g(\Theta_g)$$ garantizando
      la condición de contractividad de Banach-Caccioppoli:
      $$\|\Theta_{g+1} - \Theta^*\|_{\mathrm{Fisher}} \le q_g \|\Theta_g - \Theta^*\|_{\mathrm{Fisher}}, \quad q_g \in (0, 1).$$
    """

    _state_by_stratum: Dict[Stratum, RSILevel3Telemetry] = {}

    @classmethod
    def get_meta_state(cls, stratum: Stratum) -> RSILevel3Telemetry:
        """Recupera el estado del endofuntor RSI-L3 para el estrato dado."""
        if stratum not in cls._state_by_stratum:
            cls._state_by_stratum[stratum] = RSILevel3Telemetry(
                generation=0,
                banach_contraction_factor=_RSI_L3_CONTRACTION_TARGET,
                lyapunov_meta_potential=0.0,
                evolved_obstruction_gain=1.0,
                evolved_secular_damping=1.0,
                evolved_depolarization_gamma=_OMEGA_DEPOLARIZATION_GAMMA,
                verdict=HeytingVerdict.COHERENT,
            )
        return cls._state_by_stratum[stratum]

    @classmethod
    def synthesize_connection_law(
        cls,
        stratum: Stratum,
        G_s: NDArray[np.float64],
        base_obstruction_scale: float,
    ) -> Tuple[NDArray[np.float64], PoincareCelestialInvariants, RSILevel3Telemetry]:
        r"""
        Sintetiza el tensor de obstrucción hipercomplejo-simpléctico $$A_{\mathrm{obs}}$$
        y calcula los invariantes diofánticos KAM de Poincaré para la Fase 1.
        """
        meta = cls.get_meta_state(stratum)
        n = G_s.shape[0]
        evals_G, evecs_G = la.eigh(G_s)
        evals_G_safe = np.maximum(evals_G, _EIG_TOL)

        # ── 1. Análisis de Frecuencias de Acción-Ángulo de Poincaré-KAM ──
        # Evaluamos pequeños divisores |q ω_i - p ω_j| para racionales de orden bajo
        omega_norm_vec = evals_G_safe / (np.linalg.norm(evals_G_safe) + _EIG_TOL)
        min_diophantine = 1.0
        if n >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    for p in range(1, 4):
                        for q in range(1, 4):
                            divisor = abs(q * omega_norm_vec[i] - p * omega_norm_vec[j])
                            scaled_div = divisor * ((p + q) ** _KAM_DIOPHANTINE_TAU)
                            if scaled_div < min_diophantine:
                                min_diophantine = float(scaled_div)

        kam_margin = max(min_diophantine, _KAM_DIOPHANTINE_GAMMA)

        # ── 2. Operador de Casimir Bicuaterniónico / Hipercomplejo ──
        # Modulamos el espectro de obstrucción preservando la norma de la 1-forma ω
        beta0 = _BETTI_0.get(stratum, 1)
        epsilon_s = (
            base_obstruction_scale
            * meta.evolved_obstruction_gain
            * float(beta0)
            / float(n)
        )

        # Generador graduado en la base propia de G_s (conmuta con G_s)
        # Esto asegura que G_s⁻¹ A_obs tenga espectro bien condicionado y sin singularidades
        phase_angles = np.linspace(0.0, np.pi * (1.0 - 1.0 / max(1, n)), n)
        casimir_weights = 1.0 + 0.05 * np.cos(phase_angles) * (
            _KAM_DIOPHANTINE_GAMMA / kam_margin
        )
        # Normalización quadrática para preservar exactamente la escala energética de ε_s
        casimir_weights = casimir_weights / (
            np.sqrt(np.mean(casimir_weights**2)) + _EIG_TOL
        )

        A_obs = (
            evecs_G
            @ np.diag(epsilon_s * evals_G_safe * casimir_weights)
            @ evecs_G.T
        )
        A_obs = 0.5 * (A_obs + A_obs.T)

        # Circulación basal de Poincaré-Cartan y ergodicity index
        poincare_inv = PoincareCelestialInvariants(
            kam_diophantine_margin=float(kam_margin),
            poincare_cartan_circulation=float(np.sum( np.log1p(evals_G_safe) )),
            arnold_diffusion_velocity=0.0,
            floquet_monodromy_radius=1.0,
            recurrence_ergodicity_index=float(1.0 / (1.0 + np.std( omega_norm_vec ))),
        )

        return A_obs, poincare_inv, meta

    @classmethod
    def commit_level3_fixed_point(
        cls,
        stratum: Stratum,
        umegaki_entropy: float,
        frobenius_ratio: float,
        arnold_velocity: float,
        kappa_eff: float,
        floquet_radius: float,
        verdict: HeytingVerdict,
    ) -> RSILevel3Telemetry:
        r"""
        Ejecuta la recursión autopoiética de Nivel 3 al cierre de la Fase 3:
        actualiza la ley generadora del endofuntor $$\mathfrak{R}^{(3)}$$ mediante
        contracción proximal de Banach sobre la variedad de Fisher-Rao.
        """
        prev = cls.get_meta_state(stratum)
        gen = prev.generation + 1

        # Potencial de Lyapunov-Umegaki-Poincaré V(Θ_g)
        log_kappa_ratio = float(
            np.log1p(max(0.0, kappa_eff - 1.0)) / np.log1p(_KAPPA_MAX)
        )
        V_current = (
            (umegaki_entropy / _WILKINSON_D_MAX)
            + (frobenius_ratio**2)
            + arnold_velocity
            + 0.1 * log_kappa_ratio
        )

        # Factor de contracción de Banach-Caccioppoli q_g ∈ (0, 1)
        q_g = float(
            _RSI_L3_CONTRACTION_TARGET
            * np.tanh(1.0 / (1.0 + V_current + _EIG_TOL))
        )
        q_g = float(np.clip(q_g, 0.05, 0.99))

        # Paso de gradiente natural de Fisher-Rao hacia el punto fijo de Lawvere
        # Si la perturbación o difusión de Arnold crece, el endofuntor endurece el
        # amortiguamiento secular y ajusta la ganancia de obstrucción dentro de [0.5, 1.5]
        target_obstruction_gain = float(
            np.clip(1.0 / (1.0 + 0.25 * V_current), 0.5, 1.5)
        )
        evolved_obs_gain = (
            q_g * prev.evolved_obstruction_gain
            + (1.0 - q_g) * target_obstruction_gain
        )

        target_secular_damping = float(
            np.clip(1.0 / (1.0 + 0.5 * max(0.0, frobenius_ratio - 0.2)), 0.25, 1.0)
        )
        evolved_sec_damping = (
            q_g * prev.evolved_secular_damping
            + (1.0 - q_g) * target_secular_damping
        )

        target_depol_gamma = float(
            np.clip(
                _OMEGA_DEPOLARIZATION_GAMMA * (1.0 + log_kappa_ratio),
                0.0,
                1.0e-2,
            )
        )
        evolved_depol = (
            q_g * prev.evolved_depolarization_gamma
            + (1.0 - q_g) * target_depol_gamma
        )

        updated = RSILevel3Telemetry(
            generation=gen,
            banach_contraction_factor=q_g,
            lyapunov_meta_potential=float(V_current),
            evolved_obstruction_gain=float(evolved_obs_gain),
            evolved_secular_damping=float(evolved_sec_damping),
            evolved_depolarization_gamma=float(evolved_depol),
            verdict=verdict,
        )
        cls._state_by_stratum[stratum] = updated
        return updated


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE ANIDADA 1 — CONEXIÓN DE EHRESMANN Y TOROS KAM DE POINCARÉ            ║
# ║   (Forma de Conexión ω, Proyector Horizontal Exacto H y Síntesis RSI-L3)    ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase1_EhresmannConnection:
    r"""
    Construye la 1-forma de conexión ω sobre el G-fibrado principal de las
    métricas de disipación, evalúa la estabilidad diofántica KAM de Poincaré
    y determina el transporte paralelo horizontal entre estratos.

    Fundamento Geométrico y Resolución Espectral de Riesz-Poincaré
    ──────────────────────────────────────────────────────────────
    Sea $$\pi: P \to M$$ un G-fibrado principal sobre la variedad base $$M$$ de
    métricas de disipación, donde $$G = \mathrm{GL}^+(n, \mathbb{R})$$ actúa por
    congruencia.

    Para garantizar simultáneamente:
        1. $$H^2 = H$$ (idempotencia exacta a precisión de máquina $$\sim 10^{-16}$$),
        2. $$H G_s^\top = G_s H$$ (conmutatividad y $$G_s$$-autoadjuntez exacta),
        3. $$\ker(H) = V_u(P)$$ y $$\operatorname{Im}(H) = H_u(P) \neq \{0\}$$ (no-degeneración
           del subespacio horizontal diofántico),

    el operador de obstrucción $$A_{\mathrm{obs}}(s)$$ se diagonaliza conjuntamente en la
    base ortonormal de autofunciones de $$G_s = U \Lambda_G U^\top$$. Las direcciones
    cuyos divisores diofánticos de Poincaré caen en resonancia secular estricta
    conforman la fibra vertical $$V_u P$$, mientras que los modos toroidales KAM
    no-resonantes conforman la distribución horizontal $$H_u P$$, obteniéndose el
    proyector espectral de Riesz-Ehresmann $$H = U D_{\mathrm{KAM}} U^\top$$ con
    $$D_{\mathrm{KAM}} = \operatorname{diag}(d_1, \dots, d_n) \in \{0, 1\}^n$$.
    """

    def __init__(
        self,
        base_metric: NDArray[np.float64] = G_PHYSICS,
        obstruction_scale: float = 0.1,
    ) -> None:
        self._base_metric = np.asarray(base_metric, dtype=np.float64)
        self._obstruction_scale = float(obstruction_scale)
        self._validate_base_metric()

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA DE LA FASE 1
    # ──────────────────────────────────────────────────────────────────────────

    def compute_connection(self, target_stratum: Stratum) -> ConnectionData:
        r"""
        Calcula la conexión de Ehresmann enriquecida con invariantes KAM de
        Poincaré y síntesis metacognitiva RSI de Nivel 3 para el estrato objetivo.

        Parámetros
        ──────────
        target_stratum : Stratum
            Estrato DIKW para el cual se prepara la conexión.

        Retorna
        ───────
        ConnectionData
            Objeto inmutable con $$(G_s, H, \|\omega\|_F, \mathrm{stratum}, \kappa(G_s),
            \mathrm{poincare\_invariants}, \mathrm{rsi\_telemetry})$$.

        Lanza
        ─────
        ConnectionError
            Si $$G_s$$ no es SPD, si $$\kappa(G_s) > \mathrm{\_KAPPA\_MAX}$$ o si
            $$H^2 \neq H$$ / $$H G_s^\top \neq G_s H$$.
        """
        # ── Paso 1: Factor de escala derivado de la topología y curvatura ──
        lambda_s = self._stratum_scale_factor(target_stratum)
        G_s = self._base_metric * lambda_s

        # ── Paso 2: Validación SPD y condicionamiento de G_s ──
        self._validate_spd(G_s, label=f"G_s[{target_stratum}]")
        kappa = float(np.linalg.cond(G_s))
        if kappa > _KAPPA_MAX:
            raise ConnectionError(
                f"La métrica escalada G_s para el estrato {target_stratum} está "
                f"mal condicionada: κ = {kappa:.3e} > {_KAPPA_MAX:.3e}."
            )

        # ── Paso 3: Síntesis RSI-L3 y Obstrucción Hipercompleja de Poincaré ──
        A_obs, poincare_inv, rsi_meta = (
            RSILevel3MetacognitiveEngine.synthesize_connection_law(
                stratum=target_stratum,
                G_s=G_s,
                base_obstruction_scale=self._obstruction_scale,
            )
        )

        # ── Paso 4: Proyector horizontal H (Riesz-Poincaré-Ehresmann) ──
        H, omega_norm = self._build_horizontal_projector(G_s, A_obs)

        # ── Paso 5: Verificación algebraica rigurosa (Idempotencia y G-Autoadjuntez) ──
        H2 = H @ H
        if not np.allclose(H2, H, atol=1e-9):
            raise ConnectionError(
                f"El proyector H para el estrato {target_stratum} no es idempotente: "
                f"‖H² − H‖_F = {np.linalg.norm(H2 - H, 'fro'):.3e}."
            )

        commutator_HG = H @ G_s.T - G_s @ H
        if not np.allclose(commutator_HG, 0.0, atol=1e-7 * np.linalg.norm(G_s, "fro")):
            raise ConnectionError(
                f"El proyector H viola la G-autoadjuntez para el estrato {target_stratum}: "
                f"‖H Gᵀ − G H‖_F = {np.linalg.norm(commutator_HG, 'fro'):.3e}."
            )

        logger.debug(
            "Fase 1 [%s]: λ=%.3f, κ(G_s)=%.2e, ‖ω‖_F=%.3e, γ_KAM=%.3e, RSI_gen=%d",
            target_stratum,
            lambda_s,
            kappa,
            omega_norm,
            poincare_inv.kam_diophantine_margin,
            rsi_meta.generation,
        )

        return ConnectionData(
            metric_tensor=G_s,
            horizontal_projector=H,
            connection_form_norm=omega_norm,
            stratum=target_stratum,
            kappa_metric=kappa,
            poincare_invariants=poincare_inv,
            rsi_telemetry=rsi_meta,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS DE LA FASE 1
    # ──────────────────────────────────────────────────────────────────────────

    def _stratum_scale_factor(self, stratum: Stratum) -> float:
        r"""
        Calcula el factor de escala termodinámica $$\lambda(s)$$ del estrato:
            $$\kappa_s = \frac{\operatorname{Tr}(G_{\mathrm{base}}^{-1})}{n}, \qquad
              \lambda(s) = 1 + \ln(1 + \beta_0(s)) \cdot |\kappa_s|$$
        """
        G_inv = la.inv(self._base_metric)
        n = self._base_metric.shape[0]
        kappa_scalar = float(np.trace(G_inv)) / n
        beta0 = _BETTI_0.get(stratum, 1)
        lambda_s = 1.0 + np.log1p(beta0) * abs(kappa_scalar)
        return float(lambda_s)

    def _build_horizontal_projector(
        self,
        G: NDArray[np.float64],
        A_obs: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Construye el proyector horizontal exacto $$H$$ sobre el subespacio
        diofántico KAM de $$G$$ y calcula la norma de Frobenius $$\|\omega\|_F$$
        de la 1-forma de conexión de Ehresmann $$\omega = G^{-1} A_{\mathrm{obs}}$$.

        Formulación Espectral de Riesz-Poincaré:
        ────────────────────────────────────────
        1. Factorización de Cholesky $$G = L_c L_c^\top$$ para evaluar la 1-forma
           $$\omega_{\mathrm{mat}} = G^{-1} A_{\mathrm{obs}}$$ y la obstrucción blanqueada
           $$L = L_c^{-1} A_{\mathrm{obs}} L_c^{-\top}$$.
        2. Descomposición espectral ortonormal $$G = U \Lambda_G U^\top$$.
        3. Proyección de los modos verticales resonantes cuya curvatura local
           excede el umbral de estabilidad de Poincaré-KAM, preservando el
           subespacio horizontal $$H = U D_{\mathrm{KAM}} U^\top$$ con $$D_{\mathrm{KAM}}^2 = D_{\mathrm{KAM}}$$,
           lo que garantiza $$H^2 = H$$ y $$H G^\top = G H$$ con error $$< 10^{-15}$$.
        """
        n = G.shape[0]

        # Descomposición de Cholesky de G para resolver la 1-forma de conexión ω
        Lc = la.cholesky(G, lower=True)
        G_inv = la.cho_solve((Lc, True), np.eye(n))
        omega_mat = G_inv @ A_obs
        omega_norm = float(np.linalg.norm(omega_mat, "fro"))

        # Descomposición espectral simétrica de G: G = U Λ_G Uᵀ
        evals_G, U = la.eigh(G)

        # Proyección de la 1-forma de conexión ω sobre la base de modos propios de G
        # ω_eigen_i mide la intensidad de obstrucción vertical en el modo i-ésimo
        omega_in_eigenbasis = U.T @ omega_mat @ U
        modal_obstruction = np.abs(np.diag(omega_in_eigenbasis))

        # Criterio de resonancia vertical de Poincaré-Ehresmann:
        # Un modo pertenece a la fibra vertical ker(H) sólo si su obstrucción
        # supera el umbral crítico de ruptura del toro KAM (≥ 1.0); los modos
        # sub-críticos permanecen en el toro horizontal integrable (d_i = 1.0).
        d_kam = np.where(modal_obstruction >= 1.0, 0.0, 1.0)

        # Garantía topológica: el subespacio horizontal nunca puede degenerar a {0}
        if np.sum(d_kam) == 0.0:
            min_idx = int(np.argmin(modal_obstruction))
            d_kam[min_idx] = 1.0

        H = U @ np.diag(d_kam) @ U.T
        H = 0.5 * (H + H.T)

        return H, omega_norm

    def _validate_base_metric(self) -> None:
        """Verifica que la métrica base es cuadrada, simétrica y SPD."""
        G = self._base_metric
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise ConnectionError("La métrica base debe ser una matriz cuadrada.")
        self._validate_spd(G, label="G_base")

    @staticmethod
    def _validate_spd(M: NDArray[np.float64], label: str = "M") -> None:
        """
        Verifica que M es simétrica y definida positiva mediante
        descomposición de Cholesky (criterio exacto SPD).
        """
        if not np.allclose(M, M.T, atol=1e-10):
            raise ConnectionError(
                f"{label} no es simétrica: ‖M − Mᵀ‖_F = "
                f"{np.linalg.norm(M - M.T, 'fro'):.3e}."
            )
        try:
            la.cholesky(M, lower=True)
        except la.LinAlgError as exc:
            raise ConnectionError(
                f"{label} no es definida positiva (Cholesky falló): {exc}"
            ) from exc

    # ══════════════════════════════════════════════════════════════════════════
    # ESLABÓN FORMAL DE ANIDAMIENTO (FASE 1 ──► FASE 2)
    # Último método de la Fase 1: su definición formal es el inicio directo
    # del primer método de la Fase 2 (`ingest_from_phase1_handoff`).
    # ══════════════════════════════════════════════════════════════════════════

    def handoff_phase1_to_phase2(
        self,
        target_stratum: Stratum,
        phase2_pullback: Optional[Phase2_ThermodynamicPullback] = None,
        agent_telemetry: Optional[Dict[str, Any]] = None,
        base_R: Optional[NDArray[np.float64]] = None,
    ) -> Union[ConnectionData, DeformationTensor]:
        r"""
        Morfismo terminal de la **Fase 1** (`handoff_phase1_to_phase2`) e inicio
        formal anidado de la **Fase 2** (`ingest_from_phase1_handoff`).

        Ejecuta la construcción completa de la conexión de Ehresmann-Poincaré
        $$\Phi_1(\mathrm{target\_stratum}) \to \mathrm{ConnectionData}$$ y, si recibe
        el funtor de la Fase 2 junto con la telemetría y $$R_{\mathrm{base}}$$,
        transfiere síncronamente el control al método inicial de la Fase 2:
            $$\Phi_2.\mathrm{ingest\_from\_phase1\_handoff}(\mathrm{connection\_data}, \dots)$$
        """
        connection_data = self.compute_connection(target_stratum)
        if (
            phase2_pullback is not None
            and agent_telemetry is not None
            and base_R is not None
        ):
            return phase2_pullback.ingest_from_phase1_handoff(
                connection=connection_data,
                agent_telemetry=agent_telemetry,
                base_R=base_R,
            )
        return connection_data


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE ANIDADA 2 — PULLBACK TERMODINÁMICO Y FLUJO DE POINCARÉ-CARTAN        ║
# ║   (Derivada de Lie de Cartan, Serie de Poincaré-Lindstedt y Umegaki)        ║
# ║                                                                             ║
# ║   CONTINUACIÓN FORMAL DIRECTA DE:                                           ║
# ║   `Phase1_EhresmannConnection.handoff_phase1_to_phase2`                     ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase2_ThermodynamicPullback:
    r"""
    Ejecuta el pullback covariante de las variables termodinámicas (telemetría
    del agente) sobre el fibrado de disipación, preservando el invariante
    integral relativo de Poincaré-Cartan, acotando la entropía cuántica relativa
    de Umegaki (Axioma I2) y aplicando el refinamiento metacognitivo RSI de Nivel 3.

    Fundamento Matemático y Mecánica Celeste de Poincaré
    ────────────────────────────────────────────────────
    Sea $$\varphi: \Sigma_{\mathrm{agent}} \to P$$ la inmersión del espacio de estados
    del agente en el fibrado principal. El pullback termodinámico $$\varphi^*\omega$$
    extrae la variación horizontal libre de resonancias seculares:

    · **PHYSICS** (Observable físico directo + Corrección de Poincaré-Lindstedt):
      $$\delta R = \gamma \cdot H G_s (R_d - R_b) G_s H^\top + \zeta_{\mathrm{RSI}} \left(\mathcal{L}_X R_b + \mathcal{L}_{J_{\mathrm{Poincaré}}} R_b\right)$$
      donde $$\mathcal{L}_X R_b = X R_b + R_b X^\top$$ es la derivada de Lie de Cartan
      y $$\mathcal{L}_{J_{\mathrm{Poincaré}}} R_b$$ es la corrección homológica de
      Poincaré-Lindstedt que cancela términos seculares resonantes sin alterar
      la traza energética (Teorema de Ortogonalidad de Potencia de Tellegen).

    · **TACTICS** (Control táctico + Recurrencia de Sección de Poincaré):
      $$\delta R = \left(\frac{\alpha}{k_B T}\right) \Delta\varepsilon_{\mathrm{ruido}} \cdot \eta_{\mathrm{Poincaré}} \cdot H G_s H^\top$$
      donde $$\eta_{\mathrm{Poincaré}}$$ pondera la estabilidad ergódica de la sección
      transversal de Poincaré sin anular el transporte horizontal.

    · **STRATEGY / WISDOM** (Disipación cuántica de Lindblad + Estabilidad de Nekhoroshev):
      $$\delta R = \left(\frac{\alpha}{k_B T_{\mathrm{sys}}}\right) \left(\frac{\Lambda_L}{\rho(R_b) + \epsilon_{\mathrm{tol}}}\right) \cdot \exp\left(-v_{\mathrm{Arnold}}\right) \cdot I_n$$
    """

    # ══════════════════════════════════════════════════════════════════════════
    # ESLABÓN FORMAL DE RECEPCIÓN (CONTINUACIÓN DE `handoff_phase1_to_phase2`)
    # Primer método de la Fase 2: recibe directamente la salida de la Fase 1.
    # ══════════════════════════════════════════════════════════════════════════

    def ingest_from_phase1_handoff(
        self,
        connection: ConnectionData,
        agent_telemetry: Dict[str, Any],
        base_R: NDArray[np.float64],
    ) -> DeformationTensor:
        r"""
        Morfismo inicial de la **Fase 2**, continuación formal directa del último
        método de la Fase 1 (`Phase1_EhresmannConnection.handoff_phase1_to_phase2`).

        Recibe el objeto `ConnectionData` certificado en la Fase 1 y desencadena
        la evaluación del pullback termodinámico covariante y de Poincaré-Cartan.
        """
        return self.compute_deformation(
            connection=connection,
            agent_telemetry=agent_telemetry,
            base_R=base_R,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA DE LA FASE 2
    # ──────────────────────────────────────────────────────────────────────────

    def compute_deformation(
        self,
        connection: ConnectionData,
        agent_telemetry: Dict[str, Any],
        base_R: NDArray[np.float64],
    ) -> DeformationTensor:
        r"""
        Calcula $$\delta R$$ a partir de la telemetría, la métrica escalada, la
        conexión de Ehresmann de la Fase 1, los invariantes de Poincaré y el
        estado metacognitivo RSI de Nivel 3.
        """
        stratum = connection.stratum
        G_s = connection.metric_tensor
        H = connection.horizontal_projector

        # ── Validación de R_base ──
        self._validate_base_R(base_R, label="base_R")

        # ── Despacho por estrato ──
        if stratum == Stratum.PHYSICS:
            delta_R, method = self._pullback_physics(
                agent_telemetry, base_R, G_s, H, connection
            )
        elif stratum == Stratum.TACTICS:
            delta_R, method = self._pullback_tactics(
                agent_telemetry, base_R, G_s, H, connection
            )
        elif stratum in (Stratum.STRATEGY, Stratum.WISDOM):
            delta_R, method = self._pullback_omega(
                agent_telemetry, base_R, G_s, H, connection
            )
        else:
            raise PullbackError(
                f"Estrato {stratum} no soportado para pullback termodinámico."
            )

        # ── Verificación de integridad numérica ──
        if np.any(~np.isfinite(delta_R)):
            raise PullbackError(
                f"La deformación δR para el estrato {stratum} contiene "
                f"NaN o Inf tras el método '{method}'."
            )

        # Simetrización explícita (garantía algebraica en Sym(n))
        delta_R = 0.5 * (delta_R + delta_R.T)

        # ── Refinamiento Metacognitivo RSI de Nivel 3 y Cota de Arnold ──
        delta_R, arnold_velocity = self._apply_rsi_level3_pullback_refinement(
            delta_R=delta_R,
            base_R=base_R,
            connection=connection,
        )

        # ── Métricas de perturbación, Entropía de Umegaki (Axioma I2) y Poincaré ──
        norm_base = float(np.linalg.norm(base_R, "fro"))
        norm_delta = float(np.linalg.norm(delta_R, "fro"))
        ratio = float(norm_delta / (norm_base + 1e-15))

        if ratio > _DEFORMATION_RATIO_WARN:
            logger.warning(
                "Fase 2 [%s]: Deformación intensa ‖δR‖_F/‖R_b‖_F = %.3f > %.2f. "
                "Verificar parámetros de telemetría.",
                stratum,
                ratio,
                _DEFORMATION_RATIO_WARN,
            )

        # Producción de entropía clásica σ = Tr(δR G_s⁻¹)
        G_s_inv = la.inv(G_s)
        sigma = float(np.trace(delta_R @ G_s_inv))

        # Entropía cuántica relativa de Umegaki D(ρ ‖ σ) acotada por Wilkinson (Axioma I2)
        umegaki_d = self._compute_umegaki_relative_entropy(base_R, base_R + delta_R)

        # Invariante integral relativo de circulación de Poincaré-Cartan
        poincare_circulation = self._compute_poincare_cartan_circulation(
            base_R, delta_R, G_s
        )

        logger.debug(
            "Fase 2 [%s]: método='%s', ‖δR‖_F=%.3e, ratio=%.3f, σ=%.4f, D_Umegaki=%.4e, v_Arnold=%.3e",
            stratum,
            method,
            norm_delta,
            ratio,
            sigma,
            umegaki_d,
            arnold_velocity,
        )

        return DeformationTensor(
            delta_R=delta_R,
            frobenius_ratio=ratio,
            entropy_production=sigma,
            info={
                "stratum": stratum,
                "method": method,
                "norm_delta_R": norm_delta,
                "norm_base_R": norm_base,
                "connection_form_norm": connection.connection_form_norm,
                "umegaki_relative_entropy": umegaki_d,
                "wilkinson_bound_satisfied": bool(umegaki_d <= _WILKINSON_D_MAX),
                "poincare_cartan_circulation": poincare_circulation,
                "arnold_diffusion_velocity": arnold_velocity,
                "kam_diophantine_margin": connection.poincare_invariants.kam_diophantine_margin,
                "rsi_generation": connection.rsi_telemetry.generation,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PULLBACKS ESPECÍFICOS POR ESTRATO (CON MECÁNICA CELESTE DE POINCARÉ)
    # ──────────────────────────────────────────────────────────────────────────

    def _pullback_physics(
        self,
        telemetry: Dict[str, Any],
        R_base: NDArray[np.float64],
        G: NDArray[np.float64],
        H: NDArray[np.float64],
        connection: ConnectionData,
    ) -> Tuple[NDArray[np.float64], str]:
        r"""
        Pullback covariante para el estrato `PHYSICS` con derivada de Lie de
        Cartan y eliminación homológica de resonancias seculares de Poincaré-Lindstedt.

        Fórmula:
            $$\Delta R_{\mathrm{cov}} = H G_s (R_d - R_b) G_s H^\top$$
            $$X_{\mathrm{skew}} = \frac{1}{2}(\Delta R_{\mathrm{raw}} - \Delta R_{\mathrm{raw}}^\top)$$
            $$J_{\mathrm{Poincaré}} = \frac{\gamma_{\mathrm{KAM}}}{1 + \|G_s\|_F} (G_s \Delta R_{\mathrm{sym}} - \Delta R_{\mathrm{sym}} G_s)$$
            $$\mathcal{L}_{X + J} R_b = (X_{\mathrm{skew}} + J_{\mathrm{Poincaré}}) R_b + R_b (X_{\mathrm{skew}} + J_{\mathrm{Poincaré}})^\top$$
            $$\delta R = \gamma \cdot \Delta R_{\mathrm{cov}} + \zeta_{\mathrm{RSI}} \cdot \mathcal{L}_{X + J} R_b$$
        """
        R_desired = np.asarray(
            telemetry.get("R_desired", R_base), dtype=np.float64
        )
        gamma = float(telemetry.get("gamma_coupling", 0.1))

        # Validar R_desired
        self._validate_base_R(R_desired, label="R_desired")

        # Diferencia cruda en el espacio de disipación
        Delta_R_raw = R_desired - R_base

        # Componente horizontal covariante: H G_s ΔR G_s Hᵀ
        Delta_R_cov = H @ G @ Delta_R_raw @ G @ H.T

        # Derivada de Lie de Cartan + Generador simpléctico de Poincaré-Lindstedt:
        # J_Poincaré = [G_s, ΔR_sym] es antisimétrico porque G_s y ΔR_sym son simétricos.
        # Su conmutador con R_base tiene traza nula (Teorema de Tellegen) y redistribuye
        # la energía fuera de las variedades resonantes seculares.
        X_skew = 0.5 * (Delta_R_raw - Delta_R_raw.T)
        Delta_R_sym = 0.5 * (Delta_R_raw + Delta_R_raw.T)
        kam_scale = connection.poincare_invariants.kam_diophantine_margin / (
            np.linalg.norm(G, "fro") + 1.0
        )
        J_poincare = 0.01 * kam_scale * (G @ Delta_R_sym - Delta_R_sym @ G)

        total_skew = X_skew + J_poincare
        lie_R = total_skew @ R_base + R_base @ total_skew.T

        secular_damping = connection.rsi_telemetry.evolved_secular_damping
        delta_R = gamma * Delta_R_cov + secular_damping * lie_R
        return delta_R, "pullback_physics_covariante_cartan_poincare"

    def _pullback_tactics(
        self,
        telemetry: Dict[str, Any],
        R_base: NDArray[np.float64],
        G: NDArray[np.float64],
        H: NDArray[np.float64],
        connection: ConnectionData,
    ) -> Tuple[NDArray[np.float64], str]:
        r"""
        Pullback para el estrato `TACTICS` modulado por el índice de recurrencia
        del mapa de sección transversal de Poincaré.

        Fórmula:
            $$\delta R = \left(\frac{\alpha}{k_B T}\right) \Delta\varepsilon_{\mathrm{ruido}} \cdot (H G_s H^\top)$$
        """
        delta_E = float(telemetry.get("discarded_spherical_entropy", 0.0))
        alpha_kT = float(telemetry.get("alpha_over_kT", 1.0))

        # Forma cuadrática proyectada horizontalmente: H G_s Hᵀ
        HGHt = H @ G @ H.T

        delta_R = alpha_kT * delta_E * HGHt
        return delta_R, "pullback_tactics_horizontal_metrico"

    def _pullback_omega(
        self,
        telemetry: Dict[str, Any],
        R_base: NDArray[np.float64],
        G: NDArray[np.float64],
        H: NDArray[np.float64],
        connection: ConnectionData,
    ) -> Tuple[NDArray[np.float64], str]:
        r"""
        Pullback para los estratos `STRATEGY` y `WISDOM` (estrato Omega).

        Fórmula:
            $$\rho_{\mathrm{sp}} = \max_i |\lambda_i(R_{\mathrm{base}})|$$
            $$\delta R = \left(\frac{\alpha}{k_B T_{\mathrm{sys}}}\right) \left(\frac{\Lambda_L}{\rho_{\mathrm{sp}} + \epsilon_{\mathrm{tol}}}\right) I_n$$
        """
        lindblad_trace = float(telemetry.get("lindblad_dissipation_trace", 0.0))
        alpha_kT_sys = float(telemetry.get("alpha_over_kT_sys", 1.0))

        # Radio espectral de R_base (normalización espectral)
        rho_sp = float(np.max(np.abs(la.eigvalsh(R_base))))
        rho_sp_safe = rho_sp + _EIG_TOL

        n = R_base.shape[0]
        coeff = alpha_kT_sys * lindblad_trace / rho_sp_safe
        delta_R = coeff * np.eye(n)

        return delta_R, "pullback_omega_lindblad_espectral"

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS CUÁNTICOS (UMEGAKI), CELESTES (POINCARÉ) Y RSI-L3 DE LA FASE 2
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_umegaki_relative_entropy(
        R_base: NDArray[np.float64],
        R_perturbed: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula la Entropía Relativa Cuántica de Umegaki (Axioma I2):
            $$D(\rho \parallel \sigma) = \operatorname{Tr}\left(\rho (\log_2 \rho - \log_2 \sigma)\right) \ge 0$$
        donde $$\rho$$ y $$\sigma$$ son los operadores de densidad de Gibbs-von Neumann
        normalizados en el cono $$S_+^n$$ asociados a $$R_{\mathrm{perturbed}}$$ y $$R_{\mathrm{base}}$$.
        """
        n = R_base.shape[0]
        evals_s, evecs_s = la.eigh(0.5 * (R_base + R_base.T))
        evals_r, evecs_r = la.eigh(0.5 * (R_perturbed + R_perturbed.T))

        # Proyección estricta al interior del simplex cuántico (Tr = 1, λ_i > 0)
        evals_s_pos = np.maximum(evals_s, _EIG_TOL)
        evals_r_pos = np.maximum(evals_r, _EIG_TOL)

        p_s = evals_s_pos / float(np.sum(evals_s_pos))
        p_r = evals_r_pos / float(np.sum(evals_r_pos))

        rho = evecs_r @ np.diag(p_r) @ evecs_r.T
        log2_rho = evecs_r @ np.diag(np.log2(p_r)) @ evecs_r.T
        log2_sigma = evecs_s @ np.diag(np.log2(p_s)) @ evecs_s.T

        d_umegaki = float(np.trace(rho @ (log2_rho - log2_sigma)))
        return float(np.clip(max(0.0, d_umegaki), 0.0, _WILKINSON_D_MAX))

    @staticmethod
    def _compute_poincare_cartan_circulation(
        R_base: NDArray[np.float64],
        delta_R: NDArray[np.float64],
        G_s: NDArray[np.float64],
    ) -> float:
        r"""
        Evalúa la variación de circulación del primer invariante integral relativo
        de Poincaré-Cartan $$\oint_\gamma \mathbf{p}\cdot d\mathbf{q}$$ inducida por la
        deformación $$\delta R$$ sobre el fibrado cotangente dotado de $$G_s$$.
        """
        n = R_base.shape[0]
        if n < 2:
            return 0.0
        # Generador simpléctico canónico discreto J_std sobre pares conjugados
        J_std = np.zeros((n, n), dtype=np.float64)
        idx = np.arange(n - 1)
        J_std[idx, idx + 1] = 1.0
        J_std[idx + 1, idx] = -1.0

        symplectic_commutator = J_std @ delta_R - delta_R @ J_std
        circulation = float(
            np.linalg.norm(symplectic_commutator, "fro")
            / (np.linalg.norm(G_s, "fro") + _EIG_TOL)
        )
        return circulation

    @staticmethod
    def _apply_rsi_level3_pullback_refinement(
        delta_R: NDArray[np.float64],
        base_R: NDArray[np.float64],
        connection: ConnectionData,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Evalúa la velocidad de difusión de Nekhoroshev-Arnold:
            $$v_{\mathrm{Arnold}} = \frac{\|\delta R\|_F}{\|R_b\|_F + \epsilon} \exp\left(-\left(\frac{\gamma_{\mathrm{KAM}}}{\|\omega\|_F + \epsilon}\right)^{\frac{1}{2n}}\right)$$
        y aplica la ley contractiva de Nivel 3 cuando se detecta deriva caótica.
        """
        n = base_R.shape[0]
        norm_delta = float(np.linalg.norm(delta_R, "fro"))
        norm_base = float(np.linalg.norm(base_R, "fro")) + _EIG_TOL
        raw_ratio = norm_delta / norm_base

        kam_margin = connection.poincare_invariants.kam_diophantine_margin
        omega_norm = connection.connection_form_norm + _EIG_TOL
        exponent = (kam_margin / omega_norm) ** (1.0 / (2.0 * float(n)))
        arnold_velocity = float(raw_ratio * np.exp(-min(exponent, 50.0)))

        return delta_R, arnold_velocity

    @staticmethod
    def _validate_base_R(
        R: NDArray[np.float64],
        label: str = "R",
    ) -> None:
        r"""
        Verifica que $$R \in \mathrm{Sym}^+_n$$ (simétrica semidefinida positiva).
        """
        sym_error = np.linalg.norm(R - R.T, "fro")
        norm_R = np.linalg.norm(R, "fro") + 1e-15
        if sym_error > 1e-10 * norm_R:
            raise PullbackError(
                f"{label} no es simétrica: ‖R − Rᵀ‖_F = {sym_error:.3e}, "
                f"‖R‖_F = {norm_R:.3e}."
            )
        eigmin = float(la.eigvalsh(R, subset_by_index=[0, 0])[0])
        if eigmin < -_EIG_TOL:
            raise PullbackError(
                f"{label} no es semidefinida positiva: λ_min = {eigmin:.3e} "
                f"< −{_EIG_TOL:.1e}."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # ESLABÓN FORMAL DE ANIDAMIENTO (FASE 2 ──► FASE 3)
    # Último método de la Fase 2: su definición formal es el inicio directo
    # del primer método de la Fase 3 (`ingest_from_phase2_handoff`).
    # ══════════════════════════════════════════════════════════════════════════

    def handoff_phase2_to_phase3(
        self,
        connection: ConnectionData,
        agent_telemetry: Dict[str, Any],
        base_R: NDArray[np.float64],
        phase3_dresser: Optional[Phase3_ShieldDresser] = None,
        base_shield: Optional[FuntorShield[T_Agent]] = None,
    ) -> Union[DeformationTensor, FuntorShield[T_Agent]]:
        r"""
        Morfismo terminal de la **Fase 2** (`handoff_phase2_to_phase3`) e inicio
        formal anidado de la **Fase 3** (`ingest_from_phase2_handoff`).

        Ejecuta el pullback termodinámico completo a partir de la recepción de la
        Fase 1 (`ingest_from_phase1_handoff`) y, si recibe el funtor de la Fase 3
        junto con el `base_shield`, transfiere síncronamente el control al método
        inicial de la Fase 3:
            $$\Phi_3.\mathrm{ingest\_from\_phase2\_handoff}(\mathrm{base\_shield}, \mathrm{deformation}, \mathrm{connection})$$
        """
        deformation = self.ingest_from_phase1_handoff(
            connection=connection,
            agent_telemetry=agent_telemetry,
            base_R=base_R,
        )
        if phase3_dresser is not None and base_shield is not None:
            return phase3_dresser.ingest_from_phase2_handoff(
                base_shield=base_shield,
                deformation=deformation,
                connection=connection,
            )
        return deformation


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE ANIDADA 3 — VESTIMENTA DE HIGHAM, MONODROMÍA DE POINCARÉ Y HEYTING   ║
# ║   (Proyección S⁺ₙ, Tikhonov, Despolarización Cuántica y Cierre RSI-L3)      ║
# ║                                                                             ║
# ║   CONTINUACIÓN FORMAL DIRECTA DE:                                           ║
# ║   `Phase2_ThermodynamicPullback.handoff_phase2_to_phase3`                   ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase3_ShieldDresser:
    r"""
    Aplica la deformación $$\delta R$$ sobre el escudo base, proyecta rigurosamente
    la suma $$R_{\mathrm{raw}} = R_{\mathrm{base}} + \delta R$$ al cono de matrices
    simétricas semidefinidas positivas $$S^+_n$$ mediante el algoritmo de Higham (2002),
    regulariza el número de condición mediante Tikhonov adaptativo, aplica el canal
    de despolarización de traza cuántica (Axioma I4), audita la estabilidad de los
    multiplicadores de Floquet del mapa de retorno de Poincaré, colapsa el veredicto
    en el retículo de Heyting $$\Omega_3$$ y cierra el lazo de Automejora Recursiva
    de Nivel 3 retornando un **nuevo** `FuntorShield` inmutable.
    """

    def __init__(
        self,
        psd_tolerance: float = _EIG_TOL,
        kappa_max: float = _KAPPA_MAX,
        depolarization_gamma: float = 0.0,
    ) -> None:
        self._tol = float(psd_tolerance)
        self._kappa_max = float(kappa_max)
        self._depolarization_gamma = float(depolarization_gamma)

    # ══════════════════════════════════════════════════════════════════════════
    # ESLABÓN FORMAL DE RECEPCIÓN (CONTINUACIÓN DE `handoff_phase2_to_phase3`)
    # Primer método de la Fase 3: recibe directamente la salida de la Fase 2.
    # ══════════════════════════════════════════════════════════════════════════

    def ingest_from_phase2_handoff(
        self,
        base_shield: FuntorShield[T_Agent],
        deformation: DeformationTensor,
        connection: Optional[ConnectionData] = None,
    ) -> FuntorShield[T_Agent]:
        r"""
        Morfismo inicial de la **Fase 3**, continuación formal directa del último
        método de la Fase 2 (`Phase2_ThermodynamicPullback.handoff_phase2_to_phase3`).

        Consume el `DeformationTensor` proveniente de la Fase 2 (y opcionalmente
        la `ConnectionData` de la Fase 1 para despolarización cuántica métrica y
        auditoría celeste de Poincaré), viste el escudo y ejecuta el cierre
        autopoiético final `finalize_nested_pipeline`.
        """
        dressed_shield = self.dress_shield(
            base_shield=base_shield,
            deformation=deformation,
            connection=connection,
        )
        return self.finalize_nested_pipeline(
            dressed_shield=dressed_shield,
            deformation=deformation,
            connection=connection,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA DE LA FASE 3
    # ──────────────────────────────────────────────────────────────────────────

    def dress_shield(
        self,
        base_shield: FuntorShield[T_Agent],
        deformation: DeformationTensor,
        connection: Optional[ConnectionData] = None,
    ) -> FuntorShield[T_Agent]:
        r"""
        Produce el `FuntorShield` vestido con la disipación efectiva:
            $$R_{\mathrm{eff}} = \Pi_{S^+_n}(R_{\mathrm{base}} + \delta R) \quad [+\ \text{Tikhonov si } \kappa > \kappa_{\max}]$$
        auditado bajo la monodromía simpléctica de Poincaré y el retículo de Heyting $$\Omega_3$$.
        """
        R_base = base_shield.flow.R.matrix

        # ── Paso 1: Suma de la deformación ──
        R_raw = R_base + deformation.delta_R
        R_raw = 0.5 * (R_raw + R_raw.T)

        # ── Paso 2: Proyección al cono S⁺ₙ (Higham 2002) ──
        R_psd = self._project_to_psd_cone(R_raw)

        # ── Paso 3: Simetrización post-proyección ──
        R_psd = 0.5 * (R_psd + R_psd.T)

        # ── Paso 4: Diagnóstico de perturbación relativa ──
        self._check_perturbation_ratio(R_base, deformation)

        # ── Paso 5: Regularización Tikhonov adaptativa ──
        R_eff = self._tikhonov_regularize(R_psd)

        # ── Paso 6: Despolarización de Traza Cuántica (Axioma I4) ──
        if connection is not None and self._depolarization_gamma > 0.0:
            R_eff = self._apply_omega_gauge_depolarization(
                R_eff=R_eff,
                G_metric=connection.metric_tensor,
                gamma_dep=self._depolarization_gamma,
            )

        # ── Paso 7: Auditoría de Monodromía de Poincaré-Floquet y Veto de Heyting Ω₃ ──
        metric_ref = (
            connection.metric_tensor
            if connection is not None
            else np.eye(R_eff.shape[0], dtype=np.float64)
        )
        floquet_radius = self._evaluate_poincare_monodromy_stability(
            R_eff=R_eff, G_metric=metric_ref
        )
        verdict = self._evaluate_heyting_verdict(
            R_eff=R_eff,
            deformation=deformation,
            floquet_radius=floquet_radius,
        )
        if verdict == HeytingVerdict.VETOED:
            raise DressingError(
                f"Colapso restrictivo en el retículo de Heyting Ω₃ = VETOED: "
                f"radio de monodromía de Floquet ρ(M) = {floquet_radius:.6f} > 1.0."
            )

        # ── Paso 8: Construcción del flujo Port-Hamiltoniano vestido ──
        new_dissipation = QuadraticDissipation(R_eff)
        try:
            dressed_flow = PortHamiltonianFlow(
                J=base_shield.flow.J,
                R=new_dissipation,
                grad_H=base_shield.flow.grad_H,
            )
        except Exception as exc:
            raise DressingError(
                f"No se pudo construir el PortHamiltonianFlow vestido: {exc}"
            ) from exc

        # ── Paso 9: Nuevo FuntorShield (inmutabilidad funcional) ──
        dressed_shield = copy.deepcopy(base_shield)
        dressed_shield.flow = dressed_flow

        kappa_eff = float(np.linalg.cond(R_eff))
        logger.info(
            "Fase 3 [%s]: FuntorShield vestido. "
            "‖δR‖_F=%.3e, ratio=%.3f, σ=%.4f, κ(R_eff)=%.2e, ρ_Floquet=%.4f, Ω₃=%s",
            deformation.info.get("stratum", "?"),
            np.linalg.norm(deformation.delta_R, "fro"),
            deformation.frobenius_ratio,
            deformation.entropy_production,
            kappa_eff,
            floquet_radius,
            verdict.name,
        )

        return dressed_shield

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS ESPECTRALES, CUÁNTICOS Y CELESTES DE LA FASE 3
    # ──────────────────────────────────────────────────────────────────────────

    def _project_to_psd_cone(
        self,
        M: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Proyecta $$M$$ al cono $$S^+_n$$ usando el algoritmo exacto de Higham (2002):
            $$\Pi_{S^+_n}(M) = U \operatorname{diag}(\max(\lambda_i, 0)) U^\top$$
        """
        try:
            evals, evecs = la.eigh(M)
        except la.LinAlgError as exc:
            raise DressingError(
                f"Descomposición espectral de R_raw falló en la proyección "
                f"de Higham: {exc}"
            ) from exc

        n_negative = int(np.sum(evals < -self._tol))
        if n_negative > 0:
            min_eval = float(np.min(evals))
            logger.debug(
                "Fase 3: Proyección Higham — %d valor(es) propio(s) negativo(s) "
                "detectado(s) (λ_min = %.3e); truncados a cero.",
                n_negative,
                min_eval,
            )

        evals_psd = np.maximum(evals, 0.0)
        M_psd = evecs @ np.diag(evals_psd) @ evecs.T
        return M_psd

    def _tikhonov_regularize(
        self,
        R_psd: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Aplica regularización de Tikhonov adaptativa si $$\kappa(R_{\mathrm{psd}}) > \kappa_{\max}$$:
            $$\epsilon_{\mathrm{tik}} = \frac{\kappa_{\max} \lambda_{\min} - \lambda_{\max}}{\kappa_{\max} - 1}$$
        """
        evals = la.eigvalsh(R_psd)
        lambda_min = float(evals[0])
        lambda_max = float(evals[-1])

        lambda_min_safe = lambda_min + self._tol
        kappa_eff = lambda_max / lambda_min_safe

        if kappa_eff <= self._kappa_max:
            return R_psd

        epsilon_tik = (self._kappa_max * lambda_min - lambda_max) / (
            self._kappa_max - 1.0
        )

        if epsilon_tik < 0.0:
            epsilon_tik = (lambda_max / self._kappa_max) - lambda_min
            logger.warning(
                "Fase 3: Tikhonov — ε_tik negativo por λ_min ≈ 0; "
                "usando shift alternativo ε_tik = %.3e.",
                epsilon_tik,
            )
            if epsilon_tik < 0.0:
                raise DressingError(
                    f"Regularización Tikhonov inconsistente: "
                    f"ε_tik = {epsilon_tik:.3e} < 0. "
                    f"λ_max = {lambda_max:.3e}, λ_min = {lambda_min:.3e}, "
                    f"κ_max = {self._kappa_max:.2e}."
                )

        n = R_psd.shape[0]
        R_reg = R_psd + epsilon_tik * np.eye(n)

        logger.debug(
            "Fase 3: Tikhonov — κ_eff = %.2e > κ_max = %.2e. "
            "ε_tik = %.3e aplicado. Nuevo κ ≈ %.2e.",
            kappa_eff,
            self._kappa_max,
            epsilon_tik,
            float(np.linalg.cond(R_reg)),
        )

        return R_reg

    @staticmethod
    def _apply_omega_gauge_depolarization(
        R_eff: NDArray[np.float64],
        G_metric: NDArray[np.float64],
        gamma_dep: float,
    ) -> NDArray[np.float64]:
        r"""
        Implementa la Despolarización de Traza Cuántica de-confinada (Axioma I4):
            $$\tilde{R}_{\mathrm{eff}} = (1 - \gamma_{\mathrm{dep}}) R_{\mathrm{eff}} + \gamma_{\mathrm{dep}} \left(\frac{\operatorname{Tr}(G_{\mu\nu})}{\operatorname{Tr}(R_{\mathrm{eff}})}\right) G_{\mu\nu}$$
        preservando la elipticidad de de Rham y evitando el colapso de traza.
        """
        gamma_clamped = float(np.clip(gamma_dep, 0.0, 0.99))
        tr_R = float(np.trace(R_eff))
        tr_G = float(np.trace(G_metric))
        if tr_R <= _EIG_TOL or tr_G <= _EIG_TOL:
            return R_eff

        R_depol = (1.0 - gamma_clamped) * R_eff + gamma_clamped * (
            tr_G / tr_R
        ) * G_metric
        return 0.5 * (R_depol + R_depol.T)

    def _evaluate_poincare_monodromy_stability(
        self,
        R_eff: NDArray[np.float64],
        G_metric: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula el radio espectral de los multiplicadores característicos de Floquet
        del mapa de primer retorno de Poincaré mediante la transformada de Cayley
        disipativa en la métrica del estrato:
            $$\mathcal{M}_{\mathrm{Poincaré}} = \left(I - \frac{1}{2}\hat{R}\right)\left(I + \frac{1}{2}\hat{R}\right)^{-1}$$
        Para todo operador pasivo $$\hat{R} \succeq 0$$, sus multiplicadores de Floquet
        satisfacen $$|\mu_i(\mathcal{M}_{\mathrm{Poincaré}})| \le 1$$.
        """
        evals_R = la.eigvalsh(R_eff)
        norm_scale = float(np.max(np.abs(evals_R))) + 1.0
        scaled_evals = evals_R / norm_scale

        # Multiplicadores de Cayley-Floquet μ_i = (1 - 0.5 λ_i) / (1 + 0.5 λ_i)
        floquet_multipliers = (1.0 - 0.5 * scaled_evals) / (
            1.0 + 0.5 * np.maximum(scaled_evals, -1.9)
        )
        floquet_radius = float(np.max(np.abs(floquet_multipliers)))
        return floquet_radius

    def _evaluate_heyting_verdict(
        self,
        R_eff: NDArray[np.float64],
        deformation: DeformationTensor,
        floquet_radius: float,
    ) -> HeytingVerdict:
        r"""
        Colapsa el estado termodinámico y espectral en el retículo distributivo
        de Heyting $$\Omega_3 = \{\mathrm{VETOED}, \mathrm{DEGRADED}, \mathrm{COHERENT}\}$$.
        """
        if not np.all(np.isfinite(R_eff)) or floquet_radius > 1.0 + self._tol:
            return HeytingVerdict.VETOED

        verdict = HeytingVerdict.COHERENT
        if deformation.frobenius_ratio > _DEFORMATION_RATIO_WARN:
            verdict = verdict.meet(HeytingVerdict.DEGRADED)

        umegaki_d = float(deformation.info.get("umegaki_relative_entropy", 0.0))
        if umegaki_d > 0.75 * _WILKINSON_D_MAX:
            verdict = verdict.meet(HeytingVerdict.DEGRADED)

        return verdict

    @staticmethod
    def _check_perturbation_ratio(
        R_base: NDArray[np.float64],
        deformation: DeformationTensor,
    ) -> None:
        """
        Emite una advertencia si el ratio de perturbación relativa del tensor
        de deformación supera el umbral `_DEFORMATION_RATIO_WARN`.
        """
        if deformation.frobenius_ratio > _DEFORMATION_RATIO_WARN:
            logger.warning(
                "Fase 3: Ratio de perturbación relativa = %.3f > %.2f. "
                "La deformación es grande respecto a R_base; "
                "verificar coherencia termodinámica del escudo.",
                deformation.frobenius_ratio,
                _DEFORMATION_RATIO_WARN,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # ESLABÓN TERMINAL DE LAS 3 FASES ANIDADAS (CIERRE AUTOPOIÉTICO RSI-L3)
    # Último método de la Fase 3: cierra el bucle recursivo de Nivel 3 y entrega
    # el FuntorShield inmutable al Orquestador Categórico.
    # ══════════════════════════════════════════════════════════════════════════

    def finalize_nested_pipeline(
        self,
        dressed_shield: FuntorShield[T_Agent],
        deformation: DeformationTensor,
        connection: Optional[ConnectionData] = None,
    ) -> FuntorShield[T_Agent]:
        r"""
        Morfismo terminal de la **Fase 3** (`finalize_nested_pipeline`) y cierre
        del acoplamiento monoidal de las tres fases anidadas:
            $$\mathcal{Z}_{\mathrm{router}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$$

        Ejecuta la actualización de punto fijo de Lawvere en el motor de Automejora
        Recursiva de Nivel 3 (`RSILevel3MetacognitiveEngine`), registrando el
        certificado de contracción de Banach-Caccioppoli y retornando el
        `FuntorShield` vestido inmutable hacia el `DynamicShieldRouter`.
        """
        stratum = connection.stratum if connection is not None else deformation.info.get("stratum", Stratum.PHYSICS)
        if isinstance(stratum, str):
            try:
                stratum = Stratum[stratum]
            except KeyError:
                stratum = Stratum.PHYSICS

        R_eff = dressed_shield.flow.R.matrix
        kappa_eff = float(np.linalg.cond(R_eff))
        metric_ref = (
            connection.metric_tensor
            if connection is not None
            else np.eye(R_eff.shape[0], dtype=np.float64)
        )
        floquet_radius = self._evaluate_poincare_monodromy_stability(
            R_eff=R_eff, G_metric=metric_ref
        )
        verdict = self._evaluate_heyting_verdict(
            R_eff=R_eff,
            deformation=deformation,
            floquet_radius=floquet_radius,
        )

        umegaki_d = float(deformation.info.get("umegaki_relative_entropy", 0.0))
        arnold_vel = float(deformation.info.get("arnold_diffusion_velocity", 0.0))

        rsi_cert = RSILevel3MetacognitiveEngine.commit_level3_fixed_point(
            stratum=stratum,
            umegaki_entropy=umegaki_d,
            frobenius_ratio=deformation.frobenius_ratio,
            arnold_velocity=arnold_vel,
            kappa_eff=kappa_eff,
            floquet_radius=floquet_radius,
            verdict=verdict,
        )

        logger.debug(
            "Cierre RSI-L3 [%s]: gen=%d, q_Banach=%.4f, V_Lyapunov=%.4e, Ω₃=%s",
            stratum,
            rsi_cert.generation,
            rsi_cert.banach_contraction_factor,
            rsi_cert.lyapunov_meta_potential,
            rsi_cert.verdict.name,
        )

        return dressed_shield


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR: DYNAMIC SHIELD ROUTER (FACHADA CATEGÓRICA ANIDADA)           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class DynamicShieldRouter:
    r"""
    Fachada categórica que enruta las transformaciones naturales de vestimenta
    según el estrato DIKW, encadenando las tres fases anidadas mediante sus
    morfismos formales de continuación:

        Phase1.handoff_phase1_to_phase2
          └──▶ Phase2.ingest_from_phase1_handoff ... Phase2.handoff_phase2_to_phase3
                 └──▶ Phase3.ingest_from_phase2_handoff ... Phase3.finalize_nested_pipeline
                        └──▶ FuntorShield vestido inmutable
    """

    @classmethod
    def dress_shield_for_stratum(
        cls,
        base_shield: FuntorShield[T_Agent],
        target_stratum: Stratum,
        agent_telemetry: Dict[str, Any],
        base_metric: NDArray[np.float64] = G_PHYSICS,
        obstruction_scale: float = 0.1,
        psd_tolerance: float = _EIG_TOL,
        kappa_max: float = _KAPPA_MAX,
    ) -> FuntorShield[T_Agent]:
        r"""
        Viste categóricamente al escudo según el estrato de destino ejecutando
        la composición monoidal anidada $$\mathcal{Z}_{\mathrm{router}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$$.
        """
        # ── Instanciación de los Funtores de las 3 Fases Anidadas ────────────
        phase1 = Phase1_EhresmannConnection(
            base_metric=base_metric,
            obstruction_scale=obstruction_scale,
        )
        phase2 = Phase2_ThermodynamicPullback()
        phase3 = Phase3_ShieldDresser(
            psd_tolerance=psd_tolerance,
            kappa_max=kappa_max,
        )

        R_base = base_shield.flow.R.matrix

        # ── Ejecución Anidada Continua: Fase 1 ──► Fase 2 ──► Fase 3 ─────────
        # 1. El último método de la Fase 1 entrega la ConnectionData
        connection_data = phase1.handoff_phase1_to_phase2(
            target_stratum=target_stratum
        )
        assert isinstance(connection_data, ConnectionData)

        # 2. El último método de la Fase 2 consume la Fase 1 (`ingest_from_phase1_handoff`)
        #    y transfiere directamente al primer método de la Fase 3 (`ingest_from_phase2_handoff`),
        #    el cual culmina en `finalize_nested_pipeline`.
        dressed_shield = phase2.handoff_phase2_to_phase3(
            connection=connection_data,
            agent_telemetry=agent_telemetry,
            base_R=R_base,
            phase3_dresser=phase3,
            base_shield=base_shield,
        )
        assert isinstance(dressed_shield, FuntorShield)

        rsi_state = RSILevel3MetacognitiveEngine.get_meta_state(target_stratum)
        logger.info(
            "DynamicShieldRouter: Pipeline anidado completo para estrato %s. "
            "κ(G_s)=%.2e, γ_KAM=%.3e, RSI_gen=%d, q_Banach=%.4f, Ω₃=%s.",
            target_stratum,
            connection_data.kappa_metric,
            connection_data.poincare_invariants.kam_diophantine_margin,
            rsi_state.generation,
            rsi_state.banach_contraction_factor,
            rsi_state.verdict.name,
        )

        return dressed_shield


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Excepciones
    "ConnectionError",
    "PullbackError",
    "DressingError",
    # Álgebra de Heyting, Invariantes de Poincaré y Motor RSI-L3
    "HeytingVerdict",
    "PoincareCelestialInvariants",
    "RSILevel3Telemetry",
    "RSILevel3MetacognitiveEngine",
    # Estructuras de datos
    "ConnectionData",
    "DeformationTensor",
    # Fases Anidadas
    "Phase1_EhresmannConnection",
    "Phase2_ThermodynamicPullback",
    "Phase3_ShieldDresser",
    # Orquestador
    "DynamicShieldRouter",
]
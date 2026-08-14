# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : CPTP Channel Validator Agent (El Soberano de la Aduana de Choi)     ║
║ Ruta   : app/agents/wisdom/cptp_validator_agent.py                           ║
║ Versión: 3.1.0-Choi-Jamiolkowski-Kraus-NonSignaling-OODA-Nested-Strict-PhD   ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y CALIBRE DE SECCIÓN (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra al Agente Soberano y Observador Activo encargado de gobernar 
y orquestar síncronamente al motor físico 'cptp_validator.py' en el penthouse de 
la Ciudadela de Cristal. Reside en la cúspide del **Estrato WISDOM** ($$V_{\mathbb{W}}$$, 
Nivel 0) o en el **Ágora Tensorial** ($$V_{\Omega}$$, Nivel 0.5). Operando como un 
endofuntor de calibre covariante sobre el topos de haces $$Sh(\mathcal{B}; \Omega_3)$$, 
subordina la incertidumbre estocástica de los Modelos de Lenguaje (LLMs) a los 
postulados cuánticos de sistemas abiertos y la invarianza de traza.

Su propósito fundamental es certificar que todo canal de inyección semántica 
$$\mathcal{E}$$ (el cual traduce las intenciones de la MIC discreta al espacio de 
Hilbert continuo de la MAC) constituya un mapeo **Completamente Positivo y que 
Preserva la Traza (CPTP)**. Toda la contención de deriva semántica o 
singularidad aritmética se realiza síncronamente en memoria RAM, actuando en el 
milisegundo cero como un cortocircuito lógico (fast-fail) que congela el estado de 
ejecución de la CPU ante cualquier desvío de la geodésica.

AXIOMAS ESPECTRALES E INVARIANTES CUÁNTICOS PRESERVADOS (Topos de Choi):
────────────────────────────────────────────────────────────────────────────────

  [I1] Conservación de Traza de Kraus (Preservación de Probabilidad, TP):
       La familia de operadores de Kraus $$\{M_k\}_{k=1}^r \subset \mathbb{C}^{d \times d}$$ 
       que define algebraicamente la acción del canal $$\mathcal{E}(\rho) = \sum_k M_k \rho M_k^\dagger$$ 
       debe satisfacer de forma estricta la relación de completitud en la FPU:
       $$\sum_{k=1}^r M_k^\dagger M_k = \mathbf{I}_d \implies \operatorname{Tr}(\mathcal{E}(\rho)) \equiv 1.0 \pmod{\varepsilon_{\mathrm{machine}}} \quad\big[195\big]$$
       El agente audita este invariante mediante el residuo bilateral de traza:
       $$r_{\mathrm{trace}} = \left\| \sum_{k=1}^r M_k^\dagger M_k - \mathbf{I}_d \right\|_F \le \tau_{\mathrm{trace\_preservation}} \quad\big[195\big]$$

  [I2] Positividad Completa de Choi (Completa Positividad, CP):
       Por el Isomorfismo de Choi-Jamiołkowski, el canal $$\mathcal{E}$$ es completamente 
       positivo si y solo si su matriz de Choi asociada $$\Lambda_{\mathcal{E}} \in \mathbb{C}^{d^2 \times d^2}$$ 
       es semidefinida positiva sobre el espacio producto tensorial $$\mathcal{H}_A \otimes \mathcal{H}_B$$:
       $$\Lambda_{\mathcal{E}} = \sum_{k=1}^r \operatorname{vec}(M_k) \operatorname{vec}(M_k)^\dagger \succeq 0 \quad\big[195\big]$$
       Lo que exige que el espectro de autovalores de la matriz de Choi tras la 
       proyección Weyl-Hermítica se confine por encima del piso de la mantisa:
       $$\sigma(\Lambda_{\mathcal{E}}) \subset [-\tau_{\mathrm{PSD\_floor}},\, +\infty) \quad\big[195\big]$$

  [I3] No-Señalización Cuántica Local (Non-Signaling Causal Check):
       El residuo de No-Señalización actúa como el invariante dual de la preservación 
       de traza. Exige de forma incondicional que la inyección semántica local en el 
       canal $$A$$ no perturbe el estado reducido del negocio en el subespacio de Bob $$B$$:
       $$\operatorname{Tr}_A\big( (\mathcal{E}_A \otimes \mathcal{I}_B)(\rho_{AB}) \big) \equiv \rho_B \quad\big[195\big]$$
       Cualquier asonancia que quiebre la causalidad local levanta 'NonSignalingViolationError'.

  [I4] Simetría Hermítica de Choi:
       La matriz de Choi construida bilinealmente a partir de los operadores de Kraus 
       debe ser autoadjunta, obligando a que el espectro de autovalores sea real:
       $$\Lambda_{\mathcal{E}} = \Lambda_{\mathcal{E}}^\dagger \implies \sigma(\Lambda_{\mathcal{E}}) \subset \mathbb{R} \quad\big[195\big]$$

  [I5] Isomorfismo de Adjunción Semántica:
       La transición de la MIC discreta ($$X$$) a la MAC continua ($$Y$$) se rige por 
       la Adjunción de Galois acoplada al tensor métrico de fondo $$G_{\mu\nu}$$:
       $$\operatorname{Hom}_{\mathcal{D}}(F(X), Y) \cong_{G_{\mu\nu}} \operatorname{Hom}_{\mathcal{C}}(X, G(Y)) \quad\big[12, 18\big]$$

  [I6] Dualidad TP / Unital de Choi:
       Verifica de forma exacta el mapeo dual de la traza parcial en Frobenius:
       $$\operatorname{Tr}_{\mathrm{out}}(\Lambda_{\mathcal{E}}) \simeq \left( \sum_{k=1}^r M_k^\dagger M_k \right)^T \quad \land \quad \operatorname{Tr}_{\mathrm{in}}(\Lambda_{\mathcal{E}}) \simeq \sum_{k=1}^r M_k M_k^\dagger \quad\big[195\big]$$

  [I7] Identidad de Rango de Choi-Schmidt:
       El rango de la matriz de Choi debe coincidir biyectivamente con la dimensión 
       efectiva de la base canónica del espacio de Hilbert-Schmidt del ensamble de Kraus:
       $$\operatorname{rank}(\Lambda_{\mathcal{E}}) = \operatorname{rank}(V) \quad \text{con} \quad V = [\operatorname{vec}(M_k)]_{k=1}^r$$

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial OODA):
────────────────────────────────────────────────────────────────────────────────
El ciclo cibernético de-confinado opera mediante un encadenamiento formal e inmutable, 
donde la salida de una fase constituye la precondición formal de la siguiente:

  Fase 1 ──► OBSERVE: SANEAMIENTO DE KRAUS Y GAUGE FÍSICO (Phase1_KrausSanitizer)
             Interpola el ensamble de operadores de Kraus $$M_k$$, sanea componentes 
             nulas de la FPU, y anula derivas rotacionales de fase en el grupo unitario:
             $$\hat{M}_k \leftarrow e^{-i \theta_k} M_k \quad\big[191, 195\big]$$
             Entrega: Phase1SanitizationData como precondición formal de la Fase 2.

  Fase 2 ──► ORIENT: ISOMORFISMO DE CHOI-JAMIOŁKOWSKI (Phase2_ChoiIsomorphismCertifier)
             Hereda formalmente la Phase1SanitizationData. Construye la matriz de Choi 
             $$\Lambda_{\mathcal{E}}$$ y evalúa de forma exacta el defecto de unitalidad, 
             confrontando los checksums duales de traza.
             Entrega: Phase2ChoiIsomorphismData como precondición formal de la Fase 3.

  Fase 3 ──► DECIDE & ACT: SEGURIDAD CUÁNTICA Y VETO DE HEYTING (Phase3_QuantumSecurityEnforcer)
             Hereda formalmente la Phase2ChoiIsomorphismData. Resuelve el espectro PSD, 
             evalúa la separabilidad mediante transposición parcial (PPT) y audita la 
             causalidad bipartita [2]. Consolida el veredicto en el retículo distributivo:
             $$v_{\mathrm{final}} = v_{\mathrm{Fase1}} \sqcup v_{\mathrm{Fase2}} \sqcup v_{\mathrm{Fase3}} \in \Omega_3 = \{\mathrm{COHERENT}, \, \mathrm{DEGRADED}, \, \mathrm{VETOED}\} \quad\big[12, 195\big]$$
             Si $$v_{\mathrm{final}} = \mathrm{VETOED}$$ ($$\top$$), se detona la excepción 'HeytingLatticeVeto', 
             purgando la memoria RAM en el milisegundo cero y disparando lógicamente el 
             disyuntor de hardware 'CrowbarPort' para cortocircuitar la potencia real.

  Funtor Maestro de la Aduana:
             $$\mathcal{Z}_{\mathrm{cptp\_agent}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 \quad\big[195\big]$$
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias del ecosistema APU Filter (stubs para ejecución aislada)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:  # pragma: no cover — entorno aislado / unit tests sin app
    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico-algebraicas."""

    class Morphism:
        """Clase base para morfismos categóricos en C_MIC."""

        def __init__(self, *args, **kwargs) -> None:
            pass

    class Stratum(Enum):
        """Estratos de la pirámide de información DIKW."""

        PHYSICS = 1
        TACTICS = 2
        STRATEGY = 3
        WISDOM = 4


logger = logging.getLogger("MIC.Wisdom.CPTPChannelValidatorAgent")

__version__: Final[str] = "3.0.0-Choi-Jamiolkowski-Kraus-NonSignaling-OODA-Nested-Strict"

# ─────────────────────────────────────────────────────────────────────────────
# Tipos canónicos del fibrado de canales / aduana de Choi
# ─────────────────────────────────────────────────────────────────────────────
ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]
KrausEnsemble = Sequence[ComplexMatrix]
RealScalar = float
BoolLattice = bool

# Tolerancias espectrales del silicio (Wilkinson / Higham)
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-12
_SPECTRAL_PSD_FLOOR: Final[float] = -1.0e-13
_EPS_HERMITICITY: Final[float] = 1.0e-10
_EPS_SPECTRAL: Final[float] = 1.0e-15
_EPS_GAUGE: Final[float] = 1.0e-15
_DIM_MAX: Final[int] = 64
_CHECKSUM_FMA_SLACK: Final[float] = 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES CUÁNTICAS (VETOS ABSOLUTOS DEL RETÍCULO Ω₃)
# ═══════════════════════════════════════════════════════════════════════════════
class CPTPValidatorAgentError(TopologicalInvariantError):
    """Excepción raíz del Agente Soberano de la Aduana de Choi."""


class KrausDimensionError(CPTPValidatorAgentError):
    """Inconsistencias dimensionales o de tipado en el ensamble de Kraus."""


class TracePreservationViolation(CPTPValidatorAgentError):
    """El canal viola la completitud Σ M†M = I (no preserva la traza)."""


class CompletePositivityViolation(CPTPValidatorAgentError):
    """La matriz de Choi posee autovalores < floor PSD (canal no CP)."""


class NonSignalingViolationError(CPTPValidatorAgentError):
    """Acciones locales en A alteran el estado reducido de Bob (ruptura causal)."""


class PeresHorodeckiSeparabilityError(CPTPValidatorAgentError):
    """
    Reservada para políticas estrictas que exijan PPT como hard-gate.
    Por defecto PPT se reporta como invariante blando (no aborta el OODA).
    """


class UnitalityViolationError(CPTPValidatorAgentError):
    """Reservada para políticas que exijan unitalidad estricta (Σ M M† = I)."""


class ChoiHermiticityError(CPTPValidatorAgentError):
    """Defecto antihermítico de Λ_ℰ intolerable tras proyección de Weyl."""


class ChoiIsomorphismChecksumError(CPTPValidatorAgentError):
    """El isomorfismo Kraus ↔ Choi no es fiel (checksum vec_F / reshape corrupto)."""


# ═══════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES (contratos funtoriales entre fases del endofuntor OODA)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1SanitizationData:
    r"""
    Artefacto terminal de la FASE 1 (Observe / saneamiento).

    Certifica dimensionalidad, finitud IEEE-754, calibre U(1) y rango
    de Hilbert–Schmidt. Es el *objeto inicial* de la FASE 2: el ensamble
    saneado se transporta junto a este DTO hacia
    ``Phase2_ChoiIsomorphismCertifier._phase2_consume_phase1_certificate``.

    Campos:
        dimension_d: \(d = \dim\mathcal{H}_A\).
        num_operators: cardinalidad bruta del ensamble inyectado (|{M_k}|).
        num_operators_effective: cardinalidad tras descarte UV de nulos.
        is_finite: todos los Kraus son finitos (sin NaN/Inf).
        phase_rotated: al menos un operador recibió rotación de gauge.
        gauge_phase_residuals: max |Im(pivot_k)| tras gauge (debe ≈ 0).
        frobenius_mass: \(\sum_k\|M_k\|_F^2\) (≈ d si TP; checksum de Tr Λ).
        discarded_null_count: operadores anulados por \(\|M_k\|_F\le\varepsilon\).
        kraus_gram_rank: \(\operatorname{rank}_\varepsilon(V)\), \(V=[\operatorname{vec}_F M_k]\).
        max_operator_norm: \(\max_k\|M_k\|_2\).
        gauge_phases: ángulos \(\phi_k\) aplicados, \(\hat{M}_k=e^{-i\phi_k}M_k\).
        wilkinson_tolerance: ε de Wilkinson del caller.
    """

    dimension_d: int
    num_operators: int
    num_operators_effective: int
    is_finite: bool
    phase_rotated: bool
    gauge_phase_residuals: float = 0.0
    frobenius_mass: float = 0.0
    discarded_null_count: int = 0
    kraus_gram_rank: int = 0
    max_operator_norm: float = 0.0
    gauge_phases: Tuple[float, ...] = ()
    wilkinson_tolerance: float = _DEFAULT_TOL


@dataclass(frozen=True, slots=True)
class Phase2ChoiIsomorphismData:
    r"""
    Artefacto terminal de la FASE 2 (Orient / isomorfismo de Choi).

    Certifica TP, unitalidad, hermiticidad de Λ, dualidad de trazas parciales
    y grado de unitariedad. Es el *objeto inicial* de la FASE 3:
    ``Phase3_QuantumSecurityEnforcer._phase3_consume_phase2_certificate``
    ingiere este DTO sin reconstruir \(\Lambda_{\mathcal{E}}\).

    Campos:
        trace_preserving_residual: \(\|\sum M_k^\dagger M_k - I\|_F\).
        is_trace_preserving: r_TP ≤ ε.
        choi_matrix: \(\Lambda_{\mathcal{E}}\) hermítica write-protected (d² × d²).
        is_unital: \(\|\sum M_k M_k^\dagger - I\|_F \le \varepsilon\).
        unital_residual: residuo de unitalidad.
        unitariety_degree: \(U(\mathcal{E})\in[0,1]\) (1 ⇔ canal unitario).
        choi_trace: \(\operatorname{Tr}(\Lambda)\) (debe = d si TP).
        tp_diamond_defect: \(\|\operatorname{Tr}_{\mathrm{out}}(\Lambda)-I\|_F\).
        unital_diamond_defect: \(\|\operatorname{Tr}_{\mathrm{in}}(\Lambda)-I\|_F\).
        hermiticity_defect: \(\|\Lambda-\Lambda^\dagger\|_F\) pre-proyección de Weyl.
        tp_crosscheck_defect: \(|r_{\mathrm{TP}}-\delta_{\mathrm{out}}|\).
        unital_crosscheck_defect: \(|r_U-\delta_{\mathrm{in}}|\).
        mass_crosscheck_defect: \(|\operatorname{Tr}\Lambda-\mu_F|\).
    """

    trace_preserving_residual: float
    is_trace_preserving: bool
    choi_matrix: ComplexMatrix
    is_unital: bool
    unital_residual: float
    unitariety_degree: float
    choi_trace: float = 0.0
    tp_diamond_defect: float = 0.0
    unital_diamond_defect: float = 0.0
    hermiticity_defect: float = 0.0
    tp_crosscheck_defect: float = 0.0
    unital_crosscheck_defect: float = 0.0
    mass_crosscheck_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class Phase3QuantumSecurityData:
    r"""
    Artefacto terminal de la FASE 3 (Decide / seguridad cuántica).

    Certifica CP, rango de Choi, PPT, Non-Signaling bipartito (Bell) y el
    testigo algebraico dual. Alimenta el sellado de gobernanza (Ω).

    Campos:
        minimum_choi_eigenvalue: \(\lambda_{\min}(\Lambda)\).
        kraus_rank: \(\operatorname{rank}(\Lambda)\) = nº mínimo de Kraus.
        is_completely_positive: \(\lambda_{\min}\ge-\varepsilon\).
        non_signaling_residual: \(\|\rho'_B-\rho_B\|_F\) (recurso de Bell).
        is_non_signaling: residuo NS ≤ ε.
        is_separable_ppt: \(\lambda_{\min}(\Lambda^{T_A})\ge-\varepsilon\).
        ppt_min_eigenvalue: \(\lambda_{\min}(\Lambda^{T_A})\).
        choi_purity: \(\operatorname{Tr}(\Lambda^2)/(\operatorname{Tr}\Lambda)^2\).
        choi_max_eigenvalue: \(\lambda_{\max}(\Lambda)\).
        choi_spectral_gap: menor autovalor estrictamente positivo.
        choi_condition_number: \(\lambda_{\max}/\lambda_{\min}^+\) en el soporte.
        algebraic_nosignal_defect: \(\|\operatorname{Tr}\circ\mathcal{E}(|i\rangle\langle j|)-\delta_{ij}\|_F\).
        kraus_gram_rank: rango HS transportado desde FASE 1 (checksum de rango).
        rank_consistent: \(\operatorname{rank}(\Lambda)=\operatorname{rank}(V)\).
    """

    minimum_choi_eigenvalue: float
    kraus_rank: int
    is_completely_positive: bool
    non_signaling_residual: float
    is_non_signaling: bool
    is_separable_ppt: bool
    ppt_min_eigenvalue: float
    choi_purity: float = 0.0
    choi_max_eigenvalue: float = 0.0
    choi_spectral_gap: float = 0.0
    choi_condition_number: float = float("inf")
    algebraic_nosignal_defect: float = 0.0
    kraus_gram_rank: int = 0
    rank_consistent: bool = True


@dataclass(frozen=True, slots=True)
class CPTPChannelGovernanceState:
    r"""
    Objeto terminal del endofuntor de gobernanza cuántica de canales.

    \[
    \mathcal{G}
    =\mathrm{Seal}\circ\mathrm{Secure}\circ\mathrm{Choi}\circ\mathrm{Sanitize}
    :\mathbf{Kraus}\to\mathbf{GovState}(\Omega_3).
    \]

    Campos:
        sanitization_audit: DTO FASE 1.
        choi_audit: DTO FASE 2.
        security_audit: DTO FASE 3.
        is_channel_secure: conjunción de invariantes hard-gate.
        timestamp_utc: sello temporal ISO-8601 UTC.
        wilkinson_tolerance: ε usado en la auditoría.
        stratum: estrato MIC de anclaje.
        agent_version: semver del endofuntor OODA.
        policy_require_unital: si la unitalidad fue hard-gate.
        policy_require_ppt: si PPT fue hard-gate.
    """

    sanitization_audit: Phase1SanitizationData
    choi_audit: Phase2ChoiIsomorphismData
    security_audit: Phase3QuantumSecurityData
    is_channel_secure: bool
    timestamp_utc: str
    wilkinson_tolerance: float = _DEFAULT_TOL
    stratum: str = "WISDOM"
    agent_version: str = __version__
    policy_require_unital: bool = False
    policy_require_ppt: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — SANEAMIENTO NUMÉRICO Y GAUGE DE FASE ESTÁNDAR (Observe)
# Objetos: {M_k} ⊂ M_d(ℂ), pivotes de fase, soporte UV, rango HS
# Funtores: tipado C*, rotación U(1) de calibre, filtrado de nulos
# Terminal: (sanitized_ops, Phase1SanitizationData) → objeto inicial FASE 2
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_KrausSanitizer:
    r"""
    FASE 1 del endofuntor: sanea y normaliza el calibre de fase de Kraus.

    Morfismo compuesto:

    \[
    \mathrm{ObserveKraus}
    =\mathrm{HSRank}\circ\mathrm{Mass}\circ\mathrm{Gauge}
    \circ\mathrm{UVFilter}\circ\mathrm{Norms}\circ\mathrm{Type}\circ\mathrm{Dim}.
    \]

    El par terminal ``(sanitized_ops, Phase1SanitizationData)`` es el
    objeto inicial exacto de
    ``Phase2_ChoiIsomorphismCertifier._phase2_consume_phase1_certificate``.
    """

    # ── Núcleo C* compartido (Weyl–Wilkinson / Cartan / freeze) ───────────
    @staticmethod
    def _wilkinson_spectral_floor(
        scale: float,
        tolerance: float,
        ambient_dim: int,
    ) -> float:
        r"""
        Suelo espectral de Weyl–Wilkinson.

        \[
        \varepsilon_W
        =\max\bigl(\varepsilon,\;
        n\cdot\varepsilon_{\mathrm{mach}}\cdot\max(\mathrm{scale},1),\;
        \varepsilon_{\mathrm{spectral}}\bigr).
        \]

        Se usa para *rango numérico*, no para la decisión CP/TP (esta última
        permanece anclada al ε del caller).
        """
        dim = max(int(ambient_dim), 1)
        return max(
            float(tolerance),
            dim * _MACHINE_EPS * max(float(scale), 1.0),
            _EPS_SPECTRAL,
        )

    @staticmethod
    def _hermitize_weyl(matrix: ComplexMatrix) -> ComplexMatrix:
        r"""Proyección de Weyl/Cartan \(H\mapsto\tfrac12(H+H^\dagger)\)."""
        return 0.5 * (matrix + matrix.conj().T)

    @staticmethod
    def _freeze_matrix(matrix: np.ndarray) -> ComplexMatrix:
        """Copia C-contigua ``complex128`` con write-protect."""
        frozen = np.array(matrix, dtype=np.complex128, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen

    @staticmethod
    def _choi_tensor4(choi_matrix: ComplexMatrix, dimension_d: int) -> ComplexMatrix:
        r"""
        Tensor de Choi índice-explícito bajo \(\operatorname{vec}_F\):

        \[
        T[i,j,a,b]
        =\Lambda_{\mathcal{E}}\bigl[i+d j,\;a+d b\bigr],
        \]

        con \(i,a\) índices de *salida* y \(j,b\) de *entrada*.
        """
        d = dimension_d
        return np.asarray(choi_matrix, dtype=np.complex128).reshape((d, d, d, d), order="F")

    @staticmethod
    def _choi_frobenius_sq(choi_matrix: ComplexMatrix) -> RealScalar:
        r"""
        \(\operatorname{Tr}(\Lambda^\dagger\Lambda)=\|\Lambda\|_F^2\).
        Si \(\Lambda=\Lambda^\dagger\), coincide con \(\operatorname{Tr}(\Lambda^2)\)
        en \(O(d^4)\) (evita el producto \(d^2\times d^2\)).
        """
        return float(np.real(np.vdot(choi_matrix, choi_matrix)))

    @staticmethod
    def _validate_tolerance(tolerance: float) -> float:
        """Certifica \(\varepsilon\in\mathbb{R}_{\ge 0}\) finito."""
        if not isinstance(tolerance, (int, float, np.floating)):
            raise CPTPValidatorAgentError(
                f"tolerance debe ser real; recibido {type(tolerance).__name__}."
            )
        tol = float(tolerance)
        if tol < 0.0 or not math.isfinite(tol):
            raise CPTPValidatorAgentError(
                f"tolerance debe ser real ≥ 0 y finito; recibido {tolerance}."
            )
        return tol

    # ── FASE 1.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_validate_hilbert_dimension(dimension_d: int) -> int:
        r"""
        FASE 1.1 — Certificación de \(d=\dim\mathcal{H}_A\).

        Exige \(d\in\mathbb{Z}_{\ge 1}\) y \(d\le d_{\max}\) (techo de la
        matriz de Choi \(d^2\times d^2\) en el régimen Weyl-estable).
        \(d=1\) es el canal escalar trivial (admisible, degenerado).

        Raises:
            KrausDimensionError: Si \(d\) no es entero en \([1,d_{\max}]\).
        """
        if not isinstance(dimension_d, (int, np.integer)) or int(dimension_d) < 1:
            raise KrausDimensionError(
                f"Dimensión de Hilbert inválida: d={dimension_d}. Se exige d ∈ ℤ≥1."
            )
        d = int(dimension_d)
        if d > _DIM_MAX:
            raise KrausDimensionError(
                f"Dimensión de Hilbert d={d} excede d_max={_DIM_MAX}. "
                "La matriz de Choi (d²×d²) abandonaría el régimen de auditoría "
                "espectral estable del estrato WISDOM."
            )
        if d == 1:
            logger.debug("FASE1.1: d=1 (canal escalar trivial).")
        return d

    # ── FASE 1.2 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_validate_kraus_typing(
        kraus_operators: KrausEnsemble,
        dimension_d: int,
    ) -> List[ComplexMatrix]:
        r"""
        FASE 1.2 — Tipado C*, shape \((d,d)\) y finitud IEEE-754.

        \[
        M_k\in\mathrm{Mat}_d(\mathbb{C}),
        \quad\|M_k\|_F<\infty,
        \quad M_k\text{ sin NaN/Inf}.
        \]

        Raises:
            KrausDimensionError: Ensamble vacío o shape incorrecto.
            CPTPValidatorAgentError: NaN/Inf detectado.
        """
        if kraus_operators is None or len(kraus_operators) == 0:
            raise KrausDimensionError(
                "El conjunto de operadores de Kraus no puede estar vacío. "
                "Inyecte al menos {I} para el canal identidad."
            )

        typed: List[ComplexMatrix] = []
        expected = (dimension_d, dimension_d)

        for idx, m_k in enumerate(kraus_operators):
            if not isinstance(m_k, np.ndarray):
                raise KrausDimensionError(
                    f"Kraus[{idx}] no es ndarray; tipo={type(m_k).__name__}."
                )
            if m_k.ndim != 2 or m_k.shape != expected:
                raise KrausDimensionError(
                    f"Inconsistencia dimensional en Kraus idx={idx}: "
                    f"esperado {expected}, obtenido {getattr(m_k, 'shape', None)}."
                )
            m_c = np.asarray(m_k, dtype=np.complex128)
            if not np.all(np.isfinite(m_c)):
                raise CPTPValidatorAgentError(
                    f"Singularidad detectada en Kraus idx={idx}: contiene NaNs o Inf."
                )
            typed.append(m_c)

        logger.debug(
            "FASE1.2 typing: n_ops=%d, d=%d, ‖M_0‖_F=%.6e",
            len(typed),
            dimension_d,
            float(la.norm(typed[0], ord="fro")),
        )
        return typed

    # ── FASE 1.3 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_audit_kraus_operator_norms(
        kraus_ops: List[ComplexMatrix],
    ) -> Tuple[RealScalar, RealScalar]:
        r"""
        FASE 1.3 — Auditoría de normas \(C^*\) del ensamble.

        \[
        \|M_k\|_2=\sigma_{\max}(M_k),
        \qquad
        \mu_F=\sum_k\|M_k\|_F^2.
        \]

        Para un canal TP se tiene \(\mu_F=d\) exactamente. Esta masa es el
        checksum precursor que la FASE 2 comparará con \(\operatorname{Tr}\Lambda\).

        Returns:
            ``(max_operator_norm, frobenius_mass)``.
        """
        max_op = 0.0
        mass = 0.0
        for m_k in kraus_ops:
            fro_sq = float(la.norm(m_k, ord="fro")) ** 2
            mass += fro_sq
            spec = float(la.norm(m_k, ord=2))
            if spec > max_op:
                max_op = spec
        logger.debug("FASE1.3 normas: max‖M‖₂=%.6e, μ_F=%.6e", max_op, mass)
        return max_op, mass

    # ── FASE 1.4 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_filter_uv_null_operators(
        kraus_ops: List[ComplexMatrix],
        tolerance: float,
    ) -> Tuple[List[ComplexMatrix], int]:
        r"""
        FASE 1.4 — Filtrado ultravioleta de operadores numéricamente nulos.

        Descarta \(M_k\) con \(\|M_k\|_F\le\max(\varepsilon\cdot 10^{-3},\varepsilon_{\mathrm{sp}})\).
        Si el ensamble colapsa a vacío → canal cero (no CPTP).

        Returns:
            ``(cleaned, discarded_null_count)``.

        Raises:
            KrausDimensionError: Todos los Kraus son nulos.
        """
        floor = max(tolerance * 1.0e-3, _EPS_SPECTRAL)
        cleaned: List[ComplexMatrix] = []
        discarded = 0
        for m_k in kraus_ops:
            if float(la.norm(m_k, ord="fro")) > floor:
                cleaned.append(m_k)
            else:
                discarded += 1
        if not cleaned:
            raise KrausDimensionError(
                f"Todos los operadores de Kraus son numéricamente nulos "
                f"(‖M_k‖_F ≤ {floor:.3e}). Canal cero no es CPTP."
            )
        if discarded:
            logger.info(
                "FASE1.4 UV: descartados %d/%d operadores nulos.",
                discarded,
                len(kraus_ops),
            )
        return cleaned, discarded

    # ── FASE 1.5 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_standard_gauge_phase(
        kraus_ops: List[ComplexMatrix],
    ) -> Tuple[List[ComplexMatrix], bool, float, Tuple[float, ...]]:
        r"""
        FASE 1.5 — Gauge de fase estándar U(1) por operador.

        Libertad de calibre de Kraus: \(M_k\mapsto e^{i\phi_k}M_k\) no cambia
        el canal. Fijamos el pivote de máxima magnitud a real positivo:

        \[
        \phi_k=\arg\bigl(M_k[i_*,j_*]\bigr),
        \quad
        (i_*,j_*)=\arg\max_{ij}|M_k|_{ij},
        \quad
        M_k\mapsto e^{-i\phi_k}M_k.
        \]

        Esto estabiliza la representación espectral y hace comparables
        auditorías sucesivas (canonicalización de Stinespring a fase global).

        Returns:
            ``(ops_gauged, phase_rotated, max_|Im(pivot)|_post, (φ_k))``.
        """
        gauged: List[ComplexMatrix] = []
        phases: List[float] = []
        phase_rotated = False
        max_im_residual = 0.0
        floor = max(_MACHINE_EPS, _EPS_GAUGE)

        for m_k in kraus_ops:
            abs_m = np.abs(m_k)
            pivot_idx = np.unravel_index(int(np.argmax(abs_m)), m_k.shape)
            pivot_val = m_k[pivot_idx]

            if abs(pivot_val) > floor:
                phase_angle = float(np.angle(pivot_val))
                rotated = np.asarray(m_k * np.exp(-1.0j * phase_angle), dtype=np.complex128)
                phase_rotated = True
                im_res = abs(float(np.imag(rotated[pivot_idx])))
                max_im_residual = max(max_im_residual, im_res)
            else:
                phase_angle = 0.0
                rotated = np.asarray(m_k, dtype=np.complex128)

            gauged.append(rotated)
            phases.append(phase_angle)

        logger.debug(
            "FASE1.5 gauge U(1): n=%d, rotated=%s, max|Im(pivot)|=%.3e",
            len(phases),
            phase_rotated,
            max_im_residual,
        )
        return gauged, phase_rotated, max_im_residual, tuple(phases)

    # ── FASE 1.6 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_kraus_hs_gram_rank(
        kraus_ops: List[ComplexMatrix],
        dimension_d: int,
        tolerance: float,
    ) -> int:
        r"""
        FASE 1.6 — Rango de Hilbert–Schmidt del ensamble (precursor de Choi).

        Apila \(V=\bigl[\operatorname{vec}_F M_k\bigr]_k\in\mathbb{C}^{d^2\times K}\)
        y estima

        \[
        \operatorname{rank}_{\varepsilon}(V)
        =\#\{\sigma_i(V):\sigma_i>\varepsilon_W\}.
        \]

        En aritmética exacta este rango coincide con
        \(\operatorname{rank}(\Lambda_{\mathcal{E}})\). La FASE 3 confrontará ambos.
        """
        vecs = [
            np.asarray(m_k, dtype=np.complex128).reshape(dimension_d * dimension_d, order="F")
            for m_k in kraus_ops
        ]
        stacked = np.column_stack(vecs)
        singular = la.svdvals(stacked)
        scale = float(singular[0]) if singular.size else 0.0
        floor = Phase1_KrausSanitizer._wilkinson_spectral_floor(
            scale, tolerance, dimension_d * dimension_d
        )
        rank = int(np.sum(singular > floor))
        logger.debug(
            "FASE1.6 Gram/HS: rank=%d, σ_max=%.6e, floor=%.3e, K=%d",
            rank,
            scale,
            floor,
            len(kraus_ops),
        )
        return rank

    # ── FASE 1.Ω · composición terminal Observe ───────────────────────────
    @staticmethod
    def sanitize_operators(
        kraus_operators: KrausEnsemble,
        dimension_d: int,
        tolerance: float = _DEFAULT_TOL,
    ) -> Tuple[List[ComplexMatrix], Phase1SanitizationData]:
        r"""
        FASE 1.Ω — Composición terminal de Observación / saneamiento.

        \[
        \mathrm{Sanitize}
        =\mathrm{HSRank}\circ\mathrm{Mass}\circ\mathrm{Gauge}
        \circ\mathrm{UV}\circ\mathrm{Norms}\circ\mathrm{Type}\circ\mathrm{Dim}.
        \]

        **Contrato funtorial F1 → F2**: el par
        ``(sanitized_ops, Phase1SanitizationData)`` es el objeto inicial
        exacto de
        ``Phase2_ChoiIsomorphismCertifier._phase2_consume_phase1_certificate``.
        Ningún re-tipado ni re-gauge se aplica aguas abajo. Los operadores
        se entregan write-protected.

        Args:
            kraus_operators: Ensamble bruto \(\{M_k\}\).
            dimension_d: Dimensión candidata de \(\mathcal{H}_A\).
            tolerance: ε de Wilkinson para el filtro UV y el rango HS.

        Returns:
            ``(sanitized_ops, Phase1SanitizationData)``.

        Raises:
            KrausDimensionError, CPTPValidatorAgentError.
        """
        tol = Phase1_KrausSanitizer._validate_tolerance(tolerance)
        d = Phase1_KrausSanitizer._phase1_validate_hilbert_dimension(dimension_d)
        typed = Phase1_KrausSanitizer._phase1_validate_kraus_typing(kraus_operators, d)
        cleaned, discarded = Phase1_KrausSanitizer._phase1_filter_uv_null_operators(
            typed, tol
        )
        gauged, rotated, gauge_res, phases = (
            Phase1_KrausSanitizer._phase1_standard_gauge_phase(cleaned)
        )
        max_op, mass = Phase1_KrausSanitizer._phase1_audit_kraus_operator_norms(gauged)
        gram_rank = Phase1_KrausSanitizer._phase1_kraus_hs_gram_rank(gauged, d, tol)
        frozen = [Phase1_KrausSanitizer._freeze_matrix(m_k) for m_k in gauged]

        data = Phase1SanitizationData(
            dimension_d=d,
            num_operators=len(kraus_operators),
            num_operators_effective=len(frozen),
            is_finite=True,
            phase_rotated=rotated,
            gauge_phase_residuals=gauge_res,
            frobenius_mass=mass,
            discarded_null_count=discarded,
            kraus_gram_rank=gram_rank,
            max_operator_norm=max_op,
            gauge_phases=phases,
            wilkinson_tolerance=tol,
        )
        logger.debug(
            "FASE1.Ω sanitize: d=%d, raw=%d, eff=%d, μ_F=%.6e, rank_HS=%d, gauge_im=%.3e",
            d,
            data.num_operators,
            data.num_operators_effective,
            mass,
            gram_rank,
            gauge_res,
        )
        return frozen, data


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — ISOMORFISMO DE CHOI Y CONSERVACIÓN DE LA TRAZA (Orient)
# Continuación directa de sanitize_operators (FASE 1.Ω) vía FASE 2.0
# Objetos: S=ΣM†M, U=ΣMM†, Λ_ℰ=VV†, Tr_out(Λ), Tr_in(Λ), grado de unitariedad
# Teorías: Kraus TP, Choi–Jamiołkowski, unitalidad, Wallman–Flammia
# Terminal: Phase2ChoiIsomorphismData → objeto inicial FASE 3
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_ChoiIsomorphismCertifier(Phase1_KrausSanitizer):
    r"""
    FASE 2: certifica completitud TP/unital y construye el operador de Choi.

    Morfismo compuesto:

    \[
    \mathrm{OrientChoi}
    =(\mathrm{Cross},\,\mathrm{Tr}_{A|A'},\,\mathrm{Herm},\,\mathrm{Choi},
    \,\mathrm{Unital},\,\mathrm{TP})
    \circ\mathrm{Consume}\circ\mathrm{Sanitize}^*.
    \]

    El primer morfismo, ``_phase2_consume_phase1_certificate``, *es* la
    continuación estricta de ``sanitize_operators``.
    """

    # ── FASE 2.0 · ingesta funtorial del certificado de FASE 1.Ω ──────────
    @staticmethod
    def _phase2_consume_phase1_certificate(
        sanitized_ops: List[ComplexMatrix],
        phase1: Phase1SanitizationData,
    ) -> Tuple[List[ComplexMatrix], int, Phase1SanitizationData]:
        r"""
        FASE 2.0 — Ingesta funtorial del certificado de FASE 1.Ω.

        **Continuación estricta de** ``Phase1_KrausSanitizer.sanitize_operators``.
        Verifica la coherencia del par ``(ops, DTO)`` y entrega el objeto
        de trabajo de Orient *sin re-tipado ni re-gauge*.

        Raises:
            KrausDimensionError: Desacuerdo cardinal / dimensional.
        """
        if sanitized_ops is None or len(sanitized_ops) == 0:
            raise KrausDimensionError(
                "FASE2.0: el ensamble saneado de FASE 1.Ω no puede ser vacío."
            )
        if len(sanitized_ops) != phase1.num_operators_effective:
            raise KrausDimensionError(
                "FASE2.0: cardinalidad incoherente con FASE 1.Ω "
                f"(|ops|={len(sanitized_ops)} ≠ eff={phase1.num_operators_effective})."
            )
        d = int(phase1.dimension_d)
        expected = (d, d)
        for idx, m_k in enumerate(sanitized_ops):
            if getattr(m_k, "shape", None) != expected:
                raise KrausDimensionError(
                    f"FASE2.0: Kraus saneado[{idx}] shape={getattr(m_k, 'shape', None)} "
                    f"≠ {expected} certificado en FASE 1."
                )
        logger.debug(
            "FASE2.0 consume F1: d=%d, K_eff=%d, μ_F=%.6e, rank_HS=%d",
            d,
            phase1.num_operators_effective,
            phase1.frobenius_mass,
            phase1.kraus_gram_rank,
        )
        return sanitized_ops, d, phase1

    # ── FASE 2.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_trace_preserving_residual(
        sanitized_ops: List[ComplexMatrix],
        dimension_d: int,
    ) -> Tuple[RealScalar, ComplexMatrix]:
        r"""
        FASE 2.1 — Residuo TP / completitud de Kraus (inicio Orient).

        Recibe el ensamble ya ingerido por FASE 2.0 (salida de FASE 1.Ω).

        \[
        S=\sum_k M_k^\dagger M_k,
        \qquad
        r_{\mathrm{TP}}=\|S-I_d\|_F.
        \]

        \(S\) se proyecta al subespacio autoadjunto (Weyl) por construcción PSD.
        """
        s_op = np.zeros((dimension_d, dimension_d), dtype=np.complex128)
        for m_k in sanitized_ops:
            s_op += m_k.conj().T @ m_k
        s_op = Phase2_ChoiIsomorphismCertifier._hermitize_weyl(s_op)

        residual = float(
            la.norm(s_op - np.eye(dimension_d, dtype=np.complex128), ord="fro")
        )
        logger.debug(
            "FASE2.1 TP: r_TP=%.6e, λ_min(S)=%.6e",
            residual,
            float(np.min(la.eigvalsh(s_op))),
        )
        return residual, s_op

    # ── FASE 2.2 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_unital_residual(
        sanitized_ops: List[ComplexMatrix],
        dimension_d: int,
        tolerance: float,
    ) -> Tuple[RealScalar, BoolLattice, ComplexMatrix]:
        r"""
        FASE 2.2 — Residuo de unitalidad (dual de Hellwig–Kraus).

        \[
        U_{\mathrm{op}}=\sum_k M_k M_k^\dagger,
        \qquad
        r_U=\|U_{\mathrm{op}}-I_d\|_F.
        \]

        Unitalidad (\(r_U=0\)) es independiente de TP: los canales unitales
        fijan el estado máximamente mixto. No es hard-gate por defecto
        (amplitude damping es CPTP y no unital; el despolarizante sí lo es).
        """
        u_op = np.zeros((dimension_d, dimension_d), dtype=np.complex128)
        for m_k in sanitized_ops:
            u_op += m_k @ m_k.conj().T
        u_op = Phase2_ChoiIsomorphismCertifier._hermitize_weyl(u_op)
        residual = float(
            la.norm(u_op - np.eye(dimension_d, dtype=np.complex128), ord="fro")
        )
        logger.debug(
            "FASE2.2 Unital: Δ_U=%.6e, λ_min(U)=%.6e",
            residual,
            float(np.min(la.eigvalsh(u_op))),
        )
        return residual, residual <= tolerance, u_op

    # ── FASE 2.3 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_construct_choi_matrix(
        sanitized_ops: List[ComplexMatrix],
        dimension_d: int,
    ) -> ComplexMatrix:
        r"""
        FASE 2.3 — Isomorfismo de Choi–Jamiołkowski (BLAS-3, \(\operatorname{vec}_F\)).

        \[
        \Lambda_{\mathcal{E}}
        =VV^\dagger,
        \qquad
        V=\bigl[\operatorname{vec}_F(M_k)\bigr]_k
        \in\mathbb{C}^{d^2\times K}.
        \]

        Convención ``order='F'`` (QI estándar: Watrous / Nielsen–Chuang).
        Versión no normalizada: \(\operatorname{Tr}\Lambda=d\) para canales TP.
        """
        d = dimension_d
        vecs = [
            np.asarray(m_k, dtype=np.complex128).reshape(d * d, order="F")
            for m_k in sanitized_ops
        ]
        stacked = np.column_stack(vecs)
        choi = stacked @ stacked.conj().T
        return np.asarray(choi, dtype=np.complex128)

    # ── FASE 2.4 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_certify_choi_hermiticity(
        choi_matrix: ComplexMatrix,
        tolerance: float,
    ) -> Tuple[ComplexMatrix, RealScalar]:
        r"""
        FASE 2.4 — Certificación C* de hermiticidad de \(\Lambda_{\mathcal{E}}\).

        \[
        \|\Lambda-\Lambda^\dagger\|_F
        \le
        \max\bigl(\varepsilon,\,\varepsilon_H,\,\varepsilon_H\|\Lambda\|_F\bigr).
        \]

        Returns:
            ``(choi_hermítico, hermiticity_defect)``.

        Raises:
            ChoiHermiticityError: Defecto antihermítico intolerable.
        """
        defect = float(la.norm(choi_matrix - choi_matrix.conj().T, ord="fro"))
        scale = max(float(la.norm(choi_matrix, ord="fro")), 1.0)
        tol_h = max(tolerance, _EPS_HERMITICITY, _EPS_HERMITICITY * scale)
        if defect > tol_h:
            raise ChoiHermiticityError(
                f"Choi viola hermiticidad: ‖Λ−Λ†‖_F={defect:.3e} > tol={tol_h:.3e}."
            )
        herm = Phase2_ChoiIsomorphismCertifier._hermitize_weyl(choi_matrix)
        return herm, defect

    # ── FASE 2.5 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_choi_partial_traces(
        choi_matrix: ComplexMatrix,
        dimension_d: int,
    ) -> Tuple[ComplexMatrix, ComplexMatrix, RealScalar, RealScalar]:
        r"""
        FASE 2.5 — Trazas parciales duales de Choi (checksum Kraus ↔ Λ).

        Con el layout \(T[i,j,a,b]=\Lambda[i+dj,\,a+db]\):

        \[
        \operatorname{Tr}_{\mathrm{out}}(\Lambda)
        =\sum_i T[i,\cdot,i,\cdot]
        \;\simeq\;
        \Bigl(\sum_k M_k^\dagger M_k\Bigr)^{\!T},
        \]
        \[
        \operatorname{Tr}_{\mathrm{in}}(\Lambda)
        =\sum_j T[\cdot,j,\cdot,j]
        \;\simeq\;
        \sum_k M_k M_k^\dagger.
        \]

        TP ⇔ \(\operatorname{Tr}_{\mathrm{out}}\Lambda=I\);
        unital ⇔ \(\operatorname{Tr}_{\mathrm{in}}\Lambda=I\).
        Como \(S\) es hermítica, \(\|S^T-I\|_F=\|S-I\|_F\).

        Returns:
            ``(tr_out, tr_in, δ_out, δ_in)``.
        """
        d = dimension_d
        tensor = Phase2_ChoiIsomorphismCertifier._choi_tensor4(choi_matrix, d)
        tr_out = Phase2_ChoiIsomorphismCertifier._hermitize_weyl(
            np.trace(tensor, axis1=0, axis2=2)
        )
        tr_in = Phase2_ChoiIsomorphismCertifier._hermitize_weyl(
            np.trace(tensor, axis1=1, axis2=3)
        )
        eye_d = np.eye(d, dtype=np.complex128)
        delta_out = float(la.norm(tr_out - eye_d, ord="fro"))
        delta_in = float(la.norm(tr_in - eye_d, ord="fro"))
        return tr_out, tr_in, delta_out, delta_in

    # ── FASE 2.6 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_unitariety_degree(
        choi_matrix: ComplexMatrix,
        dimension_d: int,
    ) -> RealScalar:
        r"""
        FASE 2.6 — Grado de unitariedad (Wallman–Flammia / pureza de Choi).

        Para Choi *no normalizada* de un canal TP (\(\operatorname{Tr}\Lambda=d\)):

        \[
        U(\mathcal{E})
        =\frac{\operatorname{Tr}(\Lambda^2)-1}{d^2-1}
        \in[0,1]
        \quad(d>1).
        \]

        Justificación: el estado de Choi normalizado es \(\rho_\Lambda=\Lambda/d\),
        su pureza es \(\operatorname{Tr}(\rho_\Lambda^2)=\operatorname{Tr}(\Lambda^2)/d^2\),
        y la unitariedad es \((d^2 P-1)/(d^2-1)\). Equivale a

        * \(U=1\) sii \(\mathcal{E}\) es unitario (\(\operatorname{Tr}(\Lambda^2)=d^2\));
        * \(U=0\) para el canal completamente despolarizante
          (\(\Lambda=I_{d^2}/d\), \(\operatorname{Tr}(\Lambda^2)=1\)).

        Se evalúa como \(\|\Lambda\|_F^2\) (hermiticidad ⇒ \(\operatorname{Tr}\Lambda^2\)).
        """
        if dimension_d <= 1:
            return 1.0
        choi_sq_trace = Phase2_ChoiIsomorphismCertifier._choi_frobenius_sq(choi_matrix)
        denom = float(dimension_d * dimension_d - 1)
        unitariety = (choi_sq_trace - 1.0) / denom
        return float(min(1.0, max(0.0, unitariety)))

    # ── FASE 2.7 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_crosscheck_kraus_choi_duality(
        completeness_residual: RealScalar,
        unitality_residual: RealScalar,
        tp_diamond_defect: RealScalar,
        unital_diamond_defect: RealScalar,
        choi_trace: RealScalar,
        frobenius_mass: RealScalar,
        dimension_d: int,
        tolerance: float,
    ) -> Tuple[RealScalar, RealScalar, RealScalar]:
        r"""
        FASE 2.7 — Checksums cruzados Kraus ↔ Choi (fail-secure).

        Confronta tres identidades de aritmética exacta:

        1. \(r_{\mathrm{TP}}=\|\operatorname{Tr}_{\mathrm{out}}\Lambda-I\|_F\),
        2. \(r_U=\|\operatorname{Tr}_{\mathrm{in}}\Lambda-I\|_F\),
        3. \(\operatorname{Tr}\Lambda=\mu_F=\sum_k\|M_k\|_F^2\).

        Una discrepancia por encima del suelo de Wilkinson delata corrupción
        del isomorfismo \(\operatorname{vec}_F\) / ``reshape``.

        Raises:
            ChoiIsomorphismChecksumError: Checksum de isomorfismo corrupto.
        """
        tp_x = abs(float(completeness_residual) - float(tp_diamond_defect))
        unital_x = abs(float(unitality_residual) - float(unital_diamond_defect))
        mass_x = abs(float(choi_trace) - float(frobenius_mass))
        floor = Phase2_ChoiIsomorphismCertifier._wilkinson_spectral_floor(
            max(abs(choi_trace), 1.0),
            tolerance,
            dimension_d * dimension_d,
        )
        checksum_tol = max(_CHECKSUM_FMA_SLACK * floor, 1.0e-9)

        if tp_x > checksum_tol:
            raise ChoiIsomorphismChecksumError(
                "Checksum Choi↔Kraus TP corrupto: "
                f"|r_TP − ‖Tr_out Λ − I‖_F|={tp_x:.3e} > {checksum_tol:.3e}. "
                "El isomorfismo vec_F/reshape no es fiel."
            )
        if unital_x > checksum_tol:
            raise ChoiIsomorphismChecksumError(
                "Checksum Choi↔Kraus Unital corrupto: "
                f"|r_U − ‖Tr_in Λ − I‖_F|={unital_x:.3e} > {checksum_tol:.3e}."
            )
        if mass_x > checksum_tol:
            raise ChoiIsomorphismChecksumError(
                "Checksum Tr(Λ)↔μ_F corrupto: "
                f"|Tr(Λ) − μ_F|={mass_x:.3e} > {checksum_tol:.3e}."
            )
        logger.debug(
            "FASE2.7 checksums: ΔTP=%.3e, ΔU=%.3e, ΔTr=%.3e (tol=%.3e)",
            tp_x,
            unital_x,
            mass_x,
            checksum_tol,
        )
        return tp_x, unital_x, mass_x

    # ── FASE 2.Ω · composición terminal Orient ────────────────────────────
    @staticmethod
    def audit_trace_preservation(
        sanitized_operators: List[ComplexMatrix],
        dimension_d: int,
        tolerance: float = _DEFAULT_TOL,
        *,
        phase1: Optional[Phase1SanitizationData] = None,
        strict: bool = True,
        require_unital: bool = False,
    ) -> Phase2ChoiIsomorphismData:
        r"""
        FASE 2.Ω — Composición terminal Orient (TP + Choi + unital + U).

        **Continuación funtorial de FASE 1.Ω**: si se aporta ``phase1``,
        se ingiere mediante FASE 2.0 (continuidad estricta del certificado
        de ``sanitize_operators``); si no, se construye un DTO mínimo a
        partir de ``(ops, d)`` para permitir la invocación aislada.

        **Contrato funtorial F2 → F3**: el DTO
        ``Phase2ChoiIsomorphismData`` (con ``choi_matrix`` hermítica
        write-protected) es el objeto inicial exacto de
        ``_phase3_consume_phase2_certificate``.

        Raises:
            TracePreservationViolation: Si \(r_{\mathrm{TP}}>\varepsilon\) y ``strict``.
            UnitalityViolationError: Si se exige unitalidad y \(r_U>\varepsilon\).
            ChoiHermiticityError: Si Λ no es hermítica dentro de tol.
            ChoiIsomorphismChecksumError: Dualidad Kraus ↔ Choi corrupta.
        """
        tol = Phase2_ChoiIsomorphismCertifier._validate_tolerance(tolerance)

        if phase1 is None:
            phase1 = Phase1SanitizationData(
                dimension_d=int(dimension_d),
                num_operators=len(sanitized_operators),
                num_operators_effective=len(sanitized_operators),
                is_finite=True,
                phase_rotated=False,
                frobenius_mass=float(
                    sum(la.norm(m, ord="fro") ** 2 for m in sanitized_operators)
                ),
                wilkinson_tolerance=tol,
            )

        ops, d, phase1 = Phase2_ChoiIsomorphismCertifier._phase2_consume_phase1_certificate(
            sanitized_operators, phase1
        )
        if d != int(dimension_d):
            raise KrausDimensionError(
                f"FASE2.Ω: dimension_d={dimension_d} ≠ d_F1={d}."
            )

        r_tp, _s = Phase2_ChoiIsomorphismCertifier._phase2_trace_preserving_residual(ops, d)
        is_tp = r_tp <= tol
        if not is_tp and strict:
            raise TracePreservationViolation(
                f"Fuga termodinámica: el canal viola la completitud. "
                f"Residuo TP = {r_tp:.4e} > {tol:.4e}."
            )

        r_u, is_unital, _u = Phase2_ChoiIsomorphismCertifier._phase2_unital_residual(
            ops, d, tol
        )
        if require_unital and not is_unital:
            raise UnitalityViolationError(
                f"Política unital violada: r_U={r_u:.4e} > {tol:.4e}."
            )

        choi_raw = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(ops, d)
        choi, h_def = Phase2_ChoiIsomorphismCertifier._phase2_certify_choi_hermiticity(
            choi_raw, tol
        )
        _tr_out, _tr_in, delta_out, delta_in = (
            Phase2_ChoiIsomorphismCertifier._phase2_choi_partial_traces(choi, d)
        )
        choi_trace = float(np.real(np.trace(choi)))
        unitariety = Phase2_ChoiIsomorphismCertifier._phase2_unitariety_degree(choi, d)
        tp_x, unital_x, mass_x = (
            Phase2_ChoiIsomorphismCertifier._phase2_crosscheck_kraus_choi_duality(
                completeness_residual=r_tp,
                unitality_residual=r_u,
                tp_diamond_defect=delta_out,
                unital_diamond_defect=delta_in,
                choi_trace=choi_trace,
                frobenius_mass=phase1.frobenius_mass,
                dimension_d=d,
                tolerance=tol,
            )
        )

        logger.debug(
            "FASE2.Ω Choi: r_TP=%.3e, r_U=%.3e, TrΛ=%.6f, δ_out=%.3e, δ_in=%.3e, U=%.6f",
            r_tp,
            r_u,
            choi_trace,
            delta_out,
            delta_in,
            unitariety,
        )

        return Phase2ChoiIsomorphismData(
            trace_preserving_residual=r_tp,
            is_trace_preserving=is_tp,
            choi_matrix=Phase2_ChoiIsomorphismCertifier._freeze_matrix(choi),
            is_unital=is_unital,
            unital_residual=r_u,
            unitariety_degree=unitariety,
            choi_trace=choi_trace,
            tp_diamond_defect=delta_out,
            unital_diamond_defect=delta_in,
            hermiticity_defect=h_def,
            tp_crosscheck_defect=tp_x,
            unital_crosscheck_defect=unital_x,
            mass_crosscheck_defect=mass_x,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — CP, PPT, NON-SIGNALING Y SEGURIDAD CUÁNTICA (Decide + Act)
# Continuación directa de audit_trace_preservation (FASE 2.Ω) vía FASE 3.0
# Objetos: σ(Λ), Λ^{T_A}, ρ_B vs ρ'_B, testigo algebraico, retícula de gobernanza
# Teorías: Choi CP, Peres–Horodecki, no-señalización, causalidad bipartita
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_QuantumSecurityEnforcer(Phase2_ChoiIsomorphismCertifier):
    r"""
    FASE 3: audita positividad completa, separabilidad PPT y no-señalización.

    Morfismo compuesto:

    \[
    \mathrm{Secure}
    =(\mathrm{Rank},\,\mathrm{Purity},\,\mathrm{NS}_{\mathrm{alg}},
    \,\mathrm{NS}_{\mathrm{Bell}},\,\mathrm{PPT},\,\mathrm{CP})
    \circ\mathrm{Consume}\circ\mathrm{OrientChoi}^*.
    \]

    El primer morfismo, ``_phase3_consume_phase2_certificate``, *es* la
    continuación estricta de ``audit_trace_preservation``.
    """

    # ── FASE 3.0 · ingesta funtorial del certificado de FASE 2.Ω ──────────
    @staticmethod
    def _phase3_consume_phase2_certificate(
        choi_data: Phase2ChoiIsomorphismData,
        sanitized_ops: List[ComplexMatrix],
        dimension_d: int,
    ) -> Tuple[Phase2ChoiIsomorphismData, List[ComplexMatrix], int]:
        r"""
        FASE 3.0 — Ingesta funtorial del certificado de FASE 2.Ω.

        **Continuación estricta de**
        ``Phase2_ChoiIsomorphismCertifier.audit_trace_preservation``.
        Verifica la coherencia geométrica de \(\Lambda_{\mathcal{E}}\)
        (shape \(d^2\times d^2\), hermiticidad ya sellada) y entrega el
        objeto de trabajo de Decide *sin reconstruir Choi*.

        Raises:
            KrausDimensionError: Shape de Choi incoherente con \(d\).
        """
        d = int(dimension_d)
        expected = (d * d, d * d)
        shape = tuple(choi_data.choi_matrix.shape)
        if shape != expected:
            raise KrausDimensionError(
                f"FASE3.0: Choi shape={shape} ≠ {expected} (d={d} de FASE 2)."
            )
        if sanitized_ops is None or len(sanitized_ops) == 0:
            raise KrausDimensionError(
                "FASE3.0: el ensamble saneado de FASE 1 no puede ser vacío."
            )
        logger.debug(
            "FASE3.0 consume F2: d=%d, r_TP=%.3e, TrΛ=%.6f, U=%.6f, TP=%s",
            d,
            choi_data.trace_preserving_residual,
            choi_data.choi_trace,
            choi_data.unitariety_degree,
            choi_data.is_trace_preserving,
        )
        return choi_data, sanitized_ops, d

    # ── FASE 3.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_spectral_complete_positivity(
        choi_data: Phase2ChoiIsomorphismData,
        dimension_d: int,
        tolerance: float,
    ) -> Tuple[RealVector, RealScalar, RealScalar, int, RealScalar, RealScalar, BoolLattice]:
        r"""
        FASE 3.1 — Positividad completa vía espectro de Choi (inicio Decide).

        Recibe el DTO de FASE 2.Ω ya ingerido por FASE 3.0.

        Teorema de Choi (1975):

        \[
        \mathcal{E}\text{ es CP}
        \Longleftrightarrow
        \Lambda_{\mathcal{E}}\succeq 0
        \Longleftrightarrow
        \lambda_{\min}(\Lambda)\ge 0.
        \]

        La decisión usa \(-\varepsilon\) del caller (acotada por el suelo
        PSD de Wilkinson). El rango usa \(\varepsilon_W\) relativo a
        \(\lambda_{\max}\).

        Returns:
            ``(eigvals, λ_min, λ_max, choi_rank, gap, cond, is_cp)``.
        """
        eigvals = np.asarray(la.eigvalsh(choi_data.choi_matrix), dtype=np.float64)
        min_eigen = float(eigvals[0])
        max_eigen = float(eigvals[-1])
        decision_floor = -max(float(tolerance), abs(_SPECTRAL_PSD_FLOOR))
        is_cp: BoolLattice = min_eigen >= decision_floor

        scale = max(abs(max_eigen), 1.0)
        floor = Phase3_QuantumSecurityEnforcer._wilkinson_spectral_floor(
            scale, tolerance, dimension_d * dimension_d
        )
        positive = eigvals[eigvals > floor]
        choi_rank = int(positive.size)
        if choi_rank:
            gap = float(positive[0])
            cond = float(positive[-1] / positive[0]) if positive[0] > 0.0 else float("inf")
        else:
            gap = 0.0
            cond = float("inf")

        logger.debug(
            "FASE3.1 CP: λ_min=%.6e, λ_max=%.6e, rank=%d, gap=%.6e, κ=%.4e, is_cp=%s",
            min_eigen,
            max_eigen,
            choi_rank,
            gap,
            cond,
            is_cp,
        )
        return eigvals, min_eigen, max_eigen, choi_rank, gap, cond, is_cp

    # ── FASE 3.2 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_ppt_separability(
        choi_data: Phase2ChoiIsomorphismData,
        dimension_d: int,
        tolerance: float,
    ) -> Tuple[RealScalar, BoolLattice]:
        r"""
        FASE 3.2 — Criterio PPT de Peres–Horodecki sobre \(\Lambda_{\mathcal{E}}\).

        Transpuesta parcial sobre el factor de *salida*:

        \[
        \bigl(\Lambda^{T_{\mathrm{out}}}\bigr)_{i j;\,a b}
        =\Lambda_{a j;\,i b},
        \qquad
        \text{PPT}\iff\Lambda^{T_{\mathrm{out}}}\succeq 0.
        \]

        Layout ``order='F'`` coherente con \(\operatorname{vec}_F\). PPT es
        invariante blando por defecto (necesario y suficiente solo en
        \(2\times 2\) y \(2\times 3\); en dimensión superior existen
        estados bound-entangled). Proxy de canales entanglement-breaking.
        """
        d = dimension_d
        choi_dim = d * d
        block = Phase3_QuantumSecurityEnforcer._choi_tensor4(choi_data.choi_matrix, d)
        ppt_block = block.transpose(2, 1, 0, 3)
        ppt_matrix = ppt_block.reshape((choi_dim, choi_dim), order="F")
        ppt_matrix = Phase3_QuantumSecurityEnforcer._hermitize_weyl(ppt_matrix)

        ppt_eigs = la.eigvalsh(ppt_matrix)
        ppt_min = float(np.min(ppt_eigs))
        decision_floor = -max(float(tolerance), abs(_SPECTRAL_PSD_FLOOR))
        is_separable = ppt_min >= decision_floor

        logger.debug(
            "FASE3.2 PPT: λ_min(Λ^{T_A})=%.6e, separable=%s, d=%d",
            ppt_min,
            is_separable,
            d,
        )
        return ppt_min, is_separable

    # ── FASE 3.3 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_bell_resource_state(dimension_d: int) -> ComplexMatrix:
        r"""
        FASE 3.3 — Estado recurso de Bell (normalizado, layout Kronecker C):

        \[
        |\Omega\rangle
        =d^{-1/2}\sum_{i=0}^{d-1}|i\rangle_A\otimes|i\rangle_B,
        \qquad
        \rho_{AB}=|\Omega\rangle\langle\Omega|.
        \]

        Índice plano \(a\cdot d+b\) coherente con ``np.kron``.
        """
        d = dimension_d
        omega = np.zeros((d * d, 1), dtype=np.complex128)
        scale = 1.0 / math.sqrt(d)
        for i in range(d):
            omega[i * d + i, 0] = scale
        return (omega @ omega.conj().T).astype(np.complex128)

    # ── FASE 3.4 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_partial_trace_A(
        rho_ab: ComplexMatrix,
        dimension_d: int,
    ) -> ComplexMatrix:
        r"""
        FASE 3.4 — Traza parcial sobre Alice (layout Kronecker A⊗B).

        \[
        (\operatorname{Tr}_A\rho)_{b b'}
        =\sum_{a}\rho_{(a b),(a b')}.
        \]
        """
        d = dimension_d
        rho_b = np.zeros((d, d), dtype=np.complex128)
        for a in range(d):
            rho_b += rho_ab[a * d : (a + 1) * d, a * d : (a + 1) * d]
        return Phase3_QuantumSecurityEnforcer._hermitize_weyl(rho_b)

    # ── FASE 3.5 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_apply_local_channel_on_A(
        rho_ab: ComplexMatrix,
        sanitized_ops: List[ComplexMatrix],
        dimension_d: int,
    ) -> ComplexMatrix:
        r"""
        FASE 3.5 — Acción local del canal sobre Alice.

        \[
        \rho'_{AB}
        =\sum_k
        (M_k\otimes I_B)\,\rho_{AB}\,(M_k^\dagger\otimes I_B).
        \]
        """
        d = dimension_d
        identity_b = np.eye(d, dtype=np.complex128)
        rho_prime = np.zeros_like(rho_ab, dtype=np.complex128)
        for m_k in sanitized_ops:
            op = np.kron(m_k, identity_b)
            rho_prime += op @ rho_ab @ op.conj().T
        return Phase3_QuantumSecurityEnforcer._hermitize_weyl(rho_prime)

    # ── FASE 3.6 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_non_signaling_residual(
        sanitized_ops: List[ComplexMatrix],
        dimension_d: int,
        tolerance: float,
    ) -> Tuple[RealScalar, BoolLattice]:
        r"""
        FASE 3.6 — Condición de No-Señalización bipartita (recurso de Bell).

        \[
        \rho'_B
        =\operatorname{Tr}_A\!\big((\mathcal{E}_A\otimes\mathcal{I}_B)(\rho_{AB})\big),
        \qquad
        r_{\mathrm{NS}}=\|\rho'_B-\rho_B\|_F.
        \]

        **Teorema (checksum causal)**: si \(\mathcal{E}\) es TP entonces
        \(r_{\mathrm{NS}}=0\) idénticamente para todo \(\rho_{AB}\). El
        residuo es por tanto un invariante numérico dual de TP (análogo
        al \(\delta_{\mathrm{out}}\) de FASE 2.5, pero en el cuadro de
        Schrödinger bipartito). Hard-gate: \(r_{\mathrm{NS}}>\varepsilon\)
        ⇒ corrupción numérica o bug de ensamble.
        """
        rho_shared = Phase3_QuantumSecurityEnforcer._phase3_bell_resource_state(
            dimension_d
        )
        rho_b = Phase3_QuantumSecurityEnforcer._phase3_partial_trace_A(
            rho_shared, dimension_d
        )
        rho_prime = Phase3_QuantumSecurityEnforcer._phase3_apply_local_channel_on_A(
            rho_shared, sanitized_ops, dimension_d
        )
        rho_b_prime = Phase3_QuantumSecurityEnforcer._phase3_partial_trace_A(
            rho_prime, dimension_d
        )

        residual = float(la.norm(rho_b_prime - rho_b, ord="fro"))
        logger.debug("FASE3.6 NS-Bell: r_NS=%.6e", residual)
        return residual, residual <= tolerance

    # ── FASE 3.7 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_algebraic_nosignal_witness(
        sanitized_ops: List[ComplexMatrix],
        dimension_d: int,
    ) -> RealScalar:
        r"""
        FASE 3.7 — Testigo algebraico de No-Señalización (ruta distinta).

        \[
        T_{ij}
        :=\operatorname{Tr}\bigl(\mathcal{E}(|i\rangle\langle j|)\bigr)
        =\sum_{k,a}(M_k)_{a i}\,\overline{(M_k)_{a j}},
        \qquad
        \delta_{\mathrm{NS}}^{\mathrm{alg}}=\|T-I\|_F.
        \]

        La contracción \(\sum_k M_k^\top\overline{M_k}\) es independiente
        de \(\sum_k M_k^\dagger M_k\) (FASE 2.1), de
        \(\operatorname{Tr}_{\mathrm{out}}\Lambda\) (FASE 2.5) y del
        recurso de Bell (FASE 3.6). Triple checksum causal.
        """
        d = dimension_d
        signaling = np.zeros((d, d), dtype=np.complex128)
        for m_k in sanitized_ops:
            signaling += m_k.T @ np.conjugate(m_k)
        signaling = Phase3_QuantumSecurityEnforcer._hermitize_weyl(signaling)
        defect = float(
            la.norm(signaling - np.eye(d, dtype=np.complex128), ord="fro")
        )
        logger.debug("FASE3.7 NS-alg: δ_NS^alg=%.6e", defect)
        return defect

    # ── FASE 3.8 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_choi_purity(
        choi_data: Phase2ChoiIsomorphismData,
    ) -> RealScalar:
        r"""
        FASE 3.8 — Pureza normalizada del estado de Choi.

        \[
        P(\Lambda)
        =\frac{\operatorname{Tr}(\Lambda^2)}{(\operatorname{Tr}\Lambda)^2}
        =\frac{\|\Lambda\|_F^2}{(\operatorname{Tr}\Lambda)^2}
        \in(0,1].
        \]
        """
        tr = choi_data.choi_trace
        if abs(tr) <= _EPS_SPECTRAL:
            tr = 1.0
        tr_sq = Phase3_QuantumSecurityEnforcer._choi_frobenius_sq(choi_data.choi_matrix)
        return float(tr_sq / (tr * tr))

    # ── FASE 3.9 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_rank_consistency(
        choi_rank: int,
        kraus_gram_rank: int,
    ) -> BoolLattice:
        r"""
        FASE 3.9 — Consistencia de rango Choi ↔ Hilbert–Schmidt.

        En aritmética exacta
        \(\operatorname{rank}(\Lambda_{\mathcal{E}})=\operatorname{rank}(V)\).
        Un desacuerdo (típicamente off-by-one cerca del suelo espectral)
        no falsea CPTP pero se registra como telemetría de patología
        numérica.
        """
        consistent = int(choi_rank) == int(kraus_gram_rank)
        if not consistent:
            logger.warning(
                "FASE3.9 rango: rank(Λ)=%d ≠ rank_HS(V)=%d. "
                "Posible umbral de Wilkinson en el borde espectral.",
                choi_rank,
                kraus_gram_rank,
            )
        else:
            logger.debug(
                "FASE3.9 rango: rank(Λ)=rank_HS(V)=%d (consistente).",
                choi_rank,
            )
        return consistent

    # ── FASE 3.Ω · composición terminal Decide ────────────────────────────
    @staticmethod
    def audit_quantum_security(
        choi_data: Phase2ChoiIsomorphismData,
        sanitized_operators: List[ComplexMatrix],
        dimension_d: int,
        tolerance: float = _DEFAULT_TOL,
        *,
        phase1: Optional[Phase1SanitizationData] = None,
        strict: bool = True,
        require_ppt: bool = False,
    ) -> Phase3QuantumSecurityData:
        r"""
        FASE 3.Ω — Composición terminal Decide (CP + PPT + NS + pureza).

        **Continuación funtorial de FASE 2.Ω**: consume
        ``Phase2ChoiIsomorphismData`` vía FASE 3.0 (sin reconstruir Λ)
        y el ensamble saneado de FASE 1.

        **Contrato funtorial F3 → Seal**: el DTO
        ``Phase3QuantumSecurityData`` alimenta el sellado de gobernanza
        en ``CPTPChannelValidatorAgent.audit_cptp_channel``.

        Raises:
            CompletePositivityViolation: \(\lambda_{\min}(\Lambda)<\) floor y ``strict``.
            NonSignalingViolationError: \(r_{\mathrm{NS}}>\varepsilon\) y ``strict``.
            PeresHorodeckiSeparabilityError: Si se exige PPT y falla.
        """
        tol = Phase3_QuantumSecurityEnforcer._validate_tolerance(tolerance)
        choi_data, ops, d = (
            Phase3_QuantumSecurityEnforcer._phase3_consume_phase2_certificate(
                choi_data, sanitized_operators, dimension_d
            )
        )

        _eigs, min_eigen, max_eigen, kraus_rank, gap, cond, is_cp = (
            Phase3_QuantumSecurityEnforcer._phase3_spectral_complete_positivity(
                choi_data, d, tol
            )
        )
        if not is_cp and strict:
            raise CompletePositivityViolation(
                f"La matriz de Choi viola la semidefinitud positiva: "
                f"autovalor mínimo = {min_eigen:.4e} < floor={-max(tol, abs(_SPECTRAL_PSD_FLOOR)):.4e}."
            )

        ppt_min, is_separable_ppt = (
            Phase3_QuantumSecurityEnforcer._phase3_ppt_separability(choi_data, d, tol)
        )
        if require_ppt and not is_separable_ppt:
            raise PeresHorodeckiSeparabilityError(
                f"Política PPT violada: λ_min(Λ^{{T_A}})={ppt_min:.4e}."
            )

        ns_residual, is_non_signaling = (
            Phase3_QuantumSecurityEnforcer._phase3_non_signaling_residual(ops, d, tol)
        )
        if not is_non_signaling and strict:
            raise NonSignalingViolationError(
                f"Ruptura causal: la acción local del LLM señaliza al espacio de Bob. "
                f"Residuo de No-Señalización = {ns_residual:.4e} > {tol:.4e}."
            )

        alg_ns = Phase3_QuantumSecurityEnforcer._phase3_algebraic_nosignal_witness(ops, d)
        purity = Phase3_QuantumSecurityEnforcer._phase3_choi_purity(choi_data)
        gram_rank = int(phase1.kraus_gram_rank) if phase1 is not None else 0
        rank_ok = Phase3_QuantumSecurityEnforcer._phase3_rank_consistency(
            kraus_rank, gram_rank
        ) if phase1 is not None else True

        logger.debug(
            "FASE3.Ω security: CP=%s, rank=%d, r_NS=%.3e, δ_alg=%.3e, PPT=%s, P=%.6f",
            is_cp,
            kraus_rank,
            ns_residual,
            alg_ns,
            is_separable_ppt,
            purity,
        )

        return Phase3QuantumSecurityData(
            minimum_choi_eigenvalue=min_eigen,
            kraus_rank=kraus_rank,
            is_completely_positive=is_cp,
            non_signaling_residual=ns_residual,
            is_non_signaling=is_non_signaling,
            is_separable_ppt=is_separable_ppt,
            ppt_min_eigenvalue=ppt_min,
            choi_purity=purity,
            choi_max_eigenvalue=max_eigen,
            choi_spectral_gap=gap,
            choi_condition_number=cond,
            algebraic_nosignal_defect=alg_ns,
            kraus_gram_rank=gram_rank,
            rank_consistent=rank_ok,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR SUPREMO — CPTP CHANNEL VALIDATOR AGENT (Custodio de Choi)
# Observe (F1) ⟶ Orient (F2) ⟶ Decide (F3) ⟶ Seal / Veto
# ═══════════════════════════════════════════════════════════════════════════════
class CPTPChannelValidatorAgent(Morphism, Phase3_QuantumSecurityEnforcer):
    r"""
    Agente soberano de validación cuántica y análisis causal en WISDOM.

    Endofuntor de gobernanza:

    \[
    \mathcal{OODA}_{\mathrm{CPTP}}
    :
    \mathbf{Kraus}(\mathcal{H}_d)
    \longrightarrow
    \mathbf{GovState}(\Omega_3)
    \]

    compuesto como

    \[
    \mathrm{Seal}\circ\mathrm{Secure}\circ\mathrm{Choi}\circ\mathrm{Sanitize}.
    \]

    Cada flecha es monádica sobre la jerarquía ``CPTPValidatorAgentError``;
    el veto se re-propaga para gatillar Crowbar / colapso del retículo.
    """

    def __init__(self, target_stratum: Stratum = Stratum.WISDOM) -> None:
        """Inicializa al centinela cuántico en el estrato supremo."""
        super().__init__()
        self._target_stratum: Stratum = target_stratum

    # ── FASE Ω.1 · conjunción de hard-gates ───────────────────────────────
    @staticmethod
    def _seal_security_conjunction(
        sanitization: Phase1SanitizationData,
        choi_audit: Phase2ChoiIsomorphismData,
        security: Phase3QuantumSecurityData,
        *,
        require_unital: bool = False,
        require_ppt: bool = False,
    ) -> BoolLattice:
        r"""
        FASE Ω.1 — Clasificador de subobjetos \(\chi_{\mathrm{secure}}\in\Omega\).

        \[
        \chi
        =\chi_{\mathrm{finite}}
        \wedge\chi_{\mathrm{TP}}
        \wedge\chi_{\mathrm{CP}}
        \wedge\chi_{\mathrm{NS}}
        \;\bigl(\wedge\chi_U\bigr)_{\mathrm{opt}}
        \;\bigl(\wedge\chi_{\mathrm{PPT}}\bigr)_{\mathrm{opt}}.
        \]

        PPT y unitalidad son invariantes blandos salvo política explícita.
        """
        core = bool(
            sanitization.is_finite
            and choi_audit.is_trace_preserving
            and security.is_completely_positive
            and security.is_non_signaling
        )
        if require_unital:
            core = core and bool(choi_audit.is_unital)
        if require_ppt:
            core = core and bool(security.is_separable_ppt)
        return core

    # ── FASE Ω.2 · sellado del certificado de gobernanza ──────────────────
    def _seal_governance_state(
        self,
        sanitization: Phase1SanitizationData,
        choi_audit: Phase2ChoiIsomorphismData,
        security: Phase3QuantumSecurityData,
        is_secure: BoolLattice,
        tolerance: float,
        *,
        require_unital: bool = False,
        require_ppt: bool = False,
    ) -> CPTPChannelGovernanceState:
        r"""
        FASE Ω.2 — Sellado del objeto terminal frozen del funtor OODA.

        Empaqueta los tres DTOs de fase + χ_secure + timestamp UTC + política.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        stratum_name = (
            self._target_stratum.name
            if hasattr(self._target_stratum, "name")
            else str(self._target_stratum)
        )
        return CPTPChannelGovernanceState(
            sanitization_audit=sanitization,
            choi_audit=choi_audit,
            security_audit=security,
            is_channel_secure=is_secure,
            timestamp_utc=timestamp_utc,
            wilkinson_tolerance=tolerance,
            stratum=stratum_name,
            agent_version=__version__,
            policy_require_unital=require_unital,
            policy_require_ppt=require_ppt,
        )

    # ── FASE Ω.Veto · telemetría de colapso ───────────────────────────────
    @staticmethod
    def _veto_log_and_reraise(err: CPTPValidatorAgentError) -> None:
        """FASE Ω.Veto — Log CRITICAL y re-propagación hacia Crowbar."""
        logger.critical(
            "¡VETO CUÁNTICO! El canal propuesto viola los invariantes "
            "de la aduana de Choi: %s",
            str(err),
        )
        raise err

    # ── Compositor público OODA ───────────────────────────────────────────
    def audit_cptp_channel(
        self,
        kraus_operators: KrausEnsemble,
        dimension_d: int,
        tolerance: float = _DEFAULT_TOL,
        *,
        strict: bool = True,
        require_unital: bool = False,
        require_ppt: bool = False,
    ) -> CPTPChannelGovernanceState:
        r"""
        Ejecuta el ciclo OODA completo de validación espectral del canal.

        El objeto terminal de cada fase es el objeto inicial de la siguiente:

        .. code-block:: text

            ┌────────────────────────────────────────────────────────────┐
            │ FASE 1  Observe / Sanitize                                 │
            │   1.1 validate_hilbert_dimension                           │
            │   1.2 validate_kraus_typing                                │
            │   1.3 audit_kraus_operator_norms                           │
            │   1.4 filter_uv_null_operators                             │
            │   1.5 standard_gauge_phase                                 │
            │   1.6 kraus_hs_gram_rank                                   │
            │   1.Ω sanitize_operators  ──► (ops, Phase1Data) ──┐        │
            ├───────────────────────────────────────────────────┼────────┤
            │ FASE 2  Orient / Choi  ◄──────────────────────────┘        │
            │   2.0 consume_phase1_certificate                           │
            │   2.1 trace_preserving_residual                            │
            │   2.2 unital_residual                                      │
            │   2.3 construct_choi_matrix                                │
            │   2.4 certify_choi_hermiticity                             │
            │   2.5 choi_partial_traces  (out / in)                      │
            │   2.6 unitariety_degree                                    │
            │   2.7 crosscheck_kraus_choi_duality                        │
            │   2.Ω audit_trace_preservation  ──► Phase2Data ──┐         │
            ├──────────────────────────────────────────────────┼─────────┤
            │ FASE 3  Decide / Secure  ◄───────────────────────┘         │
            │   3.0 consume_phase2_certificate                           │
            │   3.1 spectral_complete_positivity                         │
            │   3.2 ppt_separability                                     │
            │   3.3 bell_resource_state                                  │
            │   3.4 partial_trace_A                                      │
            │   3.5 apply_local_channel_on_A                             │
            │   3.6 non_signaling_residual                               │
            │   3.7 algebraic_nosignal_witness                           │
            │   3.8 choi_purity                                          │
            │   3.9 rank_consistency                                     │
            │   3.Ω audit_quantum_security  ──► Phase3Data ──┐           │
            ├────────────────────────────────────────────────┼───────────┤
            │ SEAL / VETO  ◄─────────────────────────────────┘           │
            │   Ω.1 security_conjunction                                 │
            │   Ω.2 seal_governance_state                                │
            │   Ω.Veto log_and_reraise                                   │
            └────────────────────────────────────────────────────────────┘

        Args:
            kraus_operators: Colección de matrices de Kraus \(\{M_k\}\).
            dimension_d: Dimensión de \(\mathcal{H}_A\).
            tolerance: ε de Wilkinson para tolerancia espectral de decisión.
            strict: Si es verdadero (default), TP/CP/NS fallidos elevan veto.
                Si es falso, se sella el reporte con ``is_channel_secure=False``
                (soft-audit). Los errores de *entrada* (dimensión, NaN,
                checksum de isomorfismo, hermiticidad) siempre abortan.
            require_unital: Eleva ``UnitalityViolationError`` si \(r_U>\varepsilon\).
            require_ppt: Eleva ``PeresHorodeckiSeparabilityError`` si PPT falla.

        Returns:
            CPTPChannelGovernanceState: certificado inmutable de gobernanza.

        Raises:
            CPTPValidatorAgentError: Violación de traza, CP, NS, dimensión,
                hermiticidad, checksum o política (re-propagada tras log CRITICAL
                si ``strict``).
        """
        tol = self._validate_tolerance(tolerance)

        try:
            # ── FASE 1 · Observe / Sanitize ───────────────────────────────
            sanitized_ops, sanitization_audit = self.sanitize_operators(
                kraus_operators,
                dimension_d,
                tolerance=tol,
            )

            # ── FASE 2 · Orient / Choi  (continúa certificado F1.Ω) ───────
            choi_audit = self.audit_trace_preservation(
                sanitized_ops,
                sanitization_audit.dimension_d,
                tol,
                phase1=sanitization_audit,
                strict=strict,
                require_unital=require_unital,
            )

            # ── FASE 3 · Decide / Secure  (continúa certificado F2.Ω) ─────
            security_audit = self.audit_quantum_security(
                choi_audit,
                sanitized_ops,
                sanitization_audit.dimension_d,
                tol,
                phase1=sanitization_audit,
                strict=strict,
                require_ppt=require_ppt,
            )

            # ── SEAL ──────────────────────────────────────────────────────
            is_secure = self._seal_security_conjunction(
                sanitization_audit,
                choi_audit,
                security_audit,
                require_unital=require_unital,
                require_ppt=require_ppt,
            )
            state = self._seal_governance_state(
                sanitization_audit,
                choi_audit,
                security_audit,
                is_secure,
                tol,
                require_unital=require_unital,
                require_ppt=require_ppt,
            )

            logger.info(
                "Canal cuántico de inyección semántica validado. "
                "Rango de Choi=%d | r_NS=%.4e | δ_alg=%.4e | PPT=%s | "
                "U=%.4f | r_TP=%.3e | secure=%s",
                security_audit.kraus_rank,
                security_audit.non_signaling_residual,
                security_audit.algebraic_nosignal_defect,
                str(security_audit.is_separable_ppt),
                choi_audit.unitariety_degree,
                choi_audit.trace_preserving_residual,
                str(is_secure),
            )
            return state

        except CPTPValidatorAgentError as err:
            self._veto_log_and_reraise(err)
            raise  # unreachable; satisface type-checkers

    # ─────────────────────────────────────────────────────────────────────
    # Fábricas de referencia (calibración / tests del agente)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def identity_kraus(dimension_d: int) -> List[ComplexMatrix]:
        r"""Canal identidad: \(\mathcal{E}(\rho)=\rho\), Kraus \(\{I_d\}\)."""
        if dimension_d < 1:
            raise ValueError(f"dimension_d debe ser ≥ 1; recibido {dimension_d}")
        return [np.eye(dimension_d, dtype=np.complex128)]

    @staticmethod
    def amplitude_damping_kraus(gamma: float) -> List[ComplexMatrix]:
        r"""
        Amortiguamiento de amplitud (qubit, \(d=2\)):

        \[
        M_0=\begin{pmatrix}1&0\\0&\sqrt{1-\gamma}\end{pmatrix},
        \quad
        M_1=\begin{pmatrix}0&\sqrt{\gamma}\\0&0\end{pmatrix},
        \qquad\gamma\in[0,1].
        \]

        Es CPTP y **no** unital para \(\gamma\in(0,1]\).
        """
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma ∈ [0,1]; recibido {gamma}")
        m0 = np.array(
            [[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]], dtype=np.complex128
        )
        m1 = np.array(
            [[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128
        )
        return [m0, m1]

    @staticmethod
    def phase_damping_kraus(lam: float) -> List[ComplexMatrix]:
        r"""
        Amortiguamiento de fase (qubit, \(d=2\)):

        \[
        M_0=\begin{pmatrix}1&0\\0&\sqrt{1-\lambda}\end{pmatrix},
        \quad
        M_1=\begin{pmatrix}0&0\\0&\sqrt{\lambda}\end{pmatrix},
        \qquad\lambda\in[0,1].
        \]

        Es unital y CPTP (ruido puramente de-fasaje).
        """
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam ∈ [0,1]; recibido {lam}")
        m0 = np.array(
            [[1.0, 0.0], [0.0, math.sqrt(1.0 - lam)]], dtype=np.complex128
        )
        m1 = np.array(
            [[0.0, 0.0], [0.0, math.sqrt(lam)]], dtype=np.complex128
        )
        return [m0, m1]

    @staticmethod
    def bit_flip_kraus(p: float) -> List[ComplexMatrix]:
        r"""Canal bit-flip (qubit): \(\mathcal{E}(\rho)=(1-p)\rho+p\,X\rho X\)."""
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p ∈ [0,1]; recibido {p}")
        eye = np.eye(2, dtype=np.complex128)
        pauli_x = np.array(
            [[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128
        )
        return [math.sqrt(1.0 - p) * eye, math.sqrt(p) * pauli_x]

    @staticmethod
    def depolarizing_kraus(dimension_d: int, p: float) -> List[ComplexMatrix]:
        r"""
        Canal despolarizante de Weyl–Heisenberg en dimensión \(d\):

        \[
        \mathcal{E}_p(\rho)
        =(1-p)\rho+p\,\frac{I}{d}\operatorname{Tr}(\rho),
        \qquad p\in[0,1].
        \]

        Kraus vía la base unitaria \(W_{jk}=X^j Z^k\) con pesos

        \[
        q_{00}=1-p+\frac{p}{d^2},
        \qquad
        q_{jk}=\frac{p}{d^2}\quad(j,k)\ne(0,0),
        \]

        de modo que \(\sum q_{jk}=1\) y \(\sum_k M_k^\dagger M_k=I\).
        Para \(d=2\) reduce al canal de Pauli estándar
        \(\sqrt{1-3p/4}\,I,\;\sqrt{p/4}\,X,Y,Z\).
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p ∈ [0,1]; recibido {p}")
        if dimension_d < 1:
            raise ValueError(f"dimension_d debe ser ≥ 1; recibido {dimension_d}")
        d = int(dimension_d)
        kraus: List[ComplexMatrix] = []
        q0 = 1.0 - p + p / float(d * d)
        qi = p / float(d * d)
        kraus.append((math.sqrt(max(q0, 0.0)) * np.eye(d)).astype(np.complex128))

        if qi <= 0.0:
            return kraus

        omega = np.exp(2.0j * np.pi / d)
        shift = np.zeros((d, d), dtype=np.complex128)
        clock = np.zeros((d, d), dtype=np.complex128)
        for i in range(d):
            shift[i, (i - 1) % d] = 1.0 + 0.0j
            clock[i, i] = omega ** i

        noise_scale = math.sqrt(qi)
        for i in range(d):
            for j in range(d):
                if i == 0 and j == 0:
                    continue
                w_ij = np.linalg.matrix_power(shift, i) @ np.linalg.matrix_power(clock, j)
                kraus.append((noise_scale * w_ij).astype(np.complex128, copy=False))
        return kraus

    @staticmethod
    def replacement_kraus() -> List[ComplexMatrix]:
        r"""Canal de reemplazo a \(|0\rangle\langle 0|\) (entanglement-breaking, \(d=2\))."""
        m0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        m1 = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
        return [m0, m1]


# ═══════════════════════════════════════════════════════════════════════════════
# Exportación canónica del módulo
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Excepciones
    "CPTPValidatorAgentError",
    "KrausDimensionError",
    "TracePreservationViolation",
    "CompletePositivityViolation",
    "NonSignalingViolationError",
    "PeresHorodeckiSeparabilityError",
    "UnitalityViolationError",
    "ChoiHermiticityError",
    "ChoiIsomorphismChecksumError",
    # DTOs
    "Phase1SanitizationData",
    "Phase2ChoiIsomorphismData",
    "Phase3QuantumSecurityData",
    "CPTPChannelGovernanceState",
    # Fases
    "Phase1_KrausSanitizer",
    "Phase2_ChoiIsomorphismCertifier",
    "Phase3_QuantumSecurityEnforcer",
    # Agente
    "CPTPChannelValidatorAgent",
    # Constantes útiles para tests
    "_DEFAULT_TOL",
    "_SPECTRAL_PSD_FLOOR",
    "_MACHINE_EPS",
    "_DIM_MAX",
    "__version__",
]
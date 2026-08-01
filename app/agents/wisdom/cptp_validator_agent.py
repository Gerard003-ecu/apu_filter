### -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Módulo : CPTP Channel Validator Agent (El Soberano de la Aduana de Choi)    ║
║  Ruta   : app/agents/wisdom/cptp_validator_agent.py                          ║
║  Versión: 2.0.0-Choi-Jamiolkowski-Kraus-NonSignaling-OODA-Nested-Strict      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  NATURALEZA CIBER-FÍSICA Y RIGOR DOCTORAL:                                   ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Este módulo consagra el agente soberano que gobierna y orquesta el          ║
║  cptp_validator.py en el Estrato WISDOM. Su propósito es asegurar            ║
║  que todo canal de inyección semántica $\mathcal{E}$ cumpla con los          ║
║  postulados cuánticos de sistemas abiertos y de conservación de traza,       ║
║  así como la condición de no-señalización local (Non-Signaling).             ║
║                                                                              ║
║  Arquitectura de Fases Anidadas (OODA espectral sobre Ω₃):                   ║
║    FASE 1 — Saneamiento Kraus, gauge de fase y certificado dimensional       ║
║    FASE 2 — TP / unitalidad / Choi–Jamiołkowski / unitariedad  (cont. F1)    ║
║    FASE 3 — CP, PPT, Non-Signaling y gobernanza sellada        (cont. F2)    ║
║                                                                              ║
║  Invariantes Matemáticos Preservados:                                        ║
║  $$\Lambda_{\mathcal{E}}\succeq 0
║    \;\land\;
║    \sum_k M_k^\dagger M_k = I
║    \;\land\;
║    \operatorname{Tr}_A\!\big((\mathcal{E}_A\otimes\mathcal{I}_B)(\rho_{AB})\big)
║    =\rho_B$$                                                                 ║
║                                                                              ║
║  Teorema (checksum causal): todo canal TP implica Non-Signaling sobre el     ║
║  estado reducido de Bob; el residuo NS es un invariante numérico dual de TP. ║
╚══════════════════════════════════════════════════════════════════════════════╝
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


# ═══════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES (contratos funtoriales entre fases del endofuntor OODA)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1SanitizationData:
    r"""
    Artefacto terminal de la FASE 1 (Observe / saneamiento).

    Certifica dimensionalidad, finitud IEEE-754 y aplicación del gauge de fase
    estándar. Es el *objeto inicial* de la FASE 2: el ensamble saneado se
    transporta junto a este DTO hacia TP/Choi.

    Campos:
        dimension_d: \(d = \dim\mathcal{H}_A\).
        num_operators: cardinalidad bruta del ensamble inyectado (|{M_k}|).
        num_operators_effective: cardinalidad tras descarte UV de nulos.
        is_finite: todos los Kraus son finitos (sin NaN/Inf).
        phase_rotated: al menos un operador recibió rotación de gauge.
        gauge_phase_residuals: max |Im(pivot_k)| tras gauge (debe ≈ 0).
        frobenius_mass: \(\sum_k\|M_k\|_F^2\) (≈ d si TP).
    """
    dimension_d: int
    num_operators: int
    num_operators_effective: int
    is_finite: bool
    phase_rotated: bool
    gauge_phase_residuals: float = 0.0
    frobenius_mass: float = 0.0


@dataclass(frozen=True, slots=True)
class Phase2ChoiIsomorphismData:
    r"""
    Artefacto terminal de la FASE 2 (Orient / isomorfismo de Choi).

    Certifica TP, unitalidad, hermiticidad de Λ y grado de unitariedad.
    Es el *objeto inicial* de la FASE 3: `choi_matrix` + residuales alimentan
    CP / PPT / Non-Signaling.

    Campos:
        trace_preserving_residual: \(\|\sum M_k^\dagger M_k - I\|_F\).
        is_trace_preserving: r_TP ≤ ε.
        choi_matrix: \(\Lambda_{\mathcal{E}}\) hermítica (d² × d²).
        is_unital: \(\|\sum M_k M_k^\dagger - I\|_F \le \varepsilon\).
        unital_residual: residuo de unitalidad.
        unitariety_degree: \(U(\mathcal{E})\in[0,1]\) (1 ⇔ canal unitario).
        choi_trace: \(\operatorname{Tr}(\Lambda)\) (debe = d si TP).
        tp_diamond_defect: \(\|\operatorname{Tr}_B(\Lambda)-I\|_F\) (checksum dual).
    """
    trace_preserving_residual: float
    is_trace_preserving: bool
    choi_matrix: ComplexMatrix
    is_unital: bool
    unital_residual: float
    unitariety_degree: float
    choi_trace: float = 0.0
    tp_diamond_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class Phase3QuantumSecurityData:
    r"""
    Artefacto terminal de la FASE 3 (Decide / seguridad cuántica).

    Certifica CP, rango de Choi, PPT y Non-Signaling bipartito.

    Campos:
        minimum_choi_eigenvalue: \(\lambda_{\min}(\Lambda)\).
        kraus_rank: \(\operatorname{rank}(\Lambda)\) = nº mínimo de Kraus.
        is_completely_positive: \(\lambda_{\min}\ge\) floor PSD.
        non_signaling_residual: \(\|\rho'_B-\rho_B\|_F\).
        is_non_signaling: residuo NS ≤ ε.
        is_separable_ppt: \(\lambda_{\min}(\Lambda^{T_A})\ge\) floor PSD.
        ppt_min_eigenvalue: \(\lambda_{\min}(\Lambda^{T_A})\).
        choi_purity: \(\operatorname{Tr}(\Lambda^2)/(\operatorname{Tr}\Lambda)^2\).
    """
    minimum_choi_eigenvalue: float
    kraus_rank: int
    is_completely_positive: bool
    non_signaling_residual: float
    is_non_signaling: bool
    is_separable_ppt: bool
    ppt_min_eigenvalue: float
    choi_purity: float = 0.0


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
    """
    sanitization_audit: Phase1SanitizationData
    choi_audit: Phase2ChoiIsomorphismData
    security_audit: Phase3QuantumSecurityData
    is_channel_secure: bool
    timestamp_utc: str
    wilkinson_tolerance: float = _DEFAULT_TOL
    stratum: str = "WISDOM"


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — SANEAMIENTO NUMÉRICO Y GAUGE DE FASE ESTÁNDAR (Observe)
# Objetos: {M_k} ⊂ M_d(ℂ), pivotes de fase, soporte UV
# Funtores: tipado C*, rotación U(1) de calibre, filtrado de nulos
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_KrausSanitizer:
    r"""
    FASE 1 del endofuntor: sanea y normaliza el calibre de fase de Kraus.

    Morfismo compuesto:

    \[
    \mathrm{ObserveKraus}
    =\mathrm{Gauge}\circ\mathrm{UVFilter}\circ\mathrm{Type}\circ\mathrm{Dim}.
    \]
    """

    # ── FASE 1.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_validate_hilbert_dimension(dimension_d: int) -> int:
        r"""
        FASE 1.1 — Certificación de \(d=\dim\mathcal{H}_A\).

        Exige \(d\in\mathbb{Z}_{\ge 1}\). d=1 es el canal escalar trivial
        (admisible, degenerado). Análogo de circuitos: d = nº de puertos
        del multipolo de scattering semántico.

        Raises:
            KrausDimensionError: Si d no es entero ≥ 1.
        """
        if not isinstance(dimension_d, (int, np.integer)) or int(dimension_d) < 1:
            raise KrausDimensionError(
                f"Dimensión de Hilbert inválida: d={dimension_d}. Se exige d ∈ ℤ≥1."
            )
        d = int(dimension_d)
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
        FASE 1.2 — Tipado C*, shape (d,d) y finitud IEEE-754.

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
            if m_k.shape != expected:
                raise KrausDimensionError(
                    f"Inconsistencia dimensional en Kraus idx={idx}: "
                    f"esperado {expected}, obtenido {m_k.shape}."
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
    def _phase1_filter_uv_null_operators(
        kraus_ops: List[ComplexMatrix],
        tolerance: float,
    ) -> List[ComplexMatrix]:
        r"""
        FASE 1.3 — Filtrado ultravioleta de operadores numéricamente nulos.

        Descarta \(M_k\) con \(\|M_k\|_F\le\max(\varepsilon\cdot 10^{-3},\varepsilon_{\mathrm{sp}})\).
        Si el ensamble colapsa a vacío → canal cero (no CPTP).

        Raises:
            KrausDimensionError: Todos los Kraus son nulos.
        """
        floor = max(tolerance * 1.0e-3, _EPS_SPECTRAL)
        cleaned = [m for m in kraus_ops if float(la.norm(m, ord="fro")) > floor]
        if not cleaned:
            raise KrausDimensionError(
                f"Todos los operadores de Kraus son numéricamente nulos "
                f"(‖M_k‖_F ≤ {floor:.3e}). Canal cero no es CPTP."
            )
        if len(cleaned) < len(kraus_ops):
            logger.info(
                "FASE1.3 UV: descartados %d/%d operadores nulos.",
                len(kraus_ops) - len(cleaned),
                len(kraus_ops),
            )
        return cleaned

    # ── FASE 1.4 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_standard_gauge_phase(
        kraus_ops: List[ComplexMatrix],
    ) -> Tuple[List[ComplexMatrix], bool, float]:
        r"""
        FASE 1.4 — Gauge de fase estándar U(1) por operador.

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
            (ops_gauged, phase_rotated, max_|Im(pivot)|_post).
        """
        gauged: List[ComplexMatrix] = []
        phase_rotated = False
        max_im_residual = 0.0

        for m_k in kraus_ops:
            abs_m = np.abs(m_k)
            pivot_idx = np.unravel_index(int(np.argmax(abs_m)), m_k.shape)
            pivot_val = m_k[pivot_idx]

            if abs(pivot_val) > max(_MACHINE_EPS, _EPS_GAUGE):
                phase_angle = float(np.angle(pivot_val))
                rotated = (m_k * np.exp(-1j * phase_angle)).astype(np.complex128)
                phase_rotated = True
                # pivote debe quedar real positivo
                im_res = abs(float(np.imag(rotated[pivot_idx])))
                max_im_residual = max(max_im_residual, im_res)
            else:
                rotated = m_k

            gauged.append(rotated)

        return gauged, phase_rotated, max_im_residual

    # ── FASE 1.5 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_frobenius_mass(kraus_ops: List[ComplexMatrix]) -> RealScalar:
        r"""
        FASE 1.5 — Masa de Frobenius del ensamble.

        \[
        \mu_F=\sum_k\|M_k\|_F^2=\sum_k\operatorname{Tr}(M_k^\dagger M_k).
        \]

        Si el canal es TP, \(\mu_F=d\). Se reporta como invariante blando
        de telemetría (no hard-gate).
        """
        return float(sum(la.norm(m, ord="fro") ** 2 for m in kraus_ops))

    # ── FASE 1.Ω · composición terminal Observe ───────────────────────────
    @staticmethod
    def sanitize_operators(
        kraus_operators: List[ComplexMatrix],
        dimension_d: int,
        tolerance: float = _DEFAULT_TOL,
    ) -> Tuple[List[ComplexMatrix], Phase1SanitizationData]:
        r"""
        FASE 1.Ω — Composición terminal de Observación / saneamiento.

        \[
        \mathrm{Sanitize}
        =\mathrm{Mass}\circ\mathrm{Gauge}\circ\mathrm{UV}\circ\mathrm{Type}\circ\mathrm{Dim}.
        \]

        **Contrato funtorial F1 → F2**: el par
        ``(sanitized_ops, Phase1SanitizationData)`` es el objeto inicial
        exacto de `_phase2_trace_preserving_residual` /
        `audit_trace_preservation`. Ningún re-tipado ni re-gauge se aplica
        aguas abajo.

        Args:
            kraus_operators: Ensamble bruto \(\{M_k\}\).
            dimension_d: Dimensión candidata de \(\mathcal{H}_A\).
            tolerance: ε de Wilkinson para el filtro UV.

        Returns:
            (sanitized_ops, Phase1SanitizationData).

        Raises:
            KrausDimensionError, CPTPValidatorAgentError.
        """
        d = Phase1_KrausSanitizer._phase1_validate_hilbert_dimension(dimension_d)
        typed = Phase1_KrausSanitizer._phase1_validate_kraus_typing(
            kraus_operators, d
        )
        cleaned = Phase1_KrausSanitizer._phase1_filter_uv_null_operators(
            typed, tolerance
        )
        gauged, rotated, gauge_res = (
            Phase1_KrausSanitizer._phase1_standard_gauge_phase(cleaned)
        )
        mass = Phase1_KrausSanitizer._phase1_frobenius_mass(gauged)

        data = Phase1SanitizationData(
            dimension_d=d,
            num_operators=len(kraus_operators),
            num_operators_effective=len(gauged),
            is_finite=True,
            phase_rotated=rotated,
            gauge_phase_residuals=gauge_res,
            frobenius_mass=mass,
        )
        logger.debug(
            "FASE1.Ω sanitize: d=%d, raw=%d, eff=%d, μ_F=%.6e, gauge_im=%.3e",
            d,
            data.num_operators,
            data.num_operators_effective,
            mass,
            gauge_res,
        )
        return gauged, data


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — ISOMORFISMO DE CHOI Y CONSERVACIÓN DE LA TRAZA (Orient)
# Continuación directa del ensamble saneado de FASE 1.Ω
# Objetos: S=ΣM†M, U=ΣMM†, Λ_ℰ, Tr_B(Λ), grado de unitariedad
# Teorías: Kraus TP, Choi–Jamiołkowski, unitalidad, pureza de canal
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_ChoiIsomorphismCertifier(Phase1_KrausSanitizer):
    r"""
    FASE 2: certifica completitud TP/unital y construye el operador de Choi.

    Morfismo compuesto:

    \[
    \mathrm{OrientChoi}
    =(\mathrm{TP},\,\mathrm{Unital},\,\mathrm{Choi},\,\mathrm{Tr}_B,\,U)
    \circ\mathrm{Sanitize}^*.
    \]
    """

    # ── FASE 2.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_trace_preserving_residual(
        sanitized_ops: List[ComplexMatrix],
        dimension_d: int,
    ) -> Tuple[RealScalar, ComplexMatrix]:
        r"""
        FASE 2.1 — Residuo TP / completitud de Kraus (inicio Orient).

        **Continuación funtorial de FASE 1.Ω**: recibe `sanitized_ops`
        y `dimension_d` del certificado de saneamiento.

        \[
        S=\sum_k M_k^\dagger M_k,
        \qquad
        r_{\mathrm{TP}}=\|S-I_d\|_F.
        \]

        S se proyecta al subespacio autoadjunto (Weyl) por construcción PSD.
        """
        s_op = np.zeros((dimension_d, dimension_d), dtype=np.complex128)
        for m_k in sanitized_ops:
            s_op += m_k.conj().T @ m_k
        s_op = 0.5 * (s_op + s_op.conj().T)

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
    ) -> Tuple[RealScalar, BoolLattice]:
        r"""
        FASE 2.2 — Residuo de unitalidad.

        \[
        U_{\mathrm{op}}=\sum_k M_k M_k^\dagger,
        \qquad
        r_U=\|U_{\mathrm{op}}-I_d\|_F.
        \]

        Unitalidad (r_U=0) es independiente de TP: los canales unitales
        fijan el estado máximamente mixto. No es hard-gate por defecto
        (el depolarizante con p>0 es unital; amplitude damping no lo es).
        """
        u_op = np.zeros((dimension_d, dimension_d), dtype=np.complex128)
        for m_k in sanitized_ops:
            u_op += m_k @ m_k.conj().T
        u_op = 0.5 * (u_op + u_op.conj().T)
        residual = float(
            la.norm(u_op - np.eye(dimension_d, dtype=np.complex128), ord="fro")
        )
        return residual, residual <= _DEFAULT_TOL  # flag preliminar; tol real en compositor

    # ── FASE 2.3 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_construct_choi_matrix(
        sanitized_ops: List[ComplexMatrix],
        dimension_d: int,
    ) -> ComplexMatrix:
        r"""
        FASE 2.3 — Isomorfismo de Choi–Jamiołkowski (vec column-major).

        \[
        \Lambda_{\mathcal{E}}
        =\sum_k\operatorname{vec}_F(M_k)\,\operatorname{vec}_F(M_k)^\dagger
        \in\mathrm{Mat}_{d^2}(\mathbb{C}).
        \]

        Convención ``order='F'`` (QI estándar: Watrous / Nielsen–Chuang).
        Proyección de Weyl al subespacio autoadjunto post-ensamblaje.
        """
        choi_dim = dimension_d * dimension_d
        choi = np.zeros((choi_dim, choi_dim), dtype=np.complex128)
        for m_k in sanitized_ops:
            vec_m = np.asarray(m_k.flatten(order="F"), dtype=np.complex128)
            choi += np.outer(vec_m, vec_m.conj())
        choi = 0.5 * (choi + choi.conj().T)
        return choi

    # ── FASE 2.4 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_certify_choi_hermiticity(
        choi_matrix: ComplexMatrix,
        tolerance: float,
    ) -> ComplexMatrix:
        r"""
        FASE 2.4 — Certificación C* de hermiticidad de Λ_ℰ.

        \[
        \|\Lambda-\Lambda^\dagger\|_F\le\max(\varepsilon,\varepsilon_H).
        \]

        Raises:
            ChoiHermiticityError: Defecto antihermítico intolerable.
        """
        defect = float(la.norm(choi_matrix - choi_matrix.conj().T, ord="fro"))
        tol_h = max(tolerance, _EPS_HERMITICITY)
        if defect > tol_h:
            raise ChoiHermiticityError(
                f"Choi viola hermiticidad: ‖Λ−Λ†‖_F={defect:.3e} > tol={tol_h:.3e}."
            )
        # Re-Weyl por seguridad numérica
        return 0.5 * (choi_matrix + choi_matrix.conj().T)

    # ── FASE 2.5 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_choi_partial_trace_tp_defect(
        choi_matrix: ComplexMatrix,
        dimension_d: int,
    ) -> RealScalar:
        r"""
        FASE 2.5 — Checksum dual TP vía traza parcial de Choi.

        Para Choi no normalizada de un canal TP:

        \[
        \operatorname{Tr}_B(\Lambda_{\mathcal{E}})=I_A,
        \qquad
        \delta_{\diamond}=\|\operatorname{Tr}_B(\Lambda)-I\|_F.
        \]

        Layout ``order='F'`` coherente con ``vec_F`` de FASE 2.3.
        """
        d = dimension_d
        tensor = choi_matrix.reshape((d, d, d, d), order="F")
        # ejes: 0=a, 1=b, 2=a', 3=b'  → Tr_B contrae b=b'
        partial = np.trace(tensor, axis1=1, axis2=3)
        partial = 0.5 * (partial + partial.conj().T)
        return float(
            la.norm(partial - np.eye(d, dtype=np.complex128), ord="fro")
        )

    # ── FASE 2.6 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_unitariety_degree(
        choi_matrix: ComplexMatrix,
        dimension_d: int,
    ) -> RealScalar:
        r"""
        FASE 2.6 — Grado de unitariedad del canal.

        \[
        U(\mathcal{E})
        =\frac{d\cdot\operatorname{Tr}(\Lambda^2)-1}{d^2-1}
        \in[0,1]
        \quad(d>1),
        \]

        con \(U=1\) sii \(\mathcal{E}\) es unitario (Λ es proyector de rango 1
        sobre el estado máximamente entrelazado, a escala d) y \(U=0\) para
        el canal completamente despolarizante. Fórmula de pureza de canal
        de Nielsen / Wallman–Flammia (adaptada a Choi no normalizada:
        \(\operatorname{Tr}\Lambda=d\Rightarrow\operatorname{Tr}(\Lambda^2)=d^2\)
        en el caso unitario).
        """
        if dimension_d <= 1:
            return 1.0
        choi_sq_trace = float(np.real(np.trace(choi_matrix @ choi_matrix)))
        choi_dim = dimension_d * dimension_d
        unitariety = (dimension_d * choi_sq_trace - 1.0) / (choi_dim - 1.0)
        # clamp numérico a [0, 1]
        return float(min(1.0, max(0.0, unitariety)))

    # ── FASE 2.Ω · composición terminal Orient ────────────────────────────
    @staticmethod
    def audit_trace_preservation(
        sanitized_operators: List[ComplexMatrix],
        dimension_d: int,
        tolerance: float = _DEFAULT_TOL,
    ) -> Phase2ChoiIsomorphismData:
        r"""
        FASE 2.Ω — Composición terminal Orient (TP + Choi + unital + U).

        **Continuación funtorial de FASE 1.Ω**: consume `sanitized_operators`
        emitidos por `sanitize_operators`.

        **Contrato funtorial F2 → F3**: el DTO
        ``Phase2ChoiIsomorphismData`` (con `choi_matrix` hermítica) es el
        objeto inicial exacto de `audit_quantum_security`.

        Raises:
            TracePreservationViolation: Si r_TP > ε (hard-gate).
            ChoiHermiticityError: Si Λ no es hermítica dentro de tol.
        """
        r_tp, _s = Phase2_ChoiIsomorphismCertifier._phase2_trace_preserving_residual(
            sanitized_operators, dimension_d
        )
        is_tp = r_tp <= tolerance
        if not is_tp:
            raise TracePreservationViolation(
                f"Fuga termodinámica: el canal viola la completitud. "
                f"Residuo TP = {r_tp:.4e} > {tolerance:.4e}."
            )

        r_u, _ = Phase2_ChoiIsomorphismCertifier._phase2_unital_residual(
            sanitized_operators, dimension_d
        )
        is_unital = r_u <= tolerance

        choi = Phase2_ChoiIsomorphismCertifier._phase2_construct_choi_matrix(
            sanitized_operators, dimension_d
        )
        choi = Phase2_ChoiIsomorphismCertifier._phase2_certify_choi_hermiticity(
            choi, tolerance
        )
        choi_trace = float(np.real(np.trace(choi)))
        delta_tp = (
            Phase2_ChoiIsomorphismCertifier._phase2_choi_partial_trace_tp_defect(
                choi, dimension_d
            )
        )
        unitariety = Phase2_ChoiIsomorphismCertifier._phase2_unitariety_degree(
            choi, dimension_d
        )

        logger.debug(
            "FASE2.Ω Choi: r_TP=%.3e, r_U=%.3e, TrΛ=%.6f, δ_⋄=%.3e, U=%.6f",
            r_tp,
            r_u,
            choi_trace,
            delta_tp,
            unitariety,
        )

        return Phase2ChoiIsomorphismData(
            trace_preserving_residual=r_tp,
            is_trace_preserving=is_tp,
            choi_matrix=choi,
            is_unital=is_unital,
            unital_residual=r_u,
            unitariety_degree=unitariety,
            choi_trace=choi_trace,
            tp_diamond_defect=delta_tp,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — CP, PPT, NON-SIGNALING Y SEGURIDAD CUÁNTICA (Decide + Act)
# Continuación directa de la geometría de Choi de FASE 2.Ω
# Objetos: σ(Λ), Λ^{T_A}, ρ_B vs ρ'_B, retícula de gobernanza
# Teorías: Choi CP, Peres–Horodecki, no-señalización, causalidad bipartita
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_QuantumSecurityEnforcer(Phase2_ChoiIsomorphismCertifier):
    r"""
    FASE 3: audita positividad completa, separabilidad PPT y no-señalización.

    Morfismo compuesto:

    \[
    \mathrm{Secure}
    =(\mathrm{CP},\,\mathrm{PPT},\,\mathrm{NS},\,\mathrm{Purity})
    \circ\mathrm{OrientChoi}^*.
    \]
    """

    # ── FASE 3.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_spectral_complete_positivity(
        choi_matrix: ComplexMatrix,
        tolerance: float,
    ) -> Tuple[RealVector, RealScalar, int, BoolLattice]:
        r"""
        FASE 3.1 — Positividad completa vía espectro de Choi (inicio Decide).

        **Continuación funtorial de FASE 2.Ω**: recibe `choi_matrix`.

        Teorema de Choi (1975):

        \[
        \mathcal{E}\text{ es CP}
        \Longleftrightarrow
        \Lambda_{\mathcal{E}}\succeq 0
        \Longleftrightarrow
        \lambda_{\min}(\Lambda)\ge 0.
        \]

        Se admite suelo de Wilkinson ``_SPECTRAL_PSD_FLOOR`` por redondeo FPU.
        """
        eigvals = la.eigvalsh(choi_matrix).astype(np.float64)
        min_eigen = float(np.min(eigvals))
        is_cp: BoolLattice = min_eigen >= _SPECTRAL_PSD_FLOOR
        kraus_rank = int(np.sum(eigvals > tolerance))

        logger.debug(
            "FASE3.1 CP: λ_min=%.6e, rank=%d, is_cp=%s",
            min_eigen,
            kraus_rank,
            is_cp,
        )
        return eigvals, min_eigen, kraus_rank, is_cp

    # ── FASE 3.2 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_ppt_separability(
        choi_matrix: ComplexMatrix,
        dimension_d: int,
    ) -> Tuple[RealScalar, BoolLattice]:
        r"""
        FASE 3.2 — Criterio PPT de Peres–Horodecki sobre Λ_ℰ.

        \[
        \bigl(\Lambda^{T_A}\bigr)_{a b;\,a'b'}
        =\Lambda_{a'b;\,ab'},
        \qquad
        \text{PPT}\iff\Lambda^{T_A}\succeq 0.
        \]

        Layout ``order='F'`` coherente con ``vec_F``. PPT es invariante
        blando: se reporta, no aborta (salvo política externa que eleve
        ``PeresHorodeckiSeparabilityError``).
        """
        d = dimension_d
        choi_dim = d * d
        block = choi_matrix.reshape((d, d, d, d), order="F")
        # Transpuesta parcial sobre A: intercambiar a ↔ a' (ejes 0 ↔ 2)
        ppt_block = block.transpose(2, 1, 0, 3)
        ppt_matrix = ppt_block.reshape((choi_dim, choi_dim), order="F")
        ppt_matrix = 0.5 * (ppt_matrix + ppt_matrix.conj().T)

        ppt_eigs = la.eigvalsh(ppt_matrix)
        ppt_min = float(np.min(ppt_eigs))
        is_separable = ppt_min >= _SPECTRAL_PSD_FLOOR

        logger.debug(
            "FASE3.2 PPT: λ_min(Λ^{T_A})=%.6e, separable=%s",
            ppt_min,
            is_separable,
        )
        return ppt_min, is_separable

    # ── FASE 3.3 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_bell_resource_state(dimension_d: int) -> ComplexMatrix:
        r"""
        FASE 3.3 — Estado recurso de Bell (no normalizado a traza-1 en escala
        de probabilidad, sí normalizado):

        \[
        |\Omega\rangle
        =d^{-1/2}\sum_{i=0}^{d-1}|i\rangle_A\otimes|i\rangle_B,
        \qquad
        \rho_{AB}=|\Omega\rangle\langle\Omega|.
        \]
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
        FASE 3.4 — Traza parcial sobre Alice.

        \[
        (\operatorname{Tr}_A\rho)_{b b'}
        =\sum_{a}\rho_{(a b),(a b')}.
        \]

        Layout: Kronecker A⊗B con bloques d×d (fila-bloque = índice A).
        """
        d = dimension_d
        rho_b = np.zeros((d, d), dtype=np.complex128)
        for a in range(d):
            rho_b += rho_ab[a * d : (a + 1) * d, a * d : (a + 1) * d]
        return 0.5 * (rho_b + rho_b.conj().T)

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
        rho_prime = np.zeros_like(rho_ab)
        for m_k in sanitized_ops:
            op = np.kron(m_k, identity_b)
            rho_prime += op @ rho_ab @ op.conj().T
        # Weyl: el resultado debe ser hermítico
        return 0.5 * (rho_prime + rho_prime.conj().T)

    # ── FASE 3.6 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_non_signaling_residual(
        sanitized_ops: List[ComplexMatrix],
        dimension_d: int,
    ) -> Tuple[RealScalar, BoolLattice]:
        r"""
        FASE 3.6 — Condición de No-Señalización bipartita.

        \[
        \rho'_B
        =\operatorname{Tr}_A\!\big((\mathcal{E}_A\otimes\mathcal{I}_B)(\rho_{AB})\big),
        \qquad
        r_{\mathrm{NS}}=\|\rho'_B-\rho_B\|_F.
        \]

        **Teorema (checksum causal)**: si \(\mathcal{E}\) es TP entonces
        \(r_{\mathrm{NS}}=0\) idénticamente para todo \(\rho_{AB}\). El
        residuo es por tanto un invariante numérico dual de TP (análogo
        al \(\delta_\diamond\) de FASE 2.5, pero en el cuadro de Schrödinger
        bipartito). Hard-gate: r_NS > ε ⇒ corrupción numérica o bug de
        ensamble.

        Implementación: recurso de Bell + traza parcial A.
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
        return residual, residual <= _DEFAULT_TOL  # flag preliminar

    # ── FASE 3.7 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_choi_purity(
        choi_matrix: ComplexMatrix,
        choi_trace: RealScalar,
    ) -> RealScalar:
        r"""
        FASE 3.7 — Pureza normalizada del estado de Choi.

        \[
        P(\Lambda)
        =\frac{\operatorname{Tr}(\Lambda^2)}{(\operatorname{Tr}\Lambda)^2}
        \in(0,1].
        \]
        """
        tr = choi_trace if abs(choi_trace) > _EPS_SPECTRAL else 1.0
        tr_sq = float(np.real(np.trace(choi_matrix @ choi_matrix)))
        return float(tr_sq / (tr * tr))

    # ── FASE 3.Ω · composición terminal Decide ────────────────────────────
    @staticmethod
    def audit_quantum_security(
        choi_data: Phase2ChoiIsomorphismData,
        sanitized_operators: List[ComplexMatrix],
        dimension_d: int,
        tolerance: float = _DEFAULT_TOL,
    ) -> Phase3QuantumSecurityData:
        r"""
        FASE 3.Ω — Composición terminal Decide (CP + PPT + NS + pureza).

        **Continuación funtorial de FASE 2.Ω**: consume
        ``Phase2ChoiIsomorphismData`` y el ensamble saneado de FASE 1.

        **Contrato funtorial F3 → Seal**: el DTO
        ``Phase3QuantumSecurityData`` alimenta el sellado de gobernanza
        en `CPTPChannelValidatorAgent.audit_cptp_channel`.

        Raises:
            CompletePositivityViolation: λ_min(Λ) < floor PSD.
            NonSignalingViolationError: r_NS > ε.
        """
        choi_matrix = choi_data.choi_matrix

        _eigs, min_eigen, kraus_rank, is_cp = (
            Phase3_QuantumSecurityEnforcer._phase3_spectral_complete_positivity(
                choi_matrix, tolerance
            )
        )
        if not is_cp:
            raise CompletePositivityViolation(
                f"La matriz de Choi viola la semidefinitud positiva: "
                f"autovalor mínimo = {min_eigen:.4e} < floor={_SPECTRAL_PSD_FLOOR:.4e}."
            )

        ppt_min, is_separable_ppt = (
            Phase3_QuantumSecurityEnforcer._phase3_ppt_separability(
                choi_matrix, dimension_d
            )
        )

        ns_residual, _ = (
            Phase3_QuantumSecurityEnforcer._phase3_non_signaling_residual(
                sanitized_operators, dimension_d
            )
        )
        is_non_signaling = ns_residual <= tolerance
        if not is_non_signaling:
            raise NonSignalingViolationError(
                f"Ruptura causal: la acción local del LLM señaliza al espacio de Bob. "
                f"Residuo de No-Señalización = {ns_residual:.4e} > {tolerance:.4e}."
            )

        purity = Phase3_QuantumSecurityEnforcer._phase3_choi_purity(
            choi_matrix, choi_data.choi_trace
        )

        logger.debug(
            "FASE3.Ω security: CP=%s, rank=%d, r_NS=%.3e, PPT=%s, P=%.6f",
            is_cp,
            kraus_rank,
            ns_residual,
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
    ) -> BoolLattice:
        r"""
        FASE Ω.1 — Clasificador de subobjetos \(\chi_{\mathrm{secure}}\in\Omega\).

        \[
        \chi
        =\chi_{\mathrm{finite}}
        \wedge\chi_{\mathrm{TP}}
        \wedge\chi_{\mathrm{CP}}
        \wedge\chi_{\mathrm{NS}}.
        \]

        PPT y unitalidad son invariantes blandos (no entran en χ).
        """
        return bool(
            sanitization.is_finite
            and choi_audit.is_trace_preserving
            and security.is_completely_positive
            and security.is_non_signaling
        )

    # ── FASE Ω.2 · sellado del certificado de gobernanza ──────────────────
    def _seal_governance_state(
        self,
        sanitization: Phase1SanitizationData,
        choi_audit: Phase2ChoiIsomorphismData,
        security: Phase3QuantumSecurityData,
        is_secure: BoolLattice,
        tolerance: float,
    ) -> CPTPChannelGovernanceState:
        r"""
        FASE Ω.2 — Sellado del objeto terminal frozen del funtor OODA.

        Empaqueta los tres DTOs de fase + χ_secure + timestamp UTC.
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
        kraus_operators: List[ComplexMatrix],
        dimension_d: int,
        tolerance: float = _DEFAULT_TOL,
    ) -> CPTPChannelGovernanceState:
        r"""
        Ejecuta el ciclo OODA completo de validación espectral del canal.

        .. code-block:: text

            ┌────────────────────────────────────────────────────────────┐
            │ FASE 1  Observe / Sanitize                                 │
            │   1.1 validate_hilbert_dimension                           │
            │   1.2 validate_kraus_typing                                │
            │   1.3 filter_uv_null_operators                             │
            │   1.4 standard_gauge_phase                                 │
            │   1.5 frobenius_mass                                       │
            │   1.Ω sanitize_operators  ──┐                              │
            ├─────────────────────────────┼──────────────────────────────┤
            │ FASE 2  Orient / Choi  ◄────┘                              │
            │   2.1 trace_preserving_residual                            │
            │   2.2 unital_residual                                      │
            │   2.3 construct_choi_matrix                                │
            │   2.4 certify_choi_hermiticity                             │
            │   2.5 choi_partial_trace_tp_defect                         │
            │   2.6 unitariety_degree                                    │
            │   2.Ω audit_trace_preservation  ──┐                        │
            ├───────────────────────────────────┼────────────────────────┤
            │ FASE 3  Decide / Secure  ◄────────┘                        │
            │   3.1 spectral_complete_positivity                         │
            │   3.2 ppt_separability                                     │
            │   3.3 bell_resource_state                                  │
            │   3.4 partial_trace_A                                      │
            │   3.5 apply_local_channel_on_A                             │
            │   3.6 non_signaling_residual                               │
            │   3.7 choi_purity                                          │
            │   3.Ω audit_quantum_security  ──┐                          │
            ├─────────────────────────────────┼──────────────────────────┤
            │ SEAL / VETO  ◄──────────────────┘                          │
            │   Ω.1 security_conjunction                                 │
            │   Ω.2 seal_governance_state                                │
            │   Ω.Veto log_and_reraise                                   │
            └────────────────────────────────────────────────────────────┘

        Args:
            kraus_operators: Colección de matrices de Kraus \(\{M_k\}\).
            dimension_d: Dimensión de \(\mathcal{H}_A\).
            tolerance: ε de Wilkinson para tolerancia espectral.

        Returns:
            CPTPChannelGovernanceState: certificado inmutable de gobernanza.

        Raises:
            CPTPValidatorAgentError: Violación de traza, CP, NS, dimensión
                o hermiticidad de Choi (re-propagada tras log CRITICAL).
        """
        if tolerance < 0.0 or not math.isfinite(tolerance):
            raise CPTPValidatorAgentError(
                f"tolerance debe ser real ≥ 0 y finito; recibido {tolerance}."
            )

        try:
            # ── FASE 1 · Observe / Sanitize ───────────────────────────────
            sanitized_ops, sanitization_audit = self.sanitize_operators(
                kraus_operators,
                dimension_d,
                tolerance=tolerance,
            )

            # ── FASE 2 · Orient / Choi  (continúa ensamble F1) ────────────
            choi_audit = self.audit_trace_preservation(
                sanitized_ops,
                sanitization_audit.dimension_d,
                tolerance,
            )

            # ── FASE 3 · Decide / Secure  (continúa Choi F2) ──────────────
            security_audit = self.audit_quantum_security(
                choi_audit,
                sanitized_ops,
                sanitization_audit.dimension_d,
                tolerance,
            )

            # ── SEAL ──────────────────────────────────────────────────────
            is_secure = self._seal_security_conjunction(
                sanitization_audit, choi_audit, security_audit
            )
            state = self._seal_governance_state(
                sanitization_audit,
                choi_audit,
                security_audit,
                is_secure,
                tolerance,
            )

            logger.info(
                "Canal cuántico de inyección semántica validado con éxito. "
                "Rango de Choi=%d | r_NS=%.4e | PPT=%s | U=%.4f | secure=%s",
                security_audit.kraus_rank,
                security_audit.non_signaling_residual,
                str(security_audit.is_separable_ppt),
                choi_audit.unitariety_degree,
                str(is_secure),
            )
            return state

        except CPTPValidatorAgentError as err:
            self._veto_log_and_reraise(err)
            raise  # unreachable; satisface type-checkers
        return None  # pragma: no cover

    # ─────────────────────────────────────────────────────────────────────
    # Fábricas de referencia (calibración / tests del agente)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def identity_kraus(dimension_d: int) -> List[ComplexMatrix]:
        """Canal identidad: \(\{I_d\}\)."""
        return [np.eye(dimension_d, dtype=np.complex128)]

    @staticmethod
    def amplitude_damping_kraus(gamma: float) -> List[ComplexMatrix]:
        r"""
        Amortiguamiento de amplitud (qubit):

        \[
        M_0=\begin{pmatrix}1&0\\0&\sqrt{1-\gamma}\end{pmatrix},
        \quad
        M_1=\begin{pmatrix}0&\sqrt{\gamma}\\0&0\end{pmatrix}.
        \]
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
    def depolarizing_kraus(dimension_d: int, p: float) -> List[ComplexMatrix]:
        r"""
        Canal despolarizante de Weyl en dimensión d
        (Heisenberg–Weyl shift/clock; p ∈ [0,1]).
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p ∈ [0,1]; recibido {p}")
        d = dimension_d
        kraus: List[ComplexMatrix] = []
        m0_scale = math.sqrt(max(1.0 - p + p / d, 0.0))
        kraus.append((m0_scale * np.eye(d)).astype(np.complex128))

        omega = np.exp(2j * np.pi / d)
        x = np.zeros((d, d), dtype=np.complex128)
        z = np.zeros((d, d), dtype=np.complex128)
        for i in range(d):
            x[i, (i - 1) % d] = 1.0 + 0.0j
            z[i, i] = omega ** i

        noise_scale = math.sqrt(max(p / d, 0.0) / d)
        for i in range(d):
            for j in range(d):
                if i == 0 and j == 0:
                    continue
                xi = np.linalg.matrix_power(x, i)
                zj = np.linalg.matrix_power(z, j)
                kraus.append((noise_scale * (xi @ zj)).astype(np.complex128))
        return kraus

    @staticmethod
    def replacement_kraus() -> List[ComplexMatrix]:
        r"""Canal de reemplazo a \(|0\rangle\langle 0|\) (EB, d=2)."""
        m0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        m1 = np.array([[0, 1], [0, 0]], dtype=np.complex128)
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
]
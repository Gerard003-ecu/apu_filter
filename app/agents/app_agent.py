# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : App Agent (El Custodio de la Variedad de Frontera Ciber-Física)     ║
║ Ruta   : app/agents/app_agent.py                                             ║
║ Versión: 3.0.0-Triadic-Nested-Spectral-Categorical-Quantum-Governance        ║
╚══════════════════════════════════════════════════════════════════════════════╝

EVOLUCIÓN RIGUROSA (Artesanía Senior + Física-Matemática de Doctorado):
────────────────────────────────────────────────────────────────────────────────
• Fase 1 → Fibración de Estado + Impedancia Activa Port-Hamiltoniana.
  - [Teoría de grupos / Álgebra de Boole] Distancia de Hamming en el grupo
    abeliano (GF(2)^n, XOR) como certificado forense adicional a la
    comparación de tiempo constante (que se preserva intacta).
  - [Teoría espectral] Certificado espectral de J: los autovalores de un
    generador antisimétrico real son puramente imaginarios (teorema
    espectral para operadores normales antisimétricos).
  - [Álgebra lineal / Ley de Sylvester] Inercia (n₊, n₀, n₋) de R, invariante
    bajo toda transformación de congruencia — más barata y más informativa
    que un único chequeo de semidefinitud.
  - [Física de circuitos / Teoría espectral de grafos] R_sym se interpreta
    como matriz de conductancias de una red disipativa: se certifica
    dominancia diagonal (condición de M-matriz / red pasiva sin
    acoplamientos parásitos) y se reporta un análogo de conectividad
    algebraica (valor de Fiedler) del tejido disipativo.

• Fase 2 → Cierre Categórico-Topológico del Poset DIKW.
  - [Teoría de grafos + Álgebra de Boole] La relación "requiere" se modela
    como un grafo dirigido con adyacencia inmediata; su clausura transitiva
    se calcula mediante el **algoritmo de Warshall** sobre el semianillo
    booleano (∨, ∧), en vez de un rango hardcodeado.
  - [Teoría de categorías] Un poset es una categoría delgada (thin category);
    se certifican explícitamente sus tres axiomas — reflexividad,
    antisimetría, transitividad (idempotencia del operador de clausura:
    cl∘cl = cl) — como invariantes estructurales del módulo.

• Fase 3 → Proyección Cuántica-Booleana MIC + Síntesis de Anomalía Global.
  - [Mecánica cuántica / Teorema espectral] Todo proyector ortogonal
    hermítico satisface σ(P) ⊆ {0,1}; se certifica el residuo espectral
    binario vía `eigvalsh`.
  - [Mecánica cuántica / Regla de Born] Se computa la probabilidad de
    colapso p_i = ‖P_i x‖² / ‖x‖² del vector de intención sobre el
    subespacio dirigido.
  - [Álgebra de Boole / Lógica cuántica de von Neumann–Birkhoff] Se certifican
    los axiomas de reticulado ortocomplementado para pares de proyectores
    conmutantes: meet = P_iP_j ≈ 0, join = P_i+P_j-P_iP_j idempotente.
  - [Teoría de cuerdas — mecanismo de Green–Schwarz, por analogía formal]
    La consistencia global de la frontera se certifica como una
    **cancelación de anomalías**: cada sector (Φ₁, Φ₂, Φ₃) aporta una carga
    binaria de inconsistencia; el vacío ciber-físico solo es estable si el
    índice de anomalía total es idénticamente cero.
  - [Curry–Howard / tipificación de excepciones] Los validadores numéricos
    genéricos ahora reciben `exception_cls` para que cada dominio semántico
    (impedancia vs. proyección) levante su propio tipo de excepción.

ANIDAMIENTO FUNTORIAL (3 fases):
────────────────────────────────────────────────────────────────────────────────
Phase1_SpectralFibrationImpedanceCertifier
  └─ último método: _phase1_terminal_bridge_to_phase2
       └─ Phase2_CategoricalDIKWClosureInstantiator
            └─ último método: _phase2_terminal_bridge_to_phase3
                 └─ Phase3_QuantumBooleanMICProjectorSynthesizer
                      └─ último método: _phase3_terminal_synthesis
                           └─ AppAgent.execute_gateway_governance
"""

from __future__ import annotations

import hmac
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Tuple, Type

import numpy as np
from numpy.typing import NDArray

try:
    import scipy.linalg as la
except ImportError:  # pragma: no cover
    import numpy.linalg as la  # type: ignore

try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos MIC."""
        pass

    class Morphism:
        """Clase base de morfismos del Topos cuando no existe dependencia externa."""
        pass


logger = logging.getLogger("MIC.Boundary.AppAgent")
logger.addHandler(logging.NullHandler())


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, TERMODINÁMICAS Y ESTRUCTURALES DE FRONTERA
# ══════════════════════════════════════════════════════════════════════════════

_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)
_BASE_TOLERANCE: float = 1e-12
_DISSIPATION_TOLERANCE: float = 1e-9
_ORTHOGONALITY_TOLERANCE: float = 1e-12
_PSD_TOLERANCE: float = 1e-10

# Estructura fija del poset DIKW como grafo dirigido de adyacencia inmediata.
# adjacency[i][j] = True  ⇔  el estrato i requiere directamente (o es) el estrato j.
# La diagonal es reflexiva por definición (todo estrato se requiere a sí mismo).
_DIKW_LEVEL_NAMES: Tuple[str, ...] = ("PHYSICS", "TACTICS", "STRATEGY", "WISDOM")

_DIKW_IMMEDIATE_ADJACENCY: NDArray[np.bool_] = np.array(
    [
        [True, False, False, False],   # PHYSICS
        [True, True, False, False],    # TACTICS  requiere PHYSICS
        [False, True, True, False],    # STRATEGY requiere TACTICS
        [False, False, True, True],    # WISDOM   requiere STRATEGY
    ],
    dtype=np.bool_,
)


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES DE FRONTERA
# ══════════════════════════════════════════════════════════════════════════════

class AppAgentError(TopologicalInvariantError):
    """Excepción raíz del Custodio de la Variedad de Frontera."""
    pass


class StateSymmetryBreakingError(AppAgentError):
    r"""Detonada si la norma residual del tensor de estado criptográfico es ≠ 0."""
    pass


class ImpedanceInputError(AppAgentError):
    """Detonada cuando los operadores Port-Hamiltonianos tienen formato inválido."""
    pass


class ResonanceCatastropheVeto(AppAgentError):
    r"""Detonada si Ḣ > 0. El Rate Limiter bloquea el flujo energético."""
    pass


class TopologicalPassportError(AppAgentError):
    r"""Detonada si se viola la Ley de Clausura Transitiva en el TelemetryContext."""
    pass


class OrthogonalProjectionInputError(AppAgentError):
    """Detonada cuando los proyectores o el vector de intención son inválidos."""
    pass


class OrthogonalityViolationError(AppAgentError):
    r"""Detonada si una petición activa componentes fuera de su base ortogonal asignada."""
    pass


class GlobalAnomalyCancellationError(AppAgentError):
    r"""
    Detonada si el índice de anomalía global (Σ cargas de inconsistencia por
    sector) no se anula. Análogo formal al mecanismo de Green-Schwarz: una
    teoría de frontera con anomalía neta no nula no define un vacío estable.
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs de la Categoría de Frontera)
# ══════════════════════════════════════════════════════════════════════════════

def _utc_timestamp() -> str:
    """Devuelve marca de tiempo UTC ISO-8601 para trazabilidad auditable."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class StateFibrationData:
    r"""
    Artefacto de Fase 1 (sub-certificado criptográfico).
    Certificado de isomorfismo en la sesión criptográfica, enriquecido con
    la distancia de Hamming en el grupo (GF(2)^n, XOR) como huella forense.
    """
    hash_residual: float
    is_symmetric: bool
    hash_length_t0: int
    hash_length_t: int
    constant_time_compared: bool
    hamming_distance_bits: int
    bit_length_reference: int
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ImpedanceControlData:
    r"""
    Artefacto de Fase 1 (sub-certificado Port-Hamiltoniano).
    Certificado de decaimiento, pasividad de red y topología espectral
    del par estructural (J, R).
    """
    dimension: int
    h_dot: float
    dissipation_term: float
    exogenous_work_gu: float
    tolerance: float
    j_antisymmetry_residual: float
    j_spectral_max_real_part: float
    r_symmetry_residual: float
    r_min_eigenvalue: float
    r_inertia_positive: int
    r_inertia_zero: int
    r_inertia_negative: int
    r_diagonally_dominant: bool
    r_algebraic_connectivity_analog: float
    structural_valid: bool
    is_passively_stable: bool
    is_strictly_dissipative: bool
    is_asymptotically_stable: bool
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ThermodynamicPassportData:
    r"""
    Artefacto de Fase 2.
    Certificado de filtración DIKW derivado de la clausura transitiva
    booleana (Warshall) sobre el grafo de dependencias del poset.
    """
    assigned_stratum_level: int
    requested_levels: Tuple[int, ...]
    missing_lower_levels: Tuple[int, ...]
    is_transitively_closed: bool
    is_valid_poset_structure: bool
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OrthogonalProjectionData:
    r"""
    Artefacto de Fase 3.
    Certificado de proyección ortogonal en la MIC, enriquecido con el
    teorema espectral de proyectores, la regla de Born y los axiomas de
    reticulado booleano ortocomplementado.
    """
    intent_dimension: int
    projector_rank: Optional[int]
    idempotence_residual: float
    symmetry_residual: float
    spectral_binary_residual: float
    operator_orthogonality_residual: float
    state_orthogonality_residual: float
    boolean_meet_residual: float
    boolean_join_idempotence_residual: float
    born_rule_probability: float
    projected_norm: float
    tolerance: float
    is_mutually_orthogonal: bool
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GatewayGovernanceState:
    r"""
    Objeto final del endofuntor:
        Z_Gateway = Φ₃ ∘ Φ₂ ∘ Φ₁
    """
    governance_id: str
    fibration_audit: StateFibrationData
    impedance_audit: ImpedanceControlData
    passport_audit: ThermodynamicPassportData
    projection_audit: OrthogonalProjectionData
    anomaly_index: int
    anomaly_failing_sectors: Tuple[str, ...]
    is_gateway_secure: bool
    generated_at_utc: str


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1: FIBRACIÓN CRIPTOGRÁFICA DE ESTADO + IMPEDANCIA ACTIVA               ║
# ║ PORT-HAMILTONIANA CON CERTIFICACIÓN ESPECTRAL Y DE CIRCUITOS                ║
# ║                                                                             ║
# ║ El último método de esta fase es el puente formal hacia Fase 2.             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase1_SpectralFibrationImpedanceCertifier:
    r"""
    Unifica dos certificaciones de estado sobre el mismo objeto de frontera:

    1) Isomorfismo criptográfico de la sesión (grupo discreto GF(2)^n).
    2) Pasividad Port-Hamiltoniana del flujo de energía/tasa de peticiones,
       verificada no solo por norma sino por **espectro** (teorema espectral
       de operadores normales), **inercia de Sylvester** (invariante de
       congruencia) y **topología de circuitos** (dominancia diagonal /
       conectividad algebraica análoga a un grafo de disipación).
    """

    # ────────────────────────────────────────────────────────────────────
    # §1.1 — Utilidades numéricas genéricas (tipificadas por dominio)
    # ────────────────────────────────────────────────────────────────────

    def _as_finite_vector(
        self,
        name: str,
        value: NDArray[Any],
        *,
        exception_cls: Type[AppAgentError] = ImpedanceInputError,
    ) -> NDArray[np.float64]:
        """
        Convierte una entrada a vector float64 finito.

        Mejora rigurosa (Curry-Howard): el tipo de excepción es inyectable,
        de modo que cada dominio semántico (impedancia, proyección MIC)
        levanta su propio tipo de error, preservando la correspondencia
        entre categoría de fallo y categoría de dato.
        """
        arr = np.asarray(value, dtype=np.float64)

        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim == 2:
            if 1 in arr.shape:
                arr = arr.reshape(-1)
            else:
                raise exception_cls(
                    f"{name} debe ser un vector 1D o una matriz columna/fila 2D. "
                    f"Se recibió shape={arr.shape}."
                )
        elif arr.ndim != 1:
            raise exception_cls(
                f"{name} debe ser un vector. Se recibió ndim={arr.ndim}."
            )

        if not np.all(np.isfinite(arr)):
            raise exception_cls(
                f"{name} contiene valores no finitos (NaN/Inf)."
            )

        return arr

    def _as_finite_square_matrix(
        self,
        name: str,
        value: NDArray[Any],
        dimension: int,
        *,
        exception_cls: Type[AppAgentError] = ImpedanceInputError,
    ) -> NDArray[np.float64]:
        """
        Convierte una entrada a matriz cuadrada float64 finita de dimensión dada.
        """
        matrix = np.asarray(value, dtype=np.float64)

        if matrix.ndim != 2:
            raise exception_cls(
                f"{name} debe ser una matriz 2D. Se recibió ndim={matrix.ndim}."
            )

        expected_shape = (dimension, dimension)
        if matrix.shape != expected_shape:
            raise exception_cls(
                f"{name} debe tener shape={expected_shape}. "
                f"Se recibió shape={matrix.shape}."
            )

        if not np.all(np.isfinite(matrix)):
            raise exception_cls(
                f"{name} contiene valores no finitos (NaN/Inf)."
            )

        return matrix

    def _operator_tolerance(
        self,
        dimension: int,
        scale: float,
        base: float = _BASE_TOLERANCE,
    ) -> float:
        """
        Tolerancia dinámica para operadores matriciales.

        Modelo:
            tol = max(base, 100 · n · ε_machine · max(1, scale))
        """
        dim = max(1, int(dimension))
        return float(max(base, 100.0 * dim * _MACHINE_EPSILON * max(1.0, float(scale))))

    # ────────────────────────────────────────────────────────────────────
    # §1.2 — Certificación criptográfica (grupo abeliano GF(2)^n)
    # ────────────────────────────────────────────────────────────────────

    def _normalize_hash_label(
        self,
        name: str,
        value: Any,
        *,
        normalize_case: bool,
    ) -> Tuple[str, Tuple[str, ...]]:
        """
        Normaliza una etiqueta hash a texto UTF-8, sin espacios laterales.

        Retorna:
            (texto_normalizado, tupla_de_violaciones)
        """
        violations: List[str] = []

        if isinstance(value, bytes):
            try:
                text = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                violations.append(
                    f"{name}: no se pudo decodificar bytes como UTF-8: {exc}"
                )
                text = ""
        elif isinstance(value, str):
            text = value
        else:
            violations.append(
                f"{name}: se esperaba str o bytes, se recibió {type(value).__name__}."
            )
            text = ""

        text = text.strip()

        if normalize_case:
            text = text.lower()

        return text, tuple(violations)

    def _compute_hamming_distance_bits(self, a: str, b: str) -> int:
        r"""
        Distancia de Hamming en el grupo abeliano (GF(2)^N, XOR) sobre la
        codificación UTF-8 de ambos textos.

        d_H(a, b) = |{k : a_k ⊕ b_k = 1}|

        Si las longitudes en bytes difieren, cada byte faltante se penaliza
        con 8 bits de discrepancia máxima, ya que dos cadenas de distinta
        longitud jamás pueden ser el mismo elemento del grupo discreto.

        Este certificado es puramente forense/diagnóstico: la decisión de
        aceptación permanece binaria y se toma exclusivamente vía
        `hmac.compare_digest` (tiempo constante), preservando la doctrina
        "sin tolerancia continua en el espacio discreto de hashes".
        """
        bytes_a = a.encode("utf-8")
        bytes_b = b.encode("utf-8")

        common_len = min(len(bytes_a), len(bytes_b))
        distance = 0

        for i in range(common_len):
            distance += bin(bytes_a[i] ^ bytes_b[i]).count("1")

        length_diff = abs(len(bytes_a) - len(bytes_b))
        distance += length_diff * 8

        return int(distance)

    def _certify_state_fibration_isomorphism(
        self,
        hash_t0: str,
        hash_t: str,
        *,
        require_nonempty_hash: bool = True,
        normalize_hash_case: bool = False,
        raise_on_veto: bool = True,
    ) -> StateFibrationData:
        r"""
        Verifica el difeomorfismo de la carga útil evaluando el residual de
        hashes y su distancia de Hamming en el grupo discreto (GF(2)^n, XOR).

        Contrato:
            hash_t0 == hash_t  ⇔  residual = 0  ⇔  d_H(hash_t0, hash_t) = 0.

        Seguridad:
            La decisión se toma con hmac.compare_digest (tiempo constante).
            La distancia de Hamming se calcula únicamente con fines de
            auditoría forense post-decisión y no participa en el veto.
        """
        h0, violations_t0 = self._normalize_hash_label(
            "hash_t0",
            hash_t0,
            normalize_case=normalize_hash_case,
        )
        ht, violations_t = self._normalize_hash_label(
            "hash_t",
            hash_t,
            normalize_case=normalize_hash_case,
        )

        violations: List[str] = []
        violations.extend(violations_t0)
        violations.extend(violations_t)

        if require_nonempty_hash and (not h0 or not ht):
            violations.append(
                "Hash vacío o no imprimible. Una sesión criptográfica válida "
                "requiere identificadores de estado no vacíos."
            )

        try:
            match = hmac.compare_digest(
                h0.encode("utf-8"),
                ht.encode("utf-8"),
            )
        except Exception as exc:  # pragma: no cover
            violations.append(f"Comparación criptográfica fallida: {exc}")
            match = False

        is_symmetric = bool(match and not violations)

        hamming_distance = self._compute_hamming_distance_bits(h0, ht)
        bit_length_reference = max(len(h0), len(ht)) * 8

        # Certificación cruzada grupo-teórica: si el comparador de tiempo
        # constante certifica igualdad, la distancia de Hamming DEBE ser 0.
        # Cualquier discrepancia es una alarma de inconsistencia algebraica
        # interna (defecto de codificación, no de seguridad).
        if is_symmetric and hamming_distance != 0:
            violations.append(
                "Inconsistencia algebraica: hmac.compare_digest certifica "
                f"igualdad, pero d_H(GF(2)^n) = {hamming_distance} ≠ 0."
            )
            is_symmetric = False

        residual = 0.0 if is_symmetric else 1.0

        audit = StateFibrationData(
            hash_residual=residual,
            is_symmetric=is_symmetric,
            hash_length_t0=len(h0),
            hash_length_t=len(ht),
            constant_time_compared=True,
            hamming_distance_bits=hamming_distance,
            bit_length_reference=bit_length_reference,
            notes=tuple(violations),
        )

        if not is_symmetric and raise_on_veto:
            raise StateSymmetryBreakingError(
                "Ruptura de simetría detectada: el tensor de estado criptográfico "
                "mutó entre t₀ y t. La carga útil de la petición ha sido contaminada."
            )

        return audit

    # ────────────────────────────────────────────────────────────────────
    # §1.3 — Certificación espectral, de inercia y de circuitos (J, R)
    # ────────────────────────────────────────────────────────────────────

    def _spectral_antisymmetry_certificate(self, J: NDArray[np.float64]) -> float:
        r"""
        Teorema espectral para operadores normales antisimétricos reales:
        el espectro de J es puramente imaginario (pares conjugados ±iλ_k) o
        cero. Un residuo de parte real no nula denota fuga disipativa
        parásita mezclada en el tejido conservativo del sistema.

        Retorna:
            max_k |Re(λ_k)|
        """
        if J.shape[0] == 0:
            return 0.0
        eigenvalues = np.linalg.eigvals(J)
        return float(np.max(np.abs(eigenvalues.real))) if eigenvalues.size > 0 else 0.0

    def _sylvester_inertia_certificate(
        self,
        R_sym: NDArray[np.float64],
        tolerance: float,
    ) -> Tuple[int, int, int]:
        r"""
        Calcula la inercia de Sylvester (n₊, n₀, n₋) de la matriz simétrica R.

        Ley de inercia de Sylvester:
            La signatura (n₊, n₀, n₋) es invariante bajo toda transformación
            de congruencia X ↦ AᵀXA con A no singular; es, por tanto, un
            invariante topológico de la forma cuadrática asociada.

        R es semidefinida positiva  ⇔  n₋ = 0.
        """
        if R_sym.shape[0] == 0:
            return (0, 0, 0)
        eigenvalues = np.linalg.eigvalsh(R_sym)
        n_pos = int(np.sum(eigenvalues > tolerance))
        n_neg = int(np.sum(eigenvalues < -tolerance))
        n_zero = int(eigenvalues.size - n_pos - n_neg)
        return (n_pos, n_zero, n_neg)

    def _graph_theoretic_dissipation_topology(
        self,
        R_sym: NDArray[np.float64],
    ) -> Tuple[bool, float, float]:
        r"""
        Interpreta R_sym como matriz de conductancias de una red resistiva
        disipativa (física de circuitos + teoría espectral de grafos):

        - Dominancia diagonal:
              R_ii ≥ Σ_{j≠i} |R_ij|  ∀i
          condición suficiente de M-matriz / red pasiva sin acoplamientos
          parásitos dominantes entre nodos.

        - Conectividad algebraica análoga (valor de Fiedler):
          el segundo autovalor más pequeño de R_sym, en analogía formal con
          el Laplaciano de un grafo, cuantifica la conectividad del tejido
          disipativo (0 ⇒ posible desacoplo en componentes independientes).

        Retorna:
            (es_diagonalmente_dominante, margen_de_dominancia, conectividad_análoga)
        """
        n = R_sym.shape[0]
        if n == 0:
            return True, 0.0, 0.0

        diag = np.diag(R_sym)
        off_diag_abs_sum = np.sum(np.abs(R_sym), axis=1) - np.abs(diag)
        dominance_margin = float(np.min(diag - off_diag_abs_sum))
        is_diagonally_dominant = bool(dominance_margin >= -_BASE_TOLERANCE)

        eigenvalues = np.sort(np.linalg.eigvalsh(R_sym))
        algebraic_connectivity_analog = (
            float(eigenvalues[1]) if eigenvalues.size > 1 else float(eigenvalues[0])
        )

        return is_diagonally_dominant, dominance_margin, algebraic_connectivity_analog

    def _enforce_active_impedance_control(
        self,
        grad_H: NDArray[Any],
        J_matrix: NDArray[Any],
        R_matrix: NDArray[Any],
        exogenous_work_gu: float,
        *,
        dissipation_tolerance: float = _DISSIPATION_TOLERANCE,
        psd_tolerance: float = _PSD_TOLERANCE,
        raise_on_veto: bool = True,
    ) -> ImpedanceControlData:
        r"""
        Computa la disipación térmica y veta frentes de onda que inyecten
        energía superior a la resistencia R(x) del servidor, certificando
        estructura tanto por norma como por espectro, inercia y topología
        de circuitos.

        Veto:
            Ḣ > tol ⇒ ResonanceCatastropheVeto.
        """
        grad = self._as_finite_vector("grad_H", grad_H)
        n = int(grad.size)

        J = self._as_finite_square_matrix("J_matrix", J_matrix, n)
        R = self._as_finite_square_matrix("R_matrix", R_matrix, n)

        try:
            work = float(exogenous_work_gu)
        except Exception as exc:
            raise ImpedanceInputError(
                f"exogenous_work_gu debe ser un escalar numérico finito. "
                f"Se recibió: {exogenous_work_gu!r}."
            ) from exc

        if not np.isfinite(work):
            raise ImpedanceInputError(
                f"exogenous_work_gu no es finito: {work}."
            )

        j_norm = float(la.norm(J, ord=np.inf)) if n > 0 else 0.0
        r_norm = float(la.norm(R, ord=np.inf)) if n > 0 else 0.0

        tol_j = self._operator_tolerance(n, j_norm)
        tol_r = self._operator_tolerance(n, r_norm)

        antisymmetry_residual = (
            float(la.norm(J + J.T, ord=np.inf)) if n > 0 else 0.0
        )
        symmetry_residual = (
            float(la.norm(R - R.T, ord=np.inf)) if n > 0 else 0.0
        )

        j_spectral_max_real_part = self._spectral_antisymmetry_certificate(J)

        R_sym = 0.5 * (R + R.T)

        if n > 0:
            eigenvalues = np.linalg.eigvalsh(R_sym)
            min_eigenvalue = (
                float(np.min(eigenvalues)) if eigenvalues.size > 0 else 0.0
            )
        else:
            min_eigenvalue = 0.0

        psd_tol = max(float(psd_tolerance), tol_r)

        n_pos, n_zero, n_neg = self._sylvester_inertia_certificate(R_sym, psd_tol)
        (
            is_diagonally_dominant,
            _dominance_margin,
            algebraic_connectivity_analog,
        ) = self._graph_theoretic_dissipation_topology(R_sym)

        dissipation_raw = float(grad @ R_sym @ grad) if n > 0 else 0.0

        violations: List[str] = []
        structural_valid = True

        if antisymmetry_residual > tol_j:
            structural_valid = False
            violations.append(
                f"J no es antisimétrica: ||J + J^T||_∞ = "
                f"{antisymmetry_residual:.6e} > tol={tol_j:.6e}."
            )

        if j_spectral_max_real_part > tol_j:
            structural_valid = False
            violations.append(
                f"Espectro de J no es puramente imaginario: "
                f"max|Re(λ_k)| = {j_spectral_max_real_part:.6e} > tol={tol_j:.6e}."
            )

        if symmetry_residual > tol_r:
            structural_valid = False
            violations.append(
                f"R no es simétrica: ||R - R^T||_∞ = "
                f"{symmetry_residual:.6e} > tol={tol_r:.6e}."
            )

        if min_eigenvalue < -psd_tol:
            structural_valid = False
            violations.append(
                f"R no es semidefinida positiva: λ_min(R_sym) = "
                f"{min_eigenvalue:.6e} < -tol={-psd_tol:.6e}."
            )

        if n_neg > 0:
            structural_valid = False
            violations.append(
                f"Inercia de Sylvester inválida: n₋ = {n_neg} > 0 "
                f"(signatura ({n_pos}, {n_zero}, {n_neg}))."
            )

        if dissipation_raw < -psd_tol:
            structural_valid = False
            violations.append(
                f"Disipación cuadrática negativa: ∇H^T R_sym ∇H = "
                f"{dissipation_raw:.6e} < -tol={-psd_tol:.6e}."
            )

        dissipation_term = float(dissipation_raw)
        if dissipation_term < 0.0 and dissipation_term >= -psd_tol:
            dissipation_term = 0.0

        h_dot = -dissipation_term + work

        h_tolerance = float(
            max(
                float(dissipation_tolerance),
                _BASE_TOLERANCE * max(1.0, abs(dissipation_term), abs(work)),
            )
        )

        is_passively_stable = bool(structural_valid and h_dot <= h_tolerance)
        is_strictly_dissipative = bool(is_passively_stable and h_dot < -h_tolerance)
        is_asymptotically_stable = bool(is_strictly_dissipative)

        audit = ImpedanceControlData(
            dimension=n,
            h_dot=float(h_dot),
            dissipation_term=float(dissipation_term),
            exogenous_work_gu=float(work),
            tolerance=h_tolerance,
            j_antisymmetry_residual=antisymmetry_residual,
            j_spectral_max_real_part=j_spectral_max_real_part,
            r_symmetry_residual=symmetry_residual,
            r_min_eigenvalue=min_eigenvalue,
            r_inertia_positive=n_pos,
            r_inertia_zero=n_zero,
            r_inertia_negative=n_neg,
            r_diagonally_dominant=is_diagonally_dominant,
            r_algebraic_connectivity_analog=algebraic_connectivity_analog,
            structural_valid=structural_valid,
            is_passively_stable=is_passively_stable,
            is_strictly_dissipative=is_strictly_dissipative,
            is_asymptotically_stable=is_asymptotically_stable,
            notes=tuple(violations),
        )

        if not is_passively_stable and raise_on_veto:
            if violations:
                detail = " | ".join(violations)
                raise ResonanceCatastropheVeto(
                    f"Catástrofe de resonancia o estructura Port-Hamiltoniana inválida: {detail}"
                )

            raise ResonanceCatastropheVeto(
                f"Catástrofe de resonancia (Rate Limit excedido): "
                f"Ḣ = {h_dot:.6e} > tol={h_tolerance:.6e}."
            )

        return audit

    # ────────────────────────────────────────────────────────────────────
    # §1.4 — Puente terminal hacia Fase 2 (stub + composición)
    # ────────────────────────────────────────────────────────────────────

    def _instantiate_thermodynamic_passport(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ThermodynamicPassportData:
        """
        Stub formal de continuación hacia Fase 2.

        La implementación real vive en Phase2_CategoricalDIKWClosureInstantiator.
        """
        raise NotImplementedError(
            "Phase 2 must be mixed in. "
            "Este método es la continuación formal hacia el cierre categórico DIKW."
        )

    def _phase1_terminal_bridge_to_phase2(
        self,
        hash_t0: str,
        hash_t: str,
        grad_H: NDArray[Any],
        J_matrix: NDArray[Any],
        R_matrix: NDArray[Any],
        exogenous_work_gu: float,
        req_physics: bool,
        req_tactics: bool,
        req_strategy: bool,
        req_wisdom: bool,
        *,
        require_nonempty_hash: bool = True,
        normalize_hash_case: bool = False,
        dissipation_tolerance: float = _DISSIPATION_TOLERANCE,
        psd_tolerance: float = _PSD_TOLERANCE,
        allow_empty_request: bool = False,
        raise_on_veto: bool = True,
    ) -> Tuple[StateFibrationData, ImpedanceControlData, ThermodynamicPassportData]:
        """
        Último método de Fase 1: puente funtorial hacia Fase 2.

        Composición:
            Φ₁(hash, grad_H, J, R, gu) → (StateFibrationData, ImpedanceControlData)
            Φ₁ ▷ Φ₂(DIKW) → (StateFibrationData, ImpedanceControlData,
                              ThermodynamicPassportData)
        """
        fibration_audit = self._certify_state_fibration_isomorphism(
            hash_t0,
            hash_t,
            require_nonempty_hash=require_nonempty_hash,
            normalize_hash_case=normalize_hash_case,
            raise_on_veto=raise_on_veto,
        )

        impedance_audit = self._enforce_active_impedance_control(
            grad_H,
            J_matrix,
            R_matrix,
            exogenous_work_gu,
            dissipation_tolerance=dissipation_tolerance,
            psd_tolerance=psd_tolerance,
            raise_on_veto=raise_on_veto,
        )

        passport_audit = self._instantiate_thermodynamic_passport(
            req_physics,
            req_tactics,
            req_strategy,
            req_wisdom,
            allow_empty_request=allow_empty_request,
            raise_on_veto=raise_on_veto,
        )

        return fibration_audit, impedance_audit, passport_audit


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2: CIERRE CATEGÓRICO-TOPOLÓGICO DEL POSET DE CUSTODIA DIKW             ║
# ║ Clausura transitiva booleana (Warshall) + axiomas de orden parcial          ║
# ║ (categoría delgada).                                                        ║
# ║                                                                             ║
# ║ El último método de esta fase es el puente formal hacia Fase 3.             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase2_CategoricalDIKWClosureInstantiator(Phase1_SpectralFibrationImpedanceCertifier):
    r"""
    Asegura que la petición API solicite y pre-reserve la clausura geométrica
    correcta de los estratos DIKW antes de inyectarse en el núcleo.

    Modelo formal:
        El poset (Σ, ≤) con Σ = {PHYSICS, TACTICS, STRATEGY, WISDOM} es una
        **categoría delgada**: existe a lo sumo un morfismo i → j y este
        existe sii i ≤ j (i.e., j requiere a i). La relación "requiere" se
        codifica como grafo dirigido de adyacencia inmediata y su clausura
        transitiva se obtiene mediante el **algoritmo de Warshall** sobre el
        semianillo booleano (∨, ∧), certificando explícitamente los tres
        axiomas de orden parcial.
    """

    # ────────────────────────────────────────────────────────────────────
    # §2.1 — Álgebra de Boole sobre grafos: clausura transitiva de Warshall
    # ────────────────────────────────────────────────────────────────────

    def _warshall_boolean_transitive_closure(
        self,
        adjacency: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        r"""
        Calcula la clausura transitiva de una relación binaria mediante el
        algoritmo de Warshall sobre el semianillo booleano (∨, ∧):

            cl[i,j] ← cl[i,j] ∨ (cl[i,k] ∧ cl[k,j])   ∀ k = 0..n-1

        Este operador de clausura es monótono, extensivo e idempotente
        (cl∘cl = cl), propiedades que se certifican explícitamente en
        `_certify_poset_axioms`.
        """
        closure = adjacency.copy()
        n = closure.shape[0]

        for k in range(n):
            closure = closure | (closure[:, [k]] & closure[[k], :])

        return closure

    def _certify_poset_axioms(
        self,
        closure: NDArray[np.bool_],
    ) -> Tuple[bool, bool, bool]:
        r"""
        Certifica que `closure` define efectivamente un orden parcial
        (categoría delgada):

            1) Reflexividad:  ∀i, (i,i) ∈ ≤
            2) Antisimetría:  (i,j) ∈ ≤ ∧ (j,i) ∈ ≤ ∧ i≠j  ⇒  contradicción
            3) Transitividad: cl(cl(R)) = cl(R)  (idempotencia del operador
               de clausura — forma canónica de exigir transitividad sin
               enumerar explícitamente triples (i,j,k)).

        Retorna:
            (es_reflexiva, es_antisimétrica, es_transitiva)
        """
        n = closure.shape[0]

        is_reflexive = bool(np.all(np.diag(closure)))

        mutual = closure & closure.T
        np.fill_diagonal(mutual, False)
        is_antisymmetric = not bool(np.any(mutual))

        re_closure = self._warshall_boolean_transitive_closure(closure)
        is_transitive = bool(np.array_equal(re_closure, closure))

        return is_reflexive, is_antisymmetric, is_transitive

    # ────────────────────────────────────────────────────────────────────
    # §2.2 — Instanciación del pasaporte termodinámico (implementación real)
    # ────────────────────────────────────────────────────────────────────

    def _instantiate_thermodynamic_passport(
        self,
        req_physics: bool,
        req_tactics: bool,
        req_strategy: bool,
        req_wisdom: bool,
        *,
        allow_empty_request: bool = False,
        raise_on_veto: bool = True,
    ) -> ThermodynamicPassportData:
        r"""
        Fuerza la monotonicidad de los subespacios métricos en el
        TelemetryContext usando la clausura transitiva booleana del poset
        DIKW en lugar de un rango hardcodeado.
        """
        closure = self._warshall_boolean_transitive_closure(_DIKW_IMMEDIATE_ADJACENCY)
        reflexive, antisymmetric, transitive = self._certify_poset_axioms(closure)
        is_valid_poset_structure = bool(reflexive and antisymmetric and transitive)

        if not is_valid_poset_structure:  # pragma: no cover — red de seguridad estructural
            raise TopologicalPassportError(
                "El poset DIKW estructural (constante del módulo) viola axiomas "
                f"de orden parcial: reflexiva={reflexive}, "
                f"antisimétrica={antisymmetric}, transitiva={transitive}."
            )

        requested_flags: Tuple[bool, ...] = (
            bool(req_physics),
            bool(req_tactics),
            bool(req_strategy),
            bool(req_wisdom),
        )

        requested_levels: List[int] = [
            level for level, active in enumerate(requested_flags) if active
        ]

        violations: List[str] = []

        if not requested_levels:
            if allow_empty_request:
                return ThermodynamicPassportData(
                    assigned_stratum_level=-1,
                    requested_levels=(),
                    missing_lower_levels=(),
                    is_transitively_closed=True,
                    is_valid_poset_structure=True,
                    notes=(
                        "Petición vacía permitida por allow_empty_request=True. "
                        "Ningún estrato DIKW fue solicitado.",
                    ),
                )

            violations.append(
                "Ningún estrato DIKW fue solicitado. Una petición de frontera "
                "válida debe requerir al menos PHYSICS, salvo allow_empty_request=True."
            )

            audit = ThermodynamicPassportData(
                assigned_stratum_level=-1,
                requested_levels=(),
                missing_lower_levels=(),
                is_transitively_closed=False,
                is_valid_poset_structure=True,
                notes=tuple(violations),
            )

            if raise_on_veto:
                raise TopologicalPassportError(
                    "Petición DIKW vacía: se requiere al menos un estrato basal."
                )

            return audit

        assigned_level = max(requested_levels)
        requested_set = set(requested_levels)

        required_set = {
            j for j in range(len(_DIKW_LEVEL_NAMES)) if bool(closure[assigned_level, j])
        }
        missing_lower_levels: Tuple[int, ...] = tuple(
            sorted(required_set - requested_set)
        )

        is_transitively_closed = len(missing_lower_levels) == 0

        if not is_transitively_closed:
            missing_names = ", ".join(
                _DIKW_LEVEL_NAMES[level] for level in missing_lower_levels
            )
            violations.append(
                f"Violación de clausura transitiva categórica (poset DIKW): el "
                f"estrato {_DIKW_LEVEL_NAMES[assigned_level]} fue solicitado sin "
                f"los estratos requeridos por la clausura de Warshall: {missing_names}."
            )

        audit = ThermodynamicPassportData(
            assigned_stratum_level=assigned_level,
            requested_levels=tuple(sorted(requested_levels)),
            missing_lower_levels=missing_lower_levels,
            is_transitively_closed=is_transitively_closed,
            is_valid_poset_structure=is_valid_poset_structure,
            notes=tuple(violations),
        )

        if not is_transitively_closed and raise_on_veto:
            raise TopologicalPassportError(
                "Inconsistencia en fibrado DIKW: la petición intentó invocar un "
                "estrato superior sin asegurar la estabilidad de la variedad basal."
            )

        return audit

    # ────────────────────────────────────────────────────────────────────
    # §2.3 — Puente terminal hacia Fase 3 (stub + composición)
    # ────────────────────────────────────────────────────────────────────

    def _project_intention_to_mic_basis(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> OrthogonalProjectionData:
        """
        Stub formal de continuación hacia Fase 3.

        La implementación real vive en Phase3_QuantumBooleanMICProjectorSynthesizer.
        """
        raise NotImplementedError(
            "Phase 3 must be mixed in. "
            "Este método es la continuación formal hacia la proyección cuántica MIC."
        )

    def _phase2_terminal_bridge_to_phase3(
        self,
        hash_t0: str,
        hash_t: str,
        grad_H: NDArray[Any],
        J_matrix: NDArray[Any],
        R_matrix: NDArray[Any],
        exogenous_work_gu: float,
        req_physics: bool,
        req_tactics: bool,
        req_strategy: bool,
        req_wisdom: bool,
        intent_x: NDArray[Any],
        projector_Pi: NDArray[Any],
        other_projectors: Optional[Sequence[NDArray[Any]]],
        *,
        require_nonempty_hash: bool = True,
        normalize_hash_case: bool = False,
        dissipation_tolerance: float = _DISSIPATION_TOLERANCE,
        psd_tolerance: float = _PSD_TOLERANCE,
        allow_empty_request: bool = False,
        orthogonality_tolerance: float = _ORTHOGONALITY_TOLERANCE,
        check_operator_orthogonality: bool = True,
        require_hermitian_projectors: bool = True,
        raise_on_veto: bool = True,
    ) -> Tuple[
        StateFibrationData,
        ImpedanceControlData,
        ThermodynamicPassportData,
        OrthogonalProjectionData,
    ]:
        """
        Último método de Fase 2: puente funtorial hacia Fase 3.

        Composición:
            Φ₂ ∘ Φ₁ → (StateFibrationData, ImpedanceControlData,
                       ThermodynamicPassportData)
            (Φ₂ ∘ Φ₁) ▷ Φ₃(x, P_i, {P_j}) → (..., OrthogonalProjectionData)
        """
        (
            fibration_audit,
            impedance_audit,
            passport_audit,
        ) = self._phase1_terminal_bridge_to_phase2(
            hash_t0,
            hash_t,
            grad_H,
            J_matrix,
            R_matrix,
            exogenous_work_gu,
            req_physics,
            req_tactics,
            req_strategy,
            req_wisdom,
            require_nonempty_hash=require_nonempty_hash,
            normalize_hash_case=normalize_hash_case,
            dissipation_tolerance=dissipation_tolerance,
            psd_tolerance=psd_tolerance,
            allow_empty_request=allow_empty_request,
            raise_on_veto=raise_on_veto,
        )

        projection_audit = self._project_intention_to_mic_basis(
            intent_x,
            projector_Pi,
            other_projectors,
            orthogonality_tolerance=orthogonality_tolerance,
            check_operator_orthogonality=check_operator_orthogonality,
            require_hermitian_projectors=require_hermitian_projectors,
            raise_on_veto=raise_on_veto,
        )

        return fibration_audit, impedance_audit, passport_audit, projection_audit


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3: PROYECCIÓN CUÁNTICA-BOOLEANA A LA BASE CANÓNICA MIC                 ║
# ║ + SÍNTESIS DE CANCELACIÓN DE ANOMALÍA GLOBAL                               ║
# ║                                                                            ║
# ║ El último método de esta fase sintetiza el objeto final de gobernanza.     ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase3_QuantumBooleanMICProjectorSynthesizer(Phase2_CategoricalDIKWClosureInstantiator):
    r"""
    Asegura que el Tool Dispatcher dirija la petición a un vector ortogonal
    puro de la Matriz de Interacción Central (MIC), certificado con el
    aparato completo de la lógica cuántica de proyecciones:

        P_i² = P_i                 (idempotencia)
        P_i = P_iᵀ                 (hermiticidad real / simetría)
        σ(P_i) ⊆ {0,1}             (teorema espectral de proyectores)
        P_i P_j = 0    (i≠j)       (ortogonalidad de operadores — "meet" nulo)
        P_i ∨ P_j idempotente      (axioma de "join" en el reticulado booleano)
        ⟨P_i x, P_j x⟩ = 0         (ortogonalidad de estados proyectados)
        p_i = ‖P_i x‖² / ‖x‖²      (regla de Born, probabilidad de colapso)

    y finalmente sintetiza el veredicto de frontera mediante una condición
    de **cancelación de anomalía global**, análoga formal al mecanismo de
    Green-Schwarz en teoría de cuerdas.
    """

    # ────────────────────────────────────────────────────────────────────
    # §3.1 — Certificados cuánticos elementales de un proyector
    # ────────────────────────────────────────────────────────────────────

    def _certify_projector_spectral_binary(
        self,
        Pi: NDArray[np.float64],
        tolerance: float,
    ) -> float:
        r"""
        Teorema espectral para proyectores ortogonales hermíticos:
            σ(P) ⊆ {0, 1}.

        Retorna el residuo espectral máximo respecto al espectro binario
        admisible:
            max_k min(|λ_k|, |λ_k - 1|)
        """
        n = Pi.shape[0]
        if n == 0:
            return 0.0

        Pi_sym = 0.5 * (Pi + Pi.T)
        eigenvalues = np.linalg.eigvalsh(Pi_sym)
        residual_to_zero = np.abs(eigenvalues)
        residual_to_one = np.abs(eigenvalues - 1.0)
        residual = np.minimum(residual_to_zero, residual_to_one)

        return float(np.max(residual)) if residual.size > 0 else 0.0

    def _certify_boolean_lattice_pair(
        self,
        Pi: NDArray[np.float64],
        Pj: NDArray[np.float64],
    ) -> Tuple[float, float]:
        r"""
        Certifica los axiomas de reticulado ortocomplementado (lógica
        cuántica de proyecciones de von Neumann-Birkhoff) para el par
        (P_i, P_j), bajo la hipótesis de conmutatividad garantizada por la
        ortogonalidad de operador:

            Meet:  P_i ∧ P_j := P_i P_j             (≈ 0 si son disjuntos)
            Join:  P_i ∨ P_j := P_i + P_j - P_iP_j  (debe ser idempotente)

        Retorna:
            (residuo_de_meet, residuo_de_idempotencia_del_join)
        """
        n = Pi.shape[0]
        if n == 0:
            return 0.0, 0.0

        meet = Pi @ Pj
        meet_residual = float(la.norm(meet, ord=np.inf))

        join = Pi + Pj - meet
        join_squared = join @ join
        join_idempotence_residual = float(la.norm(join_squared - join, ord=np.inf))

        return meet_residual, join_idempotence_residual

    def _compute_born_rule_probability(
        self,
        projected_norm: float,
        x_norm: float,
    ) -> float:
        r"""
        Regla de Born: la probabilidad de que una medición de von Neumann
        sobre el vector de intención |x⟩ colapse en el subespacio dirigido
        por P_i es:

            p_i = ‖P_i x‖² / ‖x‖²   ,   p_i ∈ [0, 1]
        """
        if x_norm <= 0.0:
            return 0.0
        probability = (projected_norm ** 2) / (x_norm ** 2)
        return float(min(max(probability, 0.0), 1.0))

    # ────────────────────────────────────────────────────────────────────
    # §3.2 — Proyección ortogonal MIC (implementación real, Fase 3)
    # ────────────────────────────────────────────────────────────────────

    def _project_intention_to_mic_basis(
        self,
        intent_x: NDArray[Any],
        projector_Pi: NDArray[Any],
        other_projectors: Optional[Sequence[NDArray[Any]]],
        *,
        orthogonality_tolerance: float = _ORTHOGONALITY_TOLERANCE,
        check_operator_orthogonality: bool = True,
        require_hermitian_projectors: bool = True,
        raise_on_veto: bool = True,
    ) -> OrthogonalProjectionData:
        r"""
        Verifica la idempotencia, hermiticidad, espectro binario y
        producto interno nulo del proyector P_i contra las bases
        ortogonales restantes, junto con los axiomas de reticulado
        booleano y la probabilidad de colapso de Born.
        """
        x = self._as_finite_vector(
            "intent_x", intent_x, exception_cls=OrthogonalProjectionInputError
        )
        n = int(x.size)

        Pi = self._as_finite_square_matrix(
            "projector_Pi", projector_Pi, n, exception_cls=OrthogonalProjectionInputError
        )

        if other_projectors is None:
            others: List[NDArray[Any]] = []
        else:
            try:
                others = list(other_projectors)
            except TypeError as exc:
                raise OrthogonalProjectionInputError(
                    "other_projectors debe ser una secuencia de matrices."
                ) from exc

        pi_norm = float(la.norm(Pi, ord=np.inf)) if n > 0 else 0.0
        x_norm = float(np.linalg.norm(x)) if n > 0 else 0.0

        scale = max(1.0, pi_norm, x_norm * x_norm)
        tolerance = float(
            max(
                float(orthogonality_tolerance),
                self._operator_tolerance(n, scale, _ORTHOGONALITY_TOLERANCE),
            )
        )

        violations: List[str] = []

        Pi_squared = Pi @ Pi if n > 0 else Pi.copy()
        idempotence_residual = (
            float(la.norm(Pi_squared - Pi, ord=np.inf)) if n > 0 else 0.0
        )

        if idempotence_residual > tolerance:
            violations.append(
                f"Proyector P_i no idempotente: ||P_i² - P_i||_∞ = "
                f"{idempotence_residual:.6e} > tol={tolerance:.6e}."
            )

        symmetry_residual = 0.0
        if require_hermitian_projectors:
            symmetry_residual = (
                float(la.norm(Pi - Pi.T, ord=np.inf)) if n > 0 else 0.0
            )
            if symmetry_residual > tolerance:
                violations.append(
                    f"Proyector P_i no simétrico: ||P_i - P_i^T||_∞ = "
                    f"{symmetry_residual:.6e} > tol={tolerance:.6e}."
                )

        spectral_binary_residual = self._certify_projector_spectral_binary(Pi, tolerance)
        if spectral_binary_residual > tolerance:
            violations.append(
                f"Espectro de P_i fuera de {{0,1}} (teorema espectral de "
                f"proyectores): residuo = {spectral_binary_residual:.6e} > "
                f"tol={tolerance:.6e}."
            )

        Pi_x = Pi @ x if n > 0 else np.asarray([], dtype=np.float64)
        projected_norm = float(np.linalg.norm(Pi_x)) if Pi_x.size > 0 else 0.0
        born_rule_probability = self._compute_born_rule_probability(projected_norm, x_norm)

        operator_orthogonality_residual = 0.0
        state_orthogonality_residual = 0.0
        boolean_meet_residual = 0.0
        boolean_join_idempotence_residual = 0.0

        for idx, Pj_raw in enumerate(others):
            Pj_name = f"other_projectors[{idx}]"
            Pj = self._as_finite_square_matrix(
                Pj_name, Pj_raw, n, exception_cls=OrthogonalProjectionInputError
            )

            Pj_squared = Pj @ Pj if n > 0 else Pj.copy()
            Pj_idempotence_residual = (
                float(la.norm(Pj_squared - Pj, ord=np.inf)) if n > 0 else 0.0
            )

            if Pj_idempotence_residual > tolerance:
                violations.append(
                    f"{Pj_name} no idempotente: ||P_j² - P_j||_∞ = "
                    f"{Pj_idempotence_residual:.6e} > tol={tolerance:.6e}."
                )

            if require_hermitian_projectors:
                Pj_symmetry_residual = (
                    float(la.norm(Pj - Pj.T, ord=np.inf)) if n > 0 else 0.0
                )
                if Pj_symmetry_residual > tolerance:
                    violations.append(
                        f"{Pj_name} no simétrico: ||P_j - P_j^T||_∞ = "
                        f"{Pj_symmetry_residual:.6e} > tol={tolerance:.6e}."
                    )

            if check_operator_orthogonality:
                op_ij = float(la.norm(Pi @ Pj, ord=np.inf)) if n > 0 else 0.0
                op_ji = float(la.norm(Pj @ Pi, ord=np.inf)) if n > 0 else 0.0
                operator_residual = max(op_ij, op_ji)

                operator_orthogonality_residual = max(
                    operator_orthogonality_residual,
                    operator_residual,
                )

                if operator_residual > tolerance:
                    violations.append(
                        f"{Pj_name} no ortogonal a P_i a nivel de operador: "
                        f"max(||P_iP_j||∞, ||P_jP_i||∞) = "
                        f"{operator_residual:.6e} > tol={tolerance:.6e}."
                    )

            meet_residual, join_idempotence_residual = self._certify_boolean_lattice_pair(
                Pi, Pj
            )
            boolean_meet_residual = max(boolean_meet_residual, meet_residual)
            boolean_join_idempotence_residual = max(
                boolean_join_idempotence_residual, join_idempotence_residual
            )

            if join_idempotence_residual > tolerance:
                violations.append(
                    f"{Pj_name}: el 'join' booleano P_i∨P_j no es idempotente "
                    f"(reticulado ortocomplementado roto): residuo = "
                    f"{join_idempotence_residual:.6e} > tol={tolerance:.6e}."
                )

            Pj_x = Pj @ x if n > 0 else np.asarray([], dtype=np.float64)
            Pj_x_norm = float(np.linalg.norm(Pj_x)) if Pj_x.size > 0 else 0.0

            inner_product = (
                float(np.abs(np.vdot(Pi_x, Pj_x)))
                if Pi_x.size > 0 and Pj_x.size > 0
                else 0.0
            )

            inner_scale = max(1.0, projected_norm * Pj_x_norm)
            inner_tolerance = float(
                max(
                    float(orthogonality_tolerance),
                    100.0 * max(1, n) * _MACHINE_EPSILON * inner_scale,
                )
            )

            state_orthogonality_residual = max(
                state_orthogonality_residual,
                inner_product,
            )

            if inner_product > inner_tolerance:
                violations.append(
                    f"{Pj_name} produce superposición parásita sobre x: "
                    f"|⟨P_i x, P_j x⟩| = {inner_product:.6e} > "
                    f"tol={inner_tolerance:.6e}."
                )

        projector_rank: Optional[int]
        try:
            projector_rank = (
                int(np.linalg.matrix_rank(Pi, tol=tolerance))
                if n > 0
                else 0
            )
        except Exception:  # pragma: no cover
            projector_rank = None

        is_mutually_orthogonal = len(violations) == 0

        audit = OrthogonalProjectionData(
            intent_dimension=n,
            projector_rank=projector_rank,
            idempotence_residual=idempotence_residual,
            symmetry_residual=symmetry_residual,
            spectral_binary_residual=spectral_binary_residual,
            operator_orthogonality_residual=operator_orthogonality_residual,
            state_orthogonality_residual=state_orthogonality_residual,
            boolean_meet_residual=boolean_meet_residual,
            boolean_join_idempotence_residual=boolean_join_idempotence_residual,
            born_rule_probability=born_rule_probability,
            projected_norm=projected_norm,
            tolerance=tolerance,
            is_mutually_orthogonal=is_mutually_orthogonal,
            notes=tuple(violations),
        )

        if not is_mutually_orthogonal and raise_on_veto:
            detail = " | ".join(violations)
            raise OrthogonalityViolationError(
                f"Colapso de aislamiento funcional: {detail}"
            )

        return audit

    # ────────────────────────────────────────────────────────────────────
    # §3.3 — Cancelación de anomalía global (analogía Green-Schwarz)
    # ────────────────────────────────────────────────────────────────────

    def _certify_global_anomaly_cancellation(
        self,
        fibration_audit: StateFibrationData,
        impedance_audit: ImpedanceControlData,
        passport_audit: ThermodynamicPassportData,
        projection_audit: OrthogonalProjectionData,
    ) -> Tuple[int, Tuple[str, ...]]:
        r"""
        Condición de cancelación global de anomalías, análoga formal al
        mecanismo de Green-Schwarz en teoría de cuerdas: la consistencia de
        una teoría de gauge (o, aquí, de una frontera ciber-física) exige
        que la suma de las contribuciones de anomalía de cada sector se
        anule idénticamente.

        Cada fase certificada aporta una "carga de anomalía" binaria
        (0 si es consistente, 1 si no lo es); el vacío de frontera solo es
        estable si el índice total Σ q_sector = 0.
        """
        contributions: List[Tuple[str, int]] = [
            (
                "Φ₁:Fibración-Impedancia",
                0 if (fibration_audit.is_symmetric and impedance_audit.is_passively_stable) else 1,
            ),
            (
                "Φ₂:Clausura-DIKW",
                0 if passport_audit.is_transitively_closed else 1,
            ),
            (
                "Φ₃:Proyección-MIC",
                0 if projection_audit.is_mutually_orthogonal else 1,
            ),
        ]

        anomaly_index = sum(charge for _name, charge in contributions)
        failing_sectors = tuple(name for name, charge in contributions if charge != 0)

        return anomaly_index, failing_sectors

    # ────────────────────────────────────────────────────────────────────
    # §3.4 — Síntesis terminal del objeto de gobernanza
    # ────────────────────────────────────────────────────────────────────

    def _phase3_terminal_synthesis(
        self,
        hash_t0: str,
        hash_t: str,
        grad_H: NDArray[Any],
        J_matrix: NDArray[Any],
        R_matrix: NDArray[Any],
        exogenous_work_gu: float,
        req_physics: bool,
        req_tactics: bool,
        req_strategy: bool,
        req_wisdom: bool,
        intent_x: NDArray[Any],
        projector_Pi: NDArray[Any],
        other_projectors: Optional[Sequence[NDArray[Any]]],
        *,
        require_nonempty_hash: bool = True,
        normalize_hash_case: bool = False,
        dissipation_tolerance: float = _DISSIPATION_TOLERANCE,
        psd_tolerance: float = _PSD_TOLERANCE,
        allow_empty_request: bool = False,
        orthogonality_tolerance: float = _ORTHOGONALITY_TOLERANCE,
        check_operator_orthogonality: bool = True,
        require_hermitian_projectors: bool = True,
        raise_on_veto: bool = True,
    ) -> GatewayGovernanceState:
        r"""
        Último método de Fase 3: síntesis del objeto final de gobernanza.

        Composición final:
            Z_Gateway = Φ₃ ∘ Φ₂ ∘ Φ₁

        La decisión de seguridad no se toma como un simple AND de
        booleanos, sino como una **condición de cancelación de anomalía
        global**: is_gateway_secure ⇔ anomaly_index = 0.
        """
        (
            fibration_audit,
            impedance_audit,
            passport_audit,
            projection_audit,
        ) = self._phase2_terminal_bridge_to_phase3(
            hash_t0,
            hash_t,
            grad_H,
            J_matrix,
            R_matrix,
            exogenous_work_gu,
            req_physics,
            req_tactics,
            req_strategy,
            req_wisdom,
            intent_x,
            projector_Pi,
            other_projectors,
            require_nonempty_hash=require_nonempty_hash,
            normalize_hash_case=normalize_hash_case,
            dissipation_tolerance=dissipation_tolerance,
            psd_tolerance=psd_tolerance,
            allow_empty_request=allow_empty_request,
            orthogonality_tolerance=orthogonality_tolerance,
            check_operator_orthogonality=check_operator_orthogonality,
            require_hermitian_projectors=require_hermitian_projectors,
            raise_on_veto=raise_on_veto,
        )

        anomaly_index, failing_sectors = self._certify_global_anomaly_cancellation(
            fibration_audit,
            impedance_audit,
            passport_audit,
            projection_audit,
        )

        is_gateway_secure = bool(anomaly_index == 0)

        if not is_gateway_secure and raise_on_veto:
            detail = ", ".join(failing_sectors)
            raise GlobalAnomalyCancellationError(
                f"Frontera ciber-física insegura: índice de anomalía global = "
                f"{anomaly_index} ≠ 0. Sectores en falta de consistencia: {detail}."
            )

        state = GatewayGovernanceState(
            governance_id=str(uuid.uuid4()),
            fibration_audit=fibration_audit,
            impedance_audit=impedance_audit,
            passport_audit=passport_audit,
            projection_audit=projection_audit,
            anomaly_index=anomaly_index,
            anomaly_failing_sectors=failing_sectors,
            is_gateway_secure=is_gateway_secure,
            generated_at_utc=_utc_timestamp(),
        )

        logger.info(
            "Frontera Ciber-Física (app.py) auditada. "
            "id=%s | hash_ok=%s | d_H=%d bits | Ḣ=%.6e | passively_stable=%s | "
            "R_inertia=(%d,%d,%d) | DIKW_level=%d | orthogonality_ok=%s | "
            "born_p=%.4f | anomaly_index=%d | secure=%s",
            state.governance_id,
            fibration_audit.is_symmetric,
            fibration_audit.hamming_distance_bits,
            impedance_audit.h_dot,
            impedance_audit.is_passively_stable,
            impedance_audit.r_inertia_positive,
            impedance_audit.r_inertia_zero,
            impedance_audit.r_inertia_negative,
            passport_audit.assigned_stratum_level,
            projection_audit.is_mutually_orthogonal,
            projection_audit.born_rule_probability,
            anomaly_index,
            state.is_gateway_secure,
        )

        return state


# ══════════════════════════════════════════════════════════════════════════════
# §D. ORQUESTADOR SUPREMO: APP AGENT (VARIEDAD DE FRONTERA)
# ══════════════════════════════════════════════════════════════════════════════

class AppAgent(Morphism, Phase3_QuantumBooleanMICProjectorSynthesizer):
    r"""
    El Custodio de la Variedad de Frontera Ciber-Física.

    Actúa en el punto de ingesta (app.py) para someter toda petición entrante
    a la composición funtorial de tres fases:

        Φ₁ : Isomorfismo criptográfico de estado + disipación Port-Hamiltoniana
              (grupo GF(2)^n, teoría espectral, inercia de Sylvester, circuitos)
        Φ₂ : Clausura transitiva categórica del poset DIKW
              (teoría de grafos + álgebra de Boole + teoría de categorías)
        Φ₃ : Proyección ortogonal cuántica-booleana MIC + cancelación de
              anomalía global (mecánica cuántica + álgebra de Boole +
              analogía de teoría de cuerdas)
    """

    def execute_gateway_governance(
        self,
        hash_t0: str,
        hash_t: str,
        grad_H: NDArray[Any],
        J_matrix: NDArray[Any],
        R_matrix: NDArray[Any],
        exogenous_work_gu: float,
        req_physics: bool,
        req_tactics: bool,
        req_strategy: bool,
        req_wisdom: bool,
        intent_x: NDArray[Any],
        projector_Pi: NDArray[Any],
        other_projectors: Optional[Sequence[NDArray[Any]]] = None,
        *,
        require_nonempty_hash: bool = True,
        normalize_hash_case: bool = False,
        dissipation_tolerance: float = _DISSIPATION_TOLERANCE,
        psd_tolerance: float = _PSD_TOLERANCE,
        allow_empty_request: bool = False,
        orthogonality_tolerance: float = _ORTHOGONALITY_TOLERANCE,
        check_operator_orthogonality: bool = True,
        require_hermitian_projectors: bool = True,
        raise_on_veto: bool = True,
    ) -> GatewayGovernanceState:
        """
        Ejecuta la composición funtorial estricta Φ₃∘Φ₂∘Φ₁ sobre la carga
        útil y la red.

        Retorna:
            GatewayGovernanceState con certificados de Fase 1, 2 y 3, más el
            índice de anomalía global.

        Veto:
            Si raise_on_veto=True, cualquier violación lanza excepción.
            Si raise_on_veto=False, devuelve estado con is_gateway_secure=False
            y anomaly_index > 0.
        """
        return self._phase3_terminal_synthesis(
            hash_t0,
            hash_t,
            grad_H,
            J_matrix,
            R_matrix,
            exogenous_work_gu,
            req_physics,
            req_tactics,
            req_strategy,
            req_wisdom,
            intent_x,
            projector_Pi,
            other_projectors,
            require_nonempty_hash=require_nonempty_hash,
            normalize_hash_case=normalize_hash_case,
            dissipation_tolerance=dissipation_tolerance,
            psd_tolerance=psd_tolerance,
            allow_empty_request=allow_empty_request,
            orthogonality_tolerance=orthogonality_tolerance,
            check_operator_orthogonality=check_operator_orthogonality,
            require_hermitian_projectors=require_hermitian_projectors,
            raise_on_veto=raise_on_veto,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §E. EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "AppAgentError",
    "StateSymmetryBreakingError",
    "ImpedanceInputError",
    "ResonanceCatastropheVeto",
    "TopologicalPassportError",
    "OrthogonalProjectionInputError",
    "OrthogonalityViolationError",
    "GlobalAnomalyCancellationError",
    "StateFibrationData",
    "ImpedanceControlData",
    "ThermodynamicPassportData",
    "OrthogonalProjectionData",
    "GatewayGovernanceState",
    "Phase1_SpectralFibrationImpedanceCertifier",
    "Phase2_CategoricalDIKWClosureInstantiator",
    "Phase3_QuantumBooleanMICProjectorSynthesizer",
    "AppAgent",
]
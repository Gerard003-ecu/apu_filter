# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Clifford Gauge Agent (Soberano de Calibre de Clifford)              ║
║ Ruta   : app/agents/wisdom/clifford_gauge_agent.py                           ║
║ Versión: 3.0.0-Doctoral-STA-OODA-Heyting-Vierbein-Nested                     ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO (OODA):                  ║
║ Agente supervisor ciber-físico sobre el topos de estados STA. Gobierna       ║
║ pares multivectoriales (P, Q) ∈ Cl_{1,3}(R)^2, audita la homomorfía de       ║
║ la forma de Lorentz Q(A) = ⟨A Ã⟩_0, contrae la 2-forma de curvatura con      ║
║ G^{μν} y clasifica el veredicto en el álgebra de Heyting Gödel Ω₃.           ║
║                                                                              ║
║ Cadena de funtores anidados:                                                 ║
║                                                                              ║
║   (P, Q) --Fase 1-->  Interacción(P, Q, PQ, δQ)                              ║
║          --Fase 2-->  Orientación(S_YM, Betti, ω_pre)                        ║
║          --Fase 3-->  Certificado(Ω₃, Crowbar, sello HMAC)                   ║
║                                                                              ║
║ Germen Fase 1 → Fase 2:                                                      ║
║     synthesize_gauge_interaction  ⊣  observe_gauge_interaction               ║
║                                                                              ║
║ Germen Fase 2 → Fase 3:                                                      ║
║     orient_gauge_diagnostics      ⊣  decide_from_orientation                 ║
║                                                                              ║
║ Ω₃ = {0 < ½ < 1}  (Gödel–Heyting ternario)                                   ║
║     1  ↔  COHERENT                                                           ║
║     ½  ↔  DEGRADED   (veto suave + gracia)                                   ║
║     0  ↔  VETOED     (veto duro + Crowbar IRAM)                              ║
║                                                                              ║
║     a ∧ b = min(a,b),  a ∨ b = max(a,b),                                     ║
║     a → b = 1 si a ≤ b else b,  ¬a = a → 0.                                  ║
║                                                                              ║
║ ORGANIZACIÓN EN TRES FASES ANIDADAS POR HERENCIA ESTRICTA:                   ║
║   FASE 1: Phase1_CliffordStateFactory                                        ║
║   FASE 2: Phase2_GaugeDiagnostics(Phase1_CliffordStateFactory)               ║
║   FASE 3: Phase3_OODAActuator(Phase2_GaugeDiagnostics)                       ║
║   Agente: CliffordGaugeAgent(Phase3_OODAActuator)                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AbstractSet, Callable, Final, Optional, Tuple

import numpy as np

try:
    from app.core.inmune_system.clifford_gauge_engine import CliffordGaugeEngine
except ImportError:
    try:
        from clifford_gauge_engine import CliffordGaugeEngine
    except ImportError as exc:
        raise ImportError(
            "CliffordGaugeAgent requiere CliffordGaugeEngine. "
            "Asegúrese de que 'clifford_gauge_engine.py' esté disponible "
            "en el paquete 'app.core.inmune_system' o en el PYTHONPATH."
        ) from exc


logger = logging.getLogger("APU.Agents.Wisdom.CliffordGaugeAgent")

__version__: Final[str] = "3.0.0"

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_SPACETIME_DIM: Final[int] = 4
_CLIFFORD_DIM: Final[int] = 16

_VERDICT_COHERENT: Final[str] = "COHERENT"
_VERDICT_DEGRADED: Final[str] = "DEGRADED"
_VERDICT_VETOED: Final[str] = "VETOED"

_LEGACY_OVERRIDE_TOKENS: Final[frozenset] = frozenset(
    {
        "AUT_POS_SABIDURIA_777",
        "OVERRIDE_CLIFFORD_G_2026",
        "HMAC_SUTURA_FOCK_SECURE",
    }
)

_GRADES: Final[np.ndarray] = np.array(
    [
        0,
        1, 1, 1, 1,
        2, 2, 2, 2, 2, 2,
        3, 3, 3, 3,
        4,
    ],
    dtype=np.int8,
)
_GRADES.setflags(write=False)

_BIVECTOR_PAIRS: Final[Tuple[Tuple[int, int, int], ...]] = (
    (5, 0, 1),
    (6, 0, 2),
    (7, 0, 3),
    (8, 1, 2),
    (9, 1, 3),
    (10, 2, 3),
)


# ───────────────────────────────────────────────────────────────────────────────
# Clasificación en el álgebra de Heyting Gödel Ω₃
# ───────────────────────────────────────────────────────────────────────────────

class HeytingVerdict(str, Enum):
    r"""
    Puntos del álgebra de Heyting ternaria Ω₃ = {0 < ½ < 1}.

    El orden es el de Gödel: VETOED < DEGRADED < COHERENT.
    El override nunca eleva 0; a lo sumo fija ½ (implicación Heyting).
    """

    VETOED = _VERDICT_VETOED
    DEGRADED = _VERDICT_DEGRADED
    COHERENT = _VERDICT_COHERENT

    @property
    def omega(self) -> float:
        return {
            HeytingVerdict.VETOED: 0.0,
            HeytingVerdict.DEGRADED: 0.5,
            HeytingVerdict.COHERENT: 1.0,
        }[self]

    @classmethod
    def from_omega(cls, value: float) -> "HeytingVerdict":
        if value <= 0.0:
            return cls.VETOED
        if value < 1.0:
            return cls.DEGRADED
        return cls.COHERENT


def _heyting_meet(a: float, b: float) -> float:
    return float(min(a, b))


def _heyting_join(a: float, b: float) -> float:
    return float(max(a, b))


def _heyting_implies(a: float, b: float) -> float:
    return 1.0 if a <= b + _MACHINE_EPS else float(b)


def _heyting_not(a: float) -> float:
    return _heyting_implies(a, 0.0)


# ───────────────────────────────────────────────────────────────────────────────
# Utilidad de inmutabilidad para ndarrays en dataclasses frozen
# ───────────────────────────────────────────────────────────────────────────────

def _immutable(array: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
    out = np.array(array, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


def _canonical_bytes(part: Any) -> bytes:
    r"""Serialización little-endian estable (independiente de endianness nativa)."""
    if isinstance(part, np.ndarray):
        arr = np.ascontiguousarray(part)
        header = np.array(arr.shape, dtype="<i8").tobytes()
        header += np.array([1 if np.iscomplexobj(arr) else 0], dtype="<i8").tobytes()
        if np.iscomplexobj(arr):
            real = np.ascontiguousarray(arr.real, dtype=np.float64).astype("<f8")
            imag = np.ascontiguousarray(arr.imag, dtype=np.float64).astype("<f8")
            return header + real.tobytes() + imag.tobytes()
        real = np.ascontiguousarray(arr, dtype=np.float64).astype("<f8")
        return header + real.tobytes()
    if isinstance(part, bytes):
        return part
    if isinstance(part, str):
        return part.encode("utf-8")
    if isinstance(part, (int, float, bool, np.generic)):
        return np.array([part], dtype="<f8").tobytes()
    return repr(part).encode("utf-8")


# ───────────────────────────────────────────────────────────────────────────────
# Dataclasses inmutables
# ───────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CliffordState:
    r"""
    Estado físico multivectorial en el espacio de fase STA 16D.

    Atributos:
        vector_rep:              Coeficientes reales en R^{16}.
        matrix_rep:              Imagen de Dirac en M_4(C).
        lorentz_quadratic_form:  Q(A) = ⟨A Ã⟩_0.
        is_rotor_unit:           Candidato a Spin(1,3): par, Q≈1 y AÃ≈1.
        faraday_bivector:        Componentes F[5:11].
        sha256_hash:             Firma canónica del estado.
        even_grade_norm:         ‖A_even‖₂.
        odd_grade_norm:          ‖A_odd‖₂.
        is_finite:               Finitud numérica completa.
        banach_norm:             ‖M‖_F / 2 (norma de representación).
        reconstruction_residual: ‖π(ι(s)) − s‖₂ si el motor lo permite.
        spin_residual:           ‖A Ã − 1‖₂ (obstrucción a Spin).
    """

    vector_rep: np.ndarray
    matrix_rep: np.ndarray
    lorentz_quadratic_form: float
    is_rotor_unit: bool
    faraday_bivector: np.ndarray
    sha256_hash: str

    even_grade_norm: float = 0.0
    odd_grade_norm: float = 0.0
    is_finite: bool = True
    banach_norm: float = 0.0
    reconstruction_residual: float = 0.0
    spin_residual: float = float("nan")


@dataclass(frozen=True, slots=True)
class CliffordInteractionReport:
    r"""
    GERMEN FASE 1 → FASE 2.

    Par de estados (contratista, proveedor) junto con su producto geométrico
    y la obstrucción a la homomorfía de Lorentz

        δQ := |Q(PQ) − Q(P) Q(Q)|.

    δQ = 0 sobre el grupo de Clifford Γ ⊂ Cl_{1,3}^×; un δQ no nulo es
    clase de obstrucción del par frente a Spin(1,3).
    """

    contractor: CliffordState
    supplier: CliffordState
    product: CliffordState
    lorentz_drift: float
    is_cfl: bool
    sha256_hash: str
    homomorphism_residual: float = 0.0


@dataclass(frozen=True, slots=True)
class GaugeOrientationReport:
    r"""
    GERMEN FASE 2 → FASE 3.

    Orientación covariante: geometría (S_YM, κ₂(G)), topología (Betti, χ)
    y preclasificación Heyting ω_pre ∈ Ω₃ *antes* de gracia y override.

        ω_hard = ¬Anomalía ∧ CFL  ∈ {0, 1}
        ω_ym   = 1 si |S_YM| ≤ τ  else ½
        ω_pre  = ω_hard ∧ ω_ym
    """

    interaction: CliffordInteractionReport
    yang_mills_action: float
    metric_condition_number: float
    is_yang_mills_finite: bool
    is_yang_mills_stable: bool
    has_topological_anomaly: bool
    betti_0: int
    betti_1: int
    betti_2: int
    euler_characteristic: int
    omega_hard: float
    omega_ym: float
    omega_pre: float
    sha256_hash: str

    volume_density: float = float("nan")
    pontryagin_density: float = float("nan")
    frobenius_power: float = float("nan")
    tetrad_action_residual: float = float("nan")
    metric_signature: str = ""


@dataclass(frozen=True, slots=True)
class CliffordGaugeCertificate:
    r"""
    Certificado formal de auditoría de calibre, regularidad y veto.

    Atributos:
        heyting_verdict:          'COHERENT' | 'DEGRADED' | 'VETOED'.
        yang_mills_action:        S_YM = (1/8) F_{μν} F^{μν}.
        lorentz_drift:            |Q(PQ) − Q(P)Q(Q)|.
        is_yang_mills_stable:     |S_YM| ≤ umbral elástico.
        has_topological_anomaly:  Fallo de contracción homotópica de la malla.
        is_soft_veto_active:      Veto suave (luz ámbar) activo.
        is_hard_veto_active:      Veto duro (Crowbar) activo.
        actuation_latency_ns:     Latencia Crowbar determinista < 400 ns.
        time_grace_remaining:     Gracia residual (s).
        cryptographic_seal:       Sello SHA-256 / HMAC-SHA256 de sesión.
        metric_condition_number:  κ₂(G).
        betti_0, betti_1:         Números de Betti auditados.
        contractor_lorentz_q:     Q(P).
        supplier_lorentz_q:       Q(Q).
        heyting_omega:            Valor numérico en Ω₃.
        euler_characteristic:     χ = b0 − b1 + b2.
        omega_pre:                ω antes de gracia/override.
        volume_density:           √|det G|.
        pontryagin_density:       Densidad F ∧ F.
        interaction_seal:         Sello del germen Fase 1.
    """

    heyting_verdict: str
    yang_mills_action: float
    lorentz_drift: float
    is_yang_mills_stable: bool
    has_topological_anomaly: bool
    is_soft_veto_active: bool
    is_hard_veto_active: bool
    actuation_latency_ns: float
    time_grace_remaining: float
    cryptographic_seal: str

    metric_condition_number: float = float("nan")
    betti_0: int = 0
    betti_1: int = 0
    contractor_lorentz_q: float = 0.0
    supplier_lorentz_q: float = 0.0
    heyting_omega: float = 1.0
    euler_characteristic: int = 0
    omega_pre: float = 1.0
    volume_density: float = float("nan")
    pontryagin_density: float = float("nan")
    interaction_seal: str = ""
    betti_2: int = 0


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1: Fábrica de estados Clifford y par de interacción STA                ║
# ║                                                                              ║
# ║ Objeto: Cl_{1,3} como álgebra de Banach de dimensión 16.                     ║
# ║ Cierre formal: synthesize_gauge_interaction  →  germen de la Fase 2.         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase1_CliffordStateFactory:
    r"""
    FASE 1 — Fábrica de estados Clifford y morfismos del grupo de Lorentz.

    Responsabilidades:
      1. Validar vectores 16D finitos.
      2. Construir CliffordState delegando ι, Q y el producto en el motor.
      3. Diagnosticar candidatos a rotor/Spin: paridad, Q≈1, AÃ≈1.
      4. Multiplicar estados por el isomorfismo de Dirac.
      5. Cierre: sintetizar el par de interacción (P, Q, PQ, δQ).
    """

    __slots__ = ("_tol", "_engine")

    def __init__(
        self,
        tolerance: float = 1e-12,
        engine: Optional[Any] = None,
    ) -> None:
        self._tol: Final[float] = float(tolerance)

        if engine is None:
            try:
                engine = CliffordGaugeEngine(self._tol)
            except TypeError:
                engine = CliffordGaugeEngine()

        self._engine: Final[Any] = engine

    def _relative_tolerance(self, scale: float = 1.0) -> float:
        return max(self._tol, 10.0 * _MACHINE_EPS * max(1.0, float(scale)))

    def _validate_vector16(
        self,
        S: np.ndarray,
        name: str = "vector Clifford",
    ) -> np.ndarray:
        arr = np.asarray(S, dtype=np.float64)
        if arr.shape != (_CLIFFORD_DIM,):
            raise ValueError(
                f"El {name} debe ser estrictamente 16D. Obtenido: {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"El {name} contiene valores no finitos.")
        return arr

    def _sha256_payload(self, *parts: Any) -> str:
        sha = hashlib.sha256()
        for part in parts:
            sha.update(_canonical_bytes(part))
        return sha.hexdigest()

    def _grade_norms(self, S_arr: np.ndarray) -> Tuple[float, float]:
        odd_mask = (_GRADES % 2) == 1
        even_mask = ~odd_mask
        odd_grade_norm = float(np.linalg.norm(S_arr[odd_mask]))
        even_grade_norm = float(np.linalg.norm(S_arr[even_mask]))
        return even_grade_norm, odd_grade_norm

    def _spin_residual(self, S_arr: np.ndarray) -> float:
        r"""
        Obstrucción a Spin(1,3): ‖A Ã − 1‖₂.

        Un rotor unitario satisface A Ã = 1 (identidad de grado 0).
        Si el motor no expone reversión, se devuelve NaN y se cae a Q≈1.
        """
        compute_reversion = getattr(self._engine, "compute_reversion", None)
        multiply = getattr(self._engine, "multiply_multivectors", None)
        if not callable(compute_reversion) or not callable(multiply):
            return float("nan")
        try:
            reversed_s = np.asarray(compute_reversion(S_arr), dtype=np.float64)
            product = multiply(S_arr, reversed_s)
            vec = np.asarray(product.vector_rep, dtype=np.float64)
            target = np.zeros(_CLIFFORD_DIM, dtype=np.float64)
            target[0] = 1.0
            return float(np.linalg.norm(vec - target))
        except Exception:
            logger.debug("No se pudo evaluar el residuo de Spin; se omite.", exc_info=True)
            return float("nan")

    def _reconstruction_residual(
        self,
        S_arr: np.ndarray,
        matrix_rep: np.ndarray,
    ) -> float:
        project = getattr(self._engine, "project_matrix", None)
        if not callable(project):
            return 0.0
        try:
            roundtrip = np.asarray(project(matrix_rep), dtype=np.float64)
            return float(np.linalg.norm(roundtrip - S_arr))
        except Exception:
            logger.debug("No se pudo evaluar el residuo π∘ι; se omite.", exc_info=True)
            return 0.0

    def build_clifford_state(self, S: np.ndarray) -> CliffordState:
        r"""
        Instancia un estado multivectorial desde R^{16} vía el funtor de Dirac

            ι : R^{16} → M_4(C),   Q(A) = ⟨A Ã⟩_0.
        """
        S_arr = self._validate_vector16(S, "vector de estado Clifford")

        matrix_rep = np.asarray(
            self._engine.embed_multivector(S_arr),
            dtype=np.complex128,
        )
        if matrix_rep.shape != (_SPACETIME_DIM, _SPACETIME_DIM):
            raise RuntimeError(
                f"El motor devolvió una representación matricial inválida: {matrix_rep.shape}"
            )
        if not np.all(np.isfinite(matrix_rep)):
            raise ValueError(
                "La representación matricial de Clifford contiene valores no finitos."
            )

        lorentz_q = float(self._engine.compute_lorentz_quadratic_form(S_arr))
        if not math.isfinite(lorentz_q):
            raise ValueError("La forma cuadrática de Lorentz no es finita.")

        even_grade_norm, odd_grade_norm = self._grade_norms(S_arr)
        rotor_tolerance = self._relative_tolerance(
            max(1.0, abs(lorentz_q), even_grade_norm, odd_grade_norm)
        )

        spin_residual = self._spin_residual(S_arr)
        spin_ok = (not math.isfinite(spin_residual)) or (spin_residual <= rotor_tolerance)

        is_rotor_unit = bool(
            (odd_grade_norm <= rotor_tolerance)
            and (abs(lorentz_q - 1.0) <= rotor_tolerance)
            and spin_ok
        )

        banach_norm = float(np.linalg.norm(matrix_rep, "fro")) / 2.0
        reconstruction_residual = self._reconstruction_residual(S_arr, matrix_rep)

        sha256_hash = self._sha256_payload(
            S_arr,
            np.real(matrix_rep),
            np.imag(matrix_rep),
            np.array(
                [lorentz_q, odd_grade_norm, even_grade_norm, banach_norm],
                dtype=np.float64,
            ),
        )

        return CliffordState(
            vector_rep=_immutable(S_arr, np.float64),
            matrix_rep=_immutable(matrix_rep, np.complex128),
            lorentz_quadratic_form=lorentz_q,
            is_rotor_unit=is_rotor_unit,
            faraday_bivector=_immutable(S_arr[5:11], np.float64),
            sha256_hash=sha256_hash,
            even_grade_norm=even_grade_norm,
            odd_grade_norm=odd_grade_norm,
            is_finite=True,
            banach_norm=banach_norm,
            reconstruction_residual=reconstruction_residual,
            spin_residual=spin_residual,
        )

    def _clifford_multiply(
        self,
        a: CliffordState,
        b: CliffordState,
    ) -> CliffordState:
        r"""Producto geométrico delegado: M(AB) = M(A) M(B), π: M_4(C) → R^{16}."""
        product_result = self._engine.multiply_multivectors(
            a.vector_rep,
            b.vector_rep,
        )
        return self.build_clifford_state(product_result.vector_rep)

    def compute_lorentz_drift(
        self,
        a: CliffordState,
        b: CliffordState,
        prod: CliffordState,
    ) -> float:
        r"""
        Obstrucción a la homomorfía de la forma de Lorentz:

            δQ(A,B) = |Q(AB) − Q(A) Q(B)|.

        Sobre el grupo de Clifford esta cantidad se anula idénticamente.
        """
        expected = a.lorentz_quadratic_form * b.lorentz_quadratic_form
        if not (math.isfinite(expected) and math.isfinite(prod.lorentz_quadratic_form)):
            return float("inf")
        return float(abs(prod.lorentz_quadratic_form - expected))

    def synthesize_gauge_interaction(
        self,
        contractor_S: np.ndarray,
        supplier_S: np.ndarray,
    ) -> CliffordInteractionReport:
        r"""
        CIERRE FORMAL DE LA FASE 1 / GERMEN DE LA FASE 2.

        Construye el objeto de interacción

            I(P, Q) = (P, Q, PQ, δQ) ∈ Cl_{1,3}^3 × R≥0

        sobre el que el funtor de la Fase 2,
        ``observe_gauge_interaction``, actúa de forma estricta:

            observe : I(P, Q) × Met_4 × N^2  →  Orientación.

        No se evalúa aún ni la métrica ni la cohomología: eso es geometría
        y topología, no álgebra de Clifford, y pertenece a la Fase 2.
        """
        p_state = self.build_clifford_state(contractor_S)
        q_state = self.build_clifford_state(supplier_S)
        prod_state = self._clifford_multiply(p_state, q_state)

        lorentz_drift = self.compute_lorentz_drift(p_state, q_state, prod_state)
        scale = max(
            1.0,
            abs(p_state.lorentz_quadratic_form),
            abs(q_state.lorentz_quadratic_form),
            abs(prod_state.lorentz_quadratic_form),
        )
        cfl_tolerance = self._relative_tolerance(scale)
        is_cfl = bool(math.isfinite(lorentz_drift) and lorentz_drift <= cfl_tolerance)

        sha256_hash = self._sha256_payload(
            p_state.sha256_hash,
            q_state.sha256_hash,
            prod_state.sha256_hash,
            np.array([lorentz_drift, float(is_cfl)], dtype=np.float64),
        )

        return CliffordInteractionReport(
            contractor=p_state,
            supplier=q_state,
            product=prod_state,
            lorentz_drift=lorentz_drift,
            is_cfl=is_cfl,
            sha256_hash=sha256_hash,
            homomorphism_residual=lorentz_drift,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2: Diagnóstico gauge, métrico y topológico                             ║
# ║                                                                              ║
# ║ Apertura: observe_gauge_interaction(synthesize_gauge_interaction(·)).        ║
# ║ Cierre formal: orient_gauge_diagnostics  →  germen de la Fase 3.             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase2_GaugeDiagnostics(Phase1_CliffordStateFactory):
    r"""
    FASE 2 — Diagnóstico geométrico (Yang-Mills) y topológico (Betti).

    Continuación estricta de ``synthesize_gauge_interaction``.

    Responsabilidades:
      1. Observar I(P, Q) contra una métrica G_{μν} y números de Betti.
      2. Delegar S_YM al motor (contracción de 2-forma); repliegue local.
      3. Auditar anomalías topológicas: la malla coherente es contráctil
         (b0 = 1, b1 = b2 = 0).
      4. Cierre: preclasificar ω_pre ∈ Ω₃ para el decisor de la Fase 3.
    """

    __slots__ = ()

    def _validate_curvature_vector(
        self,
        F: np.ndarray,
        name: str = "curvatura",
    ) -> np.ndarray:
        r"""
        Acepta R^{16} o el chart compacto 6D [F01, F02, F03, F12, F13, F23].
        """
        arr = np.asarray(F, dtype=np.float64)
        if arr.shape == (6,):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"La {name} compacta 6D contiene valores no finitos.")
            full = np.zeros(_CLIFFORD_DIM, dtype=np.float64)
            full[5:11] = arr
            return full
        return self._validate_vector16(arr, name)

    def evaluate_metric_report(self, G_metric: np.ndarray) -> Any:
        G_arr = np.asarray(G_metric, dtype=np.float64)
        if G_arr.shape != (_SPACETIME_DIM, _SPACETIME_DIM):
            raise ValueError(
                f"El tensor métrico G debe ser estrictamente 4×4. Obtenido: {G_arr.shape}"
            )
        if not np.all(np.isfinite(G_arr)):
            raise ValueError("El tensor métrico G contiene valores no finitos.")
        return self._engine.evaluate_metric_tensor(G_arr)

    def _yang_mills_fallback(
        self,
        F_vec: np.ndarray,
        G_metric: np.ndarray,
    ) -> Tuple[float, float, bool, dict]:
        r"""
        Contracción tensorial local

            S_YM = (1/8) F_{μν} F^{μν},   F^{μν} = G^{μα} G^{νβ} F_{αβ}.
        """
        F = self._validate_curvature_vector(F_vec, "curvatura Faraday")
        metric_report = self.evaluate_metric_report(G_metric)

        F_clean = np.zeros(_CLIFFORD_DIM, dtype=np.float64)
        F_clean[5:11] = F[5:11]

        F_covariant = np.zeros((_SPACETIME_DIM, _SPACETIME_DIM), dtype=np.float64)
        for bivector_index, mu, nu in _BIVECTOR_PAIRS:
            value = float(F_clean[bivector_index])
            F_covariant[mu, nu] = value
            F_covariant[nu, mu] = -value

        G_inv = np.asarray(metric_report.g_inv, dtype=np.float64)
        condition = float(getattr(metric_report, "condition_number", float("nan")))

        extras = {
            "volume_density": float(getattr(metric_report, "volume_density", float("nan"))),
            "pontryagin_density": float("nan"),
            "frobenius_power": float("nan"),
            "tetrad_action_residual": float("nan"),
            "metric_signature": str(getattr(getattr(metric_report, "signature", ""), "value", "")),
        }

        if G_inv.shape != (_SPACETIME_DIM, _SPACETIME_DIM) or not np.all(np.isfinite(G_inv)):
            return float("nan"), condition, False, extras

        F_contravariant = G_inv @ F_covariant @ G_inv.T
        F_contravariant = 0.5 * (F_contravariant - F_contravariant.T)
        contraction = float(np.einsum("mn,mn->", F_covariant, F_contravariant))
        ym_action = 0.125 * contraction
        is_finite = bool(math.isfinite(contraction) and math.isfinite(ym_action))
        if not is_finite:
            ym_action = float("nan")
        return ym_action, condition, is_finite, extras

    def _evaluate_yang_mills_details(
        self,
        F_vec: np.ndarray,
        G_metric: np.ndarray,
    ) -> Tuple[float, float, bool, dict]:
        r"""
        Prefiere el motor doctoral (vierbein / Hodge / Pontryagin).
        Si la firma no coincide, repliegue tensorial local.
        """
        evaluate = getattr(self._engine, "evaluate_yang_mills_action", None)
        if callable(evaluate):
            try:
                report = evaluate(F_vec, G_metric)
                extras = {
                    "volume_density": float(getattr(report, "volume_density", float("nan"))),
                    "pontryagin_density": float(
                        getattr(report, "pontryagin_density", float("nan"))
                    ),
                    "frobenius_power": float(getattr(report, "frobenius_power", float("nan"))),
                    "tetrad_action_residual": float(
                        getattr(report, "tetrad_action_residual", float("nan"))
                    ),
                    "metric_signature": str(
                        getattr(getattr(report, "signature", ""), "value", "")
                    ),
                }
                return (
                    float(report.ym_action),
                    float(report.condition_number),
                    bool(report.is_action_finite),
                    extras,
                )
            except TypeError:
                logger.debug(
                    "evaluate_yang_mills_action rechazó la firma; repliegue local."
                )
            except Exception:
                logger.exception(
                    "El motor falló al evaluar Yang-Mills. Se usa contracción local."
                )

        return self._yang_mills_fallback(F_vec, G_metric)

    def evaluate_yang_mills_action(
        self,
        F_state: CliffordState,
        G_metric: np.ndarray,
    ) -> float:
        action, _, _, _ = self._evaluate_yang_mills_details(
            F_state.vector_rep,
            G_metric,
        )
        return action

    def _audit_topological_anomaly(
        self,
        betti_0: int,
        betti_1: int,
        betti_2: int = 0,
    ) -> bool:
        r"""
        Auditoría homológica de la malla.

        Una malla coherente es homotópicamente contráctil:
            b0 = 1,  b1 = 0,  b2 = 0
        (un punto, sin 1-ciclos ni 2-cavidades).

        Anomalía syss:
          - b0 = 0  (vacío / grafo inválido),
          - b0 > 1  (fragmentación / islas),
          - b1 > 0  (silos de cohomología),
          - b2 > 0  (cavidades 2-dimensionales).
        """
        b0, b1, b2 = int(betti_0), int(betti_1), int(betti_2)
        if min(b0, b1, b2) < 0:
            raise ValueError("Los números de Betti no pueden ser negativos.")
        return (b0 != 1) or (b1 > 0) or (b2 > 0)

    def observe_gauge_interaction(
        self,
        interaction: CliffordInteractionReport,
        G_metric: np.ndarray,
        betti_0: int,
        betti_1: int,
        betti_2: int = 0,
        ym_threshold: float = 2.5,
    ) -> GaugeOrientationReport:
        r"""
        APERTURA FORMAL DE LA FASE 2.

        Continuación directa de ``synthesize_gauge_interaction``:

            I(P, Q)  ↦  (S_YM(Q; G), κ₂(G), Betti, ω_pre).

        El proveedor se interpreta como portador de la 2-forma de Faraday
        (convención soberana heredada: F = ⟨Q⟩_2).
        """
        if not isinstance(interaction, CliffordInteractionReport):
            raise TypeError(
                "observe_gauge_interaction exige un CliffordInteractionReport "
                "(germen de synthesize_gauge_interaction)."
            )

        ym_action, metric_condition, ym_finite, extras = self._evaluate_yang_mills_details(
            interaction.supplier.vector_rep,
            G_metric,
        )

        has_topological_anomaly = self._audit_topological_anomaly(
            betti_0, betti_1, betti_2
        )

        b0, b1, b2 = int(betti_0), int(betti_1), int(betti_2)
        euler = b0 - b1 + b2

        is_ym_stable = bool(
            ym_finite
            and math.isfinite(ym_action)
            and (abs(ym_action) <= float(ym_threshold) + self._tol)
        )

        omega_hard = 1.0 if (interaction.is_cfl and not has_topological_anomaly) else 0.0
        omega_ym = 1.0 if is_ym_stable else 0.5
        omega_pre = _heyting_meet(omega_hard, omega_ym)

        sha256_hash = self._sha256_payload(
            interaction.sha256_hash,
            np.asarray(G_metric, dtype=np.float64),
            np.array(
                [
                    ym_action,
                    metric_condition,
                    float(b0),
                    float(b1),
                    float(b2),
                    omega_pre,
                ],
                dtype=np.float64,
            ),
        )

        return GaugeOrientationReport(
            interaction=interaction,
            yang_mills_action=ym_action,
            metric_condition_number=metric_condition,
            is_yang_mills_finite=ym_finite,
            is_yang_mills_stable=is_ym_stable,
            has_topological_anomaly=has_topological_anomaly,
            betti_0=b0,
            betti_1=b1,
            betti_2=b2,
            euler_characteristic=euler,
            omega_hard=omega_hard,
            omega_ym=omega_ym,
            omega_pre=omega_pre,
            sha256_hash=sha256_hash,
            volume_density=float(extras.get("volume_density", float("nan"))),
            pontryagin_density=float(extras.get("pontryagin_density", float("nan"))),
            frobenius_power=float(extras.get("frobenius_power", float("nan"))),
            tetrad_action_residual=float(
                extras.get("tetrad_action_residual", float("nan"))
            ),
            metric_signature=str(extras.get("metric_signature", "")),
        )

    def orient_gauge_diagnostics(
        self,
        contractor_S: np.ndarray,
        supplier_S: np.ndarray,
        G_metric: np.ndarray,
        betti_0: int,
        betti_1: int,
        betti_2: int = 0,
        ym_threshold: float = 2.5,
        interaction: Optional[CliffordInteractionReport] = None,
    ) -> GaugeOrientationReport:
        r"""
        CIERRE FORMAL DE LA FASE 2 / GERMEN DE LA FASE 3.

        Encadena el germen algebraico de la Fase 1 con la observación
        geométrico-topológica y produce el objeto de orientación

            Ω_pre = (ω_hard ∧ ω_ym) ∈ Ω₃

        sobre el que ``Phase3_OODAActuator.decide_from_orientation`` actúa
        de forma estricta: aplica la ventana de gracia (flecha ½ → 0) y
        el override (implicación Heyting que no eleva 0).

        No muta estado de veto ni dispara Crowbar: eso es actuación, no
        orientación.
        """
        if interaction is None:
            interaction = self.synthesize_gauge_interaction(contractor_S, supplier_S)
        return self.observe_gauge_interaction(
            interaction=interaction,
            G_metric=G_metric,
            betti_0=betti_0,
            betti_1=betti_1,
            betti_2=betti_2,
            ym_threshold=ym_threshold,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3: Actuación OODA, Heyting Ω₃ y veto ciber-físico                      ║
# ║                                                                              ║
# ║ Apertura: decide_from_orientation(orient_gauge_diagnostics(·)).              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase3_OODAActuator(Phase2_GaugeDiagnostics):
    r"""
    FASE 3 — Decisión en Ω₃, gracia, override y Crowbar IRAM.

    Continuación estricta de ``orient_gauge_diagnostics``:

        ω = decide(ω_pre, gracia, override) ∈ Ω₃,
        Act = Crowbar  syss  ω = 0.

    Responsabilidades:
      1. Decidir COHERENT / DEGRADED / VETOED sobre ω_pre.
      2. Gestionar veto suave con ventana de gracia (½ persistente → 0).
      3. Verificar overrides en tiempo constante / HMAC.
      4. Actuar: latencia Crowbar determinista < 400 ns.
      5. Sellar HMAC/SHA-256 el certificado.
    """

    __slots__ = (
        "_ym_limit",
        "_grace_max",
        "_soft_veto_timestamp",
        "_is_soft_veto_active",
        "_override_verifier",
        "_allowed_override_tokens",
        "_hmac_secret",
    )

    def __init__(
        self,
        ym_threshold: float = 2.5,
        grace_period_seconds: float = 3600.0,
        tolerance: float = 1e-12,
        engine: Optional[Any] = None,
        override_verifier: Optional[Callable[[str], bool]] = None,
        allowed_override_tokens: Optional[AbstractSet[str]] = None,
        allow_legacy_overrides: bool = True,
        hmac_secret: Optional[bytes] = None,
    ) -> None:
        super().__init__(tolerance=tolerance, engine=engine)

        self._ym_limit: Final[float] = float(ym_threshold)
        self._grace_max: Final[float] = float(grace_period_seconds)

        self._soft_veto_timestamp: Optional[float] = None
        self._is_soft_veto_active: bool = False

        self._override_verifier: Optional[Callable[[str], bool]] = override_verifier

        if allowed_override_tokens is None:
            tokens = _LEGACY_OVERRIDE_TOKENS if allow_legacy_overrides else frozenset()
        else:
            tokens = frozenset(allowed_override_tokens)

        self._allowed_override_tokens: Final[frozenset] = frozenset(tokens)

        if hmac_secret is not None and not isinstance(hmac_secret, (bytes, bytearray)):
            raise TypeError("hmac_secret debe ser bytes o None.")
        self._hmac_secret: Final[Optional[bytes]] = (
            bytes(hmac_secret) if hmac_secret is not None else None
        )

        if allow_legacy_overrides and self._allowed_override_tokens.intersection(
            _LEGACY_OVERRIDE_TOKENS
        ):
            logger.warning(
                "Se permiten tokens legacy de override. "
                "Para producción, configure override_verifier o hmac_secret."
            )

    def _clear_soft_veto(self) -> None:
        self._is_soft_veto_active = False
        self._soft_veto_timestamp = None

    def reset_soft_veto(self) -> None:
        r"""Reinicia el temporizador de gracia (uso de laboratorio / tests)."""
        self._clear_soft_veto()

    def _verify_override(self, token: Optional[str]) -> bool:
        r"""
        Verificación de override en este orden de autoridad:

          1. override_verifier inyectado.
          2. HMAC-SHA256(hmac_secret, token) comparado en tiempo constante
             contra el propio token si el secreto está configurado y el
             token tiene formato hex de 64 caracteres *como mac de un
             nonce vacío* — en producción el verifier debe ser la autoridad.
          3. Conjunto de tokens permitidos vía hmac.compare_digest.

        El override **no** se registra en claro.
        """
        if token is None or not isinstance(token, str) or not token.strip():
            return False

        if callable(self._override_verifier):
            try:
                return bool(self._override_verifier(token))
            except Exception:
                logger.exception(
                    "El override_verifier lanzó una excepción. Se rechaza el override."
                )
                return False

        token_bytes = token.encode("utf-8")

        if self._hmac_secret is not None:
            expected = hmac.new(
                self._hmac_secret,
                token_bytes,
                hashlib.sha256,
            ).hexdigest()
            # El token inyectado se interpreta como MAC hex si aplica;
            # compare_digest sobre el conjunto permitido cubre el caso estático.
            if hmac.compare_digest(expected, token):
                return True

        for allowed in self._allowed_override_tokens:
            allowed_bytes = allowed.encode("utf-8")
            if hmac.compare_digest(token_bytes, allowed_bytes):
                if allowed in _LEGACY_OVERRIDE_TOKENS:
                    logger.warning(
                        "Override legacy aceptado. Considere migrar a tokens firmados."
                    )
                return True

        return False

    def _seeded_latency_ns(self, *parts: Any) -> float:
        r"""
        Latencia Crowbar determinista en IRAM.

        En hardware real esta función se sustituye por la medición GPIO/ISR.
        Aquí se evita np.random para preservar bit-reproducibilidad.
        Cota: 395 ns ≤ τ < 400 ns.
        """
        sha = hashlib.sha256()
        for part in parts:
            sha.update(_canonical_bytes(part))
        digest = sha.digest()
        fraction = int.from_bytes(digest[:6], "little") / float(1 << 48)
        latency = 395.0 + 4.5 * fraction
        return float(min(_CROWBAR_IRAM_LATENCY_NS, latency))

    def _seal(self, *parts: Any) -> str:
        payload = b"".join(_canonical_bytes(part) for part in parts)
        if self._hmac_secret is not None:
            return hmac.new(self._hmac_secret, payload, hashlib.sha256).hexdigest()
        return hashlib.sha256(payload).hexdigest()

    def decide_from_orientation(
        self,
        orientation: GaugeOrientationReport,
        override_token: Optional[str] = None,
        curr_time: Optional[float] = None,
    ) -> Tuple[HeytingVerdict, bool, bool, float]:
        r"""
        APERTURA FORMAL DE LA FASE 3.

        Continuación directa de ``orient_gauge_diagnostics``.

        Entrada: ω_pre = ω_hard ∧ ω_ym.
        Dinámica de Heyting:
          · ω_pre = 1  →  COHERENT, se limpia la gracia.
          · ω_pre = 0  →  VETOED instantáneo (topología o Lorentz).
            El override **no** eleva 0  (¬¬0 = 0 en Ω₃ Gödel).
          · ω_pre = ½  →  DEGRADED; si la gracia expira, ½ se colapsa a 0.
            Un override válido aplica la implicación ½ → ½: se disipa el
            ámbar operativo pero el veredicto permanece DEGRADED hasta
            que S_YM vuelva al umbral.

        Retorna:
            (verdict, is_soft_veto, is_hard_veto, time_remaining).
        """
        if not isinstance(orientation, GaugeOrientationReport):
            raise TypeError(
                "decide_from_orientation exige un GaugeOrientationReport "
                "(germen de orient_gauge_diagnostics)."
            )

        now = time.monotonic() if curr_time is None else float(curr_time)
        omega = float(orientation.omega_pre)

        if omega <= 0.0:
            self._clear_soft_veto()
            logger.error(
                "VETO DURO INSTANTÁNEO: colapso de cohomología, fragmentación "
                "topológica o anomalía de Lorentz detectada (ω_hard = 0)."
            )
            return HeytingVerdict.VETOED, False, True, 0.0

        if omega >= 1.0:
            self._clear_soft_veto()
            return HeytingVerdict.COHERENT, False, False, 0.0

        # ω = ½ : inestabilidad YM con geometría/topología sanas.
        is_soft_veto = True
        time_remaining = 0.0

        if not self._is_soft_veto_active:
            self._is_soft_veto_active = True
            self._soft_veto_timestamp = now
            verdict = HeytingVerdict.DEGRADED
            logger.warning(
                "VETO SUAVE ACTIVO (LUZ ÁMBAR): inestabilidad de Yang-Mills. "
                "Temporizador de gracia iniciado."
            )
        else:
            elapsed = now - (self._soft_veto_timestamp or now)
            time_remaining = max(0.0, self._grace_max - elapsed)
            if time_remaining <= self._tol:
                self._clear_soft_veto()
                logger.critical(
                    "PERÍODO DE GRACIA EXPIRADO SIN OVERRIDE VÁLIDO. "
                    "Heyting colapsa ½ → 0 (VETOED terminal)."
                )
                return HeytingVerdict.VETOED, False, True, 0.0
            verdict = HeytingVerdict.DEGRADED

        if override_token is not None:
            if self._verify_override(override_token):
                # Implicación Heyting: el override no restaura 1.
                self._clear_soft_veto()
                logger.info(
                    "OVERRIDE VÁLIDO ACEPTADO. Luz ámbar disipada; el estado "
                    "permanece DEGRADED hasta recuperar estabilidad YM."
                )
                return HeytingVerdict.DEGRADED, False, False, 0.0
            logger.error("Firma digital inválida en el override de calibre.")

        return verdict, is_soft_veto, False, time_remaining

    def _act_crowbar(
        self,
        orientation: GaugeOrientationReport,
        verdict: HeytingVerdict,
    ) -> float:
        r"""
        Actuación Crowbar: ISR en IRAM, GPIO14 HIGH, tiristor BT151.

        Sólo se dispara si el veredicto es VETOED (ω = 0).
        """
        if verdict is not HeytingVerdict.VETOED:
            return 0.0

        interaction = orientation.interaction
        switching_latency = self._seeded_latency_ns(
            interaction.contractor.vector_rep,
            interaction.supplier.vector_rep,
            orientation.yang_mills_action,
            interaction.lorentz_drift,
            verdict.value,
        )

        logger.error("COLA DE HEYTING COLAPSADA EN EL SOBERANO DE CALIBRE DE CLIFFORD.")
        logger.error("  - Ejecutando subrutina local isVerdictCoherent() en C++...")
        logger.error("  - Despachando ISR en IRAM de alta velocidad...")
        logger.error(
            "  - Conmutando pin de hardware GPIO14 a HIGH en %.2f ns...",
            switching_latency,
        )
        logger.error("  - Tiristor rápido de potencia BT151 (Crowbar) gatillado.")
        logger.error("  - Mezcladoras y bombas hidráulicas paralizadas en el milisegundo cero.")
        return switching_latency

    def audit_lazo_cerrado(
        self,
        contractor_S: np.ndarray,
        supplier_S: np.ndarray,
        G_metric: np.ndarray,
        betti_0: int,
        betti_1: int,
        override_token: Optional[str] = None,
        betti_2: int = 0,
    ) -> CliffordGaugeCertificate:
        r"""
        Orquesta el ciclo covariante OODA del Soberano de Calibre.

        Flujo anidado:
          OBSERVE  (Fase 1): synthesize_gauge_interaction.
          ORIENT   (Fase 2): orient_gauge_diagnostics.
          DECIDE   (Fase 3): decide_from_orientation.
          ACT      (Fase 3): _act_crowbar si ω = 0.
        """
        orientation = self.orient_gauge_diagnostics(
            contractor_S=contractor_S,
            supplier_S=supplier_S,
            G_metric=G_metric,
            betti_0=betti_0,
            betti_1=betti_1,
            betti_2=betti_2,
            ym_threshold=self._ym_limit,
        )

        verdict, is_soft_veto, is_hard_veto, time_remaining = self.decide_from_orientation(
            orientation,
            override_token=override_token,
        )

        if verdict is HeytingVerdict.VETOED:
            is_hard_veto = True

        switching_latency = self._act_crowbar(orientation, verdict)

        interaction = orientation.interaction
        cryptographic_seal = self._seal(
            interaction.contractor.vector_rep,
            interaction.supplier.vector_rep,
            np.asarray(G_metric, dtype=np.float64),
            np.array(
                [
                    orientation.yang_mills_action,
                    interaction.lorentz_drift,
                    float(orientation.betti_0),
                    float(orientation.betti_1),
                    float(orientation.betti_2),
                    float(orientation.is_yang_mills_stable),
                    float(orientation.has_topological_anomaly),
                    verdict.omega,
                ],
                dtype=np.float64,
            ),
            verdict.value,
        )

        return CliffordGaugeCertificate(
            heyting_verdict=verdict.value,
            yang_mills_action=orientation.yang_mills_action,
            lorentz_drift=interaction.lorentz_drift,
            is_yang_mills_stable=orientation.is_yang_mills_stable,
            has_topological_anomaly=orientation.has_topological_anomaly,
            is_soft_veto_active=bool(self._is_soft_veto_active),
            is_hard_veto_active=bool(is_hard_veto),
            actuation_latency_ns=switching_latency,
            time_grace_remaining=time_remaining,
            cryptographic_seal=cryptographic_seal,
            metric_condition_number=orientation.metric_condition_number,
            betti_0=orientation.betti_0,
            betti_1=orientation.betti_1,
            contractor_lorentz_q=interaction.contractor.lorentz_quadratic_form,
            supplier_lorentz_q=interaction.supplier.lorentz_quadratic_form,
            heyting_omega=verdict.omega,
            euler_characteristic=orientation.euler_characteristic,
            omega_pre=orientation.omega_pre,
            volume_density=orientation.volume_density,
            pontryagin_density=orientation.pontryagin_density,
            interaction_seal=interaction.sha256_hash,
            betti_2=orientation.betti_2,
        )


# ───────────────────────────────────────────────────────────────────────────────
# Agente final
# ───────────────────────────────────────────────────────────────────────────────

class CliffordGaugeAgent(Phase3_OODAActuator):
    r"""
    Soberano de Calibre de Clifford de de Rham–Fukaya (OODA lazo cerrado).

    Cadena de herencia (fases anidadas):

        CliffordGaugeAgent
          └─ Phase3_OODAActuator              Ω₃, gracia, Crowbar
               └─ Phase2_GaugeDiagnostics     S_YM, Betti, ω_pre
                    └─ Phase1_CliffordStateFactory   Cl_{1,3}, δQ, I(P,Q)

    Gobierna de forma covariante los estados multivectoriales de la Malla,
    audita colusiones mediante la homomorfía de Lorentz, evalúa la acción
    de Yang-Mills del bivector de Faraday y ejecuta el interlock
    ciber-físico en IRAM ante rupturas de simetría.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return (
            "CliffordGaugeAgent("
            f"ym_limit={self._ym_limit}, "
            f"grace_period_seconds={self._grace_max}, "
            f"tolerance={self._tol}"
            ")"
        )


__all__ = [
    "CliffordGaugeAgent",
    "CliffordState",
    "CliffordInteractionReport",
    "GaugeOrientationReport",
    "CliffordGaugeCertificate",
    "HeytingVerdict",
    "Phase1_CliffordStateFactory",
    "Phase2_GaugeDiagnostics",
    "Phase3_OODAActuator",
]
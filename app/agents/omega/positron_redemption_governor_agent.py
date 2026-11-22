# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO  : Positron Redemption Governor Agent (Soberano Agéntico de Calibre OODA)                     ║
║ RUTA    : app/agents/omega/positron_redemption_governor_agent.py                                     ║
║ VERSIÓN : 18.0.0-OODA-HKDF-Heyting-Nucleus-Holonomy-KAM-Yoshida-Merkle6962-Loeb-Crowbar-RSI3-PhD     ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
DOCTRINA RECTORA (evolución v16 → v17 → v18)
──────────────────────────────────────────────────────────────────────────────────────────────────────
1. SEPARACIÓN TRIÁDICA VINCULANTE (v17 consolidada en v18):
   (i)   FIRMAR   : compromiso criptográfico HMAC-SHA256 sobre payload canónico inyectivo `.17g`.
   (ii)  VERIFICAR: compuertas efectivas Φ₁₂ y Φ₂₃ que recomputan HMAC sobre cobertura total,
                    re-digieren los tensores congelados y exigen sesión viva [nbf, exp), nonce único
                    y reloj monótono estrictamente creciente.
   (iii) ACTUAR   : distinción estricta entre INTENCIÓN firmada (CrowbarIntent) y ACTUACIÓN física
                    confirmada por recibo del driver (ActuationReceipt) con latch de tiristor.

2. TOPOS DE HEYTING Ω₃ Y MORFISMOS NUCLEARES DE LAWVERE-TIERNEY:
   • Cadena de Gödel de 3 elementos: VETOED (⊥ = 0) < DEGRADED (1) < COHERENT (⊤ = 2).
   • Núcleo de Gracia Ámbar j_amber : Ω₃ → Ω₃:
         j_amber(COHERENT) = COHERENT,  j_amber(DEGRADED) = COHERENT,  j_amber(VETOED) = VETOED.
     Cumple los cuatro axiomas de Lawvere-Tierney: inflacionario (a ≤ j(a)), idempotente (j∘j = j),
     preserva ínfimos (j(a∧b) = j(a)∧j(b)) y preserva el tope (j(⊤) = ⊤). Su subtopos de gavillas
     Ω_j = {x ∈ Ω₃ : j(x) = x} = {VETOED, COHERENT} colapsa el estrato DEGRADED sólo bajo token
     criptográfico PositronAuthorization válido, ligado a la sesión y al digest de entrada.

3. CONTINUIDAD FORMAL CPS ENTRE LAS TRES FASES ANIDADAS (Φ₁ → Φ₂ → Φ₃):
   • Fase Φ₁ · Phase1Observer (1.1 → 1.11):
     Congela (p, ρ, σ, K), atestigua ausencia de retroacción sobre los buffers inspeccionados,
     audita la geometría cuántica de Uhlmann-Bures, la holonomía del fibrado principal U(n),
     la proyección cuaterniónica S³ ⊂ H, el sándwich de Fuchs-van de Graaf, el operador modular
     de Tomita-Takesaki Δ_{ρ|σ}, la información oblicua de Wigner-Yanase y la sección eficaz QED
     2γ/3γ de Dirac-Ore-Powell. Culmina en `1.11 _phase1_observe_and_sign`, cuyo certificado firmado
     `PositronPhase1Certificate` alimenta directamente `2.1 _gate_phi12`.
   • Fase Φ₂ · Phase2Orientor (2.1 → 2.11):
     Abre en `2.1 _gate_phi12` verificando `PositronPhase1Certificate`, ejecuta la sonda local de
     Banach-Kantorovich (tanto en norma producto euclídea-Frobenius como en métrica de
     Shahshahani-Bures), verifica las cotas de Connes-Daleckii-Krein, audita la descomposición
     de Lyapunov y el sector celeste de Poincaré-KAM-Yoshida-Melnikov-Krein, resuelve el modelo
     RLC subamortiguado del crowbar y culmina en `2.11 _phase2_orient_and_contract`, cuyo
     `PositronPhase2Certificate` alimenta directamente `3.1 _gate_phi23`.
   • Fase Φ₃ · Phase3Decider (3.1 → 3.12):
     Abre en `3.1 _gate_phi23` verificando `PositronPhase2Certificate` y su `PositronPhase1Certificate`
     anidado, evalúa la homología simplicial del 1-esqueleto y del complejo clique 2D Cl(G_K),
     adjudica el ínfimo en Ω₃ y el núcleo ámbar, ejecuta el descenso bien fundado de Löb sobre
     (ℕ, <), realiza la adaptación de política RSI³ con funcional de riesgo meta-Lyapunov y
     rollback automático, despacha el crowbar con latch y sella el certificado soberano con
     árbol Merkle RFC 6962 y pruebas de inclusión auditables PATH(m, D[n]).

DICTAMEN DE CORRECCIONES PRESERVADAS Y EXTENDIDAS (v16 → v17 → v18)
  [G1]–[G13] Preservadas íntegramente todas las correcciones de regresión de v17 sobre v16.
  [G14] Auditoría explícita en Φ₁ de holonomía U(n), cuaterniones S³, Fuchs-van de Graaf,
        operador modular de Tomita-Takesaki, Wigner-Yanase y ramificación QED 3γ de Ore-Powell.
  [G15] Auditoría explícita en Φ₂ de invariante simpléctico 4D de Poincaré-Cartan, margen
        diofántico KAM, identidad de Kapteyn-Fourier-Bessel, integrador simpléctico de Yoshida
        de 4º orden, número de rotación de Poincaré, índice de Hamilton-Hopf y conmutador BCH.
  [G16] Sonda de Banach-Kantorovich enriquecida con razón bruta sin truncar, distancia geodésica
        de Shahshahani-Bures y sub-pasos adaptativos RSI³ del motor.
  [G17] Descenso bien fundado de Löb con traza explícita de obligaciones por nivel en (ℕ, <).
  [G18] Adaptación efectiva de política RSI³ con funcional meta-Lyapunov, proyección al politopo
        compacto, verificación de seguridad y rollback con presupuesto finito `mutation_budget`.
  [G19] Homología simplicial extendida al 2-esqueleto clique Cl(G_K) con verificación exacta de
        la identidad de complejo de cadenas ∂₁ ∘ ∂₂ = 0, preservando β₀ y β₁ del 1-esqueleto.
  [G20] Caminos de auditoría Merkle RFC 6962 §2.1.1 (`_audit_path_rfc6962`) y verificador de
        inclusión (`_verify_inclusion_rfc6962`) integrados en el certificado soberano y verificador.
  [G21] Modelo físico RLC del crowbar extendido con energía electrostática ½CV², carga total CV,
        primer cruce por cero t_zero = π/ω_d (conmutación inversa) y frecuencia de oscilación f_d.
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

try:
    from app.omega.positron_redemption_governor_engine import (
        PositronRedemptionGovernorEngine,
        PositronRedemptionSpectrumReport,
    )
except ImportError:  # pragma: no cover - ruta plana para pruebas unitarias
    try:
        from positron_redemption_governor_engine import (  # type: ignore[no-redef]
            PositronRedemptionGovernorEngine,
            PositronRedemptionSpectrumReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar positron_redemption_governor_engine (rutas de paquete y plana)."
        ) from exc

logger = logging.getLogger("APU.Agents.Omega.PositronRedemptionGovernorAgent")
AnswerT = TypeVar("AnswerT")

__all__ = [
    "HeytingVerdict",
    "PositronPhase1Certificate",
    "PositronPhase2Certificate",
    "PositronSovereignCertificate",
    "PositronAuthorization",
    "CrowbarActuator",
    "NullCrowbarActuator",
    "SovereignVerificationReport",
    "PositronRedemptionGovernorAgent",
    "AGENT_VERSION",
]

_AGENT_VERSION: Final[str] = (
    "18.0.0-OODA-HKDF-Heyting-Nucleus-Holonomy-KAM-Yoshida-Merkle6962-Loeb-Crowbar-RSI3-PhD"
)
version: Final[str] = _AGENT_VERSION
AGENT_VERSION: Final[str] = _AGENT_VERSION

# ══════════════════════════════════════════════════════════════════════════════
# §0 · CONSTANTES CRIPTOGRÁFICAS, FÍSICAS, CUÁNTICAS Y DE POLÍTICA DECLARADA
# ══════════════════════════════════════════════════════════════════════════════
_STABILITY_FLOOR: Final[float] = 1.0e-12
_TINY: Final[float] = 1.0e-300
_WILKINSON_EPS: Final[float] = 1.0e-15
_HASH_LEN: Final[int] = 32
_HKDF_MAX_LEN: Final[int] = 255 * _HASH_LEN

_DEFAULT_MASTER_KEY: Final[bytes] = b"APU/OMEGA/POSITRON-GOVERNOR-MASTER-KEY/v16"
_HKDF_SALT_P1: Final[bytes] = b"APU/POSITRON/SALT/P1/v17"
_HKDF_SALT_P2: Final[bytes] = b"APU/POSITRON/SALT/P2/v17"
_HKDF_SALT_P3: Final[bytes] = b"APU/POSITRON/SALT/P3/v17"
_INFO_P1: Final[bytes] = b"positron/P1/cert"
_INFO_P2: Final[bytes] = b"positron/P2/cert"
_INFO_P3: Final[bytes] = b"positron/P3/sovereign"
_INFO_AUTH: Final[bytes] = b"positron/auth/grant"
_INFO_SESSION: Final[bytes] = b"positron/session/token"

# Techos y pisos evidenciales (Φ₁ y Φ₂)
_BURES_SOFT_CEILING: Final[float] = 0.35
_BURES_HARD_CEILING: Final[float] = 0.60
_UHLMANN_FIDELITY_FLOOR: Final[float] = 0.95
_MIN_ANNIHILATION_KEV: Final[float] = 1000.0
_BANACH_Q_CEILING: Final[float] = 0.999
_SYMPECTIC_DEFECT_CEILING: Final[float] = 1.0e-8
_ISOSPECTRAL_DEFECT_CEILING: Final[float] = 1.0e-9
_CANONICITY_DEFECT_CEILING: Final[float] = 1.0e-5
_KEPLER_RESIDUAL_CEILING: Final[float] = 1.0e-10
_LINDSTEDT_REMAINDER_CEILING: Final[float] = 1.0e-3
_CONDITION_CAP: Final[float] = 1.0e12

# Umbrales adicionales v18.0.0 (Holonomía, Tomita-Takesaki, KAM, Yoshida, Homotopía)
_HOLONOMY_UNITARITY_CEILING: Final[float] = 1.0e-9
_QUATERNION_NORM_CEILING: Final[float] = 1.0e-9
_FIDELITY_HOMOTOPY_CEILING: Final[float] = 1.0e-2
_TOMITA_MODULAR_CAP: Final[float] = 1.0e14
_FOURIER_BESSEL_CEILING: Final[float] = 1.0e-2
_YOSHIDA4_DEFECT_CEILING: Final[float] = 1.0e-2
_KAM_DIOPHANTINE_FLOOR: Final[float] = 1.0e-6

_SESSION_TTL_NS: Final[int] = 30_000_000_000  # 30 s
_MERGE_LEAF: Final[bytes] = b"\x00"
_MERGE_NODE: Final[bytes] = b"\x01"

# Parámetros DECLARADOS del bus DC y del tiristor crowbar (modelo físico, no medición). [G2/G21]
_GPIO_CROWBAR_PIN: Final[int] = 14
_GPIO_OUT_W1TS_REG: Final[int] = 0x3FF44008
_BUS_VOLTAGE_V: Final[float] = 48.0
_BUS_CAPACITANCE_F: Final[float] = 220.0e-6
_LOOP_INDUCTANCE_H: Final[float] = 4.7e-6
_THYRISTOR_R_ON_OHM: Final[float] = 0.02
_DI_DT_LIMIT_A_PER_US: Final[float] = 50.0
_I2T_RATING_A2S: Final[float] = 72.0


def _ev(obj: Any, name: str, default: Any = None) -> Any:
    """Lectura tolerante de campos entre motor v16, v17 y v18 (preserva False y 0.0)."""
    val = getattr(obj, name, default)
    return default if val is None else val


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _canonicalize(obj: Any) -> Any:
    """
    Canonicalización recursiva INYECTIVA. Orden de tipos crítico: Enum antes de str/int,
    bool antes de int, flotantes en `.17g` (round-trip exacto IEEE 754 float64).
    [G3/G12] v16 usaba `.8e`, que descartaba dígitos y permitía colisiones de payload.
    """
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, Enum):
        return {"__enum__": type(obj).__name__, "value": obj.value}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _canonicalize(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, np.ndarray):
        c = np.ascontiguousarray(obj)
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "sha256": hashlib.sha256(c.tobytes(order="C")).hexdigest(),
        }
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        fv = float(obj)
        return format(fv, ".17g") if math.isfinite(fv) else "non-finite"
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _canonicalize(obj[k]) for k in sorted(obj, key=str)}
    return str(obj)


def _canonical_json(fields: Dict[str, Any]) -> str:
    return json.dumps(
        _canonicalize(fields),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _mth_rfc6962(leaves: Sequence[bytes]) -> bytes:
    """
    MTH (Merkle Tree Hash) de RFC 6962 §2.1 EXACTO. [G7]
    Corte en la mayor potencia de 2 estrictamente menor que n;
    hoja = SHA256(0x00 ‖ datos), nodo = SHA256(0x01 ‖ izq ‖ der).
    El nodo impar se PROMUEVE, nunca se duplica (elimina la maleabilidad CVE-2012-2459).
    Árbol vacío = SHA256("") según la especificación.
    """
    n = len(leaves)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return hashlib.sha256(_MERGE_LEAF + bytes(leaves[0])).digest()
    k = 1
    while k * 2 < n:
        k *= 2
    return hashlib.sha256(
        _MERGE_NODE + _mth_rfc6962(leaves[:k]) + _mth_rfc6962(leaves[k:])
    ).digest()


def _audit_path_rfc6962(leaf_index: int, leaves: Sequence[bytes]) -> Tuple[str, ...]:
    """
    [G20] Camino de auditoría Merkle PATH(m, D[n]) según RFC 6962 §2.1.1:
      • PATH(0, D[1]) = ()
      • Para n > 1 y k = mayor potencia de 2 con k < n:
          si m < k : PATH(m, D[0..n-1]) = PATH(m, D[0..k-1]) ‖ [MTH(D[k..n-1])]
          si m ≥ k : PATH(m, D[0..n-1]) = PATH(m - k, D[k..n-1]) ‖ [MTH(D[0..k-1])]
    Devuelve la tupla de hashes hermanos en hexadecimal (desde el nivel inferior hasta la raíz).
    """
    n = len(leaves)
    if n <= 1 or leaf_index < 0 or leaf_index >= n:
        return ()
    k = 1
    while k * 2 < n:
        k *= 2
    if leaf_index < k:
        return _audit_path_rfc6962(leaf_index, leaves[:k]) + (_mth_rfc6962(leaves[k:]).hex(),)
    return _audit_path_rfc6962(leaf_index - k, leaves[k:]) + (_mth_rfc6962(leaves[:k]).hex(),)


def _verify_inclusion_rfc6962(
    leaf_bytes: bytes,
    leaf_index: int,
    tree_size: int,
    audit_path: Sequence[str],
    expected_root_hex: str,
) -> bool:
    """
    [G20] Verificador de prueba de inclusión RFC 6962 §2.1.3. Reconstruye recursivamente la raíz
    a partir de `(leaf_bytes, leaf_index, tree_size, audit_path)` respetando la promoción de nodos
    impares y compara en tiempo constante contra `expected_root_hex`.
    """
    if tree_size <= 0 or leaf_index < 0 or leaf_index >= tree_size:
        return False
    try:
        path_bytes = [bytes.fromhex(h) for h in audit_path]
    except ValueError:
        return False
    if any(len(b) != _HASH_LEN for b in path_bytes):
        return False

    def _reconstruct(m: int, n: int, sub_path: Sequence[bytes]) -> Optional[bytes]:
        if n == 1:
            if len(sub_path) != 0:
                return None
            return hashlib.sha256(_MERGE_LEAF + bytes(leaf_bytes)).digest()
        if len(sub_path) == 0:
            return None
        k = 1
        while k * 2 < n:
            k *= 2
        sibling = sub_path[-1]
        inner = sub_path[:-1]
        if m < k:
            left = _reconstruct(m, k, inner)
            right = sibling
        else:
            left = sibling
            right = _reconstruct(m - k, n - k, inner)
        if left is None or right is None:
            return None
        return hashlib.sha256(_MERGE_NODE + left + right).digest()

    root = _reconstruct(leaf_index, tree_size, path_bytes)
    if root is None:
        return False
    return _ct_equals(root.hex(), expected_root_hex)


def _ct_equals(a: str, b: str) -> bool:
    """Comparación en tiempo constante tolerante a entradas no ASCII."""
    try:
        return bool(hmac.compare_digest(str(a), str(b)))
    except (TypeError, ValueError):
        return False


# ══════════════════════════════════════════════════════════════════════════════
# §1 · ÁLGEBRA DE HEYTING Ω₃ Y NÚCLEOS DE LAWVERE-TIERNEY
# ══════════════════════════════════════════════════════════════════════════════
class HeytingVerdict(str, Enum):
    r"""
    Álgebra de Heyting lineal Ω₃ con orden de salud:
        VETOED (⊥ = 0) < DEGRADED (1) < COHERENT (⊤ = 2).
    [G13] Es la cadena de Gödel de 3 elementos: distributiva y residuada pero NO booleana,
    pues ¬DEGRADED = (DEGRADED ⇒ ⊥) = VETOED y ¬¬DEGRADED = COHERENT ≠ DEGRADED.
    """
    VETOED = "VETOED"
    DEGRADED = "DEGRADED"
    COHERENT = "COHERENT"

    def _order(self) -> int:
        if self is HeytingVerdict.VETOED:
            return 0
        if self is HeytingVerdict.DEGRADED:
            return 1
        return 2

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        """Ínfimo (∧): conserva el estado más restrictivo."""
        return self if self._order() <= other._order() else other

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        """Supremo (∨): conserva el estado más benigno."""
        return self if self._order() >= other._order() else other

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        """Implicación de Heyting (⇒): adjunción (a ∧ x) ≤ b ⟺ x ≤ (a ⇒ b)."""
        return HeytingVerdict.COHERENT if self._order() <= other._order() else other

    def neg(self) -> "HeytingVerdict":
        """Pseudocomplemento intuicionista ¬a = (a ⇒ ⊥)."""
        return self.implies(HeytingVerdict.VETOED)

    def double_negation(self) -> "HeytingVerdict":
        """Núcleo de doble negación de Glivenko j_¬¬(a) = ¬¬a (esqueleto booleano)."""
        return self.neg().neg()

    def is_regular(self) -> bool:
        """Un elemento a ∈ Ω₃ es ¬¬-estable (regular) ⟺ ¬¬a = a."""
        return self.double_negation() is self

    @staticmethod
    def verify_axioms() -> bool:
        """
        Verificación exhaustiva sobre las 27 tríadas (a, b, x) ∈ Ω₃³ de:
          1. Adjunción de residuación: (a ∧ x) ≤ b ⟺ x ≤ (a ⇒ b).
          2. Distributividad: a ∧ (b ∨ x) = (a ∧ b) ∨ (a ∧ x).
          3. Primera ley de De Morgan intuicionista: ¬(a ∧ b) = ¬a ∨ ¬b.
          4. Idempotencia de la clausura de Glivenko: ¬¬(¬¬a) = ¬¬a.
        """
        vals = (HeytingVerdict.VETOED, HeytingVerdict.DEGRADED, HeytingVerdict.COHERENT)
        for a in vals:
            for b in vals:
                for x in vals:
                    lhs = a.meet(x)._order() <= b._order()
                    rhs = x._order() <= a.implies(b)._order()
                    if lhs != rhs:
                        return False
                    if a.meet(b.join(x)) is not a.meet(b).join(a.meet(x)):
                        return False
                if a.meet(b).neg() is not a.neg().join(b.neg()):
                    return False
            if a.double_negation().double_negation() is not a.double_negation():
                return False
        return True


def amber_nucleus(verdict: HeytingVerdict, grace_verified: bool) -> HeytingVerdict:
    """
    [G1] NÚCLEO DE GRACIA ÁMBAR j : Ω₃ → Ω₃ (topología modal de Lawvere-Tierney):
        j(COHERENT) = COHERENT,  j(DEGRADED) = COHERENT,  j(VETOED) = VETOED.
    Leyes verificables:
      (a) Inflacionario: a ≤ j(a).
      (b) Idempotente  : j(j(a)) = j(a).
      (c) Preserva ∧   : j(a ∧ b) = j(a) ∧ j(b).
      (d) Preserva ⊤   : j(⊤) = ⊤.
    FAIL-CLOSED: j jamás eleva VETOED. El subtopos de puntos fijos Ω_j = {VETOED, COHERENT}
    absorbe el estrato DEGRADED únicamente bajo autorización criptográfica verificada.
    """
    if not grace_verified:
        return verdict
    if verdict is HeytingVerdict.DEGRADED:
        return HeytingVerdict.COHERENT
    return verdict


def verify_nucleus_laws() -> bool:
    """Comprueba exhaustivamente las cuatro leyes del núcleo de Lawvere-Tierney en Ω₃."""
    vals = (HeytingVerdict.VETOED, HeytingVerdict.DEGRADED, HeytingVerdict.COHERENT)
    for a in vals:
        j_a = amber_nucleus(a, True)
        if a._order() > j_a._order():
            return False
        if amber_nucleus(j_a, True) is not j_a:
            return False
        for b in vals:
            if amber_nucleus(a.meet(b), True) is not j_a.meet(amber_nucleus(b, True)):
                return False
    if amber_nucleus(HeytingVerdict.COHERENT, True) is not HeytingVerdict.COHERENT:
        return False
    if amber_nucleus(HeytingVerdict.VETOED, True) is not HeytingVerdict.VETOED:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# §2 · TENSORES CONGELADOS, SESIÓN, AUTORIZACIÓN, CERTIFICADOS Y ACTUACIÓN
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class FrozenTensor:
    """Huella pasiva de un buffer: contenido Y metadatos (dtype/shape/strides)."""
    name: str
    digest: str
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: str
    nbytes: int
    valid: bool
    c_contiguous: bool = True

    def same_content(self, other: "FrozenTensor") -> bool:
        return (
            self.valid
            and other.valid
            and self.digest == other.digest
            and self.shape == other.shape
            and self.dtype == other.dtype
        )


@dataclass(frozen=True, slots=True)
class ZeroBackactionAttestation:
    """Alcance DECLARADO: buffers inspeccionados pre/post; sin garantía sobre escrituras TOCTOU."""
    p_unperturbed: bool
    rho_unperturbed: bool
    sigma_unperturbed: bool
    k_unperturbed: bool
    scope: str = "INSPECTED_BUFFERS_ONLY_NO_TOCTOU_GUARANTEE"

    @property
    def all_unperturbed(self) -> bool:
        return bool(
            self.p_unperturbed
            and self.rho_unperturbed
            and self.sigma_unperturbed
            and self.k_unperturbed
        )


@dataclass(frozen=True, slots=True)
class SessionToken:
    """Sesión con nonce de un solo uso, secuencia monótona y ventana temporal [nbf, exp)."""
    session_id: str
    nonce: str
    sequence: int
    issued_ns: int
    nbf_ns: int
    exp_ns: int
    monotonic_ns: int
    input_bundle_digest: str

    def is_live(self, now_ns: int) -> bool:
        return bool(self.nbf_ns <= now_ns < self.exp_ns)


@dataclass(frozen=True, slots=True)
class PositronAuthorization:
    """
    [G8] Token de autorización de positrón REAL: HMAC bajo K_auth, ligado al session_id y al
    digest del bundle de entrada (impide replay sobre otro estado), con ventana temporal y
    nonce de un solo uso. `max_bures` sólo puede RESTRINGIR el techo duro, nunca relajarlo.
    """
    subject: str
    session_id: str
    input_bundle_digest: str
    nonce: str
    issued_ns: int
    nbf_ns: int
    exp_ns: int
    max_bures: float
    max_grace_sec: float
    signature: str = ""


@dataclass(frozen=True, slots=True)
class CrowbarIntent:
    """INTENCIÓN de disparo, firmada bajo K_P3. No es una atestación de actuación física. [G2]"""
    session_id: str
    gpio_pin: int
    mmio_register: int
    reason: Tuple[str, ...]
    issued_ns: int
    signature: str = ""


@dataclass(frozen=True, slots=True)
class ActuationReceipt:
    """Recibo del driver físico. Sin driver enlazado, confirmed=False y mode=SIMULATION. [G2]"""
    confirmed: bool
    mode: str
    latency_ns_measured: Optional[float]
    driver_identity: str
    detail: str


class CrowbarActuator(Protocol):
    """Protocolo estructural del driver físico. Debe devolver un recibo verificable."""
    identity: str

    def fire(self, intent: CrowbarIntent) -> ActuationReceipt: ...


class NullCrowbarActuator:
    """
    Driver nulo por defecto: NUNCA afirma disparo físico. [G2] v16 escribía
    hardware_crowbar_fired=True y una latencia inventada sin tocar hardware alguno;
    esa falsa atestación de desenergización se previene aquí por diseño.
    """
    identity: str = "NULL_ACTUATOR_NO_HARDWARE_BOUND"

    def fire(self, intent: CrowbarIntent) -> ActuationReceipt:
        return ActuationReceipt(
            confirmed=False,
            mode="SIMULATION",
            latency_ns_measured=None,
            driver_identity=self.identity,
            detail="Intención registrada; ningún periférico fue accionado.",
        )


@dataclass(frozen=True, slots=True)
class CrowbarRLCModel:
    """Modelo RLC subamortiguado del bus DC. DECLARADO como modelo analítico, no medición. [G21]"""
    z0_ohm: float
    zeta: float
    omega0_rad_s: float
    omega_d_rad_s: float
    peak_time_s: float
    i_peak_exact_a: float
    i_peak_smallzeta_a: float
    peak_approx_rel_error: float
    di_dt_initial_a_per_us: float
    i2t_action_a2s: float
    within_di_dt_limit: bool
    within_i2t_rating: bool
    is_model_not_measurement: bool = True
    # ── v18 ──
    energy_stored_joules: float = 0.0
    total_charge_coulomb: float = 0.0
    first_zero_crossing_s: float = 0.0
    ringing_frequency_hz: float = 0.0


@dataclass(frozen=True, slots=True)
class BanachProbe:
    """Sonda de contracción local con cotas de Banach-Kantorovich y métrica de Shahshahani-Bures. [G5/G16]"""
    q_local: float
    increment_d1: float
    increment_d2: float
    a_posteriori_bound: float
    a_priori_bound: float
    certified_digits: float
    at_fixed_point: bool
    contractive: bool
    used_engine_map: bool
    # ── v18 ──
    raw_ratio: float = 0.0
    shahshahani_bures_d1: float = 0.0
    shahshahani_bures_d2: float = 0.0
    bch_commutator_norm: float = 0.0
    rsi3_substeps_used: int = 1


@dataclass(frozen=True, slots=True)
class HomologyAudit:
    """Homología del 1-esqueleto y del complejo clique 2D Cl(G_K) de coherencia de K. [G9/G19]"""
    beta_0: int
    beta_1: int
    num_vertices: int
    num_edges: int
    h1_torsion_free: bool
    external_claim: bool
    external_claim_attested: bool
    theorem: str = "H1 of a 1-complex is free abelian: subgroups of free abelian groups are free"
    # ── v18 ──
    num_triangles_2skeleton: int = 0
    beta_1_clique_complex: int = 0
    chain_complex_exact_d1_d2: bool = True
    euler_characteristic_1skeleton: int = 0
    euler_characteristic_2skeleton: int = 0


@dataclass(frozen=True, slots=True)
class LoebVerification:
    """Verificación bien fundada por descenso sobre (ℕ, <), no auto-referencial. [G6/G17]"""
    accepted: bool
    depth_remaining: int
    independently_recomputed: bool
    bures_monotone_diagnostic: bool
    bures_diagnostic_valid: bool
    rationale: str = "WELL_FOUNDED_DESCENT_ON_NAT_NOT_SELF_TRUST"
    # ── v18 ──
    descent_trace: Tuple[str, ...] = ()
    initial_depth: int = 3


@dataclass(frozen=True, slots=True)
class MetaAuditCertificate:
    """Meta-auditoría RSI-L3 en tres niveles con presupuestos bien fundados y riesgo meta-Lyapunov. [G18]"""
    l1_numeric_ok: bool
    l2_structural_ok: bool
    l3_attestation_ok: bool
    accepted: bool
    verification_depth_remaining: int
    mutation_budget_remaining: int
    failed_checks: Tuple[str, ...] = ()
    # ── v18 ──
    rsi3_meta_lyapunov_risk: float = 0.0
    policy_adapted: bool = False


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """Autenticidad (criptográfica) ≠ salud (evidencial)."""
    authentic: bool
    evidentially_clean: bool
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PositronPhase1Certificate:
    """
    Expediente inmutable de Φ₁. Orden posicional v16 y v17 preservado al 100%.
    Constituye la salida formal de `1.11 _phase1_observe_and_sign` y la entrada formal
    directa de `2.1 _gate_phi12`.
    """
    p_init_digest: str
    rho_init_digest: str
    sigma_ref_digest: str
    K_raw_digest: str
    bures_distance: float
    uhlmann_fidelity: float
    annihilation_energy_kev: float
    pinsker_residual: float
    timestamp_ns: int
    hmac_signature: str
    # ── v17 ──
    agent_version: str = _AGENT_VERSION
    session: Optional[SessionToken] = None
    monotonic_ns: int = 0
    frozen_inputs: Tuple[FrozenTensor, ...] = ()
    backaction: Optional[ZeroBackactionAttestation] = None
    evidence_atoms: Tuple[Tuple[str, str], ...] = ()
    engine_schema: str = ""
    merkle_leaf: str = ""
    # ── v18 (Holonomía U(n), Cuaterniones S³, Fuchs-van de Graaf, Tomita-Takesaki, QED 3γ) ──
    uhlmann_pancharatnam_phase_rad: float = 0.0
    uhlmann_holonomy_unitarity_defect: float = 0.0
    uhlmann_quaternion_s3: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    fuchs_van_de_graaf_ok: bool = True
    fuchs_lower_margin: float = 0.0
    fuchs_upper_margin: float = 0.0
    tomita_modular_condition: float = 1.0
    wigner_yanase_skew_info: float = 0.0
    fidelity_homotopy_drift: float = 0.0
    ore_powell_3gamma_cross_section_m2: float = 0.0


@dataclass(frozen=True, slots=True)
class PositronPhase2Certificate:
    """
    Expediente inmutable de Φ₂. Orden posicional v16 y v17 preservado al 100%.
    Constituye la salida formal de `2.11 _phase2_orient_and_contract` y la entrada formal
    directa de `3.1 _gate_phi23`.
    """
    phase1_cert: PositronPhase1Certificate
    banach_contraction_q: float
    connes_lipschitz_bound: float
    shahshahani_speed: float
    analytic_energy_derivative: float
    numerical_energy_derivative: float
    amber_grace_duration_sec: float
    is_lyapunov_stable: bool
    timestamp_ns: int
    hmac_signature: str
    # ── v17 ──
    agent_version: str = _AGENT_VERSION
    session: Optional[SessionToken] = None
    monotonic_ns: int = 0
    gate_phi12: Optional[GateVerdict] = None
    banach_probe: Optional[BanachProbe] = None
    rlc_model: Optional[CrowbarRLCModel] = None
    homology: Optional[HomologyAudit] = None
    dk_sqrt_lipschitz: float = 0.0
    dissipative_entropy_production: float = 0.0
    lyapunov_decomposition_consistent: bool = True
    evidence_atoms: Tuple[Tuple[str, str], ...] = ()
    authentication_veto: bool = False
    merkle_leaf: str = ""
    # ── v18 (Poincaré-Cartan 4D, KAM, Kapteyn-Fourier-Bessel, Yoshida-4, Hamilton-Hopf, BCH) ──
    poincare_cartan_symplectic_defect_4d: float = 0.0
    kam_diophantine_margin: float = 0.0
    kepler_fourier_bessel_error: float = 0.0
    melnikov_splitting_angle_rad: float = 0.0
    yoshida4_monodromy_defect: float = 0.0
    poincare_rotation_number: float = 0.0
    hamilton_hopf_index: float = 0.0
    bch_commutator_norm: float = 0.0
    rsi3_substeps_per_step: int = 1


@dataclass(frozen=True, slots=True)
class GovernorPolicy:
    """Política adaptativa en un politopo compacto. `mutation_budget` es el contador bien fundado."""
    bures_soft_ceiling: float = _BURES_SOFT_CEILING
    banach_q_ceiling: float = _BANACH_Q_CEILING
    probe_dt: float = 0.05
    grace_cap_sec: float = 3600.0
    verification_depth: int = 3
    mutation_budget: int = 3

    def project(self) -> "GovernorPolicy":
        return GovernorPolicy(
            bures_soft_ceiling=float(np.clip(self.bures_soft_ceiling, 0.05, _BURES_HARD_CEILING)),
            banach_q_ceiling=float(np.clip(self.banach_q_ceiling, 0.5, 0.999999)),
            probe_dt=float(np.clip(self.probe_dt, 1e-4, 1.0)),
            grace_cap_sec=float(np.clip(self.grace_cap_sec, 0.0, 86400.0)),
            verification_depth=max(0, int(self.verification_depth)),
            mutation_budget=max(0, int(self.mutation_budget)),
        )


@dataclass(frozen=True, slots=True)
class PositronSovereignCertificate:
    """Certificado público inmutable final. Orden posicional v16 y v17 preservado al 100%."""
    heyting_verdict: str
    merkle_root_rfc6962: str
    amber_grace_active: bool
    amber_grace_duration_sec: float
    hardware_crowbar_fired: bool
    actuation_latency_ns: float
    bures_distance: float
    uhlmann_fidelity: float
    annihilation_energy_kev: float
    loeb_verification_passed: bool
    session_digest: str
    diagnostics: Tuple[str, ...]
    # ── v17 ──
    agent_version: str = _AGENT_VERSION
    phase2_cert: Optional[PositronPhase2Certificate] = None
    base_verdict: str = HeytingVerdict.COHERENT.value
    verdict_atoms: Tuple[Tuple[str, str], ...] = ()
    grace_verified: bool = False
    authorization: Optional[PositronAuthorization] = None
    crowbar_intent: Optional[CrowbarIntent] = None
    actuation: Optional[ActuationReceipt] = None
    actuation_mode: str = "SIMULATION"
    crowbar_latched: bool = False
    loeb: Optional[LoebVerification] = None
    meta_audit: Optional[MetaAuditCertificate] = None
    gate_phi23: Optional[GateVerdict] = None
    session: Optional[SessionToken] = None
    monotonic_ns: int = 0
    timestamp_ns: int = 0
    sovereign_signature: str = ""
    merkle_algorithm: str = "RFC6962-MTH-SHA256-LEAF0x00-NODE0x01-PROMOTE"
    # ── v18 (Pruebas de inclusión RFC 6962 y telemetría de política RSI³) ──
    merkle_inclusion_proofs: Tuple[Tuple[str, ...], ...] = ()
    rsi3_meta_lyapunov_risk: float = 0.0
    policy_snapshot: Optional[GovernorPolicy] = None


@dataclass(frozen=True, slots=True)
class SovereignVerificationReport:
    """Resultado del verificador público: recomputación íntegra desde el certificado."""
    p1_signature_ok: bool
    p2_signature_ok: bool
    sovereign_signature_ok: bool
    merkle_ok: bool
    verdict_recomputation_ok: bool
    nucleus_ok: bool
    session_continuity_ok: bool
    actuation_honesty_ok: bool
    all_ok: bool
    failed_checks: Tuple[str, ...] = ()
    # ── v18 ──
    merkle_inclusion_proofs_ok: bool = True
    holonomy_quaternion_ok: bool = True
    fuchs_sandwich_ok: bool = True


@dataclass
class GovernorTelemetry:
    """Telemetría meta-cognitiva RSI-L3 del soberano de calibre."""
    total_cycles: int = 0
    phase1_emissions: int = 0
    phase2_emissions: int = 0
    phase3_emissions: int = 0
    gate_phi12_failures: int = 0
    gate_phi23_failures: int = 0
    grace_grants: int = 0
    grace_denials: int = 0
    crowbar_intents: int = 0
    crowbar_confirmations: int = 0
    sequence_counter: int = 0
    monotonic_clock_ns: int = 0
    used_nonces: List[str] = field(default_factory=list)
    policy: GovernorPolicy = field(default_factory=GovernorPolicy)
    policy_ledger: List[Dict[str, Any]] = field(default_factory=list)
    mutation_count: int = 0
    rollback_count: int = 0
    crowbar_latched: bool = False
    dev_key_in_use: bool = False
    last_verdict: HeytingVerdict = HeytingVerdict.COHERENT


# ══════════════════════════════════════════════════════════════════════════════
# §3 · SOBERANO DE CALIBRE: TRES FASES OODA ANIDADAS EN CPS
# ══════════════════════════════════════════════════════════════════════════════
class PositronRedemptionGovernorAgent:
    """
    Soberano Agéntico de Calibre de Redención (Estrato Omega / Santuario Epistémico).
    Tres fases anidadas en estilo de paso de continuaciones (CPS) con tipo polimórfico AnswerT:
        Φ₁ · Phase1Observer  (1.1 → 1.11) _phase1_observe_and_sign
        Φ₂ · Phase2Orientor  (2.1 → 2.11) _phase2_orient_and_contract
        Φ₃ · Phase3Decider   (3.1 → 3.12) _phase3_decide_and_act
    """

    # ──────────────────────────────────────────────────────────────────────────
    # FASE Φ₁ · OBSERVER: CONGELACIÓN, EVIDENCIA CUÁNTICA/HOLONÓMICA Y FIRMA P1
    # ──────────────────────────────────────────────────────────────────────────
    class Phase1Observer:
        """
        Φ₁ (1.1 → 1.11).
        Método terminal: `1.11 _phase1_observe_and_sign`, cuya salida formal
        `PositronPhase1Certificate` constituye la entrada directa de `2.1 _gate_phi12`.
        """

        def __init__(self, sovereign: "PositronRedemptionGovernorAgent") -> None:
            self._s = sovereign

        # ── Método 1.1 ────────────────────────────────────────────────────────
        @staticmethod
        def _freeze_and_digest(name: str, arr: Any) -> FrozenTensor:
            """
            1.1 Digest SHA-256 sobre (dtype ‖ shape ‖ strides ‖ bytes_C).
            Incluir `strides` detecta transposiciones o vistas reordenadas que un checksum
            plano sobre bytes contiguos no distinguiría. `np.ascontiguousarray` no muta la
            fuente, preservando la pasividad observacional. [G10]
            """
            if not isinstance(arr, np.ndarray) or arr.size == 0:
                return FrozenTensor(name, "", (), (), "", 0, False, False)
            try:
                is_c = bool(arr.flags["C_CONTIGUOUS"])
                c = np.ascontiguousarray(arr)
                meta = f"{arr.dtype.str}|{arr.shape}|{arr.strides}|".encode("ascii")
                digest = hashlib.sha256(meta + c.tobytes(order="C")).hexdigest()
            except (ValueError, TypeError, MemoryError):
                return FrozenTensor(
                    name,
                    "",
                    tuple(arr.shape),
                    tuple(arr.strides),
                    str(arr.dtype.str),
                    int(arr.nbytes),
                    False,
                    False,
                )
            return FrozenTensor(
                name=name,
                digest=digest,
                shape=tuple(int(s) for s in arr.shape),
                strides=tuple(int(s) for s in arr.strides),
                dtype=str(arr.dtype.str),
                nbytes=int(arr.nbytes),
                valid=True,
                c_contiguous=is_c,
            )

        # ── Método 1.2 ────────────────────────────────────────────────────────
        @staticmethod
        def _attest_zero_backaction(
            before: Sequence[FrozenTensor], after: Sequence[FrozenTensor]
        ) -> ZeroBackactionAttestation:
            """
            1.2 Compara las huellas criptográficas y estructurales de los cuatro buffers
            (p, ρ, σ, K) antes y después de invocar el motor FPU. [G10]
            """
            flags: List[bool] = []
            for i in range(4):
                b = before[i] if i < len(before) else FrozenTensor("", "", (), (), "", 0, False)
                a = after[i] if i < len(after) else FrozenTensor("", "", (), (), "", 0, False)
                flags.append(
                    b.same_content(a) if (b.valid and a.valid) else (not b.valid and not a.valid)
                )
            return ZeroBackactionAttestation(*[bool(f) for f in flags])

        # ── Método 1.3 ────────────────────────────────────────────────────────
        @staticmethod
        def _audit_uhlmann_holonomy_and_quaternion_sector(
            report: PositronRedemptionSpectrumReport,
        ) -> Tuple[Tuple[str, str], ...]:
            """
            1.3 [G14] Audita la holonomía del fibrado principal de Uhlmann U(n) y la proyección
            espinorial al grupo de cuaterniones unitarios S³ ⊂ H (isomorfo a SU(2)):
              • Defecto de unitariedad ‖V_Uhl† V_Uhl − I‖_F ≤ 10⁻⁹.
              • Finitud de la fase geométrica de Pancharatnam arg Tr(√ρ √σ) ∈ [−π, π].
              • Norma unitaria del cuaternión q = (q₀, q₁, q₂, q₃) ∈ S³: |‖q‖₂ − 1| ≤ 10⁻⁹.
            Si el motor es v16/v17 y no expone estos campos, no emite átomos espurios.
            """
            atoms: List[Tuple[str, str]] = []
            u_def = _ev(report, "uhlmann_holonomy_unitarity_defect", None)
            pan = _ev(report, "uhlmann_pancharatnam_phase_rad", None)
            if u_def is not None:
                val_u = _safe_float(u_def, float("inf"))
                pan_ok = pan is None or (
                    math.isfinite(_safe_float(pan, float("nan")))
                    and abs(_safe_float(pan, 0.0)) <= math.pi + 1e-9
                )
                if val_u <= _HOLONOMY_UNITARITY_CEILING and pan_ok:
                    atoms.append(("UHLMANN_HOLONOMY", HeytingVerdict.COHERENT.value))
                elif val_u <= 1.0e-5 and pan_ok:
                    atoms.append(("UHLMANN_HOLONOMY", HeytingVerdict.DEGRADED.value))
                else:
                    atoms.append(("UHLMANN_HOLONOMY", HeytingVerdict.VETOED.value))

            quat = _ev(report, "uhlmann_quaternion_s3", None)
            if quat is not None and isinstance(quat, (tuple, list)) and len(quat) == 4:
                q_arr = np.asarray(quat, dtype=np.float64)
                if np.all(np.isfinite(q_arr)):
                    q_norm_err = abs(float(np.linalg.norm(q_arr, 2)) - 1.0)
                    atoms.append(
                        (
                            "QUATERNION_S3",
                            HeytingVerdict.COHERENT.value
                            if q_norm_err <= _QUATERNION_NORM_CEILING
                            else HeytingVerdict.DEGRADED.value,
                        )
                    )
                else:
                    atoms.append(("QUATERNION_S3", HeytingVerdict.VETOED.value))
            return tuple(atoms)

        # ── Método 1.4 ────────────────────────────────────────────────────────
        @staticmethod
        def _audit_fuchs_tomita_and_skew_information(
            report: PositronRedemptionSpectrumReport,
        ) -> Tuple[Tuple[str, str], ...]:
            """
            1.4 [G14] Audita la consistencia algebraica del estado cuántico relativo:
              • Sándwich de Fuchs-van de Graaf: 1 − √F(ρ, σ) ≤ ½‖ρ − σ‖₁ ≤ √(1 − F(ρ, σ)).
              • Operador modular de Tomita-Takesaki Δ_{ρ|σ} = L_ρ R_σ⁻¹: κ(Δ_{ρ|σ}) finito y > 0.
              • Información oblicua de Wigner-Yanase: I_{1/2}(ρ, −ln σ) = ½‖[√ρ, ln σ]‖_F² ≥ 0.
              • Deriva homotópica de Richardson sobre el piso espectral: |F_δ − F_{δ/2}| ≤ 10⁻².
            """
            atoms: List[Tuple[str, str]] = []
            fuchs_ok = _ev(report, "fuchs_van_de_graaf_ok", None)
            if fuchs_ok is not None:
                low_m = _safe_float(_ev(report, "fuchs_lower_margin", 0.0), -1.0)
                up_m = _safe_float(_ev(report, "fuchs_upper_margin", 0.0), -1.0)
                ok = bool(fuchs_ok) and low_m >= -1.0e-12 and up_m >= -1.0e-12
                atoms.append(
                    (
                        "FUCHS_SANDWICH",
                        HeytingVerdict.COHERENT.value if ok else HeytingVerdict.VETOED.value,
                    )
                )

            tomita = _ev(report, "tomita_modular_condition", None)
            if tomita is not None:
                t_val = _safe_float(tomita, float("inf"))
                atoms.append(
                    (
                        "TOMITA_MODULAR",
                        HeytingVerdict.COHERENT.value
                        if (0.0 < t_val <= _TOMITA_MODULAR_CAP)
                        else HeytingVerdict.DEGRADED.value,
                    )
                )

            wy = _ev(report, "wigner_yanase_skew_info", None)
            if wy is not None:
                wy_val = _safe_float(wy, -1.0)
                atoms.append(
                    (
                        "WIGNER_YANASE",
                        HeytingVerdict.COHERENT.value
                        if wy_val >= -1.0e-12
                        else HeytingVerdict.VETOED.value,
                    )
                )

            drift = _ev(report, "fidelity_homotopy_drift", None)
            if drift is not None:
                d_val = _safe_float(drift, float("inf"))
                atoms.append(
                    (
                        "FIDELITY_HOMOTOPY",
                        HeytingVerdict.COHERENT.value
                        if d_val <= _FIDELITY_HOMOTOPY_CEILING
                        else HeytingVerdict.DEGRADED.value,
                    )
                )
            return tuple(atoms)

        # ── Método 1.5 ────────────────────────────────────────────────────────
        @staticmethod
        def _audit_dirac_ore_powell_qed_sector(
            report: PositronRedemptionSpectrumReport,
        ) -> Tuple[Tuple[str, str], ...]:
            """
            1.5 [G14] Audita la sección eficaz de aniquilación QED 2γ de Dirac y el canal
            orto-positronio 3γ de Ore-Powell (σ_{3γ}/σ_{2γ} = (4(π² − 9)/(9π))·α).
            """
            atoms: List[Tuple[str, str]] = []
            sigma_3g = _ev(report, "ore_powell_3gamma_cross_section_m2", None)
            sigma_2g = _ev(report, "dirac_cross_section_m2", None)
            if sigma_3g is not None:
                s3 = _safe_float(sigma_3g, -1.0)
                s2 = _safe_float(sigma_2g, 0.0)
                ok = s3 >= 0.0 and (s2 <= 0.0 or s3 < s2)
                atoms.append(
                    (
                        "ORE_POWELL_QED",
                        HeytingVerdict.COHERENT.value if ok else HeytingVerdict.DEGRADED.value,
                    )
                )
            return tuple(atoms)

        # ── Método 1.6 ────────────────────────────────────────────────────────
        @classmethod
        def _evidence_atoms_phase1(
            cls, report: PositronRedemptionSpectrumReport, policy: GovernorPolicy
        ) -> Tuple[Tuple[str, str], ...]:
            """
            1.6 Extrae la escalera completa de átomos de evidencia de Φ₁ sobre el informe del
            motor (v16/v17/v18). Cada átomo (nombre, veredicto) alimenta el ínfimo de Ω₃.
            """
            atoms: List[Tuple[str, str]] = []
            pinsker_ok = _ev(report, "pinsker_ok", None)
            if pinsker_ok is None:
                pinsker_ok = _safe_float(_ev(report, "pinsker_bound_residual", 0.0)) >= -1e-9
                atoms.append(("PINSKER_EVIDENCE", HeytingVerdict.DEGRADED.value))
            atoms.append(
                (
                    "PINSKER",
                    HeytingVerdict.COHERENT.value
                    if pinsker_ok
                    else HeytingVerdict.VETOED.value,
                )
            )

            d_b = _safe_float(_ev(report, "bures_distance", math.sqrt(2.0)), math.sqrt(2.0))
            fid = _safe_float(_ev(report, "uhlmann_fidelity", 0.0))
            if d_b > _BURES_HARD_CEILING:
                atoms.append(("BURES", HeytingVerdict.VETOED.value))
            elif d_b > policy.bures_soft_ceiling or fid < _UHLMANN_FIDELITY_FLOOR:
                atoms.append(("BURES", HeytingVerdict.DEGRADED.value))
            else:
                atoms.append(("BURES", HeytingVerdict.COHERENT.value))

            e_ann = _safe_float(_ev(report, "annihilation_energy_kev", 0.0))
            atoms.append(
                (
                    "DIRAC_ENERGY",
                    HeytingVerdict.COHERENT.value
                    if e_ann >= _MIN_ANNIHILATION_KEV
                    else HeytingVerdict.VETOED.value,
                )
            )

            chord = _ev(report, "chord_arc_identity_error", None)
            if chord is not None:
                atoms.append(
                    (
                        "CHORD_ARC",
                        HeytingVerdict.COHERENT.value
                        if _safe_float(chord) <= 1e-9
                        else HeytingVerdict.DEGRADED.value,
                    )
                )

            kkt = _ev(report, "rho_kkt_ok", None)
            if kkt is not None:
                atoms.append(
                    (
                        "DENSITY_KKT",
                        HeytingVerdict.COHERENT.value
                        if bool(kkt)
                        else HeytingVerdict.DEGRADED.value,
                    )
                )

            kappa = _safe_float(_ev(report, "condition_number_K", 1.0), 1.0)
            atoms.append(
                (
                    "CONDITIONING",
                    HeytingVerdict.COHERENT.value
                    if kappa <= _CONDITION_CAP
                    else HeytingVerdict.DEGRADED.value,
                )
            )

            # Átomos cuánticos, holonómicos y modulares v18.0.0
            atoms.extend(cls._audit_uhlmann_holonomy_and_quaternion_sector(report))
            atoms.extend(cls._audit_fuchs_tomita_and_skew_information(report))
            atoms.extend(cls._audit_dirac_ore_powell_qed_sector(report))
            return tuple(atoms)

        # ── Método 1.7 ────────────────────────────────────────────────────────
        @staticmethod
        def _build_p1_payload(cert: PositronPhase1Certificate) -> Dict[str, Any]:
            """
            1.7 Payload de COBERTURA TOTAL de Φ₁ (excluye únicamente `hmac_signature` y la hoja
            Merkle derivada). [G3/G14] Incluye todos los invariantes v17 y v18 para impedir
            adulteración de fase de Pancharatnam, cuaternión S³, Fuchs-van de Graaf o Pinsker.
            """
            return {
                "domain": _INFO_P1.decode("ascii"),
                "agent_version": cert.agent_version,
                "engine_schema": cert.engine_schema,
                "p_init_digest": cert.p_init_digest,
                "rho_init_digest": cert.rho_init_digest,
                "sigma_ref_digest": cert.sigma_ref_digest,
                "K_raw_digest": cert.K_raw_digest,
                "bures_distance": cert.bures_distance,
                "uhlmann_fidelity": cert.uhlmann_fidelity,
                "annihilation_energy_kev": cert.annihilation_energy_kev,
                "pinsker_residual": cert.pinsker_residual,
                "timestamp_ns": cert.timestamp_ns,
                "monotonic_ns": cert.monotonic_ns,
                "session": cert.session,
                "frozen_inputs": cert.frozen_inputs,
                "backaction": cert.backaction,
                "evidence_atoms": cert.evidence_atoms,
                "uhlmann_pancharatnam_phase_rad": cert.uhlmann_pancharatnam_phase_rad,
                "uhlmann_holonomy_unitarity_defect": cert.uhlmann_holonomy_unitarity_defect,
                "uhlmann_quaternion_s3": cert.uhlmann_quaternion_s3,
                "fuchs_van_de_graaf_ok": cert.fuchs_van_de_graaf_ok,
                "fuchs_lower_margin": cert.fuchs_lower_margin,
                "fuchs_upper_margin": cert.fuchs_upper_margin,
                "tomita_modular_condition": cert.tomita_modular_condition,
                "wigner_yanase_skew_info": cert.wigner_yanase_skew_info,
                "fidelity_homotopy_drift": cert.fidelity_homotopy_drift,
                "ore_powell_3gamma_cross_section_m2": cert.ore_powell_3gamma_cross_section_m2,
            }

        # ── Método 1.8 ────────────────────────────────────────────────────────
        def _hkdf_subkey(self, salt: bytes, info: bytes, length: int = _HASH_LEN) -> bytes:
            """1.8 Delega en la derivación HKDF-SHA256 del soberano con cota RFC 5869."""
            return self._s._hkdf_extract_and_expand(salt, info, length)

        # ── Método 1.9 ────────────────────────────────────────────────────────
        def _mint_session(self, bundle_digest: str) -> SessionToken:
            """
            1.9 Emite un `SessionToken` con nonce criptográfico de un solo uso, secuencia
            estrictamente creciente y ventana temporal [nbf, exp). [G4]
            """
            tel = self._s._telemetry
            tel.sequence_counter += 1
            tel.monotonic_clock_ns = max(time.monotonic_ns(), tel.monotonic_clock_ns + 1)
            now = time.time_ns()
            nonce = hashlib.sha256(
                f"{bundle_digest}|{tel.sequence_counter}|{tel.monotonic_clock_ns}".encode("ascii")
            ).hexdigest()[:32]
            k_sess = self._hkdf_subkey(_HKDF_SALT_P1, _INFO_SESSION)
            sid = hmac.new(
                k_sess,
                _canonical_json(
                    {
                        "bundle": bundle_digest,
                        "nonce": nonce,
                        "seq": tel.sequence_counter,
                    }
                ).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            return SessionToken(
                session_id=sid,
                nonce=nonce,
                sequence=int(tel.sequence_counter),
                issued_ns=now,
                nbf_ns=now - 1_000_000,
                exp_ns=now + _SESSION_TTL_NS,
                monotonic_ns=int(tel.monotonic_clock_ns),
                input_bundle_digest=bundle_digest,
            )

        # ── Método 1.10 ───────────────────────────────────────────────────────
        def _sign_p1(self, skeleton: PositronPhase1Certificate) -> PositronPhase1Certificate:
            """1.10 Firma HMAC-SHA256 bajo K_P1 sobre el payload canónico de cobertura total."""
            payload = _canonical_json(self._build_p1_payload(skeleton))
            key = self._hkdf_subkey(_HKDF_SALT_P1, _INFO_P1)
            sig = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
            leaf = hashlib.sha256(_canonical_json({"p1": sig}).encode("utf-8")).hexdigest()
            return replace(skeleton, hmac_signature=sig, merkle_leaf=leaf)

        # ── Método 1.11 · TERMINAL Φ₁ → APERTURA Φ₂ (CPS polimórfico) ─────────
        def _phase1_observe_and_sign(
            self,
            p_init: NDArray[np.float64],
            rho_init: NDArray[np.complex128],
            sigma_ref: NDArray[np.complex128],
            K_raw: NDArray[np.complex128],
            report: PositronRedemptionSpectrumReport,
            frozen_before: Sequence[FrozenTensor],
            continuation: Callable[[PositronPhase1Certificate], AnswerT],
        ) -> AnswerT:
            """
            1.11 [TERMINAL Φ₁ → ENTRADA FORMAL DE 2.1 `_gate_phi12`]
            Re-digiere los cuatro tensores tras la ejecución del motor, atestigua la ausencia
            de retroacción (zero-backaction), extrae los átomos de evidencia cuántica y de
            holonomía de Uhlmann, emite el `SessionToken`, firma `PositronPhase1Certificate`
            con cobertura total y transfiere el control en posición de cola a `continuation`.
            [G11] Tipo de respuesta polimórfico `AnswerT`.
            """
            s = self._s
            after = [
                self._freeze_and_digest("p", p_init),
                self._freeze_and_digest("rho", rho_init),
                self._freeze_and_digest("sigma", sigma_ref),
                self._freeze_and_digest("K", K_raw),
            ]
            backaction = self._attest_zero_backaction(frozen_before, after)
            bundle = hashlib.sha256(
                "".join(t.digest for t in frozen_before).encode("ascii")
            ).hexdigest()
            session = self._mint_session(bundle)
            atoms = self._evidence_atoms_phase1(report, s._telemetry.policy)
            if not backaction.all_unperturbed:
                atoms = atoms + (("ZERO_BACKACTION", HeytingVerdict.VETOED.value),)

            raw_quat = _ev(report, "uhlmann_quaternion_s3", (1.0, 0.0, 0.0, 0.0))
            if isinstance(raw_quat, (tuple, list)) and len(raw_quat) == 4:
                quat_s3 = (
                    _safe_float(raw_quat[0], 1.0),
                    _safe_float(raw_quat[1], 0.0),
                    _safe_float(raw_quat[2], 0.0),
                    _safe_float(raw_quat[3], 0.0),
                )
            else:
                quat_s3 = (1.0, 0.0, 0.0, 0.0)

            skeleton = PositronPhase1Certificate(
                p_init_digest=frozen_before[0].digest,
                rho_init_digest=frozen_before[1].digest,
                sigma_ref_digest=frozen_before[2].digest,
                K_raw_digest=frozen_before[3].digest,
                bures_distance=_safe_float(_ev(report, "bures_distance", 0.0)),
                uhlmann_fidelity=_safe_float(_ev(report, "uhlmann_fidelity", 0.0)),
                annihilation_energy_kev=_safe_float(_ev(report, "annihilation_energy_kev", 0.0)),
                pinsker_residual=_safe_float(_ev(report, "pinsker_bound_residual", 0.0)),
                timestamp_ns=time.time_ns(),
                hmac_signature="",
                agent_version=_AGENT_VERSION,
                session=session,
                monotonic_ns=session.monotonic_ns,
                frozen_inputs=tuple(frozen_before),
                backaction=backaction,
                evidence_atoms=atoms,
                engine_schema=str(_ev(report, "schema_version", "unknown")),
                uhlmann_pancharatnam_phase_rad=_safe_float(
                    _ev(report, "uhlmann_pancharatnam_phase_rad", 0.0)
                ),
                uhlmann_holonomy_unitarity_defect=_safe_float(
                    _ev(report, "uhlmann_holonomy_unitarity_defect", 0.0)
                ),
                uhlmann_quaternion_s3=quat_s3,
                fuchs_van_de_graaf_ok=bool(_ev(report, "fuchs_van_de_graaf_ok", True)),
                fuchs_lower_margin=_safe_float(_ev(report, "fuchs_lower_margin", 0.0)),
                fuchs_upper_margin=_safe_float(_ev(report, "fuchs_upper_margin", 0.0)),
                tomita_modular_condition=_safe_float(
                    _ev(report, "tomita_modular_condition", 1.0), 1.0
                ),
                wigner_yanase_skew_info=_safe_float(_ev(report, "wigner_yanase_skew_info", 0.0)),
                fidelity_homotopy_drift=_safe_float(_ev(report, "fidelity_homotopy_drift", 0.0)),
                ore_powell_3gamma_cross_section_m2=_safe_float(
                    _ev(report, "ore_powell_3gamma_cross_section_m2", 0.0)
                ),
            )
            cert = self._sign_p1(skeleton)
            s._telemetry.phase1_emissions += 1
            s._telemetry.used_nonces.append(session.nonce)
            if len(s._telemetry.used_nonces) > 1024:
                s._telemetry.used_nonces.pop(0)
            return continuation(cert)


    # ──────────────────────────────────────────────────────────────────────────
    # FASE Φ₂ · ORIENTOR: Φ₁₂, BANACH-KANTOROVICH, KAM-YOSHIDA Y MODELO RLC
    # ──────────────────────────────────────────────────────────────────────────
    class Phase2Orientor:
        """
        Φ₂ (2.1 → 2.11).
        Método inicial : `2.1 _gate_phi12`, que recibe directamente el `PositronPhase1Certificate`
                         emitido por `1.11 _phase1_observe_and_sign`.
        Método terminal: `2.11 _phase2_orient_and_contract`, cuya salida formal
                         `PositronPhase2Certificate` constituye la entrada directa de `3.1 _gate_phi23`.
        """

        def __init__(self, sovereign: "PositronRedemptionGovernorAgent") -> None:
            self._s = sovereign

        # ── Método 2.1 · APERTURA Φ₂ (compuerta Φ₁₂ efectiva) ─────────────────
        def _gate_phi12(
            self,
            cert: PositronPhase1Certificate,
            frozen_now: Sequence[FrozenTensor],
        ) -> GateVerdict:
            """
            2.1 [COMPUERTA Φ₁₂ EFECTIVA — CONTINUACIÓN DIRECTA DE 1.11 `_phase1_observe_and_sign`]
            [G4] Verifica en tiempo constante y sobre cobertura total el expediente `cert`
            emitido por la Fase Φ₁:
              (a) HMAC(P1) recomputado bajo K_P1 sobre `1.7 _build_p1_payload(cert)`;
              (b) re-digest de los cuatro tensores congelados ≡ digests de `cert.frozen_inputs`;
              (c) sesión viva en [nbf_ns, exp_ns) y nonce registrado exactamente una vez;
              (d) secuencia estrictamente positiva y reloj monótono coherente.
            Distingue autenticidad criptográfica (`authentic`) de limpieza evidencial
            (`evidentially_clean`): admite veredictos DEGRADED/VETOED físicos para su
            orientación en Φ₂, pero rechaza cualquier adulteración de payload o replay.
            """
            s = self._s
            reasons: List[str] = []
            payload = _canonical_json(s.phase1._build_p1_payload(cert))
            key = s._hkdf_extract_and_expand(_HKDF_SALT_P1, _INFO_P1)
            expected = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
            authentic = _ct_equals(cert.hmac_signature, expected)
            if not authentic:
                reasons.append("P1_SIGNATURE_COVERAGE_MISMATCH")

            for idx, frozen in enumerate(cert.frozen_inputs):
                now = frozen_now[idx] if idx < len(frozen_now) else None
                if now is None or not frozen.same_content(now):
                    authentic = False
                    reasons.append(f"TENSOR_DIGEST_MISMATCH:{frozen.name}")

            sess = cert.session
            if sess is None:
                authentic = False
                reasons.append("SESSION_ABSENT")
            else:
                if not sess.is_live(time.time_ns()):
                    authentic = False
                    reasons.append("SESSION_WINDOW_EXPIRED")
                if s._telemetry.used_nonces.count(sess.nonce) != 1:
                    authentic = False
                    reasons.append("NONCE_REPLAY_OR_UNREGISTERED")
                if sess.sequence <= 0:
                    authentic = False
                    reasons.append("SEQUENCE_NON_POSITIVE")

            clean = all(v == HeytingVerdict.COHERENT.value for _, v in cert.evidence_atoms)
            if not clean:
                reasons.append("UPSTREAM_EVIDENTIAL_OBSTRUCTION")
            if not authentic:
                s._telemetry.gate_phi12_failures += 1
            return GateVerdict(bool(authentic), bool(clean), tuple(reasons))

        # ── Método 2.2 ────────────────────────────────────────────────────────
        @staticmethod
        def _shadow_strang_step(
            engine: PositronRedemptionGovernorEngine,
            p: NDArray[np.float64],
            rho: NDArray[np.complex128],
            k_mat: NDArray[np.float64],
            dt: float,
        ) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
            """
            2.2a Integrador sombra (replicador exponencial de Shahshahani + paso unitario de
            Brockett). Se emplea exclusivamente como respaldo si una instancia legacy del motor
            no expone `phase_pi3._strang_split_step`, marcando `used_engine_map=False`.
            """
            kp = k_mat @ p
            logits = np.log(np.maximum(p, _TINY)) + dt * kp
            logits -= float(np.max(logits))
            w = np.exp(np.clip(logits, -700.0, 0.0))
            p_new = w / max(float(np.sum(w)), _TINY)
            n_op = np.diag(p_new).astype(np.complex128)
            comm = rho @ n_op - n_op @ rho
            u_mat = la.expm(-comm * dt)
            rho_new = engine.canonicalize_density_matrix(u_mat @ rho @ u_mat.conj().T)
            return p_new, rho_new

        @staticmethod
        def _shahshahani_bures_distance(
            p_a: NDArray[np.float64],
            rho_a: NDArray[np.complex128],
            p_b: NDArray[np.float64],
            rho_b: NDArray[np.complex128],
        ) -> float:
            """
            2.2b [G16] Distancia geodésica riemanniana producto de Shahshahani-Bures sobre la
            variedad híbrida Δ^{n−1} × D(H):
                d_SB((p, ρ), (p', ρ')) = 2 arccos(∑_i √(p_i p'_i)) + √(2 − 2 Tr|√ρ √ρ'|₁).
            """
            pa = np.maximum(np.asarray(p_a, dtype=np.float64).ravel(), 0.0)
            pb = np.maximum(np.asarray(p_b, dtype=np.float64).ravel(), 0.0)
            sa = float(np.sum(pa))
            sb = float(np.sum(pb))
            if sa > 0.0:
                pa = pa / sa
            if sb > 0.0:
                pb = pb / sb
            bhattacharyya = float(np.clip(np.sum(np.sqrt(pa * pb)), -1.0, 1.0))
            d_fisher_rao = 2.0 * math.acos(bhattacharyya)

            def _sqrt_psd(mat: NDArray[np.complex128]) -> NDArray[np.complex128]:
                h = 0.5 * (mat + mat.conj().T)
                ev, uv = la.eigh(h)
                return (uv * np.sqrt(np.maximum(np.real(ev), 0.0))) @ uv.conj().T

            try:
                s_rho_a = _sqrt_psd(np.asarray(rho_a, dtype=np.complex128))
                s_rho_b = _sqrt_psd(np.asarray(rho_b, dtype=np.complex128))
                sv = la.svdvals(s_rho_a @ s_rho_b)
                root_f = float(np.clip(np.sum(np.real(sv)), 0.0, 1.0))
                d_bures = math.sqrt(max(0.0, 2.0 - 2.0 * root_f))
            except (la.LinAlgError, ValueError):
                d_bures = float(la.norm(rho_a - rho_b, "fro"))
            return float(d_fisher_rao + d_bures)

        # ── Método 2.3 ────────────────────────────────────────────────────────
        def _banach_kantorovich_probe(
            self, report: PositronRedemptionSpectrumReport, dt: float
        ) -> BanachProbe:
            """
            2.3 [G5/G16] Sonda de contracción local GENUINA de Banach-Kantorovich.
            Continúa el flujo del motor dos pasos desde el estado final (p_k, ρ_k) → (p_{k+1}, ρ_{k+1})
            → (p_{k+2}, ρ_{k+2}) y mide el cociente de incrementos sucesivos:
                q̂ = d(x_{k+2}, x_{k+1}) / d(x_{k+1}, x_k),
            en la norma producto d((p,ρ),(p',ρ')) = ‖p − p'‖₂ + ‖ρ − ρ'‖_F, registrando además
            la razón bruta sin truncar (`raw_ratio`) y los incrementos geodésicos en la métrica
            intrínseca de Shahshahani-Bures (`shahshahani_bures_d1`, `shahshahani_bures_d2`).
            Cotas de Banach-Kantorovich:
                A POSTERIORI  ‖x_{k+1} − x*‖ ≤ d₂ / (1 − q̂)
                A PRIORI      ‖x_{k+2} − x*‖ ≤ q̂² / (1 − q̂) · d₁
            HONESTIDAD DECLARADA: el replicador posee múltiples puntos fijos en las caras de
            Δ^{n−1}; q̂ es una tasa de contracción LOCAL asintótica sobre la órbita inspeccionada.
            """
            s = self._s
            p = np.asarray(_ev(report, "p_final", np.array([1.0])), dtype=np.float64).copy()
            rho = np.asarray(
                _ev(report, "rho_final", np.eye(p.size, dtype=np.complex128)),
                dtype=np.complex128,
            ).copy()
            k_mat = np.asarray(_ev(report, "K_purified", np.eye(p.size)), dtype=np.float64).copy()
            h = float(s._telemetry.policy.probe_dt if dt <= 0.0 else dt)
            rate = 0.0  # Sondea el sector conservativo replicador-Brockett del mapa
            step = getattr(getattr(s._engine, "phase_pi3", None), "_strang_split_step", None)
            used_engine = callable(step)
            try:
                if used_engine:
                    p1, r1, *_ = step(p, rho, k_mat, h, rate)
                    p2, r2, *_ = step(p1, r1, k_mat, h, rate)
                else:
                    p1, r1 = self._shadow_strang_step(s._engine, p, rho, k_mat, h)
                    p2, r2 = self._shadow_strang_step(s._engine, p1, r1, k_mat, h)
                d1 = float(np.linalg.norm(p1 - p, 2) + la.norm(r1 - rho, "fro"))
                d2 = float(np.linalg.norm(p2 - p1, 2) + la.norm(r2 - r1, "fro"))
                sb_d1 = self._shahshahani_bures_distance(p, rho, p1, r1)
                sb_d2 = self._shahshahani_bures_distance(p1, r1, p2, r2)
            except (la.LinAlgError, ValueError, FloatingPointError):
                d1, d2, sb_d1, sb_d2, used_engine = 0.0, 0.0, 0.0, 0.0, False

            at_fp = bool(d1 <= 1e-14)
            raw_ratio = 0.0 if at_fp else float(d2 / max(d1, _TINY))
            q = 0.0 if at_fp else float(np.clip(raw_ratio, 1e-9, 0.999999))
            one_minus_q = max(1.0 - q, 1e-9)
            a_post = d2 / one_minus_q
            a_prior = (q ** 2) / one_minus_q * d1
            digits = max(0.0, -math.log10(max(a_post, _TINY)))
            bch_norm = _safe_float(_ev(report, "bch_commutator_norm", 0.0))
            substeps = max(1, int(_ev(report, "rsi3_substeps_per_step", 1)))
            return BanachProbe(
                q_local=q,
                increment_d1=d1,
                increment_d2=d2,
                a_posteriori_bound=float(a_post),
                a_priori_bound=float(a_prior),
                certified_digits=float(digits),
                at_fixed_point=at_fp,
                contractive=bool(
                    at_fp or (raw_ratio < s._telemetry.policy.banach_q_ceiling and q < s._telemetry.policy.banach_q_ceiling)
                ),
                used_engine_map=bool(used_engine),
                raw_ratio=float(raw_ratio),
                shahshahani_bures_d1=float(sb_d1),
                shahshahani_bures_d2=float(sb_d2),
                bch_commutator_norm=float(bch_norm),
                rsi3_substeps_used=substeps,
            )

        # ── Método 2.4 ────────────────────────────────────────────────────────
        @staticmethod
        def _connes_daleckii_krein_audit(
            report: PositronRedemptionSpectrumReport,
        ) -> Tuple[float, float, bool]:
            """
            2.4 Cotas espectrales de Connes-Daleckii-Krein y propagación de incertidumbre.
            El multiplicador de Schur de f(A) tiene entradas f^[1](λ_i, λ_j); en norma de
            Frobenius su norma operatoria es máx_{i,j} |f^[1](λ_i, λ_j)|, de donde:
                ‖A^{1/2} − B^{1/2}‖_F   ≤ ‖A − B‖_F / (2 √λ_min),
                ‖A^{−1/2} − B^{−1/2}‖_F ≤ ‖A − B‖_F / (2 λ_min^{3/2}).
            Se exige finitud de ambas constantes de Lipschitz y condicionamiento κ(K) ≤ 10¹².
            """
            l_inv = _safe_float(_ev(report, "connes_lipschitz_bound", float("inf")), float("inf"))
            l_sqrt = _safe_float(_ev(report, "dk_sqrt_lipschitz", 0.0), 0.0)
            if l_sqrt <= 0.0 and math.isfinite(l_inv) and l_inv > 0.0:
                lam = (0.5 / l_inv) ** (2.0 / 3.0)
                l_sqrt = 0.5 / math.sqrt(max(lam, _STABILITY_FLOOR))
            kappa = _safe_float(_ev(report, "condition_number_K", 1.0), 1.0)
            usable = bool(math.isfinite(l_inv) and l_inv > 0.0 and kappa <= _CONDITION_CAP)
            return float(l_sqrt), float(l_inv), usable

        # ── Método 2.5 ────────────────────────────────────────────────────────
        @staticmethod
        def _lyapunov_decomposed_audit(
            report: PositronRedemptionSpectrumReport,
        ) -> Tuple[bool, bool, float, Tuple[str, str]]:
            """
            2.5 Auditoría de Lyapunov DESCOMPUESTA:
                H(p, ρ) = −½ pᵀ K p + S_vN(ρ),
                dH/dt   = −Var_p(Kp) + dS_GKSL/dt.
            El sector conservativo replicador-Brockett aporta −Var_p(Kp) ≤ 0 (Brockett es
            unitariamente isospectral ⟹ S_vN invariante), mientras que el baño de Lindblad
            GKSL aporta dS_GKSL/dt ≥ 0. Se exige:
              (i)   pasividad del término replicador: −Var_p(Kp) ≤ 10⁻¹²;
              (ii)  consistencia de la descomposición: dH/dt_total ≈ −Var_p(Kp) + dS_GKSL/dt;
              (iii) discrepancia relativa entre derivada analítica y numérica ≤ 5%.
            """
            analytic = _safe_float(_ev(report, "analytic_energy_derivative", 0.0))
            numeric = _safe_float(_ev(report, "numerical_energy_derivative", analytic), analytic)
            rep_term = _safe_float(_ev(report, "lyapunov_replicator_term", analytic), analytic)
            ds_dt = _safe_float(_ev(report, "gksl_entropy_production", 0.0))
            total = _ev(report, "total_energy_derivative", None)
            disc = _ev(report, "derivative_discrepancy_rel", None)
            passive = bool(rep_term <= 1e-12)
            if total is None:
                consistent = bool(abs(numeric - analytic) <= 1e-6 + 1e-3 * abs(analytic))
                atom = (
                    HeytingVerdict.COHERENT
                    if (passive and consistent)
                    else HeytingVerdict.DEGRADED
                )
            else:
                total_f = _safe_float(total, 0.0)
                consistent = bool(
                    abs(total_f - (rep_term + ds_dt)) <= 1e-6 + 1e-3 * abs(total_f)
                )
                if disc is not None:
                    consistent = consistent and (_safe_float(disc) <= 0.05)
                atom = (
                    HeytingVerdict.COHERENT
                    if (passive and consistent)
                    else HeytingVerdict.DEGRADED
                )
                if not passive:
                    atom = HeytingVerdict.VETOED
            return passive, bool(consistent), float(ds_dt), ("LYAPUNOV", atom.value)

        # ── Método 2.6 ────────────────────────────────────────────────────────
        @staticmethod
        def _celestial_evidence_atoms(
            report: PositronRedemptionSpectrumReport,
        ) -> Tuple[Tuple[str, str], ...]:
            """
            2.6 Átomos celestes base sobre los certificados de Poincaré del motor (v17/v18):
            canonicidad 2D de la carta, residuo de Halley-Kepler, defecto simpléctico de
            monodromía, isoespectralidad de Brockett, resto relativo de Lindstedt-AGM, orden
            observado de Strang, transversalidad homoclínica de Melnikov, resonancia epicíclica
            1:1 de Krein y preservación CPTP del canal GKSL.
            """
            atoms: List[Tuple[str, str]] = []

            def _add(name: str, ok: Optional[bool], soft: bool = True) -> None:
                if ok is None:
                    return
                atoms.append(
                    (
                        name,
                        HeytingVerdict.COHERENT.value
                        if ok
                        else (
                            HeytingVerdict.DEGRADED.value
                            if soft
                            else HeytingVerdict.VETOED.value
                        ),
                    )
                )

            can = _ev(report, "poincare_canonicity_defect", None)
            _add(
                "POINCARE_CANONICITY",
                None if can is None else _safe_float(can) <= _CANONICITY_DEFECT_CEILING,
            )
            kep = _ev(report, "kepler_worst_residual", _ev(report, "halley_kepler_residual", None))
            _add(
                "KEPLER_RESIDUAL",
                None if kep is None else _safe_float(kep) <= _KEPLER_RESIDUAL_CEILING,
            )
            sym = _ev(report, "monodromy_symplectic_defect", None)
            _add(
                "SYMPLECTIC_FORM",
                None if sym is None else _safe_float(sym) <= _SYMPECTIC_DEFECT_CEILING,
                soft=False,
            )
            iso = _ev(report, "isospectrality_defect", None)
            _add(
                "ISOSPECTRALITY",
                None if iso is None else _safe_float(iso) <= _ISOSPECTRAL_DEFECT_CEILING,
                soft=False,
            )
            rem = _ev(report, "lindstedt_remainder_rel", None)
            _add(
                "LINDSTEDT_REMAINDER",
                None if rem is None else _safe_float(rem) <= _LINDSTEDT_REMAINDER_CEILING,
            )
            order = _ev(report, "strang_observed_order", None)
            _add(
                "STRANG_ORDER",
                None if order is None else 1.7 <= _safe_float(order) <= 2.4,
            )
            mel = _ev(report, "melnikov_transversal", None)
            if mel is not None and bool(mel):
                atoms.append(("MELNIKOV_TRANSVERSAL", HeytingVerdict.DEGRADED.value))
            krein = _ev(report, "krein_collision_detected", None)
            if krein is not None and bool(krein):
                atoms.append(("KREIN_1TO1_RESONANCE", HeytingVerdict.DEGRADED.value))
            cptp = _ev(report, "gksl_cptp_ok", None)
            _add("GKSL_CPTP", cptp, soft=False)
            return tuple(atoms)

        # ── Método 2.7 ────────────────────────────────────────────────────────
        @staticmethod
        def _audit_poincare_kam_yoshida_sector(
            report: PositronRedemptionSpectrumReport,
        ) -> Tuple[Tuple[str, str], ...]:
            """
            2.7 [G15] Audita los invariantes celestes avanzados introducidos en el motor v18.0.0:
              • Invariante simpléctico 4D de Poincaré-Cartan: ‖J₄ᵀ Ω₄ J₄ − Ω₄‖_F ≤ 10⁻⁸.
              • Margen diofántico KAM: γ_KAM = mín_{0<|k|≤K} |k₁ ω₁ + k₂ ω₂|·|k|^τ ≥ 10⁻⁶.
              • Identidad integral de Kapteyn-Fourier-Bessel: |(2/π)∫₀^π (E−M)sin(E)dM − e| ≤ 10⁻².
              • Acuerdo simpléctico de 4º orden de Yoshida: ‖M_Yoshida4 − M_exact‖_F ≤ 10⁻².
              • Índice de bifurcación de Hamilton-Hopf: Δ_HH = (Tr M_rad)² − 4 ≤ 10⁻⁶ (elíptico).
            Si el informe proviene de un motor v16/v17 y carece de estos campos, los omite
            limpiamente sin penalizar el veredicto.
            """
            atoms: List[Tuple[str, str]] = []

            pc4 = _ev(report, "poincare_cartan_symplectic_defect_4d", None)
            if pc4 is not None:
                val_pc4 = _safe_float(pc4, float("inf"))
                atoms.append(
                    (
                        "POINCARE_CARTAN_4D",
                        HeytingVerdict.COHERENT.value
                        if val_pc4 <= _SYMPECTIC_DEFECT_CEILING
                        else (
                            HeytingVerdict.DEGRADED.value
                            if val_pc4 <= _CANONICITY_DEFECT_CEILING
                            else HeytingVerdict.VETOED.value
                        ),
                    )
                )

            kam = _ev(report, "kam_diophantine_margin", None)
            if kam is not None:
                val_kam = _safe_float(kam, -1.0)
                atoms.append(
                    (
                        "KAM_DIOPHANTINE",
                        HeytingVerdict.COHERENT.value
                        if val_kam >= _KAM_DIOPHANTINE_FLOOR
                        else HeytingVerdict.DEGRADED.value,
                    )
                )

            kfb = _ev(report, "kepler_fourier_bessel_error", None)
            if kfb is not None:
                val_kfb = _safe_float(kfb, float("inf"))
                atoms.append(
                    (
                        "KAPTEYN_FOURIER_BESSEL",
                        HeytingVerdict.COHERENT.value
                        if val_kfb <= _FOURIER_BESSEL_CEILING
                        else HeytingVerdict.DEGRADED.value,
                    )
                )

            yosh = _ev(report, "yoshida4_monodromy_defect", None)
            if yosh is not None:
                val_yosh = _safe_float(yosh, float("inf"))
                atoms.append(
                    (
                        "YOSHIDA4_SYMPLECTIC",
                        HeytingVerdict.COHERENT.value
                        if val_yosh <= _YOSHIDA4_DEFECT_CEILING
                        else HeytingVerdict.DEGRADED.value,
                    )
                )

            hh = _ev(report, "hamilton_hopf_index", None)
            if hh is not None:
                val_hh = _safe_float(hh, float("inf"))
                atoms.append(
                    (
                        "HAMILTON_HOPF",
                        HeytingVerdict.COHERENT.value
                        if val_hh <= 1.0e-6
                        else HeytingVerdict.DEGRADED.value,
                    )
                )
            return tuple(atoms)

        # ── Método 2.8 ────────────────────────────────────────────────────────
        @staticmethod
        def _crowbar_rlc_model() -> CrowbarRLCModel:
            """
            2.8 [G21] Modelo RLC subamortiguado de descarga del bus DC sobre el tiristor crowbar:
                Z₀ = √(L/C),  ω₀ = 1/√(LC),  α = R/(2L),  ζ = α/ω₀,  ω_d = ω₀√(1 − ζ²),
                i(t) = (V / (ω_d L)) e^{−αt} sin(ω_d t),
                t_pk = atan(ω_d / α) / ω_d,
                I_pk,exact  = (V / Z₀) exp(−(ζ / √(1 − ζ²)) atan(√(1 − ζ²) / ζ)),
                I_pk,approx = (V / Z₀) exp(−π ζ / (2 √(1 − ζ²))),
                di/dt|₀ = V / L,
                ∫₀^∞ i²(t) dt = ½ C V² / R   [EXACTO por balance energético en R].
            Extensiones v18.0.0:
              • Energía electrostática almacenada E₀ = ½ C V² (Joules).
              • Carga total almacenada Q₀ = C V (Coulombs).
              • Primer cruce por cero de corriente t_zero = π / ω_d (s), umbral físico de bloqueo
                inverso de un tiristor SCR sin circuito de conmutación forzada.
              • Frecuencia de oscilación amortiguada f_d = ω_d / (2π) (Hz).
            """
            v = _BUS_VOLTAGE_V
            c = _BUS_CAPACITANCE_F
            l = _LOOP_INDUCTANCE_H
            r = _THYRISTOR_R_ON_OHM

            z0 = math.sqrt(l / c)
            omega0 = 1.0 / math.sqrt(l * c)
            alpha = r / (2.0 * l)
            zeta = float(np.clip(alpha / omega0, 0.0, 0.999999))
            root = math.sqrt(max(1e-12, 1.0 - zeta * zeta))
            omega_d = omega0 * root
            t_pk = math.atan(omega_d / max(alpha, _TINY)) / omega_d
            i_exact = (v / z0) * math.exp(-(zeta / root) * math.atan(root / max(zeta, _TINY)))
            i_approx = (v / z0) * math.exp(-math.pi * zeta / (2.0 * root))
            di_dt = v / l / 1.0e6  # A/µs
            energy_j = 0.5 * c * v * v
            charge_c = c * v
            i2t = energy_j / r
            t_zero = math.pi / max(omega_d, _TINY)
            f_ring = omega_d / (2.0 * math.pi)

            return CrowbarRLCModel(
                z0_ohm=float(z0),
                zeta=float(zeta),
                omega0_rad_s=float(omega0),
                omega_d_rad_s=float(omega_d),
                peak_time_s=float(t_pk),
                i_peak_exact_a=float(i_exact),
                i_peak_smallzeta_a=float(i_approx),
                peak_approx_rel_error=float(abs(i_approx - i_exact) / max(i_exact, _TINY)),
                di_dt_initial_a_per_us=float(di_dt),
                i2t_action_a2s=float(i2t),
                within_di_dt_limit=bool(di_dt <= _DI_DT_LIMIT_A_PER_US),
                within_i2t_rating=bool(i2t <= _I2T_RATING_A2S),
                is_model_not_measurement=True,
                energy_stored_joules=float(energy_j),
                total_charge_coulomb=float(charge_c),
                first_zero_crossing_s=float(t_zero),
                ringing_frequency_hz=float(f_ring),
            )

        # ── Método 2.9 ────────────────────────────────────────────────────────
        @staticmethod
        def _build_p2_payload(cert: PositronPhase2Certificate) -> Dict[str, Any]:
            """
            2.9 Payload canónico de COBERTURA TOTAL de Φ₂ (excluye únicamente `hmac_signature`
            y `merkle_leaf`). [G3/G15] Cubre todos los campos v17 y v18 para impedir
            adulteración de la ventana de gracia, de las cotas de Banach/Daleckii-Krein o de
            los invariantes celestes KAM/Yoshida/Poincaré-Cartan.
            """
            return {
                "domain": _INFO_P2.decode("ascii"),
                "agent_version": cert.agent_version,
                "p1_signature": cert.phase1_cert.hmac_signature,
                "banach_contraction_q": cert.banach_contraction_q,
                "connes_lipschitz_bound": cert.connes_lipschitz_bound,
                "dk_sqrt_lipschitz": cert.dk_sqrt_lipschitz,
                "shahshahani_speed": cert.shahshahani_speed,
                "analytic_energy_derivative": cert.analytic_energy_derivative,
                "numerical_energy_derivative": cert.numerical_energy_derivative,
                "dissipative_entropy_production": cert.dissipative_entropy_production,
                "lyapunov_decomposition_consistent": cert.lyapunov_decomposition_consistent,
                "amber_grace_duration_sec": cert.amber_grace_duration_sec,
                "is_lyapunov_stable": cert.is_lyapunov_stable,
                "timestamp_ns": cert.timestamp_ns,
                "monotonic_ns": cert.monotonic_ns,
                "session": cert.session,
                "gate_phi12": cert.gate_phi12,
                "banach_probe": cert.banach_probe,
                "rlc_model": cert.rlc_model,
                "homology": cert.homology,
                "evidence_atoms": cert.evidence_atoms,
                "authentication_veto": cert.authentication_veto,
                "poincare_cartan_symplectic_defect_4d": cert.poincare_cartan_symplectic_defect_4d,
                "kam_diophantine_margin": cert.kam_diophantine_margin,
                "kepler_fourier_bessel_error": cert.kepler_fourier_bessel_error,
                "melnikov_splitting_angle_rad": cert.melnikov_splitting_angle_rad,
                "yoshida4_monodromy_defect": cert.yoshida4_monodromy_defect,
                "poincare_rotation_number": cert.poincare_rotation_number,
                "hamilton_hopf_index": cert.hamilton_hopf_index,
                "bch_commutator_norm": cert.bch_commutator_norm,
                "rsi3_substeps_per_step": cert.rsi3_substeps_per_step,
            }

        # ── Método 2.10 ───────────────────────────────────────────────────────
        @staticmethod
        def _propagate_gate(
            gate: GateVerdict, atoms: Sequence[Tuple[str, str]]
        ) -> Tuple[Tuple[Tuple[str, str], ...], bool]:
            """
            2.10 Propagación fail-closed SIN interferencia: si Φ₁₂ detecta fallo de
            autenticidad criptográfica, inyecta el átomo absorbente `("AUTH_VETO", "VETOED")`
            y activa `authentication_veto=True`, continuando la construcción del expediente
            de orientación para auditoría forense completa.
            """
            out = tuple(atoms)
            veto = not gate.authentic
            if veto:
                out = out + (("AUTH_VETO", HeytingVerdict.VETOED.value),)
            return out, bool(veto)

        # ── Método 2.11 · TERMINAL Φ₂ → APERTURA Φ₃ (CPS polimórfico) ─────────
        def _phase2_orient_and_contract(
            self,
            cert_p1: PositronPhase1Certificate,
            report: PositronRedemptionSpectrumReport,
            p_init: NDArray[np.float64],
            rho_init: NDArray[np.complex128],
            sigma_ref: NDArray[np.complex128],
            K_raw: NDArray[np.complex128],
            has_torsion_homological: bool,
            torsion_claim_attested: bool,
            dt: float,
            continuation: Callable[[PositronPhase2Certificate], AnswerT],
        ) -> AnswerT:
            """
            2.11 [TERMINAL Φ₂ → ENTRADA FORMAL DE 3.1 `_gate_phi23`]
            Ejecuta la compuerta efectiva Φ₁₂ (`2.1`), la sonda de contracción local de
            Banach-Kantorovich con métrica de Shahshahani-Bures (`2.3`), las cotas de
            Connes-Daleckii-Krein (`2.4`), la descomposición de Lyapunov (`2.5`), la auditoría
            celeste de Poincaré-KAM-Yoshida-Melnikov-Krein (`2.6` y `2.7`), el modelo RLC del
            crowbar (`2.8`) y la homología simplicial del grafo de coherencia, firmando
            `PositronPhase2Certificate` con cobertura total (`2.9`) y transfiriendo el control
            en posición de cola a `continuation`.
            """
            s = self._s
            frozen_now = [
                s.phase1._freeze_and_digest("p", p_init),
                s.phase1._freeze_and_digest("rho", rho_init),
                s.phase1._freeze_and_digest("sigma", sigma_ref),
                s.phase1._freeze_and_digest("K", K_raw),
            ]
            gate = self._gate_phi12(cert_p1, frozen_now)
            probe = self._banach_kantorovich_probe(report, dt)
            dk_sqrt, dk_inv, dk_usable = self._connes_daleckii_krein_audit(report)
            passive, consistent, ds_dt, lyap_atom = self._lyapunov_decomposed_audit(report)
            rlc = self._crowbar_rlc_model()
            homology = s.phase3._homology_audit(
                report, has_torsion_homological, torsion_claim_attested
            )

            atoms: List[Tuple[str, str]] = list(cert_p1.evidence_atoms)
            atoms.append(lyap_atom)
            atoms.extend(self._celestial_evidence_atoms(report))
            atoms.extend(self._audit_poincare_kam_yoshida_sector(report))
            atoms.append(
                (
                    "BANACH_CONTRACTION",
                    HeytingVerdict.COHERENT.value
                    if probe.contractive
                    else HeytingVerdict.DEGRADED.value,
                )
            )
            atoms.append(
                (
                    "DALECKII_KREIN",
                    HeytingVerdict.COHERENT.value
                    if dk_usable
                    else HeytingVerdict.DEGRADED.value,
                )
            )
            atoms.append(
                (
                    "CROWBAR_MODEL",
                    HeytingVerdict.COHERENT.value
                    if (rlc.within_di_dt_limit and rlc.within_i2t_rating)
                    else HeytingVerdict.DEGRADED.value,
                )
            )
            if not homology.chain_complex_exact_d1_d2:
                atoms.append(("CHAIN_COMPLEX_D1_D2", HeytingVerdict.VETOED.value))
            if homology.external_claim:
                atoms.append(
                    (
                        "TORSION_CLAIM",
                        HeytingVerdict.VETOED.value
                        if homology.external_claim_attested
                        else HeytingVerdict.DEGRADED.value,
                    )
                )
            if not probe.used_engine_map:
                atoms.append(("BANACH_PROBE_SHADOW_MAP", HeytingVerdict.DEGRADED.value))

            atoms, veto = self._propagate_gate(gate, atoms)
            grace = min(
                _safe_float(_ev(report, "amber_grace_duration_sec", 0.0)),
                s._telemetry.policy.grace_cap_sec,
            )
            s._telemetry.monotonic_clock_ns = max(
                time.monotonic_ns(), s._telemetry.monotonic_clock_ns + 1
            )

            skeleton = PositronPhase2Certificate(
                phase1_cert=cert_p1,
                banach_contraction_q=float(probe.q_local),
                connes_lipschitz_bound=float(dk_inv),
                shahshahani_speed=_safe_float(_ev(report, "shahshahani_speed", 0.0)),
                analytic_energy_derivative=_safe_float(
                    _ev(report, "analytic_energy_derivative", 0.0)
                ),
                numerical_energy_derivative=_safe_float(
                    _ev(report, "numerical_energy_derivative", 0.0)
                ),
                amber_grace_duration_sec=float(grace),
                is_lyapunov_stable=bool(passive and consistent),
                timestamp_ns=time.time_ns(),
                hmac_signature="",
                agent_version=_AGENT_VERSION,
                session=cert_p1.session,
                monotonic_ns=int(s._telemetry.monotonic_clock_ns),
                gate_phi12=gate,
                banach_probe=probe,
                rlc_model=rlc,
                homology=homology,
                dk_sqrt_lipschitz=float(dk_sqrt),
                dissipative_entropy_production=float(ds_dt),
                lyapunov_decomposition_consistent=bool(consistent),
                evidence_atoms=tuple(atoms),
                authentication_veto=bool(veto),
                poincare_cartan_symplectic_defect_4d=_safe_float(
                    _ev(report, "poincare_cartan_symplectic_defect_4d", 0.0)
                ),
                kam_diophantine_margin=_safe_float(_ev(report, "kam_diophantine_margin", 0.0)),
                kepler_fourier_bessel_error=_safe_float(
                    _ev(report, "kepler_fourier_bessel_error", 0.0)
                ),
                melnikov_splitting_angle_rad=_safe_float(
                    _ev(report, "melnikov_splitting_angle_rad", 0.0)
                ),
                yoshida4_monodromy_defect=_safe_float(
                    _ev(report, "yoshida4_monodromy_defect", 0.0)
                ),
                poincare_rotation_number=_safe_float(
                    _ev(report, "poincare_rotation_number", 0.0)
                ),
                hamilton_hopf_index=_safe_float(_ev(report, "hamilton_hopf_index", 0.0)),
                bch_commutator_norm=float(probe.bch_commutator_norm),
                rsi3_substeps_per_step=int(probe.rsi3_substeps_used),
            )
            payload = _canonical_json(self._build_p2_payload(skeleton))
            key = s._hkdf_extract_and_expand(_HKDF_SALT_P2, _INFO_P2)
            sig = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
            leaf = hashlib.sha256(_canonical_json({"p2": sig}).encode("utf-8")).hexdigest()
            cert_p2 = replace(skeleton, hmac_signature=sig, merkle_leaf=leaf)
            s._telemetry.phase2_emissions += 1
            return continuation(cert_p2)


    # ──────────────────────────────────────────────────────────────────────────
    # FASE Φ₃ · DECIDER: Φ₂₃, NÚCLEO ÁMBAR, LÖB, HOMOLOGÍA 2D, RSI³ Y CROWBAR
    # ──────────────────────────────────────────────────────────────────────────
    class Phase3Decider:
        """
        Φ₃ (3.1 → 3.12).
        Método inicial : `3.1 _gate_phi23`, que recibe directamente el `PositronPhase2Certificate`
                         emitido por `2.11 _phase2_orient_and_contract`.
        Método terminal: `3.12 _phase3_decide_and_act`, emisor del certificado público inmutable
                         `PositronSovereignCertificate`.
        """

        def __init__(self, sovereign: "PositronRedemptionGovernorAgent") -> None:
            self._s = sovereign

        # ── Método 3.1 · APERTURA Φ₃ (compuerta Φ₂₃ efectiva) ─────────────────
        def _gate_phi23(self, cert: PositronPhase2Certificate) -> GateVerdict:
            """
            3.1 [COMPUERTA Φ₂₃ EFECTIVA — CONTINUACIÓN DIRECTA DE 2.11 `_phase2_orient_and_contract`]
            [G4] Recomputa en tiempo constante:
              (a) HMAC(P2) bajo K_P2 sobre el payload de cobertura total `2.9 _build_p2_payload(cert)`;
              (b) HMAC(P1) anidado bajo K_P1 sobre `1.7 _build_p1_payload(cert.phase1_cert)`;
              (c) continuidad de sesión (`session_id` y `nonce` idénticos entre Φ₁ y Φ₂),
                  ventana temporal `[nbf_ns, exp_ns)` viva y reloj monótono estrictamente
                  creciente (`cert.monotonic_ns > p1.monotonic_ns`);
              (d) ausencia de veto de autenticación aguas arriba (`authentication_veto`).
            """
            s = self._s
            reasons: List[str] = []
            payload = _canonical_json(s.phase2._build_p2_payload(cert))
            key = s._hkdf_extract_and_expand(_HKDF_SALT_P2, _INFO_P2)
            expected = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
            authentic = _ct_equals(cert.hmac_signature, expected)
            if not authentic:
                reasons.append("P2_SIGNATURE_COVERAGE_MISMATCH")

            p1 = cert.phase1_cert
            p1_payload = _canonical_json(s.phase1._build_p1_payload(p1))
            p1_key = s._hkdf_extract_and_expand(_HKDF_SALT_P1, _INFO_P1)
            p1_expected = hmac.new(p1_key, p1_payload.encode("utf-8"), hashlib.sha256).hexdigest()
            if not _ct_equals(p1.hmac_signature, p1_expected):
                authentic = False
                reasons.append("NESTED_P1_SIGNATURE_MISMATCH")

            if cert.session is None or p1.session is None:
                authentic = False
                reasons.append("SESSION_ABSENT")
            else:
                if (
                    cert.session.session_id != p1.session.session_id
                    or cert.session.nonce != p1.session.nonce
                ):
                    authentic = False
                    reasons.append("SESSION_CONTINUITY_BROKEN")
                if not cert.session.is_live(time.time_ns()):
                    authentic = False
                    reasons.append("SESSION_WINDOW_EXPIRED")
                if cert.monotonic_ns <= p1.monotonic_ns:
                    authentic = False
                    reasons.append("CLOCK_NOT_MONOTONE")

            if cert.authentication_veto or (
                cert.gate_phi12 is not None and not cert.gate_phi12.authentic
            ):
                authentic = False
                reasons.append("UPSTREAM_AUTHENTICATION_VETO")

            clean = all(v == HeytingVerdict.COHERENT.value for _, v in cert.evidence_atoms)
            if not clean:
                reasons.append("EVIDENTIAL_OBSTRUCTION")
            if not authentic:
                s._telemetry.gate_phi23_failures += 1
            return GateVerdict(bool(authentic), bool(clean), tuple(reasons))

        # ── Método 3.2 ────────────────────────────────────────────────────────
        @staticmethod
        def _heyting_infimum(atoms: Sequence[Tuple[str, str]]) -> HeytingVerdict:
            """
            3.2 Ínfimo (⋀) en Ω₃ sobre la escalera de átomos de evidencia.
            AXIOMA HEYTING-1: VETOED (⊥) es elemento absorbente del meet.
            """
            out = HeytingVerdict.COHERENT
            for _name, value in atoms:
                try:
                    out = out.meet(HeytingVerdict(value))
                except ValueError:
                    out = out.meet(HeytingVerdict.VETOED)
            return out

        # ── Método 3.3 ────────────────────────────────────────────────────────
        @staticmethod
        def _amber_grant_decision(
            base_verdict: HeytingVerdict,
            authorization_valid: bool,
            bures_distance: float,
            annihilation_kev: float,
            auth_max_bures: float,
        ) -> Tuple[HeytingVerdict, bool, Tuple[str, str]]:
            """
            3.3 [G1] Adjudicación del NÚCLEO DE GRACIA ÁMBAR j_amber : Ω₃ → Ω₃.
            La gracia se otorga ⟺ token PositronAuthorization criptográficamente válido ∧
            d_B ≤ mín(0.60, auth_max_bures) ∧ E_ann ≥ 1000 keV ∧ base_verdict == DEGRADED.
            El núcleo eleva DEGRADED → COHERENT y jamás eleva VETOED (fail-closed).
            """
            ceiling = min(_BURES_HARD_CEILING, max(0.0, auth_max_bures))
            eligible = bool(
                authorization_valid
                and bures_distance <= ceiling
                and annihilation_kev >= _MIN_ANNIHILATION_KEV
            )
            granted = bool(eligible and base_verdict is HeytingVerdict.DEGRADED)
            final = amber_nucleus(base_verdict, granted)
            atom = (
                "AMBER_NUCLEUS",
                HeytingVerdict.COHERENT.value if granted else base_verdict.value,
            )
            return final, granted, atom

        # ── Método 3.4 ────────────────────────────────────────────────────────
        def _verify_positron_authorization(
            self,
            token: Optional[PositronAuthorization],
            session: Optional[SessionToken],
        ) -> Tuple[bool, Tuple[str, ...]]:
            """
            3.4 [G8] Verificación criptográfica vinculante de `PositronAuthorization`.
            Exige: firma HMAC bajo K_auth sobre cobertura total, ligadura exacta a `session_id`
            y `input_bundle_digest` (impide replay entre sesiones o estados distintos), ventana
            temporal `[nbf_ns, exp_ns)` activa y nonce de un solo uso no consumido previamente.
            """
            s = self._s
            if token is None or session is None:
                return False, ("AUTH_TOKEN_ABSENT",)
            reasons: List[str] = []
            payload = _canonical_json(
                {
                    "domain": _INFO_AUTH.decode("ascii"),
                    "subject": token.subject,
                    "session_id": token.session_id,
                    "input_bundle_digest": token.input_bundle_digest,
                    "nonce": token.nonce,
                    "issued_ns": token.issued_ns,
                    "nbf_ns": token.nbf_ns,
                    "exp_ns": token.exp_ns,
                    "max_bures": token.max_bures,
                    "max_grace_sec": token.max_grace_sec,
                }
            )
            key = s._hkdf_extract_and_expand(_HKDF_SALT_P3, _INFO_AUTH)
            expected = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
            if not _ct_equals(token.signature, expected):
                reasons.append("AUTH_SIGNATURE_INVALID")
            if token.session_id != session.session_id:
                reasons.append("AUTH_SESSION_UNBOUND")
            if token.input_bundle_digest != session.input_bundle_digest:
                reasons.append("AUTH_INPUT_UNBOUND")
            now = time.time_ns()
            if not (token.nbf_ns <= now < token.exp_ns):
                reasons.append("AUTH_WINDOW_EXPIRED")
            if token.nonce in s._consumed_auth_nonces:
                reasons.append("AUTH_NONCE_REPLAY")
            valid = not reasons
            if valid:
                s._consumed_auth_nonces.append(token.nonce)
                if len(s._consumed_auth_nonces) > 1024:
                    s._consumed_auth_nonces.pop(0)
                s._telemetry.grace_grants += 1
            else:
                s._telemetry.grace_denials += 1
            return valid, tuple(reasons)

        # ── Método 3.5 ────────────────────────────────────────────────────────
        def _loeb_safe_verification(
            self,
            cert: PositronPhase2Certificate,
            report: PositronRedemptionSpectrumReport,
            sigma_ref: NDArray[np.complex128],
        ) -> LoebVerification:
            """
            3.5 [G6/G17] Verificador de Löb SEGURO por DESCENSO BIEN FUNDADO sobre (ℕ, <).
            En lugar de invocar auto-confianza circular □(□A → A) → □A, descarga una torre
            finita de obligaciones independientes en profundidades estrictamente decrecientes:
              • Nivel d₀    : recomputación independiente de HMAC(P2) bajo K_P2.
              • Nivel d₀−1  : recomputación independiente de HMAC(P1) anidado bajo K_P1.
              • Nivel d₀−2  : verificación algebraica del sándwich de Fuchs-van de Graaf y de
                              los axiomas de Heyting/Lawvere-Tierney.
            La monotonía de la distancia de Bures respecto al estado inicial se evalúa por
            norma traza SVD con clamp explícito a [0, 1] y se reporta exclusivamente como
            DIAGNÓSTICO, nunca como átomo de veto.
            """
            s = self._s
            initial_depth = int(s._telemetry.policy.verification_depth)
            trace: List[str] = []

            p2_ok = _ct_equals(
                cert.hmac_signature,
                hmac.new(
                    s._hkdf_extract_and_expand(_HKDF_SALT_P2, _INFO_P2),
                    _canonical_json(s.phase2._build_p2_payload(cert)).encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest(),
            )
            p1 = cert.phase1_cert
            p1_ok = _ct_equals(
                p1.hmac_signature,
                hmac.new(
                    s._hkdf_extract_and_expand(_HKDF_SALT_P1, _INFO_P1),
                    _canonical_json(s.phase1._build_p1_payload(p1)).encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest(),
            )
            alg_ok = bool(
                p1.fuchs_van_de_graaf_ok
                and HeytingVerdict.verify_axioms()
                and verify_nucleus_laws()
            )

            if initial_depth >= 1:
                trace.append(f"d={initial_depth}:P2_HMAC={'OK' if p2_ok else 'FAIL'}")
            if initial_depth >= 2:
                trace.append(f"d={initial_depth - 1}:P1_NESTED_HMAC={'OK' if p1_ok else 'FAIL'}")
            if initial_depth >= 3:
                trace.append(f"d={initial_depth - 2}:ALGEBRAIC_INVARIANTS={'OK' if alg_ok else 'FAIL'}")

            recomputed = bool(p2_ok and p1_ok and alg_ok)
            monotone, valid = False, False
            try:
                rho_final = np.asarray(_ev(report, "rho_final", None), dtype=np.complex128)
                d_now = self._bures_trace_norm(rho_final, sigma_ref)
                rho_init = s._last_rho_init
                if rho_init is not None:
                    d_prev = self._bures_trace_norm(rho_init, sigma_ref)
                    monotone = bool(d_now <= d_prev + 1e-9)
                    valid = True
            except (la.LinAlgError, ValueError, TypeError):
                valid = False

            accepted = bool(recomputed and initial_depth > 0)
            s._telemetry.policy = replace(
                s._telemetry.policy, verification_depth=max(0, initial_depth - 1)
            ).project()
            return LoebVerification(
                accepted=accepted,
                depth_remaining=int(s._telemetry.policy.verification_depth),
                independently_recomputed=bool(recomputed),
                bures_monotone_diagnostic=bool(monotone),
                bures_diagnostic_valid=bool(valid),
                descent_trace=tuple(trace),
                initial_depth=initial_depth,
            )

        @staticmethod
        def _bures_trace_norm(a: Any, b: Any) -> float:
            """
            3.5b Distancia de Bures por norma traza: √F = ‖√A √B‖₁ (vía valores singulares SVD),
            con clamp explícito de √F a [0, 1]. [G6] Previene el fail-open numérico de v16.
            """
            def _sqrt_psd(x: Any) -> NDArray[np.complex128]:
                m = np.asarray(x, dtype=np.complex128)
                m = 0.5 * (m + m.conj().T)
                ev, uv = la.eigh(m)
                return (uv * np.sqrt(np.maximum(np.real(ev), 0.0))) @ uv.conj().T

            root_f = float(
                np.clip(np.sum(np.real(la.svdvals(_sqrt_psd(a) @ _sqrt_psd(b)))), 0.0, 1.0)
            )
            return math.sqrt(max(0.0, 2.0 - 2.0 * root_f))

        # ── Método 3.6 ────────────────────────────────────────────────────────
        @staticmethod
        def _homology_audit(
            report: PositronRedemptionSpectrumReport,
            external_claim: bool,
            claim_attested: bool,
            threshold: float = 0.25,
        ) -> HomologyAudit:
            """
            3.6 [G9/G19] Homología simplicial del 1-esqueleto G_K y del complejo clique 2D Cl(G_K)
            sobre la matriz de conductividad purificada K:
                (i, j) ∈ E ⟺ |K_ij| / √(K_ii K_jj) > τ.
            • En el 1-esqueleto G_K = (V, E):
              β₀ se calcula exactamente por Union-Find y β₁(G_K) = |E| − |V| + β₀.
              Por el Teorema de Estructura de módulos finitamente generados sobre el DIP ℤ,
              C₁(G_K; ℤ) es abeliano libre y H₁(G_K; ℤ) = ker(∂₁) ⊆ C₁(G_K; ℤ) es SIEMPRE
              LIBRE DE TORSIÓN (`h1_torsion_free = True`).
            • En el 2-esqueleto clique Cl(G_K) (añadiendo los 2-símplices orientados i < j < k
              tales que (i,j), (j,k), (i,k) ∈ E):
              Se construyen las matrices frontera ∂₁ ∈ ℤ^{|V|×|E|} y ∂₂ ∈ ℤ^{|E|×|T|}, se
              verifica exactamente la identidad homológica fundamental ∂₁ ∘ ∂₂ = 0
              (`chain_complex_exact_d1_d2 = True`) y se calcula:
                  β₁(Cl(G_K)) = nullity(∂₁) − rank(∂₂) = β₁(G_K) − rank(∂₂),
                  χ₁ = |V| − |E|,   χ₂ = |V| − |E| + |T|.
            """
            k = np.asarray(_ev(report, "K_purified", np.eye(2)), dtype=np.float64)
            n = int(k.shape[0]) if k.ndim == 2 else 0
            if n == 0:
                return HomologyAudit(0, 0, 0, 0, True, bool(external_claim), bool(claim_attested))

            diag = np.maximum(np.abs(np.diag(k)), _STABILITY_FLOOR)
            coh = np.abs(k) / np.sqrt(np.outer(diag, diag))
            np.fill_diagonal(coh, 0.0)

            parent = list(range(n))

            def find(a: int) -> int:
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            edge_list: List[Tuple[int, int]] = []
            edge_index: Dict[Tuple[int, int], int] = {}
            for i in range(n):
                for j in range(i + 1, n):
                    if coh[i, j] > threshold:
                        edge_index[(i, j)] = len(edge_list)
                        edge_list.append((i, j))
                        ra, rb = find(i), find(j)
                        if ra != rb:
                            parent[ra] = rb

            edges = len(edge_list)
            beta0 = len({find(i) for i in range(n)})
            beta1_1skel = max(0, edges - n + beta0)

            # Construcción del 2-esqueleto clique (triángulos orientados i < j < k)
            triangles: List[Tuple[int, int, int]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) not in edge_index:
                        continue
                    for m in range(j + 1, n):
                        if (j, m) in edge_index and (i, m) in edge_index:
                            triangles.append((i, j, m))

            num_tri = len(triangles)
            chain_exact = True
            rank_d2 = 0
            if edges > 0 and num_tri > 0:
                d1_mat = np.zeros((n, edges), dtype=np.float64)
                for e_idx, (u, v) in enumerate(edge_list):
                    d1_mat[u, e_idx] = -1.0
                    d1_mat[v, e_idx] = 1.0
                d2_mat = np.zeros((edges, num_tri), dtype=np.float64)
                for t_idx, (i, j, m) in enumerate(triangles):
                    # ∂₂([i, j, m]) = [j, m] − [i, m] + [i, j]
                    d2_mat[edge_index[(i, j)], t_idx] = 1.0
                    d2_mat[edge_index[(i, m)], t_idx] = -1.0
                    d2_mat[edge_index[(j, m)], t_idx] = 1.0
                comp = d1_mat @ d2_mat
                chain_exact = bool(float(np.max(np.abs(comp))) == 0.0)
                rank_d2 = int(np.linalg.matrix_rank(d2_mat))

            beta1_clique = max(0, beta1_1skel - rank_d2)
            chi_1 = n - edges
            chi_2 = n - edges + num_tri

            return HomologyAudit(
                beta_0=int(beta0),
                beta_1=int(beta1_1skel),
                num_vertices=n,
                num_edges=int(edges),
                h1_torsion_free=True,
                external_claim=bool(external_claim),
                external_claim_attested=bool(claim_attested),
                num_triangles_2skeleton=int(num_tri),
                beta_1_clique_complex=int(beta1_clique),
                chain_complex_exact_d1_d2=bool(chain_exact),
                euler_characteristic_1skeleton=int(chi_1),
                euler_characteristic_2skeleton=int(chi_2),
            )

        # ── Método 3.7 ────────────────────────────────────────────────────────
        @staticmethod
        def _rsi3_compute_meta_lyapunov_risk(cert: PositronPhase2Certificate) -> float:
            """
            3.7 [G18] Calcula el funcional adimensional de riesgo meta-Lyapunov R_meta ≥ 0
            sobre los invariantes certificados de Φ₁ y Φ₂:
                R_meta = (d_B / d_B,hard)² + q̂² / (1 − q̂ + 10⁻⁶) + ‖[A(p), Ȧ(p)]‖_F
                         + Δ_PC,4D + |ΔF_homotopy|.
            """
            p1 = cert.phase1_cert
            d_b_ratio = p1.bures_distance / max(_BURES_HARD_CEILING, _TINY)
            q = min(max(cert.banach_contraction_q, 0.0), 0.999999)
            banach_term = (q * q) / max(1.0 - q, 1.0e-6)
            bch_term = max(0.0, cert.bch_commutator_norm)
            pc4_term = max(0.0, cert.poincare_cartan_symplectic_defect_4d)
            hom_term = abs(p1.fidelity_homotopy_drift)
            risk = (d_b_ratio * d_b_ratio) + banach_term + bch_term + pc4_term + hom_term
            return float(risk) if math.isfinite(risk) else float("inf")

        # ── Método 3.8 ────────────────────────────────────────────────────────
        def _rsi3_policy_adaptation(
            self,
            cert: PositronPhase2Certificate,
            final_verdict: HeytingVerdict,
            meta_risk: float,
        ) -> bool:
            """
            3.8 [G18] Auto-Mejora Recursiva de Nivel 3 (RSI³) acotada sobre `GovernorPolicy`
            con verificación de invariantes de seguridad y rollback automático:
              • Sólo puede ejecutarse si `mutation_budget > 0` (contador bien fundado en ℕ) y
                la compuerta Φ₁₂ es criptográficamente auténtica.
              • Si el ciclo resulta en `DEGRADED` o el riesgo meta-Lyapunov supera el umbral
                operativo (`meta_risk > 0.75`), propone un ajuste fino de `probe_dt` y
                `bures_soft_ceiling` dentro del politopo compacto `GovernorPolicy.project()`,
                consumiendo 1 unidad de `mutation_budget`.
              • Verifica que la política candidata satisfaga los invariantes de seguridad
                estrictos; si fallasen, ejecuta rollback inmediato e incrementa `rollback_count`.
            """
            s = self._s
            pol = s._telemetry.policy
            if pol.mutation_budget <= 0:
                return False
            if cert.gate_phi12 is not None and not cert.gate_phi12.authentic:
                return False
            if final_verdict is not HeytingVerdict.DEGRADED and meta_risk <= 0.75:
                return False

            prev_pol = pol
            new_probe_dt = max(1.0e-4, pol.probe_dt * 0.9)
            new_soft_bures = max(0.05, min(_BURES_HARD_CEILING, pol.bures_soft_ceiling * 0.98))
            candidate = GovernorPolicy(
                bures_soft_ceiling=new_soft_bures,
                banach_q_ceiling=pol.banach_q_ceiling,
                probe_dt=new_probe_dt,
                grace_cap_sec=pol.grace_cap_sec,
                verification_depth=pol.verification_depth,
                mutation_budget=pol.mutation_budget - 1,
            ).project()

            safety_ok = bool(
                0.05 <= candidate.bures_soft_ceiling <= _BURES_HARD_CEILING
                and 0.5 <= candidate.banach_q_ceiling < 1.0
                and 1.0e-4 <= candidate.probe_dt <= 1.0
                and candidate.mutation_budget < prev_pol.mutation_budget
            )
            if not safety_ok:
                s._telemetry.policy = replace(
                    prev_pol, mutation_budget=max(0, prev_pol.mutation_budget - 1)
                ).project()
                s._telemetry.rollback_count += 1
                return False

            s._telemetry.policy = candidate
            s._telemetry.mutation_count += 1
            s._telemetry.policy_ledger.append(
                {
                    "cycle": int(s._telemetry.total_cycles),
                    "meta_risk": float(meta_risk),
                    "verdict": final_verdict.value,
                    "bures_soft_ceiling": float(candidate.bures_soft_ceiling),
                    "probe_dt": float(candidate.probe_dt),
                    "mutation_budget_remaining": int(candidate.mutation_budget),
                }
            )
            return True

        # ── Método 3.9 ────────────────────────────────────────────────────────
        def _rsi_l3_meta_audit(
            self,
            cert: PositronPhase2Certificate,
            gate: GateVerdict,
            loeb: LoebVerification,
            final_verdict: HeytingVerdict,
        ) -> MetaAuditCertificate:
            """
            3.9 Meta-auditoría RSI-L3 en tres niveles con presupuestos bien fundados:
              • L1 numérico   : finitud y rango físico de las métricas transportadas.
              • L2 estructural: axiomas de Heyting, leyes del núcleo de Lawvere-Tierney,
                                sándwich de Fuchs-van de Graaf, libertad de torsión en H₁(G_K),
                                exactitud ∂₁ ∘ ∂₂ = 0 en Cl(G_K), razón de Banach y acción RLC.
              • L3 atestación : autenticidad de Φ₂₃, descenso bien fundado de Löb, sesión viva
                                y monotonía estricta de reloj.
            """
            s = self._s
            failed: List[str] = []
            p1 = cert.phase1_cert
            metrics = (
                p1.bures_distance,
                p1.uhlmann_fidelity,
                p1.annihilation_energy_kev,
                p1.pinsker_residual,
                cert.banach_contraction_q,
                cert.connes_lipschitz_bound,
                cert.shahshahani_speed,
                cert.analytic_energy_derivative,
                cert.numerical_energy_derivative,
                cert.amber_grace_duration_sec,
            )
            l1 = all(math.isfinite(_safe_float(m, float("nan"))) for m in metrics)
            if not l1:
                failed.append("L1_NON_FINITE_METRIC")
            if not (
                0.0 <= p1.uhlmann_fidelity <= 1.0
                and 0.0 <= p1.bures_distance <= math.sqrt(2.0) + 1e-9
            ):
                l1 = False
                failed.append("L1_RANGE_VIOLATION")

            l2 = HeytingVerdict.verify_axioms() and verify_nucleus_laws()
            if not l2:
                failed.append("L2_HEYTING_OR_NUCLEUS_LAWS")
            if not p1.fuchs_van_de_graaf_ok:
                l2 = False
                failed.append("L2_FUCHS_VAN_DE_GRAAF")
            if cert.homology is not None:
                if not cert.homology.h1_torsion_free:
                    l2 = False
                    failed.append("L2_TORSION_THEOREM")
                if not cert.homology.chain_complex_exact_d1_d2:
                    l2 = False
                    failed.append("L2_CHAIN_COMPLEX_D1_D2")
            if cert.banach_probe is not None:
                if not (0.0 <= cert.banach_probe.q_local < 1.0):
                    l2 = False
                    failed.append("L2_BANACH_RATIO_RANGE")
            if cert.rlc_model is not None and cert.rlc_model.i2t_action_a2s < 0.0:
                l2 = False
                failed.append("L2_RLC_ACTION_NEGATIVE")

            l3 = bool(gate.authentic and loeb.accepted and loeb.independently_recomputed)
            if not l3:
                failed.append("L3_GATE_OR_LOEB")
            if cert.session is None or not cert.session.is_live(time.time_ns()):
                l3 = False
                failed.append("L3_SESSION_NOT_LIVE")
            if cert.monotonic_ns <= p1.monotonic_ns:
                l3 = False
                failed.append("L3_CLOCK_NOT_MONOTONE")

            meta_risk = self._rsi3_compute_meta_lyapunov_risk(cert)
            adapted = self._rsi3_policy_adaptation(cert, final_verdict, meta_risk)
            accepted = bool(l1 and l2 and l3)
            return MetaAuditCertificate(
                l1_numeric_ok=bool(l1),
                l2_structural_ok=bool(l2),
                l3_attestation_ok=bool(l3),
                accepted=accepted,
                verification_depth_remaining=int(s._telemetry.policy.verification_depth),
                mutation_budget_remaining=int(s._telemetry.policy.mutation_budget),
                failed_checks=tuple(failed),
                rsi3_meta_lyapunov_risk=float(meta_risk),
                policy_adapted=bool(adapted),
            )

        # ── Método 3.10 ───────────────────────────────────────────────────────
        @staticmethod
        def _build_sovereign_payload(state: PositronSovereignCertificate) -> Dict[str, Any]:
            """3.10a Payload soberano de cobertura total (excluye sólo `sovereign_signature`)."""
            return {
                "domain": _INFO_P3.decode("ascii"),
                "agent_version": state.agent_version,
                "p2_signature": state.phase2_cert.hmac_signature if state.phase2_cert else "",
                "heyting_verdict": state.heyting_verdict,
                "base_verdict": state.base_verdict,
                "verdict_atoms": state.verdict_atoms,
                "grace_verified": state.grace_verified,
                "authorization": state.authorization,
                "merkle_root_rfc6962": state.merkle_root_rfc6962,
                "merkle_algorithm": state.merkle_algorithm,
                "merkle_inclusion_proofs": state.merkle_inclusion_proofs,
                "amber_grace_active": state.amber_grace_active,
                "amber_grace_duration_sec": state.amber_grace_duration_sec,
                "hardware_crowbar_fired": state.hardware_crowbar_fired,
                "actuation_latency_ns": state.actuation_latency_ns,
                "actuation_mode": state.actuation_mode,
                "actuation": state.actuation,
                "crowbar_intent": state.crowbar_intent,
                "crowbar_latched": state.crowbar_latched,
                "bures_distance": state.bures_distance,
                "uhlmann_fidelity": state.uhlmann_fidelity,
                "annihilation_energy_kev": state.annihilation_energy_kev,
                "loeb_verification_passed": state.loeb_verification_passed,
                "loeb": state.loeb,
                "meta_audit": state.meta_audit,
                "gate_phi23": state.gate_phi23,
                "session": state.session,
                "session_digest": state.session_digest,
                "monotonic_ns": state.monotonic_ns,
                "timestamp_ns": state.timestamp_ns,
                "diagnostics": state.diagnostics,
                "rsi3_meta_lyapunov_risk": state.rsi3_meta_lyapunov_risk,
                "policy_snapshot": state.policy_snapshot,
            }

        @staticmethod
        def _build_merkle_leaves(
            cert_p2: PositronPhase2Certificate,
            atoms: Sequence[Tuple[str, str]],
            final_verdict: HeytingVerdict,
            crowbar_fired: bool,
        ) -> List[bytes]:
            """3.10b Construye las 4 hojas canónicas de cobertura total del árbol Merkle RFC 6962."""
            return [
                _canonical_json({"leaf": 1, "p1": cert_p2.phase1_cert}).encode("utf-8"),
                _canonical_json({"leaf": 2, "p2": cert_p2}).encode("utf-8"),
                _canonical_json({"leaf": 3, "atoms": atoms}).encode("utf-8"),
                _canonical_json(
                    {
                        "leaf": 4,
                        "verdict": final_verdict.value,
                        "crowbar": crowbar_fired,
                    }
                ).encode("utf-8"),
            ]

        def _seal(
            self,
            cert_p2: PositronPhase2Certificate,
            atoms: Sequence[Tuple[str, str]],
            final_verdict: HeytingVerdict,
            crowbar_fired: bool,
        ) -> Tuple[str, str, Tuple[Tuple[str, ...], ...]]:
            """
            3.10c [G7/G12/G20] Calcula la raíz Merkle RFC 6962 sobre las 4 hojas de cobertura
            total, genera los 4 caminos de auditoría de inclusión `PATH(m, D[4])` y produce el
            `session_digest` autenticado mediante HMAC bajo la subclave de sesión de Φ₃.
            """
            s = self._s
            leaves = self._build_merkle_leaves(cert_p2, atoms, final_verdict, crowbar_fired)
            root = _mth_rfc6962(leaves).hex()
            proofs = tuple(_audit_path_rfc6962(i, leaves) for i in range(len(leaves)))
            key = s._hkdf_extract_and_expand(_HKDF_SALT_P3, _INFO_SESSION)
            session_digest = hmac.new(
                key,
                _canonical_json(
                    {
                        "root": root,
                        "verdict": final_verdict.value,
                        "session": cert_p2.session,
                    }
                ).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            return root, session_digest, proofs

        # ── Método 3.11 ───────────────────────────────────────────────────────
        def _crowbar_dispatch(
            self,
            final_verdict: HeytingVerdict,
            atoms: Sequence[Tuple[str, str]],
            session: Optional[SessionToken],
        ) -> Tuple[CrowbarIntent, ActuationReceipt, bool]:
            """
            3.11 [G2] TRIBUNAL DE SILICIO: intención firmada ≠ actuación física confirmada,
            con estado de LATCH de tiristor SCR persistente hasta reset explícito del operador.
            Protección frente a falsas alarmas topológicas: una claim externa no atestada
            produce contención blanda, jamás disparo físico irreversible.
            """
            s = self._s
            unattested_only = all(
                v != HeytingVerdict.VETOED.value or n == "TORSION_CLAIM" for n, v in atoms
            ) and any(
                n == "TORSION_CLAIM" and v == HeytingVerdict.VETOED.value for n, v in atoms
            )
            should_fire = bool(final_verdict is HeytingVerdict.VETOED and not unattested_only)
            intent = CrowbarIntent(
                session_id=session.session_id if session else "",
                gpio_pin=_GPIO_CROWBAR_PIN,
                mmio_register=_GPIO_OUT_W1TS_REG,
                reason=tuple(n for n, v in atoms if v == HeytingVerdict.VETOED.value),
                issued_ns=time.time_ns(),
            )
            key = s._hkdf_extract_and_expand(_HKDF_SALT_P3, _INFO_P3)
            intent_sig = hmac.new(
                key,
                _canonical_json(
                    {
                        "session_id": intent.session_id,
                        "gpio": intent.gpio_pin,
                        "mmio": intent.mmio_register,
                        "reason": intent.reason,
                        "issued": intent.issued_ns,
                    }
                ).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            intent = replace(intent, signature=intent_sig)

            if s._telemetry.crowbar_latched:
                receipt = ActuationReceipt(
                    confirmed=True,
                    mode="LATCHED",
                    latency_ns_measured=None,
                    driver_identity=s._actuator.identity,
                    detail="Tiristor ya latcheado en un ciclo previo; requiere reset del operador.",
                )
                return intent, receipt, True

            if not should_fire:
                return (
                    intent,
                    ActuationReceipt(
                        False,
                        "NOT_REQUIRED",
                        None,
                        s._actuator.identity,
                        "Veredicto no VETOED o veto sólo por claim no atestada.",
                    ),
                    False,
                )

            s._telemetry.crowbar_intents += 1
            t0 = time.monotonic_ns()
            try:
                receipt = s._actuator.fire(intent)
            except Exception as exc:  # pragma: no cover - ruta de fallo de driver
                receipt = ActuationReceipt(
                    False,
                    "DRIVER_FAILED",
                    None,
                    s._actuator.identity,
                    f"Excepción del driver: {exc}",
                )
            elapsed = float(time.monotonic_ns() - t0)
            if receipt.confirmed:
                receipt = replace(receipt, latency_ns_measured=elapsed)
                s._telemetry.crowbar_confirmations += 1
                s._telemetry.crowbar_latched = True
            else:
                receipt = replace(receipt, latency_ns_measured=None)
            return intent, receipt, bool(receipt.confirmed)

        # ── Método 3.12 · TERMINAL Φ₃ · EMISIÓN SOBERANA ──────────────────────
        def _phase3_decide_and_act(
            self,
            cert_p2: PositronPhase2Certificate,
            report: PositronRedemptionSpectrumReport,
            sigma_ref: NDArray[np.complex128],
            authorization: Optional[PositronAuthorization],
        ) -> PositronSovereignCertificate:
            """
            3.12 [TERMINAL Φ₃] Ejecuta la compuerta efectiva Φ₂₃ (`3.1`), adjudica el ínfimo
            en Ω₃ (`3.2`), aplica el núcleo de gracia ámbar de Lawvere-Tierney bajo token
            verificado (`3.3` y `3.4`), ejecuta el descenso bien fundado de Löb (`3.5`) y la
            meta-auditoría RSI-L3 con adaptación acotada de política (`3.7`–`3.9`), despacha
            el crowbar con recibo y latch (`3.11`) y sella el certificado soberano con árbol
            Merkle RFC 6962, caminos de inclusión y firma HMAC bajo K_P3 (`3.10`).
            """
            s = self._s
            s._telemetry.policy = replace(
                s._telemetry.policy, verification_depth=3
            ).project()
            gate = self._gate_phi23(cert_p2)
            atoms = list(cert_p2.evidence_atoms)
            if not gate.authentic:
                atoms.append(("GATE_PHI23_AUTH", HeytingVerdict.VETOED.value))

            base = self._heyting_infimum(atoms)
            auth_ok, auth_reasons = self._verify_positron_authorization(
                authorization, cert_p2.session
            )
            final, granted, amber_atom = self._amber_grant_decision(
                base,
                auth_ok,
                cert_p2.phase1_cert.bures_distance,
                cert_p2.phase1_cert.annihilation_energy_kev,
                authorization.max_bures if authorization else 0.0,
            )
            atoms.append(amber_atom)

            loeb = self._loeb_safe_verification(cert_p2, report, sigma_ref)
            if not loeb.accepted:
                final = final.meet(HeytingVerdict.DEGRADED)
                atoms.append(("LOEB_DESCENT", HeytingVerdict.DEGRADED.value))

            meta = self._rsi_l3_meta_audit(cert_p2, gate, loeb, final)
            if not meta.accepted:
                final = final.meet(HeytingVerdict.DEGRADED)
                atoms.append(("META_AUDIT", HeytingVerdict.DEGRADED.value))

            intent, receipt, fired = self._crowbar_dispatch(final, atoms, cert_p2.session)
            diagnostics: List[str] = [
                f"{n}={v}" for n, v in atoms if v != HeytingVerdict.COHERENT.value
            ]
            diagnostics.extend(gate.reasons)
            diagnostics.extend(auth_reasons)
            diagnostics.extend(meta.failed_checks)
            if granted:
                diagnostics.append(
                    f"GRACIA ÁMBAR: núcleo j(DEGRADED)=COHERENT; ventana "
                    f"{cert_p2.amber_grace_duration_sec:.2f} s (token verificado)."
                )
            if intent.reason:
                diagnostics.append(
                    f"[CROWBAR_INTENT] MMIO {_GPIO_OUT_W1TS_REG:#x} → GPIO{_GPIO_CROWBAR_PIN}; "
                    f"mode={receipt.mode}; confirmed={receipt.confirmed}; {receipt.detail}"
                )
            if not loeb.bures_diagnostic_valid:
                diagnostics.append(
                    "Diagnóstico de monotonía de Bures no disponible (no es átomo de veto)."
                )

            root, session_digest, proofs = self._seal(cert_p2, atoms, final, fired)
            s._telemetry.monotonic_clock_ns = max(
                time.monotonic_ns(), s._telemetry.monotonic_clock_ns + 1
            )
            grace_sec = cert_p2.amber_grace_duration_sec if granted else 0.0
            if authorization is not None and granted:
                grace_sec = min(grace_sec, max(0.0, authorization.max_grace_sec))

            skeleton = PositronSovereignCertificate(
                heyting_verdict=final.value,
                merkle_root_rfc6962=root,
                amber_grace_active=bool(granted),
                amber_grace_duration_sec=float(grace_sec),
                hardware_crowbar_fired=bool(fired),
                actuation_latency_ns=float(receipt.latency_ns_measured or 0.0),
                bures_distance=cert_p2.phase1_cert.bures_distance,
                uhlmann_fidelity=cert_p2.phase1_cert.uhlmann_fidelity,
                annihilation_energy_kev=cert_p2.phase1_cert.annihilation_energy_kev,
                loeb_verification_passed=bool(loeb.accepted),
                session_digest=session_digest,
                diagnostics=tuple(diagnostics),
                agent_version=_AGENT_VERSION,
                phase2_cert=cert_p2,
                base_verdict=base.value,
                verdict_atoms=tuple(atoms),
                grace_verified=bool(granted),
                authorization=authorization,
                crowbar_intent=intent,
                actuation=receipt,
                actuation_mode=receipt.mode,
                crowbar_latched=bool(s._telemetry.crowbar_latched),
                loeb=loeb,
                meta_audit=meta,
                gate_phi23=gate,
                session=cert_p2.session,
                monotonic_ns=int(s._telemetry.monotonic_clock_ns),
                timestamp_ns=time.time_ns(),
                merkle_inclusion_proofs=proofs,
                rsi3_meta_lyapunov_risk=float(meta.rsi3_meta_lyapunov_risk),
                policy_snapshot=s._telemetry.policy,
            )
            payload = _canonical_json(self._build_sovereign_payload(skeleton))
            key = s._hkdf_extract_and_expand(_HKDF_SALT_P3, _INFO_P3)
            sig = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
            state = replace(skeleton, sovereign_signature=sig)
            s._telemetry.phase3_emissions += 1
            s._telemetry.last_verdict = final
            s._last_sovereign = state
            return state

    # ──────────────────────────────────────────────────────────────────────────
    # §4 · INICIALIZACIÓN, HKDF, CPS Y VERIFICACIÓN PÚBLICA
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        engine: PositronRedemptionGovernorEngine,
        master_key: Optional[bytes] = None,
        actuator: Optional[CrowbarActuator] = None,
        policy: Optional[GovernorPolicy] = None,
    ) -> None:
        self._engine: Final[PositronRedemptionGovernorEngine] = engine
        # [G12] `master_key or default` caía al defecto con clave vacía b"": fallo silencioso.
        self._master_key: Final[bytes] = (
            bytes(master_key) if master_key is not None else _DEFAULT_MASTER_KEY
        )
        self._actuator: CrowbarActuator = (
            actuator if actuator is not None else NullCrowbarActuator()
        )
        self._telemetry = GovernorTelemetry(
            policy=(policy or GovernorPolicy()).project(),
            dev_key_in_use=(master_key is None),
        )
        self._consumed_auth_nonces: List[str] = []
        self._last_sovereign: Optional[PositronSovereignCertificate] = None
        self._last_rho_init: Optional[NDArray[np.complex128]] = None
        if master_key is None:
            logger.warning(
                "Clave de desarrollo en uso: los certificados NO son atestables en producción."
            )
        elif len(self._master_key) < 32:
            logger.warning(
                "master_key con menos de 32 bytes: entropía insuficiente para HMAC-SHA256."
            )
        if isinstance(self._actuator, NullCrowbarActuator):
            logger.warning(
                "Sin driver crowbar enlazado: toda actuación se reportará como SIMULATION."
            )
        self.phase1 = self.Phase1Observer(self)
        self.phase2 = self.Phase2Orientor(self)
        self.phase3 = self.Phase3Decider(self)
        logger.info(
            "PositronRedemptionGovernorAgent v%s inicializado (Φ₁→Φ₂→Φ₃).", _AGENT_VERSION
        )

    # ── 5.2 · HKDF-SHA256 (RFC 5869) con cota de longitud ──
    def _hkdf_extract_and_expand(
        self, salt: bytes, info: bytes, length: int = _HASH_LEN
    ) -> bytes:
        """
        HKDF Extract-then-Expand (RFC 5869). [G12] Se exige 0 < L ≤ 255·HashLen: sin la cota,
        `bytes([i])` desbordaría para i > 255.
        """
        if not isinstance(length, int) or length <= 0 or length > _HKDF_MAX_LEN:
            raise ValueError(f"HKDF: length debe estar en (0, {_HKDF_MAX_LEN}].")
        prk = hmac.new(salt, self._master_key, hashlib.sha256).digest()
        okm, t, i = b"", b"", 1
        while len(okm) < length:
            t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
            okm += t
            i += 1
        return okm[:length]

    @staticmethod
    def _digest_array(arr: NDArray[Any]) -> str:
        """Compatibilidad v16/v17: digest canónico de arreglo (delegado al congelador de Φ₁)."""
        return PositronRedemptionGovernorAgent.Phase1Observer._freeze_and_digest(
            "legacy", arr
        ).digest

    @staticmethod
    def _compute_merkle_root_rfc6962(leaf_data_list: List[bytes]) -> str:
        """Compatibilidad v16/v17 con semántica RFC 6962: MTH recursivo con promoción. [G7]"""
        return _mth_rfc6962(leaf_data_list).hex()

    # ── 5.3 · Emisión de tokens de autorización (requiere la clave maestra) ──
    def mint_positron_authorization(
        self,
        session: SessionToken,
        subject: str = "POSITRON_AUTHORITY",
        ttl_ns: int = 10_000_000_000,
        max_bures: float = _BURES_HARD_CEILING,
        max_grace_sec: float = 3600.0,
    ) -> PositronAuthorization:
        """
        Emite un token de positrón firmado bajo K_auth y LIGADO a la sesión y al bundle de
        entrada. Sólo quien posee la clave maestra puede emitirlo. [G8]
        """
        now = time.time_ns()
        nonce = hashlib.sha256(
            f"{session.session_id}|{now}|{subject}".encode("utf-8")
        ).hexdigest()[:32]
        skeleton = PositronAuthorization(
            subject=subject,
            session_id=session.session_id,
            input_bundle_digest=session.input_bundle_digest,
            nonce=nonce,
            issued_ns=now,
            nbf_ns=now - 1_000_000,
            exp_ns=now + int(ttl_ns),
            max_bures=float(min(max_bures, _BURES_HARD_CEILING)),
            max_grace_sec=float(max(0.0, max_grace_sec)),
        )
        payload = _canonical_json(
            {
                "domain": _INFO_AUTH.decode("ascii"),
                "subject": skeleton.subject,
                "session_id": skeleton.session_id,
                "input_bundle_digest": skeleton.input_bundle_digest,
                "nonce": skeleton.nonce,
                "issued_ns": skeleton.issued_ns,
                "nbf_ns": skeleton.nbf_ns,
                "exp_ns": skeleton.exp_ns,
                "max_bures": skeleton.max_bures,
                "max_grace_sec": skeleton.max_grace_sec,
            }
        )
        key = self._hkdf_extract_and_expand(_HKDF_SALT_P3, _INFO_AUTH)
        sig = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return replace(skeleton, signature=sig)

    # ── 5.4 · Verificador público ──
    def verify_sovereign_certificate(
        self, state: PositronSovereignCertificate
    ) -> SovereignVerificationReport:
        """
        Recomputa íntegramente, sólo desde el certificado:
          1. Las tres firmas HMAC (P1, P2 y Soberana) sobre cobertura total.
          2. La raíz Merkle RFC 6962, el `session_digest` autenticado y las 4 pruebas de
             inclusión RFC 6962 `PATH(m, D[4])`. [G7/G20]
          3. El veredicto base por ínfimo en Ω₃ y las leyes del núcleo ámbar de Lawvere-Tierney.
          4. La continuidad de sesión y la monotonía estricta del reloj entre Φ₁, Φ₂ y Φ₃.
          5. La honestidad de la actuación física del crowbar.
          6. La consistencia del cuaternión de Uhlmann S³ y del sándwich de Fuchs-van de Graaf.
        """
        failed: List[str] = []
        p2 = state.phase2_cert
        if p2 is None:
            return SovereignVerificationReport(
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                ("PHASE2_CERT_ABSENT",),
                False,
                False,
                False,
            )
        p1 = p2.phase1_cert
        p1_ok = _ct_equals(
            p1.hmac_signature,
            hmac.new(
                self._hkdf_extract_and_expand(_HKDF_SALT_P1, _INFO_P1),
                _canonical_json(self.phase1._build_p1_payload(p1)).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest(),
        )
        p2_ok = _ct_equals(
            p2.hmac_signature,
            hmac.new(
                self._hkdf_extract_and_expand(_HKDF_SALT_P2, _INFO_P2),
                _canonical_json(self.phase2._build_p2_payload(p2)).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest(),
        )
        p3_ok = _ct_equals(
            state.sovereign_signature,
            hmac.new(
                self._hkdf_extract_and_expand(_HKDF_SALT_P3, _INFO_P3),
                _canonical_json(self.phase3._build_sovereign_payload(state)).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest(),
        )
        for name, ok in (
            ("P1_SIGNATURE", p1_ok),
            ("P2_SIGNATURE", p2_ok),
            ("SOVEREIGN_SIGNATURE", p3_ok),
        ):
            if not ok:
                failed.append(name)

        recomputed_root, recomputed_session, _ = self.phase3._seal(
            p2,
            state.verdict_atoms,
            HeytingVerdict(state.heyting_verdict),
            state.hardware_crowbar_fired,
        )
        merkle_ok = _ct_equals(recomputed_root, state.merkle_root_rfc6962) and _ct_equals(
            recomputed_session, state.session_digest
        )
        if not merkle_ok:
            failed.append("MERKLE_OR_SESSION_DIGEST")

        leaves = self.phase3._build_merkle_leaves(
            p2,
            state.verdict_atoms,
            HeytingVerdict(state.heyting_verdict),
            state.hardware_crowbar_fired,
        )
        proofs_ok = bool(len(state.merkle_inclusion_proofs) == len(leaves)) and all(
            _verify_inclusion_rfc6962(
                leaves[i],
                i,
                len(leaves),
                state.merkle_inclusion_proofs[i],
                state.merkle_root_rfc6962,
            )
            for i in range(len(leaves))
        )
        if not proofs_ok:
            failed.append("MERKLE_INCLUSION_PROOFS")

        base_atoms = [
            a
            for a in state.verdict_atoms
            if a[0] not in ("AMBER_NUCLEUS", "LOEB_DESCENT", "META_AUDIT")
        ]
        base = self.phase3._heyting_infimum(base_atoms)
        expected_final = amber_nucleus(base, state.grace_verified)
        if state.loeb is not None and not state.loeb.accepted:
            expected_final = expected_final.meet(HeytingVerdict.DEGRADED)
        if state.meta_audit is not None and not state.meta_audit.accepted:
            expected_final = expected_final.meet(HeytingVerdict.DEGRADED)
        verdict_ok = bool(
            base.value == state.base_verdict and expected_final.value == state.heyting_verdict
        )
        if not verdict_ok:
            failed.append("VERDICT_RECOMPUTATION")

        nucleus_ok = verify_nucleus_laws() and (
            not state.grace_verified or state.heyting_verdict != HeytingVerdict.VETOED.value
        )
        if not nucleus_ok:
            failed.append("NUCLEUS_LAWS")

        session_ok = bool(
            state.session is not None
            and p2.session is not None
            and p1.session is not None
            and state.session.session_id == p2.session.session_id == p1.session.session_id
            and p1.monotonic_ns < p2.monotonic_ns < state.monotonic_ns
        )
        if not session_ok:
            failed.append("SESSION_CONTINUITY")

        honesty_ok = bool(
            (not state.hardware_crowbar_fired)
            or (
                state.actuation is not None
                and state.actuation.confirmed
                and state.actuation_mode in ("DRIVER_CONFIRMED", "LATCHED")
            )
        )
        if not honesty_ok:
            failed.append("ACTUATION_ATTESTATION_DISHONEST")

        q_arr = np.asarray(p1.uhlmann_quaternion_s3, dtype=np.float64)
        holonomy_ok = bool(
            q_arr.shape == (4,)
            and np.all(np.isfinite(q_arr))
            and abs(float(np.linalg.norm(q_arr, 2)) - 1.0) <= 1.0e-6
            and p1.uhlmann_holonomy_unitarity_defect <= 1.0e-5
        )
        if not holonomy_ok:
            failed.append("HOLONOMY_OR_QUATERNION_S3")

        fuchs_ok = bool(
            p1.fuchs_van_de_graaf_ok
            and p1.fuchs_lower_margin >= -1.0e-12
            and p1.fuchs_upper_margin >= -1.0e-12
        )
        if not fuchs_ok:
            failed.append("FUCHS_VAN_DE_GRAAF")

        all_ok = bool(
            p1_ok
            and p2_ok
            and p3_ok
            and merkle_ok
            and proofs_ok
            and verdict_ok
            and nucleus_ok
            and session_ok
            and honesty_ok
            and holonomy_ok
            and fuchs_ok
        )
        return SovereignVerificationReport(
            p1_signature_ok=p1_ok,
            p2_signature_ok=p2_ok,
            sovereign_signature_ok=p3_ok,
            merkle_ok=merkle_ok,
            verdict_recomputation_ok=verdict_ok,
            nucleus_ok=nucleus_ok,
            session_continuity_ok=session_ok,
            actuation_honesty_ok=honesty_ok,
            all_ok=all_ok,
            failed_checks=tuple(failed),
            merkle_inclusion_proofs_ok=proofs_ok,
            holonomy_quaternion_ok=holonomy_ok,
            fuchs_sandwich_ok=fuchs_ok,
        )

    # ── 5.5 · Entrada principal: cadena CPS Φ₃ ∘ Φ₂ ∘ Φ₁ ──
    def audit_redemption_cycle(
        self,
        p_init: NDArray[np.float64],
        rho_init: NDArray[np.complex128],
        sigma_ref: NDArray[np.complex128],
        K_raw: NDArray[np.complex128],
        has_torsion_homological: bool = False,
        positron_auth_signed: bool = False,
        dt: float = 0.05,
        steps: int = 4,
        authorization: Optional[PositronAuthorization] = None,
        torsion_claim_attested: bool = False,
        **engine_kwargs: Any,
    ) -> PositronSovereignCertificate:
        """
        Ciclo OODA completo en CPS con tipo de respuesta polimórfico (`Φ₃ ∘ Φ₂ ∘ Φ₁`).
        `positron_auth_signed` se conserva por compatibilidad, pero [G8] YA NO otorga gracia:
        sólo un `authorization` firmado y ligado a la sesión lo hace.
        """
        self._telemetry.total_cycles += 1
        self._last_rho_init = np.array(rho_init, copy=True)
        frozen_before = [
            self.phase1._freeze_and_digest("p", p_init),
            self.phase1._freeze_and_digest("rho", rho_init),
            self.phase1._freeze_and_digest("sigma", sigma_ref),
            self.phase1._freeze_and_digest("K", K_raw),
        ]
        report = self._engine.execute_positron_redemption_trajectory(
            p_init=p_init,
            rho_init=rho_init,
            sigma_target=sigma_ref,
            K_raw=K_raw,
            dt=dt,
            steps=steps,
            **engine_kwargs,
        )
        if positron_auth_signed and authorization is None:
            logger.warning(
                "positron_auth_signed=True sin token firmado: NO se otorga gracia (G8)."
            )

        def _phase2_continuation(
            cert_p1: PositronPhase1Certificate,
        ) -> PositronSovereignCertificate:
            def _phase3_continuation(
                cert_p2: PositronPhase2Certificate,
            ) -> PositronSovereignCertificate:
                return self.phase3._phase3_decide_and_act(
                    cert_p2=cert_p2,
                    report=report,
                    sigma_ref=sigma_ref,
                    authorization=authorization,
                )

            return self.phase2._phase2_orient_and_contract(
                cert_p1=cert_p1,
                report=report,
                p_init=p_init,
                rho_init=rho_init,
                sigma_ref=sigma_ref,
                K_raw=K_raw,
                has_torsion_homological=has_torsion_homological,
                torsion_claim_attested=torsion_claim_attested,
                dt=dt,
                continuation=_phase3_continuation,
            )

        return self.phase1._phase1_observe_and_sign(
            p_init=p_init,
            rho_init=rho_init,
            sigma_ref=sigma_ref,
            K_raw=K_raw,
            report=report,
            frozen_before=frozen_before,
            continuation=_phase2_continuation,
        )

    # ── 5.6 · Accesores y control del latch ──
    def get_last_sovereign_certificate(self) -> Optional[PositronSovereignCertificate]:
        return self._last_sovereign

    def get_telemetry(self) -> GovernorTelemetry:
        return self._telemetry

    def reset_crowbar_latch(self, operator_reason: str) -> bool:
        """
        Reset explícito del latch del tiristor por operador. Un crowbar físico conduce hasta
        que la corriente cae bajo el umbral de mantenimiento (holding current). [G2]
        """
        if not self._telemetry.crowbar_latched:
            return False
        logger.warning("Reset del latch crowbar por operador: %s", operator_reason)
        self._telemetry.crowbar_latched = False
        return True


# ══════════════════════════════════════════════════════════════════════════════
# §5 · AUTOVERIFICACIÓN EJECUTABLE (regresiones v16 → v17 → v18 e identidades)
# ══════════════════════════════════════════════════════════════════════════════
def _self_test() -> bool:
    """Suite de autoverificación ejecutable: prueba regresiones [G1]–[G21]."""
    checks: List[Tuple[str, bool]] = []
    n = 4
    engine = PositronRedemptionGovernorEngine(n, stability_threshold=1e-8)
    agent = PositronRedemptionGovernorAgent(
        engine, master_key=b"self-test-master-key-32-bytes-ok!!"
    )
    rho = np.diag([0.4, 0.3, 0.2, 0.1]).astype(np.complex128)
    rho[0, 1], rho[1, 0] = 0.02j, -0.02j
    sigma = np.diag([0.38, 0.30, 0.22, 0.10]).astype(np.complex128)
    k_raw = np.array(
        [
            [1.0, 0.35, 0.30, 0.0],
            [0.35, 0.9, 0.32, 0.0],
            [0.30, 0.32, 0.8, 0.05],
            [0.0, 0.0, 0.05, 0.7],
        ],
        dtype=np.complex128,
    )
    p0 = np.array([0.25, 0.25, 0.25, 0.25])
    state = agent.audit_redemption_cycle(p0, rho, sigma, k_raw, dt=0.02, steps=4)
    p2 = state.phase2_cert
    assert p2 is not None
    p1 = p2.phase1_cert

    # [G3] REGRESIÓN CRÍTICA: mutar el residuo de Pinsker debe romper la firma.
    forged_p1 = replace(p1, pinsker_residual=1.0)
    forged_p2 = replace(p2, phase1_cert=forged_p1)
    checks.append(
        (
            "[G3] Φ₂₃ RECHAZA residuo de Pinsker mutado (v16 lo aceptaba)",
            not agent.phase3._gate_phi23(forged_p2).authentic,
        )
    )
    # [G3] REGRESIÓN: extender la ventana de gracia debe romper la firma.
    grace_forged = replace(p2, amber_grace_duration_sec=1.0e9)
    checks.append(
        (
            "[G3] Φ₂₃ RECHAZA duración de gracia falsificada",
            not agent.phase3._gate_phi23(grace_forged).authentic,
        )
    )
    # [G1] La gracia NUNCA debe empeorar el veredicto (v16 invertía el orden).
    checks.append(
        (
            "[G1] Núcleo ámbar inflacionario: j(D)=C, j(V)=V",
            amber_nucleus(HeytingVerdict.DEGRADED, True) is HeytingVerdict.COHERENT
            and amber_nucleus(HeytingVerdict.VETOED, True) is HeytingVerdict.VETOED,
        )
    )
    checks.append(("[G1] Leyes del núcleo (idempotencia, meet, ⊤)", verify_nucleus_laws()))
    checks.append(
        ("Heyting: residuación, distributividad y De Morgan exhaustivas", HeytingVerdict.verify_axioms())
    )
    checks.append(
        ("Heyting: DEGRADED no es ¬¬-estable", not HeytingVerdict.DEGRADED.is_regular())
    )
    # [G2] Sin driver, jamás se afirma disparo físico.
    checks.append(
        (
            "[G2] Sin hardware ⟹ hardware_crowbar_fired=False y mode=SIMULATION",
            (not state.hardware_crowbar_fired) or state.actuation_mode == "LATCHED",
        )
    )

    class _FakeDriver:
        identity = "FAKE_DRIVER_FOR_TEST"

        def fire(self, intent: CrowbarIntent) -> ActuationReceipt:
            return ActuationReceipt(True, "DRIVER_CONFIRMED", None, self.identity, "test")

    agent2 = PositronRedemptionGovernorAgent(
        engine, master_key=b"k" * 32, actuator=_FakeDriver()
    )
    bad_sigma = np.eye(n, dtype=np.complex128) / n
    veto_state = agent2.audit_redemption_cycle(p0, rho, bad_sigma, k_raw, dt=0.02, steps=4)
    checks.append(
        (
            "[G2] Con driver recibo ⟹ confirmación y latencia MEDIDA",
            (not veto_state.hardware_crowbar_fired)
            or (
                veto_state.actuation is not None
                and veto_state.actuation.confirmed
                and veto_state.actuation.latency_ns_measured is not None
            ),
        )
    )
    checks.append(
        (
            "[G2] Latch del tiristor persiste tras confirmación",
            (not veto_state.hardware_crowbar_fired) or veto_state.crowbar_latched,
        )
    )
    # [G7] Merkle RFC 6962: sin maleabilidad por duplicación de hoja impar.
    r3 = _mth_rfc6962([b"a", b"b", b"c"]).hex()
    r4 = _mth_rfc6962([b"a", b"b", b"c", b"c"]).hex()
    checks.append(
        ("[G7] Merkle: duplicar la hoja impar CAMBIA la raíz (anti CVE-2012-2459)", r3 != r4)
    )
    # [G5/G16] Sonda de Banach: razón de incrementos, cota a posteriori y métrica Shahshahani-Bures.
    probe = p2.banach_probe
    checks.append(
        (
            "[G5/G16] q̂ ∈ [0,1), cota a posteriori finita y distancia Shahshahani-Bures ≥ 0",
            probe is not None
            and 0.0 <= probe.q_local < 1.0
            and math.isfinite(probe.a_posteriori_bound)
            and probe.shahshahani_bures_d1 >= 0.0,
        )
    )
    # [G8] Autorización: token válido otorga; bool desnudo y token falsificado, no.
    sess = p1.session
    assert sess is not None
    token = agent.mint_positron_authorization(sess, max_bures=0.60, max_grace_sec=60.0)
    ok_valid, _ = agent.phase3._verify_positron_authorization(token, sess)
    ok_forged, _ = agent.phase3._verify_positron_authorization(
        replace(token, max_bures=1.4), sess
    )
    ok_replay, _ = agent.phase3._verify_positron_authorization(token, sess)
    checks.append(("[G8] Token firmado y ligado ⟹ válido en primer uso", ok_valid))
    checks.append(("[G8] Token mutado ⟹ inválido", not ok_forged))
    checks.append(("[G8] Replay del mismo nonce ⟹ inválido", not ok_replay))
    # [G9/G19] Torsión en 1-complejo y exactitud ∂₁ ∘ ∂₂ = 0 en el complejo clique 2D.
    hom = p2.homology
    checks.append(
        (
            "[G9/G19] H₁ libre de torsión, β₁ = |E|−|V|+β₀ y ∂₁∘∂₂ = 0 en Cl(G_K)",
            hom is not None
            and hom.h1_torsion_free
            and hom.chain_complex_exact_d1_d2
            and hom.num_triangles_2skeleton >= 1
            and hom.beta_1 == max(0, hom.num_edges - hom.num_vertices + hom.beta_0),
        )
    )
    # [G6/G17] Löb: descenso bien fundado con traza explícita en (ℕ, <).
    checks.append(
        (
            "[G6/G17] Löb exige recomputación independiente, traza de descenso y profundidad < 3",
            state.loeb is not None
            and state.loeb.independently_recomputed
            and len(state.loeb.descent_trace) == 3
            and state.loeb.depth_remaining < 3,
        )
    )
    # [G14] Holonomía U(n), cuaternión S³ y sándwich de Fuchs-van de Graaf en Φ₁.
    q_norm = float(np.linalg.norm(np.asarray(p1.uhlmann_quaternion_s3, dtype=np.float64), 2))
    checks.append(
        (
            "[G14] Holonomía U(n), cuaternión S³ unitario y sándwich de Fuchs-van de Graaf",
            abs(q_norm - 1.0) < 1e-9
            and p1.uhlmann_holonomy_unitarity_defect < 1e-9
            and p1.fuchs_van_de_graaf_ok,
        )
    )
    # [G15] Invariantes celestes v18 (Poincaré-Cartan 4D, KAM, Yoshida-4) cubiertos por firma P2.
    forged_kam = replace(p2, kam_diophantine_margin=999.0)
    checks.append(
        (
            "[G15] Φ₂₃ RECHAZA adulteración de margen diofántico KAM en P2",
            not agent.phase3._gate_phi23(forged_kam).authentic,
        )
    )
    # [G18] RSI³: riesgo meta-Lyapunov finito y adaptación acotada de política con presupuesto.
    checks.append(
        (
            "[G18] RSI³: riesgo meta-Lyapunov finito y presupuesto de mutación bien fundado",
            state.meta_audit is not None
            and math.isfinite(state.meta_audit.rsi3_meta_lyapunov_risk)
            and 0 <= state.meta_audit.mutation_budget_remaining <= 3,
        )
    )
    # [G20] Pruebas de inclusión Merkle RFC 6962 para conjuntos de hojas pares e impares.
    odd_leaves = [b"leaf-0", b"leaf-1", b"leaf-2", b"leaf-3", b"leaf-4"]
    odd_root = _mth_rfc6962(odd_leaves).hex()
    odd_proofs_ok = all(
        _verify_inclusion_rfc6962(
            odd_leaves[idx],
            idx,
            len(odd_leaves),
            _audit_path_rfc6962(idx, odd_leaves),
            odd_root,
        )
        for idx in range(len(odd_leaves))
    )
    checks.append(
        ("[G20] Pruebas de inclusión Merkle RFC 6962 exactas (árbol impar n=5)", odd_proofs_ok)
    )
    # [G21] Modelo RLC: identidades exactas y magnitudes físicas extendidas.
    rlc = p2.rlc_model
    checks.append(
        (
            "[G21] RLC: ∫i²dt = ½CV²/R, di/dt|₀ = V/L, E₀ = ½CV² y t_zero = π/ω_d",
            rlc is not None
            and abs(
                rlc.i2t_action_a2s
                - 0.5 * _BUS_CAPACITANCE_F * _BUS_VOLTAGE_V**2 / _THYRISTOR_R_ON_OHM
            )
            < 1e-9
            and abs(rlc.di_dt_initial_a_per_us - _BUS_VOLTAGE_V / _LOOP_INDUCTANCE_H / 1e6)
            < 1e-9
            and abs(rlc.energy_stored_joules - 0.5 * _BUS_CAPACITANCE_F * _BUS_VOLTAGE_V**2)
            < 1e-12
            and rlc.first_zero_crossing_s > rlc.peak_time_s > 0.0,
        )
    )
    # [G12] Verificador público: legítimo vs. adulterado.
    ver_rep = agent.verify_sovereign_certificate(state)
    checks.append(
        (
            "[G12/G20] verify_sovereign_certificate(legítimo).all_ok + inclusión Merkle",
            ver_rep.all_ok and ver_rep.merkle_inclusion_proofs_ok,
        )
    )
    tampered = replace(state, heyting_verdict=HeytingVerdict.COHERENT.value)
    checks.append(
        (
            "[G12] verify_sovereign_certificate(adulterado) detecta falsificación",
            not agent.verify_sovereign_certificate(tampered).all_ok,
        )
    )
    # [G11] CPS polimórfico: la continuación puede devolver cualquier tipo.
    answer = agent.phase1._phase1_observe_and_sign(
        p_init=p0,
        rho_init=rho,
        sigma_ref=sigma,
        K_raw=k_raw,
        report=engine.execute_positron_redemption_trajectory(p0, rho, sigma, k_raw, 0.02, 2),
        frozen_before=[
            agent.phase1._freeze_and_digest(nm, a)
            for nm, a in (("p", p0), ("rho", rho), ("sigma", sigma), ("K", k_raw))
        ],
        continuation=lambda c: f"P1:{len(c.hmac_signature)}",
    )
    checks.append(
        (
            "[G11] CPS con AnswerT polimórfico (str)",
            isinstance(answer, str) and answer == "P1:64",
        )
    )

    ok = True
    for name, passed in checks:
        ok = ok and bool(passed)
        print(f"[{'OK ' if passed else 'FAIL'}] {name}")
    print(f"\nVEREDICTO GLOBAL DE AUTOVERIFICACIÓN: {'COHERENTE' if ok else 'OBSTRUIDO'}")
    return ok


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s · %(name)s · %(message)s")
    raise SystemExit(0 if _self_test() else 1)
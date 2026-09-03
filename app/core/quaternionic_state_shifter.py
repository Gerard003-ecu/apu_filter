# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Quaternionic State Shifter (Reactor Cuaterniónico de Estado)        ║
║ Ruta   : app/core/quaternionic_state_shifter.py                              ║
║ Versión: 4.0.0-Hurwitz-Spin3-Atlas-Merkle-FPU-Secure                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Mapea S = (s_purpose, s_confidence, s_constraints, s_risk)ᵀ ∈ R⁴ al cuaternión
q = q₀ + q₁i + q₂j + q₃k ∈ H, y ejecuta el funtor compuesto

        Φ₃ ∘ Φ₂ ∘ Φ₁ :  R⁴ ──Φ₁──▶ H ──Φ₂──▶ H ──Φ₃──▶ Reporte de lazo cerrado

donde cada fase está *anidada* en la siguiente: el último método de la fase k
produce exactamente el objeto que consume el primer método de la fase k+1, y
la propia fase k verifica esa admisibilidad antes de cerrarse.

════════════════════════════════════════════════════════════════════════════════
I. ANCLAJE MATEMÁTICO (sólo enunciados demostrables)
════════════════════════════════════════════════════════════════════════════════

Def. 1 (Álgebra de Hamilton). H = R⟨1,i,j,k⟩, i² = j² = k² = ijk = −1. Es una
  R-álgebra de división asociativa, no conmutativa, de centro Z(H) = R.

Def. 2 (Composición de Hurwitz). ‖pq‖ = ‖p‖‖q‖. H es C*-álgebra real con la
  involución q* = q₀ − v: (pq)* = q*p*, q q* = ‖q‖², ‖q* q‖ = ‖q‖².

Def. 3 (Representación regular). L(q)x = qx, R(q)x = xq, L,R ∈ M₄(R).
  L(q) = q₀ I₄ + K(q),  K = −Kᵀ,  ‖K‖_F = 2‖v‖,  ‖L‖_F = 2‖q‖.
  L(q)ᵀL(q) = ‖q‖² I₄  (conforme-ortogonal; NO antisimétrica salvo q₀ = 0).
  det L(q) = ‖q‖⁴.  L es homomorfismo, R antihomomorfismo, [L(p), R(q)] = 0.
  Polinomio mínimo (q ∉ R):  m(λ) = λ² − 2q₀λ + ‖q‖²;  χ_L(λ) = m(λ)².
  σ(L(q)) = {q₀ ± i‖v‖}, cada uno con multiplicidad algebraica 2.

Def. 4 (Cayley–Dickson / Pauli). Φ_C : H ↪ M₂(C), q = α + βj, α = q₀+iq₁,
  β = q₂+iq₃:  Φ_C(q) = [[α, β], [−β̄, ᾱ]] = q₀I + i(q₁σ₃ + q₂σ₂ + q₃σ₁).
  det Φ_C(q) = ‖q‖², Φ_C(q)*Φ_C(q) = ‖q‖² I₂,  Φ_C(S³) = SU(2) ≅ Spin(3).

Def. 5 (Clases de similitud espectral). Para μ ∈ H∖R, [μ] = {sμs⁻¹} es la
  2-esfera de centro μ₀ y radio ‖Im μ‖ en R ⊕ Im(H); representante canónico
  μ₀ + i‖Im μ‖ ∈ C. Coincide con σ(Φ_C(μ)).

Def. 6 (Hopf ≡ Bloch). Con ψ = (α, β)/‖q‖ ∈ S³ ⊂ C², ρ = ψψ† es un estado
  puro (Tr ρ = Tr ρ² = 1) y su vector de Bloch
      b = (2Re(αβ̄), 2Im(αβ̄), |α|² − |β|²)/‖q‖² ∈ S²
  es exactamente la fibración de Hopf π : S³ → S², invariante bajo la fibra
  U(1) actuando por multiplicación izquierda por e^{iφ} ∈ C ⊂ H.

Def. 7 (Recubrimiento Spin(3) → SO(3)). Ad(r)x = r x r*, r ∈ S³. En Im(H),
  Ad(r) = I + 2r₀[v]ₓ + 2[v]ₓ² ∈ SO(3) (Rodrigues), ángulo 2·atan2(‖v‖, r₀).
  El núcleo es {±1}: parametrización sin singularidades (sin gimbal lock).

════════════════════════════════════════════════════════════════════════════════
II. AXIOMÁTICA NUMÉRICA
════════════════════════════════════════════════════════════════════════════════

Ax. I  (Sumación de redondeo correcto). Toda reducción escalar usa `math.fsum`
       (Shewchuk), que domina estrictamente a Kahan/Neumaier; Neumaier se usa
       como auditor independiente.
Ax. II (Condicionamiento angular). Ningún ángulo se obtiene de arccos/arcsin;
       siempre `atan2` de longitudes (error O(ε) uniforme en [0, π]).
Ax. III(Traza de von Neumann). Tr ρ = ‖ψ‖²/‖q‖² = 1 y Tr ρ² = 1, con
       |Tr ρ − 1| ≤ c·ε_mach.  (Nota: Tr Φ_C(q) = 2q₀, no ‖q‖².)
Ax. IV (Determinismo). Todo hash se calcula sobre bytes canónicos con
       cero-signado normalizado, longitud-prefijados y con token de fase.
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from dataclasses import dataclass
from typing import Any, Dict, Final, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

__all__ = [
    "QuaternionicStateShifter",
    "QuaternionicState",
    "Phase1Handoff",
    "Phase2Handoff",
    "Phase3Report",
]

__version__: Final[str] = "4.0.0-Hurwitz-Spin3-Atlas-Merkle-FPU-Secure"

logger = logging.getLogger("APU.Physics.QuaternionicStateShifter")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES IEEE-754 Y ALGEBRAICAS
# ─────────────────────────────────────────────────────────────────────────────
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_FLOAT_MAX: Final[float] = float(np.finfo(np.float64).max)
_MIN_NORMAL: Final[float] = float(np.finfo(np.float64).tiny)
_I2C: Final[np.ndarray] = np.eye(2, dtype=np.complex128)
_I3: Final[np.ndarray] = np.eye(3, dtype=np.float64)
_I4: Final[np.ndarray] = np.eye(4, dtype=np.float64)
_SIGMA1: Final[np.ndarray] = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SIGMA2: Final[np.ndarray] = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SIGMA3: Final[np.ndarray] = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_GOLDEN_ANGLE: Final[float] = math.pi * (3.0 - math.sqrt(5.0))
_PHASE1_ENTRY: Final[str] = "phase2_from_phase1"
_PHASE2_ENTRY: Final[str] = "phase3_from_phase2"
_MERKLE_GENESIS: Final[bytes] = b"\x00" * 32


# ═════════════════════════════════════════════════════════════════════════════
# NÚCLEO NUMÉRICO Y ALGEBRAICO (utilidades puras, sin estado)
# ═════════════════════════════════════════════════════════════════════════════

def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Copia contigua de sólo lectura (inmutabilidad efectiva)."""
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


def _canonicalize_signed_zero(arr: np.ndarray) -> np.ndarray:
    """−0.0 ↦ +0.0 en real y complejo (Ax. IV: bytes canónicos)."""
    src = np.asarray(arr)
    if np.iscomplexobj(src):
        re = np.array(src.real, dtype=np.float64, copy=True)
        im = np.array(src.imag, dtype=np.float64, copy=True)
        re[re == 0.0] = 0.0
        im[im == 0.0] = 0.0
        out = np.zeros(src.shape, dtype=np.complex128)
        out.real = re
        out.imag = im
        return out
    out = np.array(src, dtype=np.float64, copy=True)
    out[out == 0.0] = 0.0
    return out


def _as_finite_vector4(S: Any) -> np.ndarray:
    """Vector real finito de forma exacta (4,), float64, cero canónico."""
    arr = np.asarray(S, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(
            "La señal de estado debe ser estrictamente cuatridimensional. "
            f"Obtenida: {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("La señal de estado contiene valores NaN o infinitos.")
    return _canonicalize_signed_zero(arr.reshape(4))


def _stable_norm(arr: np.ndarray) -> Tuple[float, float]:
    """(‖x‖₂, ‖x‖₂²) estables frente a overflow/underflow (hypot + reescalado)."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0, 0.0
    values = [float(x) for x in a]
    norm = float(math.hypot(*values))
    if not math.isfinite(norm):
        scale = max(abs(x) for x in values)
        if scale == 0.0:
            return 0.0, 0.0
        ssq = max(0.0, math.fsum((x / scale) * (x / scale) for x in values))
        norm = scale * math.sqrt(ssq)
    if not math.isfinite(norm):
        raise OverflowError("La norma excede el rango representable en float64.")
    squared = norm * norm
    if not math.isfinite(squared):
        raise OverflowError("‖q‖² excede el rango representable en float64.")
    return float(norm), float(squared)


def _neumaier_sum(values: Iterable[float]) -> float:
    """Sumación Kahan–Babuška–Neumaier (auditor independiente de `fsum`)."""
    total = 0.0
    comp = 0.0
    for raw in values:
        x = float(raw)
        t = total + x
        if abs(total) >= abs(x):
            comp += (total - t) + x
        else:
            comp += (x - t) + total
        total = t
    return float(total + comp)


def _relative_residual(actual: float, expected: float) -> float:
    """|a − e| / max(1, |e|); NaN si algún operando no es finito."""
    if not math.isfinite(actual) or not math.isfinite(expected):
        return math.nan
    return abs(actual - expected) / max(1.0, abs(expected))


def _frobenius(mat: np.ndarray) -> float:
    return float(la.norm(np.asarray(mat), ord="fro"))


def _canonical_bytes(arr: np.ndarray) -> bytes:
    """dtype|shape longitud-prefijados + bytes contiguos con cero canónico."""
    a = np.ascontiguousarray(_canonicalize_signed_zero(arr))
    header = f"{a.dtype.str}|{a.shape}".encode("utf-8")
    return len(header).to_bytes(8, "little") + header + a.tobytes()


def _sha256_hex_with_token(phase_token: str, *arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        payload = _canonical_bytes(arr)
        h.update(len(payload).to_bytes(8, "little"))
        h.update(payload)
    token = phase_token.encode("utf-8")
    h.update(len(token).to_bytes(8, "little"))
    h.update(token)
    return h.hexdigest()


def _merkle_chain(links: Sequence[Tuple[str, str]]) -> str:
    """h_k = SHA256(h_{k−1} ‖ len(token) ‖ token ‖ hash_k); h₀ = 0³²."""
    digest = _MERKLE_GENESIS
    for token, state_hash in links:
        m = hashlib.sha256()
        m.update(digest)
        tok = token.encode("utf-8")
        m.update(len(tok).to_bytes(8, "little"))
        m.update(tok)
        m.update(bytes.fromhex(state_hash))
        digest = m.digest()
    return digest.hex()


def _pack_f64(value: float) -> bytes:
    """float64 little-endian con NaN canónico."""
    if math.isnan(value):
        return b"\x00\x00\x00\x00\x00\x00\xf8\x7f"
    return struct.pack("<d", float(value))


def _left_matrix(q: np.ndarray) -> np.ndarray:
    """L(q)x = qx en la base {1,i,j,k}; L(q) = q₀I + K, K antisimétrica."""
    q0, q1, q2, q3 = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array(
        [[q0, -q1, -q2, -q3],
         [q1,  q0, -q3,  q2],
         [q2,  q3,  q0, -q1],
         [q3, -q2,  q1,  q0]],
        dtype=np.float64,
    )


def _right_matrix(q: np.ndarray) -> np.ndarray:
    """R(q)x = xq. Antihomomorfismo; [L(p), R(q)] = 0 ∀ p, q (bimódulo H-H)."""
    q0, q1, q2, q3 = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array(
        [[q0, -q1, -q2, -q3],
         [q1,  q0,  q3, -q2],
         [q2, -q3,  q0,  q1],
         [q3,  q2, -q1,  q0]],
        dtype=np.float64,
    )


def _cayley_dickson_matrix(q: np.ndarray) -> np.ndarray:
    """Φ_C(q) = [[α, β], [−β̄, ᾱ]], α = q₀ + iq₁, β = q₂ + iq₃."""
    alpha = complex(float(q[0]), float(q[1]))
    beta = complex(float(q[2]), float(q[3]))
    return np.array(
        [[alpha, beta], [-beta.conjugate(), alpha.conjugate()]],
        dtype=np.complex128,
    )


def _pauli_reconstruction(q: np.ndarray) -> np.ndarray:
    """q₀I + i(q₁σ₃ + q₂σ₂ + q₃σ₁): debe coincidir con Φ_C(q)."""
    q0, q1, q2, q3 = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return q0 * _I2C + 1j * (q1 * _SIGMA3 + q2 * _SIGMA2 + q3 * _SIGMA1)


def _hamilton_vector(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Producto de Hamilton con acumulación de redondeo correcto (Ax. I)."""
    p0, p1, p2, p3 = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
    q0, q1, q2, q3 = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array(
        [
            math.fsum((p0 * q0, -p1 * q1, -p2 * q2, -p3 * q3)),
            math.fsum((p0 * q1, p1 * q0, p2 * q3, -p3 * q2)),
            math.fsum((p0 * q2, -p1 * q3, p2 * q0, p3 * q1)),
            math.fsum((p0 * q3, p1 * q2, -p2 * q1, p3 * q0)),
        ],
        dtype=np.float64,
    )


def _conjugate_vector(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _real_inner(p: np.ndarray, q: np.ndarray) -> float:
    """⟨p, q⟩ = Re(p̄ q) = Σ pᵢqᵢ (métrica redonda de S³)."""
    return float(math.fsum(float(a) * float(b) for a, b in zip(p, q)))


def _skew3(v: np.ndarray) -> np.ndarray:
    """[v]ₓ ∈ so(3): [v]ₓ w = v × w."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _orthogonality_residual(left: np.ndarray, squared_norm: float) -> float:
    """‖LᵀL − ‖q‖²I₄‖_F / max(1, ‖q‖²)."""
    if squared_norm <= 0.0:
        return _frobenius(left)
    return float(_frobenius(left.T @ left - squared_norm * _I4) / max(1.0, squared_norm))


def _cstar_residual(cd: np.ndarray, squared_norm: float) -> float:
    """‖Φ_C(q)*Φ_C(q) − ‖q‖²I₂‖_F / max(1, ‖q‖²)."""
    return float(_frobenius(cd.conj().T @ cd - squared_norm * _I2C) / max(1.0, squared_norm))


def _sinc(x: float) -> float:
    """sin(x)/x, serie de Taylor en |x| < 1e‑4 (error < 1e‑21)."""
    ax = abs(x)
    if ax < 1e-4:
        xx = x * x
        return 1.0 - xx / 6.0 + (xx * xx) / 120.0
    return math.sin(x) / x


def _angle_between_unit(u: np.ndarray, v: np.ndarray) -> float:
    """θ = 2·atan2(‖u − v‖, ‖u + v‖) ∈ [0, π] (Ax. II)."""
    diff = float(math.hypot(*(np.asarray(u) - np.asarray(v))))
    summ = float(math.hypot(*(np.asarray(u) + np.asarray(v))))
    return float(2.0 * math.atan2(diff, summ))


# ═════════════════════════════════════════════════════════════════════════════
# OBJETOS DE ESTADO Y FRONTERAS DE FASE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True, eq=False)
class QuaternionicState:
    """
    Estado canónico q ∈ H ≅ R⁴ con sus tres representaciones y auditoría local.

    Invariantes: LᵀL = ‖q‖²I₄, det L = ‖q‖⁴, det Φ_C = ‖q‖², q q* = ‖q‖²,
    theta = atan2(‖v‖, q₀) ∈ [0, π] (argumento polar en R ⊕ Im H).
    """

    vector_rep: np.ndarray
    scalar_part: float
    vector_part: np.ndarray
    vector_norm: float
    norm: float
    squared_norm: float
    theta: float
    is_unitary: bool
    unitarity_residual: float
    orthogonality_residual: float
    condition_estimate: float
    cayley_dickson_matrix: np.ndarray
    left_mult_matrix: np.ndarray
    right_mult_matrix: np.ndarray
    sha256_hash: str
    phase_token: str

    def __post_init__(self) -> None:
        for name in ("vector_rep", "vector_part", "cayley_dickson_matrix",
                     "left_mult_matrix", "right_mult_matrix"):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))

    def __hash__(self) -> int:
        return hash(self.sha256_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuaternionicState):
            return NotImplemented
        return self.sha256_hash == other.sha256_hash

    def __repr__(self) -> str:
        return (f"QuaternionicState(norm={self.norm:.17g}, theta={self.theta:.17g}, "
                f"unitary={self.is_unitary}, hash={self.sha256_hash[:12]!r})")


@dataclass(frozen=True, slots=True, eq=False)
class Phase1Handoff:
    """Frontera Φ₁→₂ : (q, δ₁) ∈ H × D₁. Codominio de Fase 1 = dominio de Fase 2."""

    state: QuaternionicState
    diagnostics: Dict[str, float]
    next_entrypoint: str

    def __hash__(self) -> int:
        return hash((self.state.sha256_hash, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase1Handoff):
            return NotImplemented
        return (self.state.sha256_hash == other.state.sha256_hash
                and self.next_entrypoint == other.next_entrypoint)


@dataclass(frozen=True, slots=True, eq=False)
class Phase2Handoff:
    """Frontera Φ₂→₃ : (q, ρ₂) ∈ H × D₂. Codominio de Fase 2 = dominio de Fase 3."""

    state: QuaternionicState
    algebra_report: Dict[str, Any]
    next_entrypoint: str

    def __hash__(self) -> int:
        return hash((self.state.sha256_hash, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase2Handoff):
            return NotImplemented
        return (self.state.sha256_hash == other.state.sha256_hash
                and self.next_entrypoint == other.next_entrypoint)


@dataclass(frozen=True, slots=True, eq=False)
class Phase3Report:
    """Reporte de lazo cerrado Φ₃ ∘ Φ₂ ∘ Φ₁ con sello Merkle."""

    state: QuaternionicState
    source_state: QuaternionicState
    spectral_audit: Dict[str, Any]
    similarity_class: Dict[str, Any]
    riemann_chart: Dict[str, Any]
    transport_metrics: Dict[str, Any]
    liouville_audit: Dict[str, Any]
    governance_seal: Dict[str, Any]
    phase_chain: Tuple[str, ...]

    def __hash__(self) -> int:
        return hash(self.governance_seal.get("seal", self.state.sha256_hash))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase3Report):
            return NotImplemented
        return hash(self) == hash(other)


# ═════════════════════════════════════════════════════════════════════════════
# REACTOR CUATERNIÓNICO — TRES FASES ANIDADAS
# ═════════════════════════════════════════════════════════════════════════════

class QuaternionicStateShifter:
    """
    FASE 1 : R⁴ → H  (validación, invariantes de Banach, tres representaciones,
             Pauli/SU(2), traza de von Neumann).         Cierre ⇒ Phase1Handoff
    FASE 2 : Álgebra de composición (Hurwitz, C*, exp/log, inverso, polinomio
             mínimo, homomorfismos L/R/Φ_C).             Cierre ⇒ Phase2Handoff
    FASE 3 : Espectro, clases de similitud, Hopf≡Bloch, atlas de Riemann,
             geodesia/SLERP, Spin(3)→SO(3), Liouville, sello Merkle.
    """

    def __init__(
        self,
        tolerance: float = 1e-12,
        regularization: float = 1e-15,
        auto_renormalize: bool = True,
    ) -> None:
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance debe ser finita y estrictamente positiva.")
        if not math.isfinite(regularization) or regularization <= 0.0:
            raise ValueError("regularization debe ser finita y estrictamente positiva.")
        self._tol = float(tolerance)
        self._reg = float(regularization)
        self._auto_renormalize = bool(auto_renormalize)

    def _tolerance_of(self, scale: float = 1.0) -> float:
        """Tolerancia mixta abs/rel acotada inferiormente por 32 ULP."""
        return max(self._tol, self._tol * abs(scale), 32.0 * _MACHINE_EPS)

    def _accept(self, residual: float, scale: float = 1.0) -> bool:
        return math.isfinite(residual) and residual <= self._tolerance_of(scale)

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 1 — INGESTIÓN, VALIDACIÓN Y CANONIZACIÓN  Φ₁ : R⁴ → H
    # ═════════════════════════════════════════════════════════════════════════

    def phase1_validate_signal(self, S: np.ndarray) -> np.ndarray:
        """Fase 1.1 — S ∈ R⁴ finito, float64, −0.0 ↦ +0.0."""
        return _as_finite_vector4(S)

    def phase1_banach_invariants(self, vector: np.ndarray) -> Dict[str, float]:
        """
        Fase 1.2 — Normas ℓ¹, ℓ², ℓ^∞ y constantes de equivalencia en R⁴.

        Cotas exactas (n = 4):  1 ≤ ‖q‖₁/‖q‖₂ ≤ 2,   1/2 ≤ ‖q‖_∞/‖q‖₂ ≤ 1.
        Auditoría cruzada fsum vs Neumaier de Σqᵢ² (Ax. I).
        """
        q = _as_finite_vector4(vector)
        abs_q = np.abs(q)
        l1 = float(math.fsum(float(x) for x in abs_q))
        l2, l2sq = _stable_norm(q)
        linf = float(np.max(abs_q))
        squares = [float(x) * float(x) for x in q]
        fsum_sq = math.fsum(squares)
        neumaier_sq = _neumaier_sum(squares)
        r12 = l1 / max(l2, _MIN_NORMAL)
        rinf2 = linf / max(l2, _MIN_NORMAL)
        tol = self._tolerance_of(1.0)
        bounds_ok = (l2 == 0.0) or (
            1.0 - tol <= r12 <= 2.0 + tol and 0.5 - tol <= rinf2 <= 1.0 + tol
        )
        return {
            "l1_banach_norm": l1,
            "l2_banach_norm": l2,
            "l2_squared": l2sq,
            "linf_banach_norm": linf,
            "fsum_square_sum": float(fsum_sq),
            "neumaier_square_sum": float(neumaier_sq),
            "summation_cross_residual": _relative_residual(neumaier_sq, fsum_sq),
            "norm_equivalence_l1_l2": float(r12),
            "norm_equivalence_linf_l2": float(rinf2),
            "equivalence_bounds_satisfied": 1.0 if bounds_ok else 0.0,
        }

    def _build_state(self, vector: np.ndarray, phase_token: str) -> QuaternionicState:
        """
        Fase 1.3 — Materialización de Φ : R⁴ → H y sus representaciones
        L(q), R(q), Φ_C(q), argumento polar θ = atan2(‖v‖, q₀) y firma por fase.
        """
        q = _as_finite_vector4(vector)
        q0 = float(q[0])
        norm, squared_norm = _stable_norm(q)
        vector_norm, _ = _stable_norm(q[1:])
        theta = math.atan2(vector_norm, q0)

        unitarity_residual = abs(norm - 1.0)
        is_unitary = unitarity_residual <= self._tolerance_of(max(1.0, norm))

        left = _left_matrix(q)
        right = _right_matrix(q)
        cd = _cayley_dickson_matrix(q)
        orthogonality_residual = _orthogonality_residual(left, squared_norm)

        # κ₂(L(q)) ≡ 1 para q ≠ 0; el exceso sobre 1 es defecto numérico puro.
        condition_estimate = math.inf if norm < self._reg else 1.0 + orthogonality_residual
        sha = _sha256_hex_with_token(phase_token, q, cd, left, right)

        return QuaternionicState(
            vector_rep=q, scalar_part=q0, vector_part=q[1:],
            vector_norm=float(vector_norm), norm=float(norm),
            squared_norm=float(squared_norm), theta=float(theta),
            is_unitary=bool(is_unitary),
            unitarity_residual=float(unitarity_residual),
            orthogonality_residual=float(orthogonality_residual),
            condition_estimate=float(condition_estimate),
            cayley_dickson_matrix=cd, left_mult_matrix=left, right_mult_matrix=right,
            sha256_hash=sha, phase_token=phase_token,
        )

    def phase1_construct_canonical_state(self, S: np.ndarray) -> QuaternionicState:
        """Fase 1.4 — Constructor público del estado canónico."""
        return self._build_state(S, "PHASE1/CANONICAL")

    def phase1_representation_audit(self, state: QuaternionicState) -> Dict[str, float]:
        """
        Fase 1.5 — Consistencia de las tres representaciones.

          · Φ_C tiene forma Cayley–Dickson y coincide con q₀I + i(q₁σ₃+q₂σ₂+q₃σ₁)
          · det Φ_C = ‖q‖², Φ_C*Φ_C = ‖q‖²I₂, Φ_C/‖q‖ ∈ SU(2)
          · tr L = 4q₀, ‖L‖_F = 2‖q‖, L = q₀I + K con K = −Kᵀ, ‖K‖_F = 2‖v‖
          · L y R comparten diagonal y difieren en el bloque Im: R = q₀I − K(q̄)ᵀ…
            (se audita R(q) = L(q̄)ᵀ, identidad exacta del antihomomorfismo)
        """
        q = state.vector_rep
        cd = np.array(state.cayley_dickson_matrix, copy=True)
        left = state.left_mult_matrix
        right = state.right_mult_matrix
        n, n2 = state.norm, state.squared_norm

        alpha = complex(float(q[0]), float(q[1]))
        beta = complex(float(q[2]), float(q[3]))
        cd_form = float(abs(cd[0, 0] - alpha) + abs(cd[0, 1] - beta)
                        + abs(cd[1, 0] + beta.conjugate()) + abs(cd[1, 1] - alpha.conjugate()))
        pauli_res = _frobenius(cd - _pauli_reconstruction(q)) / max(1.0, n)

        try:
            det_cd = complex(la.det(cd))
        except la.LinAlgError:
            det_cd = complex(math.nan, math.nan)
        det_cd_res = _relative_residual(det_cd.real, n2) + abs(det_cd.imag) / max(1.0, n2)

        if n >= self._reg:
            U = cd / n
            su2_unitarity = _frobenius(U.conj().T @ U - _I2C)
            try:
                su2_det = abs(complex(la.det(U)) - 1.0)
            except la.LinAlgError:
                su2_det = math.nan
        else:
            su2_unitarity, su2_det = math.nan, math.nan

        K = left - state.scalar_part * _I4
        skew_res = _frobenius(K + K.T) / max(1.0, n)
        k_fro_res = _relative_residual(_frobenius(K), 2.0 * state.vector_norm)
        trace_res = abs(float(np.trace(left)) - 4.0 * state.scalar_part) / max(1.0, n)
        fro_res = _relative_residual(_frobenius(left), 2.0 * n)
        # R(q) = L(q̄)ᵀ  (identidad exacta)
        rl_res = _frobenius(right - _left_matrix(_conjugate_vector(q)).T) / max(1.0, n)

        return {
            "cayley_dickson_form_residual": cd_form,
            "pauli_decomposition_residual": float(pauli_res),
            "cayley_dickson_det_residual": float(det_cd_res),
            "cstar_residual": _cstar_residual(cd, n2),
            "su2_unitarity_residual": float(su2_unitarity),
            "su2_det_residual": float(su2_det),
            "left_trace_residual": float(trace_res),
            "left_frobenius_residual": float(fro_res),
            "conformal_skew_residual": float(skew_res),
            "conformal_skew_frobenius_residual": float(k_fro_res),
            "right_equals_left_conjugate_transpose_residual": float(rl_res),
        }

    def phase1_von_neumann_trace_audit(self, state: QuaternionicState) -> Dict[str, float]:
        """
        Fase 1.6 — Axioma III corregido: ρ = ψψ†/‖q‖², ψ = (α, β).

        Tr ρ = 1, Tr ρ² = 1 (pureza), ρ = ρ†, σ(ρ) = {1, 0}. El vector de Bloch
        de ρ se compara en Fase 3.4 con la fibración de Hopf (Def. 6).
        """
        if state.norm < self._reg:
            return {"trace": math.nan, "trace_residual": math.nan,
                    "purity": math.nan, "hermiticity_residual": math.nan,
                    "spectrum_residual": math.nan}
        psi = np.array([complex(state.vector_rep[0], state.vector_rep[1]),
                        complex(state.vector_rep[2], state.vector_rep[3])],
                       dtype=np.complex128)
        rho = np.outer(psi, psi.conj()) / state.squared_norm
        tr = float(np.real(np.trace(rho)))
        purity = float(np.real(np.trace(rho @ rho)))
        herm = _frobenius(rho - rho.conj().T)
        try:
            ev = np.sort(np.real(la.eigvalsh(rho)))
            spec_res = float(math.hypot(ev[0] - 0.0, ev[1] - 1.0))
        except la.LinAlgError:
            spec_res = math.nan
        return {
            "trace": tr,
            "trace_residual": abs(tr - 1.0),
            "purity": purity,
            "purity_residual": abs(purity - 1.0),
            "hermiticity_residual": float(herm),
            "spectrum_residual": spec_res,
            "trace_of_cayley_dickson_over_2": float(np.real(np.trace(state.cayley_dickson_matrix)) / 2.0),
        }

    def phase1_close_and_open_phase2(self, S: np.ndarray) -> Phase1Handoff:
        """
        Fase 1.7 — Cierre formal de Fase 1 y apertura verificada de Fase 2.

            Φ₁→₂ : S ∈ R⁴ ↦ (q, δ₁) ∈ H × D₁

        Último método de Fase 1. Su codominio es, por definición, el dominio de
        `phase2_from_phase1`; se invoca dicho método y se exige identidad de
        firma, de modo que Fase 1 queda anidada como prefijo de Fase 2.
        """
        q_raw = self.phase1_validate_signal(S)
        state = self._build_state(q_raw, "PHASE1/CLOSED->PHASE2/OPEN")
        diagnostics: Dict[str, float] = {
            "norm": state.norm,
            "squared_norm": state.squared_norm,
            "vector_norm": state.vector_norm,
            "theta": state.theta,
            "unitarity_residual": state.unitarity_residual,
            "orthogonality_residual": state.orthogonality_residual,
            "condition_estimate": state.condition_estimate,
            "machine_epsilon": _MACHINE_EPS,
            **self.phase1_banach_invariants(state.vector_rep),
            **self.phase1_representation_audit(state),
            **{f"von_neumann_{k}": v for k, v in
               self.phase1_von_neumann_trace_audit(state).items()},
        }
        handoff = Phase1Handoff(state=state, diagnostics=diagnostics,
                                next_entrypoint=_PHASE1_ENTRY)

        opened = self.phase2_from_phase1(handoff)          # ← anidamiento Φ₁→₂
        if opened.sha256_hash != state.sha256_hash:
            raise RuntimeError("Invariante de anidamiento Φ₁→₂ violado.")
        logger.debug("Fase 1 cerrada: %s ‖q‖=%.18g θ=%.18g",
                     state.sha256_hash[:12], state.norm, state.theta)
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2 — ÁLGEBRA DE COMPOSICIÓN DE HURWITZ  Φ₂ : H → H
    # (continuación formal de phase1_close_and_open_phase2)
    # ═════════════════════════════════════════════════════════════════════════

    def phase2_from_phase1(self, handoff: Phase1Handoff) -> QuaternionicState:
        """Fase 2.0 — Entrada formal: admite Φ₁→₂ y devuelve q ∈ H."""
        if not isinstance(handoff, Phase1Handoff):
            raise TypeError("Se esperaba Phase1Handoff como frontera Φ₁→₂.")
        if handoff.next_entrypoint != _PHASE1_ENTRY:
            raise ValueError(f"Phase1Handoff inválido: se esperaba {_PHASE1_ENTRY!r}.")
        if handoff.state.vector_rep.shape != (4,):
            raise ValueError("El estado de Φ₁→₂ no es cuatridimensional.")
        return handoff.state

    def phase2_conjugate(self, state: QuaternionicState) -> QuaternionicState:
        """Fase 2.1 — Involución C*: q* = q₀ − v; (pq)* = q*p*, (q*)* = q."""
        return self._build_state(_conjugate_vector(state.vector_rep), "PHASE2/CONJUGATE")

    def phase2_inner_product(self, p: QuaternionicState, q: QuaternionicState) -> float:
        """Fase 2.2 — ⟨p, q⟩ = Re(p̄q); métrica redonda de S³."""
        return _real_inner(p.vector_rep, q.vector_rep)

    def phase2_hamilton_product(
        self, p: QuaternionicState, q: QuaternionicState, verify: bool = True,
    ) -> QuaternionicState:
        """
        Fase 2.3 — Producto pq. Triple cómputo (fsum, L(p)q, R(q)p) y, si
        `verify`, auditoría de Hurwitz y del homomorfismo Φ_C(pq) = Φ_C(p)Φ_C(q).
        """
        pv, qv = p.vector_rep, q.vector_rep
        res = _hamilton_vector(pv, qv)
        scale = max(1.0, p.norm * q.norm)
        via_left = np.asarray(p.left_mult_matrix @ qv, dtype=np.float64)
        via_right = np.asarray(q.right_mult_matrix @ pv, dtype=np.float64)
        for label, alt in (("L(p)q", via_left), ("R(q)p", via_right)):
            dev = float(math.hypot(*(res - alt))) / scale
            if dev > self._tolerance_of(1.0):
                logger.warning("Discrepancia producto vs %s: %.3e", label, dev)

        result = self._build_state(res, "PHASE2/HAMILTON_PRODUCT")
        if verify:
            hur = self.phase2_hurwitz_residual(p, q, result)
            if hur > max(10.0 * self._tol, 1e-10):
                logger.warning("Residual de Hurwitz elevado: %.3e", hur)
            cd_hom = _frobenius(result.cayley_dickson_matrix
                                - p.cayley_dickson_matrix @ q.cayley_dickson_matrix) / scale
            if cd_hom > self._tolerance_of(1.0):
                logger.warning("Residual de homomorfismo Φ_C: %.3e", cd_hom)
            if (self._auto_renormalize and p.is_unitary and q.is_unitary
                    and not result.is_unitary):
                result = self.phase2_normalize(result)
        return result

    def phase2_associator(
        self, p: QuaternionicState, q: QuaternionicState, r: QuaternionicState,
    ) -> Tuple[QuaternionicState, float]:
        """Fase 2.4 — [p,q,r] = (pq)r − p(qr) ≡ 0 (asociatividad de H)."""
        pq = self.phase2_hamilton_product(p, q, verify=False)
        qr = self.phase2_hamilton_product(q, r, verify=False)
        lhs = self.phase2_hamilton_product(pq, r, verify=False)
        rhs = self.phase2_hamilton_product(p, qr, verify=False)
        assoc = self._build_state(lhs.vector_rep - rhs.vector_rep, "PHASE2/ASSOCIATOR")
        return assoc, float(assoc.norm / max(1.0, p.norm * q.norm * r.norm))

    def phase2_commutator(
        self, p: QuaternionicState, q: QuaternionicState,
    ) -> Tuple[QuaternionicState, float]:
        """
        Fase 2.5 — [p, q] = pq − qp = 2 (v_p × v_q) ∈ Im(H).

        Se audita que la parte real sea nula (Z(H) = R) y la identidad con el
        doble producto vectorial.
        """
        pq = self.phase2_hamilton_product(p, q, verify=False)
        qp = self.phase2_hamilton_product(q, p, verify=False)
        comm_vec = pq.vector_rep - qp.vector_rep
        cross = 2.0 * np.cross(p.vector_part, q.vector_part)
        scale = max(1.0, p.norm * q.norm)
        if abs(float(comm_vec[0])) / scale > self._tolerance_of(1.0):
            logger.warning("Conmutador con parte real no nula: %.3e", comm_vec[0])
        cross_res = float(math.hypot(*(comm_vec[1:] - cross))) / scale
        if cross_res > self._tolerance_of(1.0):
            logger.warning("Residual [p,q] = 2 v_p×v_q: %.3e", cross_res)
        comm = self._build_state(comm_vec, "PHASE2/COMMUTATOR")
        return comm, float(comm.norm)

    def phase2_normalize(self, state: QuaternionicState) -> QuaternionicState:
        """Fase 2.6 — Retracción q ↦ q/‖q‖ ∈ S³."""
        if state.norm < self._reg:
            raise ZeroDivisionError("Norma nula o sub-regular; no se puede normalizar.")
        vec = state.vector_rep if state.is_unitary else state.vector_rep / state.norm
        return self._build_state(vec, "PHASE2/NORMALIZED")

    def phase2_polar_decomposition(
        self, state: QuaternionicState,
    ) -> Tuple[float, QuaternionicState]:
        """Fase 2.7 — q = ρu, ρ = ‖q‖ > 0, u ∈ S³ (≡ polar de Φ_C(q))."""
        if state.norm < self._reg:
            raise ZeroDivisionError("Descomposición polar no definida en el origen.")
        return float(state.norm), self.phase2_normalize(state)

    def phase2_exponential(self, state: QuaternionicState) -> QuaternionicState:
        """
        Fase 2.8 — exp(q₀ + v) = e^{q₀}(cos θ + sinc(θ)·v), θ = ‖v‖.

        Sin ramas: `sinc` es analítica en θ = 0. exp|Im(H) : su(2) → Spin(3).
        """
        scale = math.exp(state.scalar_part)
        if not math.isfinite(scale):
            raise OverflowError("exp(q₀) no es representable en float64.")
        theta = state.vector_norm
        c, sc = math.cos(theta), _sinc(theta)
        v = state.vector_part
        vec = np.array([scale * c, scale * sc * float(v[0]),
                        scale * sc * float(v[1]), scale * sc * float(v[2])], dtype=np.float64)
        if not np.all(np.isfinite(vec)):
            raise OverflowError("exp(q) desborda float64.")
        return self._build_state(vec, "PHASE2/EXPONENTIAL")

    def phase2_logarithm(self, state: QuaternionicState) -> QuaternionicState:
        """
        Fase 2.9 — Log principal: Log q = log‖q‖ + (θ/‖v‖)·v, θ = atan2(‖v‖, q₀).

        Corte de rama: si v = 0 y q₀ < 0, Log q no es único (cualquier û·π).
        Se toma el representante canónico π·i y se emite advertencia.
        """
        if state.norm < self._reg:
            raise ZeroDivisionError("Logaritmo no definido en el origen.")
        rho = math.log(state.norm)
        theta = state.theta
        if state.vector_norm < self._reg:
            if state.scalar_part < 0.0:
                logger.warning("Log en el corte de rama (eje real negativo): û := i.")
                vec = np.array([rho, math.pi, 0.0, 0.0], dtype=np.float64)
            else:
                vec = np.array([rho, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            factor = theta / state.vector_norm
            v = state.vector_part
            vec = np.array([rho, factor * float(v[0]), factor * float(v[1]),
                            factor * float(v[2])], dtype=np.float64)
        return self._build_state(vec, "PHASE2/LOGARITHM")

    def phase2_exp_log_roundtrip(self, state: QuaternionicState) -> Dict[str, float]:
        """
        Fase 2.10 — exp(Log q) = q ∀ q ≠ 0; Log(exp q) = q sii ‖Im q‖ < π.
        """
        if state.norm < self._reg:
            return {"exp_log_residual": math.nan, "log_exp_residual": math.nan}
        back = self.phase2_exponential(self.phase2_logarithm(state))
        exp_log = float(math.hypot(*(back.vector_rep - state.vector_rep))) / max(1.0, state.norm)
        if state.vector_norm < math.pi - self._tolerance_of(1.0):
            try:
                forth = self.phase2_logarithm(self.phase2_exponential(state))
                log_exp = float(math.hypot(*(forth.vector_rep - state.vector_rep))) / max(1.0, state.norm)
            except OverflowError:
                log_exp = math.nan
        else:
            log_exp = math.nan
        return {"exp_log_residual": exp_log, "log_exp_residual": log_exp}

    def phase2_inverse(self, state: QuaternionicState) -> QuaternionicState:
        """Fase 2.11 — q⁻¹ = q*/‖q‖²; audita q q⁻¹ = q⁻¹ q = 1."""
        if state.squared_norm < self._reg:
            raise ZeroDivisionError("Inverso no definido: norma nula o sub-regular.")
        inv = self._build_state(_conjugate_vector(state.vector_rep) / state.squared_norm,
                                "PHASE2/INVERSE")
        e0 = np.array([1.0, 0.0, 0.0, 0.0])
        for label, prod in (("q q⁻¹", _hamilton_vector(state.vector_rep, inv.vector_rep)),
                            ("q⁻¹ q", _hamilton_vector(inv.vector_rep, state.vector_rep))):
            res = float(math.hypot(*(prod - e0)))
            if res > self._tolerance_of(1.0):
                logger.warning("Residual de %s − 1: %.3e", label, res)
        return inv

    def phase2_divide(
        self, numerator: QuaternionicState, denominator: QuaternionicState,
        side: str = "right",
    ) -> QuaternionicState:
        """Fase 2.12 — División: 'right' n d⁻¹ ; 'left' d⁻¹ n (no coinciden en H)."""
        inv = self.phase2_inverse(denominator)
        if side == "right":
            return self.phase2_hamilton_product(numerator, inv, verify=True)
        if side == "left":
            return self.phase2_hamilton_product(inv, numerator, verify=True)
        raise ValueError("side debe ser 'right' o 'left'.")

    def phase2_cstar_identity(self, state: QuaternionicState) -> float:
        """Fase 2.13 — Residual de ‖q* q‖ = ‖q‖²."""
        prod = _hamilton_vector(_conjugate_vector(state.vector_rep), state.vector_rep)
        n, _ = _stable_norm(prod)
        return _relative_residual(n, state.squared_norm)

    def phase2_hurwitz_residual(
        self, p: QuaternionicState, q: QuaternionicState, product: QuaternionicState,
    ) -> float:
        """Fase 2.14 — Residual de ‖pq‖ = ‖p‖‖q‖."""
        expected = p.norm * q.norm
        return abs(product.norm) if expected <= 0.0 else abs(product.norm - expected) / max(1.0, expected)

    def phase2_bimodule_commutation(self, p: QuaternionicState, q: QuaternionicState) -> float:
        """Fase 2.15 — Residual de [L(p), R(q)] = 0."""
        lp, rq = p.left_mult_matrix, q.right_mult_matrix
        return _frobenius(lp @ rq - rq @ lp) / max(1.0, p.norm * q.norm)

    def phase2_homomorphism_residual(
        self, p: QuaternionicState, q: QuaternionicState, product: QuaternionicState,
    ) -> Dict[str, float]:
        """Fase 2.16 — L(pq)=L(p)L(q), R(pq)=R(q)R(p), Φ_C(pq)=Φ_C(p)Φ_C(q)."""
        scale = max(1.0, p.norm * q.norm)
        return {
            "left_homomorphism_residual": _frobenius(
                product.left_mult_matrix - p.left_mult_matrix @ q.left_mult_matrix) / scale,
            "right_antihomomorphism_residual": _frobenius(
                product.right_mult_matrix - q.right_mult_matrix @ p.right_mult_matrix) / scale,
            "cayley_dickson_homomorphism_residual": _frobenius(
                product.cayley_dickson_matrix
                - p.cayley_dickson_matrix @ q.cayley_dickson_matrix) / scale,
            "bimodule_commutation_residual": self.phase2_bimodule_commutation(p, q),
        }

    def phase2_minimal_polynomial_residual(self, state: QuaternionicState) -> Dict[str, float]:
        """
        Fase 2.17 — Polinomio mínimo: q² − 2q₀q + ‖q‖² = 0 en H y
        L² − 2q₀L + ‖q‖²I₄ = 0 en M₄(R). Prueba algebraica de que
        R[q] ≅ C (q ∉ R): todo cuaternión es raíz de un cuadrático real.
        """
        q = state.vector_rep
        q0, n2 = state.scalar_part, state.squared_norm
        sq = _hamilton_vector(q, q)
        alg = sq - 2.0 * q0 * q + n2 * np.array([1.0, 0.0, 0.0, 0.0])
        L = state.left_mult_matrix
        mat = L @ L - 2.0 * q0 * L + n2 * _I4
        scale = max(1.0, n2)
        return {
            "minimal_polynomial_algebraic_residual": float(math.hypot(*alg)) / scale,
            "minimal_polynomial_matrix_residual": _frobenius(mat) / scale,
        }

    def phase2_close_and_open_phase3(self, state: QuaternionicState) -> Phase2Handoff:
        """
        Fase 2.18 — Cierre formal de Fase 2 y apertura verificada de Fase 3.

            Φ₂→₃ : q ∈ H ↦ (q, ρ₂) ∈ H × D₂

        Último método de Fase 2; su codominio es el dominio de `phase3_from_phase2`.
        """
        conj = self.phase2_conjugate(state)
        qq = self.phase2_hamilton_product(state, conj, verify=False)
        algebra_report: Dict[str, Any] = {
            "phase_token": state.phase_token,
            "sha256_hash": state.sha256_hash,
            "norm": state.norm,
            "squared_norm": state.squared_norm,
            "vector_norm": state.vector_norm,
            "theta": state.theta,
            "is_unitary": state.is_unitary,
            "unitarity_residual": state.unitarity_residual,
            "orthogonality_residual": state.orthogonality_residual,
            "condition_estimate": state.condition_estimate,
            "cstar_residual": float(self.phase2_cstar_identity(state)),
            "bimodule_self_commutation_residual": float(self.phase2_bimodule_commutation(state, conj)),
            **self.phase2_homomorphism_residual(state, conj, qq),
            **self.phase2_minimal_polynomial_residual(state),
            **self.phase2_exp_log_roundtrip(state),
        }
        if state.norm >= self._reg:
            inv = self.phase2_inverse(state)
            algebra_report["inverse_hash"] = inv.sha256_hash
            algebra_report["inverse_norm_residual"] = _relative_residual(inv.norm, 1.0 / state.norm)

        handoff = Phase2Handoff(state=state, algebra_report=algebra_report,
                                next_entrypoint=_PHASE2_ENTRY)
        opened = self.phase3_from_phase2(handoff)          # ← anidamiento Φ₂→₃
        if opened.sha256_hash != state.sha256_hash:
            raise RuntimeError("Invariante de anidamiento Φ₂→₃ violado.")
        logger.debug("Fase 2 cerrada: %s auditado algebraicamente.", state.sha256_hash[:12])
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 3 — ESPECTRO, TOPOLOGÍA, TRANSPORTE Y GOBERNANZA  Φ₃
    # (continuación formal de phase2_close_and_open_phase3)
    # ═════════════════════════════════════════════════════════════════════════

    def phase3_from_phase2(self, handoff: Phase2Handoff) -> QuaternionicState:
        """Fase 3.0 — Entrada formal: admite Φ₂→₃."""
        if not isinstance(handoff, Phase2Handoff):
            raise TypeError("Se esperaba Phase2Handoff como frontera Φ₂→₃.")
        if handoff.next_entrypoint != _PHASE2_ENTRY:
            raise ValueError(f"Phase2Handoff inválido: se esperaba {_PHASE2_ENTRY!r}.")
        return handoff.state

    def phase3_characteristic_polynomial(self, state: QuaternionicState) -> Dict[str, Any]:
        """
        Fase 3.1 — χ_L(λ) = (λ² − 2q₀λ + ‖q‖²)² y verificación de Cayley–Hamilton
        χ_L(L) = 0 evaluada como (L² − 2q₀L + ‖q‖²I)².
        """
        q0, n2 = state.scalar_part, state.squared_norm
        theoretical = np.array([1.0, -4.0 * q0, 4.0 * q0 * q0 + 2.0 * n2,
                                -4.0 * q0 * n2, n2 * n2], dtype=np.float64)
        scale = max(1.0, float(np.max(np.abs(theoretical))))
        try:
            coeffs = np.asarray(np.poly(la.eigvals(state.left_mult_matrix)), dtype=np.complex128)
            numerical = np.real(coeffs)
            imag_leak = float(np.max(np.abs(np.imag(coeffs)))) / scale
            poly_res = float(math.hypot(*(numerical - theoretical))) / scale
        except la.LinAlgError as exc:
            logger.warning("Fallo en polinomio característico: %s", exc)
            numerical, imag_leak, poly_res = theoretical.copy(), math.nan, math.nan
        L = state.left_mult_matrix
        m_of_L = L @ L - 2.0 * q0 * L + n2 * _I4
        ch_res = _frobenius(m_of_L @ m_of_L) / max(1.0, n2 * n2)
        return {
            "theoretical_coefficients": theoretical.tolist(),
            "numerical_coefficients": np.asarray(numerical).tolist(),
            "characteristic_residual": poly_res,
            "characteristic_imaginary_leak": imag_leak,
            "cayley_hamilton_residual": float(ch_res),
        }

    def phase3_spectral_audit(self, state: QuaternionicState) -> Dict[str, Any]:
        """
        Fase 3.2 — σ(L) = {q₀ ± i‖v‖} (mult. 2), ρ(L) = ‖q‖, σ_i(L) = ‖q‖,
        det L = ‖q‖⁴, L normal (LLᵀ = LᵀL), κ₂(L) = 1.
        """
        M = np.array(state.left_mult_matrix, copy=True)
        n, n2 = state.norm, state.squared_norm
        try:
            eig = la.eigvals(M)
        except la.LinAlgError as exc:
            logger.warning("Fallo en autovalores: %s", exc)
            eig = np.empty(0, dtype=np.complex128)
        try:
            sv = la.svd(M, compute_uv=False)
        except la.LinAlgError as exc:
            logger.warning("Fallo en SVD: %s", exc)
            sv = np.empty(0, dtype=np.float64)
        try:
            det = float(np.real(la.det(M)))
        except la.LinAlgError as exc:
            logger.warning("Fallo en determinante: %s", exc)
            det = math.nan

        spectral_radius = float(np.max(np.abs(eig))) if eig.size else math.nan
        expected = np.array([complex(state.scalar_part, state.vector_norm),
                             complex(state.scalar_part, -state.vector_norm)] * 2)
        if eig.size == 4:
            model_res = float(math.fsum(float(np.min(np.abs(eig - e))) for e in expected) / 4.0) / max(1.0, n)
        else:
            model_res = math.nan
        if sv.size:
            singular_res = float(np.max(np.abs(sv - n))) / max(1.0, n)
            cond = float(np.max(sv) / np.min(sv)) if (n >= self._reg and np.min(sv) > 0.0) else math.inf
        else:
            singular_res, cond = math.nan, math.nan
        normal_res = _frobenius(M @ M.T - M.T @ M) / max(1.0, n2)

        return {
            "phase_token": state.phase_token,
            "eigenvalues": eig.tolist(),
            "expected_eigenvalues": expected.tolist(),
            "spectral_radius": spectral_radius,
            "spectral_residual": _relative_residual(spectral_radius, n),
            "singular_values": sv.tolist(),
            "singular_residual": singular_res,
            "determinant": det,
            "expected_determinant": n2 * n2,
            "determinant_residual": _relative_residual(det, n2 * n2),
            "eigenvalue_model_residual": model_res,
            "normality_residual": float(normal_res),
            "condition_number": cond,
            "state_norm": n,
            "state_vector_norm": state.vector_norm,
            **self.phase3_characteristic_polynomial(state),
        }

    def phase3_similarity_class(self, state: QuaternionicState) -> Dict[str, Any]:
        """
        Fase 3.3 — Clase de similitud [q] = {sqs⁻¹} (Def. 5).

        Si ‖v‖ > 0: 2-esfera de centro q₀ y radio ‖v‖; representante complejo
        canónico q₀ + i‖v‖; dirección û ∈ S² ⊂ Im(H). Se coteja con σ(Φ_C(q))
        y se verifica la invariancia numérica bajo un conjugador s genérico.
        """
        r, q0 = state.vector_norm, state.scalar_part
        out: Dict[str, Any] = {
            "center": q0,
            "radius": r,
            "canonical_complex_representative": complex(q0, r),
            "class_type": "real_point" if r <= self._tolerance_of(max(1.0, state.norm)) else "two_sphere",
            "unit_imaginary_direction": (state.vector_part / r).tolist() if r >= self._reg else None,
        }
        try:
            ev = np.sort_complex(la.eigvals(state.cayley_dickson_matrix))
            exp_ev = np.sort_complex(np.array([complex(q0, -r), complex(q0, r)]))
            out["cayley_dickson_spectrum_residual"] = float(np.max(np.abs(ev - exp_ev))) / max(1.0, state.norm)
        except la.LinAlgError:
            out["cayley_dickson_spectrum_residual"] = math.nan
        # Invariancia de (centro, radio) bajo conjugación por s genérico ∈ S³.
        s = np.array([0.5, 0.5, 0.5, 0.5])
        conj = _hamilton_vector(_hamilton_vector(s, state.vector_rep), _conjugate_vector(s))
        r_conj, _ = _stable_norm(conj[1:])
        out["conjugation_center_residual"] = abs(float(conj[0]) - q0) / max(1.0, state.norm)
        out["conjugation_radius_residual"] = abs(r_conj - r) / max(1.0, state.norm)
        return out

    def phase3_hopf_fibration(self, state: QuaternionicState) -> Dict[str, Any]:
        """
        Fase 3.4 — π : S³ → S², π(α, β) = (2αβ̄, |α|² − |β|²), generador de π₃(S²).

        Identidad Hopf ≡ Bloch (Def. 6); invariancia bajo la fibra U(1)
        (multiplicación izquierda por e^{iφ}, φ = ángulo áureo) auditada.
        """
        if state.norm < self._reg:
            return {"fiber": "null", "base_point": [0.0, 0.0, 0.0], "base_norm": 0.0,
                    "sphericity_residual": 0.0, "fiber_invariance_residual": 0.0,
                    "bloch_identity_residual": 0.0}

        def _base(vec4: np.ndarray) -> np.ndarray:
            z1 = complex(float(vec4[0]), float(vec4[1]))
            z2 = complex(float(vec4[2]), float(vec4[3]))
            w = 2.0 * z1 * z2.conjugate()
            return np.array([w.real, w.imag, abs(z1) ** 2 - abs(z2) ** 2], dtype=np.float64)

        u = state.vector_rep / state.norm
        base = _base(u)
        base_norm, _ = _stable_norm(base)
        phase = np.array([math.cos(_GOLDEN_ANGLE), math.sin(_GOLDEN_ANGLE), 0.0, 0.0])
        base_rot = _base(_hamilton_vector(phase, u))
        fiber_res = float(math.hypot(*(base - base_rot)))
        # Bloch: ρ = ψψ†, b_k = Tr(ρσ_k)
        psi = np.array([complex(u[0], u[1]), complex(u[2], u[3])])
        rho = np.outer(psi, psi.conj())
        bloch = np.array([float(np.real(np.trace(rho @ s))) for s in (_SIGMA1, _SIGMA2, _SIGMA3)])
        return {
            "fiber": "S1",
            "base_point": base.tolist(),
            "base_norm": float(base_norm),
            "sphericity_residual": abs(base_norm - 1.0),
            "fiber_invariance_residual": fiber_res,
            "bloch_vector": bloch.tolist(),
            "bloch_identity_residual": float(math.hypot(*(base - bloch))),
            "z1": complex(u[0], u[1]),
            "z2": complex(u[2], u[3]),
        }

    def phase3_riemann_sphere_chart(self, state: QuaternionicState) -> Dict[str, Any]:
        """
        Fase 3.5 — Atlas estereográfico de û = v/‖v‖ ∈ S² ⊂ Im(H).

        Norte: Z = (x+iy)/(1−z);  Sur: W = (x−iy)/(1+z);  transición Z·W = 1.
        Carta activa: norte si z ≤ 0, sur si z > 0 ⇒ |ζ| ≤ 1 (sin polos).
        Factor conforme λ = 2/(1+|ζ|²); coordenadas esféricas por atan2.
        """
        hopf = self.phase3_hopf_fibration(state)
        if state.vector_norm < self._reg:
            return {"chart": "origin", "coordinate": 0j, "coordinate_real": 0.0,
                    "coordinate_imag": 0.0, "north_coordinate": 0j, "south_coordinate": 0j,
                    "height": 0.0, "polar_angle": 0.0, "azimuth": 0.0, "metric_factor": 1.0,
                    "sphericity_residual": 0.0, "transition_residual": 0.0, "hopf": hopf}

        u = state.vector_part / state.vector_norm
        x, y, z = float(u[0]), float(u[1]), float(u[2])
        sph_res = abs(math.fsum((x * x, y * y, z * z)) - 1.0)
        north = complex(x, y) / (1.0 - z) if (1.0 - z) > self._reg else complex(math.nan, math.nan)
        south = complex(x, -y) / (1.0 + z) if (1.0 + z) > self._reg else complex(math.nan, math.nan)
        if z <= 0.0:
            chart, coord = "north", north
        else:
            chart, coord = "south", south
        if math.isfinite(north.real) and math.isfinite(south.real):
            transition = abs(north * south - 1.0)
        else:
            transition = math.nan
        a2 = abs(coord) ** 2
        return {
            "chart": chart,
            "coordinate": coord,
            "coordinate_real": float(coord.real),
            "coordinate_imag": float(coord.imag),
            "north_coordinate": north,
            "south_coordinate": south,
            "height": z,
            "polar_angle": float(math.atan2(math.hypot(x, y), z)),
            "azimuth": float(math.atan2(y, x)),
            "metric_factor": float(2.0 / (1.0 + a2)),
            "sphericity_residual": float(sph_res),
            "transition_residual": float(transition),
            "hopf": hopf,
        }

    def phase3_geodesic_distance(
        self, p: QuaternionicState, q: QuaternionicState, projective: bool = False,
    ) -> float:
        """
        Fase 3.6 — d_{S³}(û, v̂) = 2·atan2(‖û−v̂‖, ‖û+v̂‖);
        d_{RP³}(û, v̂) = 2·atan2(min, max) (identifica ±, SO(3)).  Ax. II.
        """
        if p.norm < self._reg or q.norm < self._reg:
            raise ZeroDivisionError("Distancia geodésica no definida para estados nulos.")
        pu, qu = p.vector_rep / p.norm, q.vector_rep / q.norm
        diff = float(math.hypot(*(pu - qu)))
        summ = float(math.hypot(*(pu + qu)))
        if projective:
            return float(2.0 * math.atan2(min(diff, summ), max(diff, summ)))
        return float(2.0 * math.atan2(diff, summ))

    def phase3_slerp(
        self, p: QuaternionicState, q: QuaternionicState, t: float, shortest_arc: bool = True,
    ) -> QuaternionicState:
        """
        Fase 3.7 — SLERP con pesos sinc (exactos en θ → 0, sin umbral):
            w_p = (1−t)·sinc((1−t)θ)/sinc(θ),  w_q = t·sinc(tθ)/sinc(θ).
        `shortest_arc` identifica q ~ −q (interpolación en RP³ ≅ SO(3)).
        """
        if not math.isfinite(t):
            raise ValueError("El parámetro t debe ser finito.")
        if p.norm < self._reg or q.norm < self._reg:
            raise ZeroDivisionError("SLERP no definido para estados nulos.")
        pu, qu = p.vector_rep / p.norm, q.vector_rep / q.norm
        if shortest_arc and _real_inner(pu, qu) < 0.0:
            qu = -qu
        theta = _angle_between_unit(pu, qu)
        s_theta = _sinc(theta)
        if s_theta < self._reg:
            raise ValueError("SLERP indefinido: estados antipodales (θ = π).")
        w_p = (1.0 - t) * _sinc((1.0 - t) * theta) / s_theta
        w_q = t * _sinc(t * theta) / s_theta
        result = self._build_state(w_p * pu + w_q * qu, "PHASE3/SLERP")
        if self._auto_renormalize and not result.is_unitary:
            result = self.phase2_normalize(result)
        return result

    def phase3_adjoint_rotation(self, rotor: QuaternionicState) -> Dict[str, Any]:
        """
        Fase 3.8 — Recubrimiento Spin(3) → SO(3) (Def. 7).

        Ad(r) = L(r)R(r*) ∈ M₄(R) debe ser diag(1, Rᵣ) con Rᵣ = I + 2r₀[v]ₓ + 2[v]ₓ²
        (Rodrigues), Rᵣ ∈ SO(3): RᵀR = I, det R = 1, ángulo 2·atan2(‖v‖, r₀).
        Sin gimbal lock: el mapa es una submersión sin puntos críticos.
        """
        if abs(rotor.norm - 1.0) > self._tolerance_of(1.0):
            raise ValueError("phase3_adjoint_rotation requiere rotor unitario.")
        r = rotor.vector_rep
        ad = rotor.left_mult_matrix @ _right_matrix(_conjugate_vector(r))
        R = ad[1:, 1:]
        K = _skew3(rotor.vector_part)
        rodrigues = _I3 + 2.0 * rotor.scalar_part * K + 2.0 * (K @ K)
        try:
            det_R = float(la.det(R))
        except la.LinAlgError:
            det_R = math.nan
        angle = 2.0 * math.atan2(rotor.vector_norm, rotor.scalar_part)
        axis = (rotor.vector_part / rotor.vector_norm).tolist() if rotor.vector_norm >= self._reg else None
        return {
            "adjoint_matrix": ad.tolist(),
            "rotation_matrix": R.tolist(),
            "block_scalar_residual": abs(float(ad[0, 0]) - 1.0),
            "block_offdiagonal_residual": float(math.hypot(*ad[0, 1:], *ad[1:, 0])),
            "so3_orthogonality_residual": _frobenius(R.T @ R - _I3),
            "so3_det_residual": abs(det_R - 1.0),
            "rodrigues_residual": _frobenius(R - rodrigues),
            "rotation_angle": float(angle),
            "rotation_axis": axis,
            "trace_angle_residual": abs(float(np.trace(R)) - (1.0 + 2.0 * math.cos(angle))),
        }

    def phase3_parallel_transport(
        self, state: QuaternionicState, rotor: Optional[Any] = None, mode: str = "sandwich",
    ) -> Tuple[QuaternionicState, Dict[str, Any]]:
        """
        Fase 3.9 — Transporte por rotor unitario r.

          left:     q' = r q         (L(r))
          right:    q' = q r         (R(r))
          sandwich: q' = r q r*      (Ad(r); preserva q₀ y ‖Im q‖)

        Una sola aplicación matricial (sin renormalización intermedia), cotejo
        con el producto fsum y auditoría de invariantes.
        """
        if rotor is None:
            rotor_state = self._build_state(np.array([1.0, 0.0, 0.0, 0.0]), "PHASE3/IDENTITY_ROTOR")
        elif isinstance(rotor, QuaternionicState):
            rotor_state = rotor
        else:
            rotor_state = self._build_state(np.asarray(rotor, dtype=np.float64), "PHASE3/EXTERNAL_ROTOR")
        if rotor_state.norm < self._reg:
            raise ZeroDivisionError("Rotor de norma nula o sub-regular.")
        rotor_unit_res = abs(rotor_state.norm - 1.0)
        if rotor_unit_res > self._tolerance_of(1.0):
            if not self._auto_renormalize:
                raise ValueError("El rotor no es unitario dentro de la tolerancia activa.")
            rotor_state = self.phase2_normalize(rotor_state)
            rotor_unit_res = abs(rotor_state.norm - 1.0)

        r, rc, x = rotor_state.vector_rep, _conjugate_vector(rotor_state.vector_rep), state.vector_rep
        adjoint: Dict[str, Any] = {}
        if mode == "left":
            vec = np.asarray(rotor_state.left_mult_matrix @ x, dtype=np.float64)
            check = _hamilton_vector(r, x)
        elif mode == "right":
            vec = np.asarray(rotor_state.right_mult_matrix @ x, dtype=np.float64)
            check = _hamilton_vector(x, r)
        elif mode == "sandwich":
            adjoint = self.phase3_adjoint_rotation(rotor_state)
            vec = np.asarray(np.array(adjoint["adjoint_matrix"]) @ x, dtype=np.float64)
            check = _hamilton_vector(_hamilton_vector(r, x), rc)
        else:
            raise ValueError("Modo inválido. Use 'left', 'right' o 'sandwich'.")

        cross_res = float(math.hypot(*(vec - check))) / max(1.0, state.norm)
        transported = self._build_state(vec, f"PHASE3/TRANSPORT/{mode.upper()}")
        metrics: Dict[str, Any] = {
            "mode": mode,
            "norm_drift": _relative_residual(transported.norm, state.norm),
            "matrix_vs_fsum_residual": cross_res,
            "rotor_norm": float(rotor_state.norm),
            "rotor_unitarity_residual": float(rotor_unit_res),
            "transport_is_unitary": 1.0 if transported.is_unitary else 0.0,
            "scalar_invariance_residual": (abs(transported.scalar_part - state.scalar_part)
                                           if mode == "sandwich" else math.nan),
            "imaginary_norm_invariance_residual": (abs(transported.vector_norm - state.vector_norm)
                                                   if mode == "sandwich" else math.nan),
            "adjoint": adjoint,
        }
        return transported, metrics

    def phase3_liouville_de_rham_audit(self, state: QuaternionicState) -> Dict[str, Any]:
        """
        Fase 3.10 — det L = ‖q‖⁴ (volumen); L = q₀I + K, K ∈ so(4), tr K = 0
        (divergencia nula del generador); si ‖q‖ = 1, L ∈ SO(4) preserva Haar(S³).
        """
        M = np.array(state.left_mult_matrix, copy=True)
        try:
            det = float(np.real(la.det(M)))
        except la.LinAlgError as exc:
            logger.warning("Fallo en determinante de Liouville: %s", exc)
            det = math.nan
        expected = state.squared_norm ** 2
        K = M - state.scalar_part * _I4
        return {
            "phase_token": state.phase_token,
            "determinant": det,
            "expected_determinant": expected,
            "liouville_residual": _relative_residual(det, expected) if state.norm >= self._reg else math.nan,
            "volume_ratio": (det / expected if (math.isfinite(det) and expected > 0.0) else math.nan),
            "generator_skew_residual": _frobenius(K + K.T) / max(1.0, state.norm),
            "generator_trace": float(np.trace(K)),
            "generator_frobenius_residual": _relative_residual(_frobenius(K), 2.0 * state.vector_norm),
            "trace": float(np.trace(M)),
            "infinitesimal_volume_preserving": bool(abs(state.scalar_part) <= self._tolerance_of(1.0)),
            "so4_membership_residual": (_frobenius(M.T @ M - _I4) if state.is_unitary else math.nan),
        }

    def phase3_governance_seal(
        self,
        state: QuaternionicState,
        spectral: Mapping[str, Any],
        transport_metrics: Mapping[str, Any],
        riemann: Mapping[str, Any],
        phase_links: Sequence[Tuple[str, str]] = (),
    ) -> Dict[str, Any]:
        """
        Fase 3.11 — Sello SHA-256 sobre carga binaria canónica + cadena de Merkle
        de las fronteras (token_k, hash_k) de Φ₃∘Φ₂∘Φ₁ (Ax. IV).
        """
        merkle_root = _merkle_chain(list(phase_links))
        h = hashlib.sha256()
        h.update(b"QSS/GOVERNANCE/v4")
        h.update(bytes.fromhex(state.sha256_hash))
        h.update(bytes.fromhex(merkle_root))
        tok = state.phase_token.encode("utf-8")
        h.update(len(tok).to_bytes(8, "little"))
        h.update(tok)
        payload = (
            spectral.get("spectral_radius"), spectral.get("spectral_residual"),
            spectral.get("determinant"), spectral.get("determinant_residual"),
            spectral.get("eigenvalue_model_residual"), spectral.get("cayley_hamilton_residual"),
            transport_metrics.get("norm_drift"), transport_metrics.get("scalar_invariance_residual"),
            transport_metrics.get("rotor_unitarity_residual"),
            riemann.get("height"), riemann.get("coordinate_real"), riemann.get("coordinate_imag"),
            riemann.get("metric_factor"), riemann.get("transition_residual"),
        )
        for value in payload:
            try:
                h.update(_pack_f64(float(value)))
            except (TypeError, ValueError):
                h.update(_pack_f64(math.nan))
        chart = str(riemann.get("chart", ""))
        h.update(len(chart).to_bytes(8, "little"))
        h.update(chart.encode("utf-8"))
        return {
            "seal": h.hexdigest(),
            "merkle_root": merkle_root,
            "hash_algorithm": "sha256",
            "encoding": "canonical-binary-v4",
            "state_hash": state.sha256_hash,
            "phase_links": [list(link) for link in phase_links],
        }

    def phase3_close_loop(
        self, S: np.ndarray, rotor: Optional[Any] = None, transport_mode: str = "sandwich",
    ) -> Phase3Report:
        """
        Fase 3.12 — Orquestación del funtor compuesto Φ₃ ∘ Φ₂ ∘ Φ₁.

        Cada frontera se atraviesa por su método de entrada formal; la cadena
        de tokens/hashes se sella con Merkle.
        """
        handoff_1 = self.phase1_close_and_open_phase2(S)
        state_2 = self.phase2_from_phase1(handoff_1)
        handoff_2 = self.phase2_close_and_open_phase3(state_2)
        state_3 = self.phase3_from_phase2(handoff_2)

        spectral = self.phase3_spectral_audit(state_3)
        similarity = self.phase3_similarity_class(state_3)
        riemann = self.phase3_riemann_sphere_chart(state_3)
        transported, transport = self.phase3_parallel_transport(state_3, rotor=rotor, mode=transport_mode)
        liouville = self.phase3_liouville_de_rham_audit(state_3)

        links = (
            (handoff_1.state.phase_token, handoff_1.state.sha256_hash),
            (handoff_2.state.phase_token, handoff_2.state.sha256_hash),
            (state_3.phase_token, state_3.sha256_hash),
            (transported.phase_token, transported.sha256_hash),
        )
        governance = self.phase3_governance_seal(state_3, spectral, transport, riemann, links)

        return Phase3Report(
            state=transported,
            source_state=state_3,
            spectral_audit=spectral,
            similarity_class=similarity,
            riemann_chart=riemann,
            transport_metrics=transport,
            liouville_audit=liouville,
            governance_seal=governance,
            phase_chain=tuple(tok for tok, _ in links),
        )

    # ═════════════════════════════════════════════════════════════════════════
    # COMPATIBILIDAD CON API LEGADA
    # ═════════════════════════════════════════════════════════════════════════

    def build_state(self, S: np.ndarray) -> QuaternionicState:
        """API legada — Fase 1.4."""
        return self.phase1_construct_canonical_state(S)

    def quaternionic_multiply(self, p: QuaternionicState, q: QuaternionicState) -> QuaternionicState:
        """API legada — Fase 2.3."""
        return self.phase2_hamilton_product(p, q, verify=True)

    def quaternionic_inverse(self, state: QuaternionicState) -> QuaternionicState:
        """API legada — Fase 2.11."""
        return self.phase2_inverse(state)

    def project_to_riemann_sphere(self, state: QuaternionicState) -> Tuple[complex, float]:
        """API legada — Fase 3.5: (coordenada de la carta activa, altura z)."""
        chart = self.phase3_riemann_sphere_chart(state)
        return complex(chart["coordinate"]), float(chart["height"])
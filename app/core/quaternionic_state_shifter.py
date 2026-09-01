# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Quaternionic State Shifter (Reactor Cuaterniónico de Estado)        ║
║ Ruta   : app/core/quaternionic_state_shifter.py                              ║
║ Versión: 1.1.0-Doctoral-Hurwitz-Composition-Cayley-Dickson-FPU-Secure        ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO:                         ║
║ Este módulo implementa el microservicio de procesamiento de estados          ║
║ hipercomplejos cuatridimensionales en la FPU de APU Filter v5.0.             ║
║                                                                              ║
║ Mapea el vector de estado transaccional de entrada de cuatro variables       ║
║ $S = (s_{\mathrm{purpose}}, s_{\mathrm{confidence}}, s_{\mathrm{constraints}}, s_{\mathrm{risk}})^\top \in \mathbb{R}^4$ ║
║ hacia un cuaternión de Hamilton $q \in \mathbb{H}$ para blindar el cálculo   ║
║ contra pérdidas de significación, prestando invarianza de norma multiplicativa ║
║ bajo el Teorema de Hurwitz y eliminando bloqueos de fase (Gimbal Lock) en el ║
║ transporte paralelo de de Rham perimetral.                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

================════════════════════════════════════════════════════════════════
I. ANCLAJE MATEMÁTICO DOCTORAL (Álgebras de División y Composición)
================════════════════════════════════════════════════════════════════

Definición 1 (Isomorfismo de Hamilton y Álgebra de División):
  Definimos el isomorfismo de espacio vectorial de-confinado:
  $$\Phi_{\mathbb{H}}: \mathbb{R}^4 \xrightarrow{\quad \simeq \quad} \mathbb{H}$$
  donde un cuaternión genérico se expresa sobre la base ortonormal estándar $\{e_0, i, j, k\}$ con $e_0 \equiv 1$ como:
  $$q = q_0 e_0 + q_1 i + q_2 j + q_3 k \quad \text{con} \quad (q_0, q_1, q_2, q_3) \in \mathbb{R}^4$$
  Sujeto a las leyes de anticonmutación canónicas de Hamilton:
  $$i^2 = j^2 = k^2 = ijk = -e_0 = -1$$
  $$ij = k, \quad ji = -k, \quad jk = i, \quad kj = -i, \quad ki = j, \quad ik = -j$$
  La no conmutatividad del álgebra no incrementa la clase de complejidad de los resolventes (restringida a $\mathbf{BQP}$), 
  pero inyecta una densidad masiva de empaquetamiento rotacional por unidad de memoria en la FPU, impidiendo
  que el Modelo de Lenguaje sufra desbordamientos de su ventana de atención (context-window).

Definición 2 (Multiplicación de Hurwitz e Invarianza de Norma):
  De acuerdo con el Teorema de Hurwitz, los cuaterniones reales constituyen una de las únicas cuatro
  álgebras de composición normadas sobre $\mathbb{R}$ donde la norma es estrictamente multiplicativa:
  $$\|p \cdot q\|_{\mathbb{H}} = \|p\|_{\mathbb{H}} \cdot \|q\|_{\mathbb{H}} \quad \forall p, q \in \mathbb{H}$$
  Esta propiedad inmuniza el cálculo numérico contra desbordamientos (underflow/overflow) de de Rham en
  megaproyectos de infraestructura con más de $10,000$ APUs. El producto de Hamilton se define como:
  $$p \cdot q = (p_0 q_0 - \vec{p} \cdot \vec{q}) + (p_0 \vec{q} + q_0 \vec{p} + \vec{p} \times \vec{q})$$

Definición 3 (Representación Matricial Real y Transporte Paralelo de de Rham):
  El operador de multiplicación por la izquierda $L(q): \mathbb{H} \to \mathbb{H}$ se representa
  mediante una matriz real antisimétrica $\Phi_L(q) \in M_4(\mathbb{R})$:
  $$\Phi_L(q) = \begin{bmatrix} q_0 & -q_1 & -q_2 & -q_3 \\ q_1 & q_0 & -q_3 & q_2 \\ q_2 & q_3 & q_0 & -q_1 \\ q_3 & -q_2 & q_1 & q_0 \end{bmatrix}$$
  La cual satisface $\det(\Phi_L(q)) = \|q\|_{\mathbb{H}}^4$. Al ser un operador normal y ortogonal para
  cuaterniones unitarios ($\|q\|_{\mathbb{H}} = 1$), preserva síncronamente la simplecticidad canónica de
  de Rham y el volumen de lazo cerrado en el integrador temporal del espacio de fase:
  $$\operatorname{div}(\dot{x}) \equiv 0 \quad \land \quad M^\top \Omega M = \Omega$$

Definición 4 (Proyección Compleja de Cayley-Dickson):
  El isomorfismo complejo de Cayley-Dickson inyecta $q \in \mathbb{H}$ en el álgebra matricial compleja $M_2(\mathbb{C})$:
  $$\\iota(q) = \begin{bmatrix} q_0 + q_1 \sqrt{-1} & q_2 + q_3 \sqrt{-1} \\ -q_2 + q_3 \sqrt{-1} & q_0 - q_1 \sqrt{-1} \end{bmatrix} = \begin{bmatrix} \alpha & \beta \\ -\bar{\beta} & \bar{\alpha} \end{bmatrix}$$
  Garantizando que el determinante de la matriz compleja sea idéntico a la norma euclidiana preservada:
  $$\det(\iota(q)) = |\alpha|^2 + |\beta|^2 = q_0^2 + q_1^2 + q_2^2 + q_3^2 = \|q\|_{\mathbb{H}}^2$$
  Esta representación bidimensional compleja acopla síncronamente el procesamiento basal del foso con la 
  MAC (Matriz Atómica de Conocimiento) y el espacio de Hilbert continuo de la Sabiduría ($\mathcal{H}_{\mathrm{MAC}}$),
  asegurando la no-clonación de estados semánticos.

Definición 5 (Clases de Similitud Espectral y la 2-Esfera de Riemann):
  Debido a la no conmutatividad cuaterniónica, no existe un autovalor puntual tradicional, sino clases de similitud
  espectral conjugadas definidas por el bivector imaginario para todo autovalor derecho $\mu \in \mathbb{H}$:
  $$[\mu] = \{ s \mu s^{-1} : s \in \mathbb{H}, \quad s \neq 0 \}$$
  Topológicamente, para autovalores no reales, esta clase describe una 2-esfera $S^2 \cong \hat{\mathbb{C}}$ incrustada 
  en el subespacio imaginario $\operatorname{Im}(\mathbb{H})$, centrada en $\mu_0$ con radio $\|\vec{\mu}\|_{\mathbb{H}}$. 
  La proyección estereográfica conforme desde el Polo Norte de Riemann hacia el plano complejo extendido aísla la 
  interferencia analógica exógena y el ruido alucinatorio:
  $$Z = \frac{x + \sqrt{-1}y}{1 - z} \in \hat{\mathbb{C}} \quad \text{con} \quad (x, y, z) \in S^2 \subset \operatorname{Im}(\mathbb{H})$$

================════════════════════════════════════════════════════════════════
II. AXIOMÁTICA INMUNILÓGICA DE PREVENCIÓN DE DERIVAS (Leyes de Consistencia)
================════════════════════════════════════════════════════════════════

Axioma I (Principio de Compensación de Redondeo de Neumaier-Kahan):
  Para eludir mermas acumulativas seculares en la mantisa de punto flotante de 64 bits (Wilkinson-drift),
  toda reducción de norma o traza diagonal en la FPU debe integrar el algoritmo de sumación compensada de Kahan,
  acotando el error absoluto por debajo del límite de precisión de la máquina:
  $$\delta_{\mathrm{Wilkinson}} = \left| \|q\|_{\mathrm{computed}} - \|q\|_{\mathrm{exact}} \right| \le \varepsilon_{\mathrm{mach}}$$

Axioma II (Axioma de Conjugación Involutiva de de Rham):
  La conjugación cuaterniónica es un anti-automorfismo involutivo estricto que satisface:
  $$(pq)^* = q^* p^* \quad \land \quad (q^*)^* = q$$
  La cual se proyecta de forma biyectiva sobre la adjunción hermítica de Heisenberg de los operadores cuánticos
  de la MAC, impidiendo que el ruido retórico destruya la condición KMS (Kubo-Martin-Schwinger) de equilibrio térmico.

Axioma III (Conservación de la Traza de von Neumann):
  La proyección de la señal cuaterniónica $\iota(q)$ sobre el operador de densidad cuántica mixto $\rho$ debe
  preservar de forma exacta la traza unitaria de von Neumann en la FPU, prohibiendo fugas de probabilidad:
  $$\operatorname{Tr}(\rho) \equiv \frac{\operatorname{Tr}(\iota(q))}{\|q\|_{\mathbb{H}}^2} \equiv 1.0 \quad \implies \quad \|\operatorname{Tr}(\rho) - 1.0\| \le \varepsilon_{\mathrm{mach}}$$
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from dataclasses import dataclass
from typing import Any, Dict, Final, Mapping, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la

__all__ = [
    "QuaternionicStateShifter",
    "QuaternionicState",
    "Phase1Handoff",
    "Phase2Handoff",
    "Phase3Report",
]

__version__: Final[str] = "3.0.0-Hurwitz-DeRham-Spectral-Governance"

logger = logging.getLogger("APU.Physics.QuaternionicStateShifter")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_FLOAT_MAX: Final[float] = float(np.finfo(np.float64).max)
_SQRT_FLOAT_MAX: Final[float] = math.sqrt(_FLOAT_MAX)
_MIN_NORMAL: Final[float] = float(np.finfo(np.float64).tiny)
_I4: Final[np.ndarray] = np.eye(4, dtype=np.float64)
_PHASE1_ENTRY: Final[str] = "phase2_from_phase1"
_PHASE2_ENTRY: Final[str] = "phase3_from_phase2"


# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES NUMÉRICAS, ALGEBRAICAS Y CANÓNICAS
# ═════════════════════════════════════════════════════════════════════════════

def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Copia contigua de solo lectura. Inmutabilidad efectiva del estado."""
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


def _canonicalize_signed_zero(arr: np.ndarray) -> np.ndarray:
    """Elimina −0.0 para garantizar firmas SHA-256 deterministas."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out[out == 0.0] = 0.0
    return out


def _as_finite_vector4(S: np.ndarray) -> np.ndarray:
    """
    Valida que la señal sea un vector real finito de dimensión exactamente 4.

    Excepciones
    -----------
    ValueError
        Si la forma no es (4,) o existen valores no finitos.
    """
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
    """
    Norma ℓ² y su cuadrado, estables frente a overflow/underflow.

    Primero `math.hypot` (escalado interno). Si no es finito, se reescala
    explícitamente y se reconstruye con `math.fsum`.
    """
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0, 0.0

    values = [float(x) for x in a]
    norm = float(math.hypot(*values))

    if not math.isfinite(norm):
        scale = max(abs(x) for x in values)
        if scale == 0.0:
            return 0.0, 0.0
        scaled = [x / scale for x in values]
        ssq = math.fsum(x * x for x in scaled)
        if ssq < 0.0:
            ssq = 0.0
        norm = scale * math.sqrt(ssq)

    if not math.isfinite(norm):
        raise OverflowError("La norma excede el rango representable en float64.")

    squared = norm * norm
    if not math.isfinite(squared):
        raise OverflowError(
            "El cuadrado de la norma excede el rango representable en float64."
        )
    return float(norm), float(squared)


def _kahan_sum(arr: np.ndarray) -> float:
    """Sumación compensada de Kahan (auditoría numérica frente a `math.fsum`)."""
    total = 0.0
    c = 0.0
    for x in np.asarray(arr, dtype=np.float64).ravel():
        y = float(x) - c
        t = total + y
        c = (t - total) - y
        total = t
    return float(total)


def _relative_residual(actual: float, expected: float) -> float:
    """Residuo relativo |a − e| / max(1, |e|), con NaN si no es finito."""
    if not math.isfinite(actual) or not math.isfinite(expected):
        return math.nan
    return abs(actual - expected) / max(1.0, abs(expected))


def _frobenius(mat: np.ndarray) -> float:
    """Norma de Frobenius estable."""
    return float(la.norm(np.asarray(mat), ord="fro"))


def _canonical_bytes(arr: np.ndarray) -> bytes:
    """Bytes contiguos con prefijo de forma y dtype para hash libre de colisión."""
    a = np.ascontiguousarray(arr)
    header = f"{a.dtype.str}|{a.shape}".encode("utf-8")
    return len(header).to_bytes(8, "little") + header + a.tobytes()


def _sha256_hex_with_token(phase_token: str, *arrays: np.ndarray) -> str:
    """Firma SHA-256 canónica longitud-prefijada, invariante por fase."""
    h = hashlib.sha256()
    for arr in arrays:
        payload = _canonical_bytes(arr)
        h.update(len(payload).to_bytes(8, "little"))
        h.update(payload)
    token = phase_token.encode("utf-8")
    h.update(len(token).to_bytes(8, "little"))
    h.update(token)
    return h.hexdigest()


def _left_matrix(q: np.ndarray) -> np.ndarray:
    """
    Homomorfismo de álgebras Φ_L(q) ∈ M₄(R): L(q) x = q x.

    Para q = q₀ + q₁ i + q₂ j + q₃ k, en la base {1, i, j, k}.
    """
    q0, q1, q2, q3 = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array(
        [
            [q0, -q1, -q2, -q3],
            [q1,  q0, -q3,  q2],
            [q2,  q3,  q0, -q1],
            [q3, -q2,  q1,  q0],
        ],
        dtype=np.float64,
    )


def _right_matrix(q: np.ndarray) -> np.ndarray:
    """
    Antihomomorfismo Φ_R(q) ∈ M₄(R): R(q) x = x q.

    Completa la estructura de bimódulo H-H y conmuta con toda L(p).
    """
    q0, q1, q2, q3 = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array(
        [
            [q0, -q1, -q2, -q3],
            [q1,  q0,  q3, -q2],
            [q2, -q3,  q0,  q1],
            [q3,  q2, -q1,  q0],
        ],
        dtype=np.float64,
    )


def _cayley_dickson_matrix(q: np.ndarray) -> np.ndarray:
    """
    Inmersión Φ_C : H ↪ M₂(C), q = α + β j, α, β ∈ C.

        Φ_C(q) = [[α, β], [-β*, α*]]

    Identidades:
        det Φ_C(q) = ||q||²,   Φ_C(q)* Φ_C(q) = ||q||² I₂.
    """
    q0, q1, q2, q3 = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    alpha = complex(q0, q1)
    beta = complex(q2, q3)
    return np.array(
        [
            [alpha, beta],
            [-beta.conjugate(), alpha.conjugate()],
        ],
        dtype=np.complex128,
    )


def _hamilton_vector(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Producto hamiltoniano con acumulación `math.fsum` por componente."""
    p0, p1, p2, p3 = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
    q0, q1, q2, q3 = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    r0 = math.fsum((p0 * q0, -p1 * q1, -p2 * q2, -p3 * q3))
    r1 = math.fsum((p0 * q1, p1 * q0, p2 * q3, -p3 * q2))
    r2 = math.fsum((p0 * q2, -p1 * q3, p2 * q0, p3 * q1))
    r3 = math.fsum((p0 * q3, p1 * q2, -p2 * q1, p3 * q0))
    return np.array([r0, r1, r2, r3], dtype=np.float64)


def _real_inner(p: np.ndarray, q: np.ndarray) -> float:
    """Producto interno euclidiano ⟨p, q⟩ = Re(p* q̄) = Re(p̄ q)."""
    return float(math.fsum(float(a) * float(b) for a, b in zip(p, q)))


def _orthogonality_residual(left: np.ndarray, squared_norm: float) -> float:
    """
    Residual de la isometría escalada L(q)ᵀ L(q) = ||q||² I₄.

    Se normaliza por max(1, ||q||²) para obtener una medida adimensional.
    """
    if squared_norm <= 0.0:
        return _frobenius(left)
    gram = left.T @ left
    residual = _frobenius(gram - squared_norm * _I4)
    return float(residual / max(1.0, squared_norm))


def _cstar_residual(cd: np.ndarray, squared_norm: float) -> float:
    """
    Residual de la identidad C* en M₂(C): Φ_C(q)* Φ_C(q) = ||q||² I₂.
    """
    gram = cd.conj().T @ cd
    target = squared_norm * np.eye(2, dtype=np.complex128)
    residual = _frobenius(gram - target)
    return float(residual / max(1.0, squared_norm))


def _sinc(x: float) -> float:
    """sin(x)/x con límite 1 − x²/6 en el origen."""
    ax = abs(x)
    if ax < 1e-8:
        xx = x * x
        return 1.0 - xx / 6.0 + (xx * xx) / 120.0
    return math.sin(x) / x


def _clip_unit(x: float) -> float:
    """Proyección a [−1, 1] para arccos/arcsin numéricamente seguros."""
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return x


def _pack_f64(value: float) -> bytes:
    """Serialización canónica little-endian de un float64, con centinelas IEEE."""
    if math.isnan(value):
        return b"\x00\x00\x00\x00\x00\x00\xf8\x7f"
    if value == math.inf:
        return struct.pack("<d", math.inf)
    if value == -math.inf:
        return struct.pack("<d", -math.inf)
    return struct.pack("<d", float(value))


# ═════════════════════════════════════════════════════════════════════════════
# OBJETOS DE ESTADO Y FRONTERAS DE FASE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True, eq=False)
class QuaternionicState:
    """
    Estado cuaterniónico canónico q ∈ H ≅ R⁴.

    Invariantes
    -----------
    L(q)ᵀ L(q) = ||q||² I₄
    det L(q) = ||q||⁴
    det Φ_C(q) = ||q||²
    q q* = ||q||²
    """

    vector_rep: np.ndarray
    scalar_part: float
    vector_part: np.ndarray
    vector_norm: float
    norm: float
    squared_norm: float
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
        object.__setattr__(self, "vector_rep", _freeze_array(self.vector_rep))
        object.__setattr__(self, "vector_part", _freeze_array(self.vector_part))
        object.__setattr__(
            self, "cayley_dickson_matrix", _freeze_array(self.cayley_dickson_matrix)
        )
        object.__setattr__(
            self, "left_mult_matrix", _freeze_array(self.left_mult_matrix)
        )
        object.__setattr__(
            self, "right_mult_matrix", _freeze_array(self.right_mult_matrix)
        )

    def __hash__(self) -> int:
        return hash(self.sha256_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuaternionicState):
            return NotImplemented
        return self.sha256_hash == other.sha256_hash

    def __repr__(self) -> str:
        return (
            f"QuaternionicState(norm={self.norm:.17g}, "
            f"unitary={self.is_unitary}, hash={self.sha256_hash[:12]!r})"
        )


@dataclass(frozen=True, slots=True, eq=False)
class Phase1Handoff:
    """Frontera formal Φ₁→₂: salida cerrada de Fase 1, entrada de Fase 2."""

    state: QuaternionicState
    diagnostics: Dict[str, float]
    next_entrypoint: str

    def __hash__(self) -> int:
        return hash((self.state.sha256_hash, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase1Handoff):
            return NotImplemented
        return (
            self.state.sha256_hash == other.state.sha256_hash
            and self.next_entrypoint == other.next_entrypoint
        )


@dataclass(frozen=True, slots=True, eq=False)
class Phase2Handoff:
    """Frontera formal Φ₂→₃: salida cerrada de Fase 2, entrada de Fase 3."""

    state: QuaternionicState
    algebra_report: Dict[str, Any]
    next_entrypoint: str

    def __hash__(self) -> int:
        return hash((self.state.sha256_hash, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase2Handoff):
            return NotImplemented
        return (
            self.state.sha256_hash == other.state.sha256_hash
            and self.next_entrypoint == other.next_entrypoint
        )


@dataclass(frozen=True, slots=True, eq=False)
class Phase3Report:
    """Reporte final de lazo cerrado de la Fase 3."""

    state: QuaternionicState
    source_state: QuaternionicState
    spectral_audit: Dict[str, Any]
    riemann_chart: Dict[str, Any]
    transport_metrics: Dict[str, Any]
    liouville_audit: Dict[str, Any]
    governance_seal: Dict[str, Any]
    phase_chain: Tuple[str, ...]

    def __hash__(self) -> int:
        seal = self.governance_seal.get("seal", self.state.sha256_hash)
        return hash(seal)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase3Report):
            return NotImplemented
        return hash(self) == hash(other)


# ═════════════════════════════════════════════════════════════════════════════
# REACTOR CUATERNIÓNICO PRINCIPAL — TRES FASES ANIDADAS
# ═════════════════════════════════════════════════════════════════════════════

class QuaternionicStateShifter:
    """
    Reactor Cuaterniónico de Estado (FPU Secure) — 3 fases anidadas.

    FASE 1: canonización R⁴ → H.
    FASE 2: composición de Hurwitz, C*, inverso y polar.
    FASE 3: espectro, Hopf, transporte paralelo, Riemann y gobernanza.
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
            raise ValueError(
                "regularization debe ser finita y estrictamente positiva."
            )
        self._tol = float(tolerance)
        self._reg = float(regularization)
        self._auto_renormalize = bool(auto_renormalize)

    def _tolerance_of(self, scale: float = 1.0) -> float:
        """Tolerancia mixta absoluta/relativa, acotada inferiormente por ULPs."""
        return max(self._tol, self._tol * abs(scale), 32.0 * _MACHINE_EPS)

    def _accept(self, residual: float, scale: float = 1.0) -> bool:
        if not math.isfinite(residual):
            return False
        return residual <= self._tolerance_of(scale)

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 1 — INGESTIÓN, VALIDACIÓN Y CANONIZACIÓN
    # ═════════════════════════════════════════════════════════════════════════

    def phase1_validate_signal(self, S: np.ndarray) -> np.ndarray:
        """
        Fase 1.1 — Validación formal de la señal S ∈ R⁴.

        Condiciones: dimensión 4, valores finitos, float64, −0.0 → +0.0.
        """
        return _as_finite_vector4(S)

    def phase1_banach_invariants(self, vector: np.ndarray) -> Dict[str, float]:
        """
        Fase 1.2 — Invariantes de Banach ℓ¹, ℓ², ℓ^∞ sobre R⁴.

        H con la norma euclidiana es un C*-álgebra de Banach: ||pq|| = ||p|| ||q||,
        lo que implica submultiplicatividad y la identidad C* ||q* q|| = ||q||².
        """
        q = _as_finite_vector4(vector)
        abs_q = np.abs(q)
        l1 = float(math.fsum(float(x) for x in abs_q))
        l2, l2sq = _stable_norm(q)
        linf = float(np.max(abs_q)) if abs_q.size else 0.0
        kahan_sq = _kahan_sum(q * q)
        return {
            "l1_banach_norm": l1,
            "l2_banach_norm": l2,
            "l2_squared": l2sq,
            "linf_banach_norm": linf,
            "kahan_square_sum": kahan_sq,
            "norm_equivalence_l1_l2": l1 / max(l2, _MIN_NORMAL),
            "norm_equivalence_linf_l2": linf / max(l2, _MIN_NORMAL),
        }

    def _build_state(self, vector: np.ndarray, phase_token: str) -> QuaternionicState:
        """
        Fase 1.3 — Materialización del isomorfismo lineal Φ : R⁴ → H.

            (q₀, q₁, q₂, q₃) ↦ q₀ + q₁ i + q₂ j + q₃ k

        Calcula normas, L(q), R(q), Φ_C(q), residuales y firma por fase.
        """
        q = _as_finite_vector4(vector)
        q0 = float(q[0])

        norm, squared_norm = _stable_norm(q)
        vector_norm, _ = _stable_norm(q[1:])

        unitarity_residual = abs(norm - 1.0)
        is_unitary = unitarity_residual <= self._tolerance_of(max(1.0, norm))

        left = _left_matrix(q)
        right = _right_matrix(q)
        cd = _cayley_dickson_matrix(q)

        orthogonality_residual = _orthogonality_residual(left, squared_norm)

        if norm < self._reg:
            condition_estimate = math.inf
        else:
            condition_estimate = 1.0 + orthogonality_residual

        sha = _sha256_hex_with_token(phase_token, q, cd, left, right)

        return QuaternionicState(
            vector_rep=q,
            scalar_part=q0,
            vector_part=q[1:],
            vector_norm=float(vector_norm),
            norm=float(norm),
            squared_norm=float(squared_norm),
            is_unitary=bool(is_unitary),
            unitarity_residual=float(unitarity_residual),
            orthogonality_residual=float(orthogonality_residual),
            condition_estimate=float(condition_estimate),
            cayley_dickson_matrix=cd,
            left_mult_matrix=left,
            right_mult_matrix=right,
            sha256_hash=sha,
            phase_token=phase_token,
        )

    def phase1_construct_canonical_state(self, S: np.ndarray) -> QuaternionicState:
        """Fase 1.4 — Construcción pública del estado canónico (compatibilidad)."""
        return self._build_state(S, "PHASE1/CANONICAL")

    def phase1_representation_audit(self, state: QuaternionicState) -> Dict[str, float]:
        """
        Fase 1.5 — Auditoría de las tres representaciones equivalentes.

        Comprueba:
          - det Φ_C(q) = ||q||²
          - Φ_C es de la forma Cayley–Dickson
          - tr L(q) = 4 q₀
          - ||L(q)||_F = 2 ||q||
          - residual C* en M₂(C)
        """
        q = state.vector_rep
        cd = np.array(state.cayley_dickson_matrix, copy=True)
        left = state.left_mult_matrix

        alpha = complex(float(q[0]), float(q[1]))
        beta = complex(float(q[2]), float(q[3]))

        cd_form_residual = float(
            abs(cd[0, 0] - alpha)
            + abs(cd[0, 1] - beta)
            + abs(cd[1, 0] + beta.conjugate())
            + abs(cd[1, 1] - alpha.conjugate())
        )

        try:
            det_cd = complex(la.det(cd))
        except la.LinAlgError:
            det_cd = complex(math.nan, math.nan)

        det_cd_residual = _relative_residual(det_cd.real, state.squared_norm) + abs(
            det_cd.imag
        )
        cstar = _cstar_residual(cd, state.squared_norm)
        trace_residual = abs(float(np.trace(left)) - 4.0 * state.scalar_part)
        fro_left = _frobenius(left)
        fro_expected = 2.0 * state.norm
        fro_residual = _relative_residual(fro_left, fro_expected)

        return {
            "cayley_dickson_form_residual": cd_form_residual,
            "cayley_dickson_det_residual": float(det_cd_residual),
            "cstar_residual": float(cstar),
            "left_trace_residual": float(trace_residual),
            "left_frobenius_residual": float(fro_residual),
        }

    def phase1_close_and_open_phase2(self, S: np.ndarray) -> Phase1Handoff:
        """
        Fase 1.6 — Cierre formal de Fase 1 y apertura verificada de Fase 2.

        Definición formal de frontera:

            Φ₁→₂ : S ∈ R⁴ ↦ (q, δ₁) ∈ H × D₁

        Este es el último método de la Fase 1. Su contrato es exactamente el
        dominio de `phase2_from_phase1`: consume S, produce Phase1Handoff y
        exige que la Fase 2 admita de inmediato esa frontera. Con ello la
        Fase 1 queda anidada, como prefijo functorial, dentro de la Fase 2.
        """
        q_raw = self.phase1_validate_signal(S)
        state = self._build_state(q_raw, "PHASE1/CLOSED->PHASE2/OPEN")
        banach = self.phase1_banach_invariants(state.vector_rep)
        representation = self.phase1_representation_audit(state)

        diagnostics: Dict[str, float] = {
            "norm": state.norm,
            "squared_norm": state.squared_norm,
            "vector_norm": state.vector_norm,
            "unitarity_residual": state.unitarity_residual,
            "orthogonality_residual": state.orthogonality_residual,
            "condition_estimate": state.condition_estimate,
            "machine_epsilon": _MACHINE_EPS,
            **banach,
            **representation,
        }

        handoff = Phase1Handoff(
            state=state,
            diagnostics=diagnostics,
            next_entrypoint=_PHASE1_ENTRY,
        )

        # Anidamiento Φ₁→₂: la frontera debe ser admisible por Fase 2.
        opened = self.phase2_from_phase1(handoff)
        if opened.sha256_hash != state.sha256_hash:
            raise RuntimeError(
                "Invariante de anidamiento Φ₁→₂ violado: el estado admitido "
                "por Fase 2 no coincide con el estado canónico de Fase 1."
            )

        logger.debug(
            "Fase 1 cerrada. Estado canónico %s con norma=%.18g",
            state.sha256_hash[:12],
            state.norm,
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2 — ÁLGEBRA DE COMPOSICIÓN DE HURWITZ
    # (continuación formal de phase1_close_and_open_phase2)
    # ═════════════════════════════════════════════════════════════════════════

    def phase2_from_phase1(self, handoff: Phase1Handoff) -> QuaternionicState:
        """
        Fase 2.0 — Entrada formal desde Fase 1.

        Continuación directa de `phase1_close_and_open_phase2`. Consume
        `Phase1Handoff` y devuelve el estado canónico q ∈ H.
        """
        if not isinstance(handoff, Phase1Handoff):
            raise TypeError("Se esperaba Phase1Handoff como frontera Φ₁→₂.")
        if handoff.next_entrypoint != _PHASE1_ENTRY:
            raise ValueError(
                "Phase1Handoff inválido: el punto de entrada esperado es "
                f"{_PHASE1_ENTRY!r}."
            )
        if handoff.state.vector_rep.shape != (4,):
            raise ValueError("El estado de Φ₁→₂ no es cuatridimensional.")
        return handoff.state

    def phase2_conjugate(self, state: QuaternionicState) -> QuaternionicState:
        """
        Fase 2.1 — Involución de C*: q* = q₀ − q₁ i − q₂ j − q₃ k.

        Propiedades: (pq)* = q* p*,   q q* = q* q = ||q||²,   (q*)* = q.
        """
        v = state.vector_rep
        conj_vec = np.array([v[0], -v[1], -v[2], -v[3]], dtype=np.float64)
        return self._build_state(conj_vec, "PHASE2/CONJUGATE")

    def phase2_inner_product(
        self,
        p: QuaternionicState,
        q: QuaternionicState,
    ) -> float:
        """
        Fase 2.2 — Producto interno real ⟨p, q⟩ = Re(p̄ q) = Σ pᵢ qᵢ.

        Induce la métrica riemanniana canónica de S³ ⊂ H.
        """
        return _real_inner(p.vector_rep, q.vector_rep)

    def phase2_hamilton_product(
        self,
        p: QuaternionicState,
        q: QuaternionicState,
        verify: bool = True,
    ) -> QuaternionicState:
        """
        Fase 2.3 — Producto hamiltoniano p q, identidad de los cuatro cuadrados.

        Se calcula por acumulación compensada y se cruza con L(p) q y R(q) p.
        Si `verify=True`, se audita ||p q|| = ||p|| ||q|| (Hurwitz).
        """
        pv = p.vector_rep
        qv = q.vector_rep
        res_vec = _hamilton_vector(pv, qv)

        via_left = np.asarray(p.left_mult_matrix @ qv, dtype=np.float64).reshape(4)
        via_right = np.asarray(q.right_mult_matrix @ pv, dtype=np.float64).reshape(4)

        cross_left = float(math.hypot(*(res_vec - via_left)))
        cross_right = float(math.hypot(*(res_vec - via_right)))
        if cross_left > self._tolerance_of(max(1.0, p.norm * q.norm)):
            logger.warning(
                "Discrepancia producto vs L(p)q: %.3e", cross_left
            )
        if cross_right > self._tolerance_of(max(1.0, p.norm * q.norm)):
            logger.warning(
                "Discrepancia producto vs R(q)p: %.3e", cross_right
            )

        result = self._build_state(res_vec, "PHASE2/HAMILTON_PRODUCT")

        if verify:
            residual = self.phase2_hurwitz_residual(p, q, result)
            if residual > max(10.0 * self._tol, 1e-10):
                logger.warning("Residual de norma de Hurwitz elevado: %.3e", residual)

            if (
                self._auto_renormalize
                and p.is_unitary
                and q.is_unitary
                and not result.is_unitary
            ):
                result = self.phase2_normalize(result)

        return result

    def phase2_associator(
        self,
        p: QuaternionicState,
        q: QuaternionicState,
        r: QuaternionicState,
    ) -> Tuple[QuaternionicState, float]:
        """
        Fase 2.4 — Asociador [p, q, r] = (p q) r − p (q r).

        H es asociativa: el asociador debe ser numéricamente nulo.
        """
        pq = self.phase2_hamilton_product(p, q, verify=False)
        qr = self.phase2_hamilton_product(q, r, verify=False)
        left = self.phase2_hamilton_product(pq, r, verify=False)
        right = self.phase2_hamilton_product(p, qr, verify=False)
        assoc_vec = left.vector_rep - right.vector_rep
        assoc_state = self._build_state(assoc_vec, "PHASE2/ASSOCIATOR")
        return assoc_state, float(assoc_state.norm)

    def phase2_commutator(
        self,
        p: QuaternionicState,
        q: QuaternionicState,
    ) -> Tuple[QuaternionicState, float]:
        """
        Fase 2.5 — Conmutador [p, q] = p q − q p ∈ Im(H).

        El centro de H es R. Si p o q es real, [p, q] = 0.
        """
        pq = self.phase2_hamilton_product(p, q, verify=False)
        qp = self.phase2_hamilton_product(q, p, verify=False)
        comm_vec = pq.vector_rep - qp.vector_rep
        comm_state = self._build_state(comm_vec, "PHASE2/COMMUTATOR")
        return comm_state, float(comm_state.norm)

    def phase2_normalize(self, state: QuaternionicState) -> QuaternionicState:
        """
        Fase 2.6 — Retracción rígida q ↦ q / ||q|| sobre S³ ⊂ H.
        """
        if state.norm < self._reg:
            raise ZeroDivisionError(
                "La norma del estado es nula o sub-regular; no se puede normalizar."
            )
        if state.is_unitary:
            return self._build_state(state.vector_rep, "PHASE2/NORMALIZED")
        normalized_vec = state.vector_rep / state.norm
        return self._build_state(normalized_vec, "PHASE2/NORMALIZED")

    def phase2_polar_decomposition(
        self,
        state: QuaternionicState,
    ) -> Tuple[float, QuaternionicState]:
        """
        Fase 2.7 — Descomposición polar q = ρ u, ρ = ||q||, u ∈ S³.

        Equivale a la polar de Φ_C(q) ∈ M₂(C).
        """
        if state.norm < self._reg:
            raise ZeroDivisionError(
                "La norma del estado es nula o sub-regular; la polar no está definida."
            )
        unitary = (
            state
            if state.is_unitary
            else self.phase2_normalize(state)
        )
        return float(state.norm), unitary

    def phase2_exponential(self, state: QuaternionicState) -> QuaternionicState:
        """
        Fase 2.8 — Exponencial de álgebra de Lie H → Hˣ.

            exp(q₀ + v) = e^{q₀} (cos θ + û sin θ),  θ = ||v||, û = v/θ.

        Restringida a Im(H) ≅ su(2) ≅ so(3) cubre el recubrimiento Spin(3) → SO(3).
        """
        q0 = state.scalar_part
        theta = state.vector_norm
        scale = math.exp(q0)
        if not math.isfinite(scale):
            raise OverflowError("exp(q₀) no es representable en float64.")

        if theta < self._reg:
            vec = np.array(
                [scale, scale * state.vector_part[0], scale * state.vector_part[1],
                 scale * state.vector_part[2]],
                dtype=np.float64,
            )
            # para θ ≈ 0: cos θ ≈ 1, sinc θ ≈ 1 ⇒ Im(exp q) ≈ e^{q₀} v
            vec = np.array(
                [
                    scale,
                    scale * float(state.vector_rep[1]),
                    scale * float(state.vector_rep[2]),
                    scale * float(state.vector_rep[3]),
                ],
                dtype=np.float64,
            )
        else:
            u = state.vector_part / theta
            c = math.cos(theta)
            s = math.sin(theta)
            vec = np.array(
                [
                    scale * c,
                    scale * s * float(u[0]),
                    scale * s * float(u[1]),
                    scale * s * float(u[2]),
                ],
                dtype=np.float64,
            )
        return self._build_state(vec, "PHASE2/EXPONENTIAL")

    def phase2_logarithm(self, state: QuaternionicState) -> QuaternionicState:
        """
        Fase 2.9 — Logaritmo principal Log : Hˣ → H.

            Log(q) = log ||q|| + û arccos(q₀ / ||q||),  û = Im(q)/||Im(q)||.
        """
        if state.norm < self._reg:
            raise ZeroDivisionError(
                "Logaritmo no definido en el origen del álgebra de división."
            )
        rho = math.log(state.norm)
        cosine = _clip_unit(state.scalar_part / state.norm)
        theta = math.acos(cosine)
        if state.vector_norm < self._reg:
            vec = np.array([rho, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            u = state.vector_part / state.vector_norm
            vec = np.array(
                [rho, theta * float(u[0]), theta * float(u[1]), theta * float(u[2])],
                dtype=np.float64,
            )
        return self._build_state(vec, "PHASE2/LOGARITHM")

    def phase2_inverse(self, state: QuaternionicState) -> QuaternionicState:
        """
        Fase 2.10 — Inverso en el álgebra de división: q⁻¹ = q* / ||q||².

        Garantía: q q⁻¹ = q⁻¹ q = 1, residualmente dentro de la tolerancia.
        """
        if state.squared_norm < self._reg:
            raise ZeroDivisionError(
                "La norma del estado es nula o sub-regular; el inverso no está definido."
            )
        v = state.vector_rep
        inv_vec = np.array([v[0], -v[1], -v[2], -v[3]], dtype=np.float64)
        inv_vec /= state.squared_norm
        inverse = self._build_state(inv_vec, "PHASE2/INVERSE")

        product = self.phase2_hamilton_product(state, inverse, verify=False)
        identity_residual = math.hypot(
            product.vector_rep[0] - 1.0,
            product.vector_rep[1],
            product.vector_rep[2],
            product.vector_rep[3],
        )
        if identity_residual > self._tolerance_of(max(1.0, state.condition_estimate)):
            logger.warning(
                "Residual de q q⁻¹ − 1 elevado: %.3e", identity_residual
            )
        return inverse

    def phase2_divide(
        self,
        numerator: QuaternionicState,
        denominator: QuaternionicState,
    ) -> QuaternionicState:
        """
        Fase 2.11 — División por la derecha: n / d = n d⁻¹.
        """
        inv_den = self.phase2_inverse(denominator)
        return self.phase2_hamilton_product(numerator, inv_den, verify=True)

    def phase2_cstar_identity(self, state: QuaternionicState) -> float:
        """
        Fase 2.12 — Residual de la identidad C*: ||q* q|| = ||q||².

        En H esta identidad es exacta en aritmética real; el residual mide
        únicamente el defecto de punto flotante.
        """
        conj = self.phase2_conjugate(state)
        prod = self.phase2_hamilton_product(conj, state, verify=False)
        return _relative_residual(prod.norm, state.squared_norm)

    def phase2_hurwitz_residual(
        self,
        p: QuaternionicState,
        q: QuaternionicState,
        product: QuaternionicState,
    ) -> float:
        """
        Fase 2.13 — Residual de multiplicatividad ||pq|| = ||p|| ||q||.
        """
        expected = p.norm * q.norm
        if expected <= 0.0:
            return abs(product.norm)
        return abs(product.norm - expected) / max(1.0, expected)

    def phase2_bimodule_commutation(
        self,
        p: QuaternionicState,
        q: QuaternionicState,
    ) -> float:
        """
        Fase 2.14 — Residual del bimódulo: L(p) R(q) = R(q) L(p).
        """
        lp = p.left_mult_matrix
        rq = q.right_mult_matrix
        commutator = lp @ rq - rq @ lp
        return _frobenius(commutator) / max(1.0, p.norm * q.norm)

    def phase2_homomorphism_residual(
        self,
        p: QuaternionicState,
        q: QuaternionicState,
        product: QuaternionicState,
    ) -> Dict[str, float]:
        """
        Fase 2.15 — Residual de homomorfismo Φ_L(pq) = Φ_L(p) Φ_L(q)
        y antihomomorfismo Φ_R(pq) = Φ_R(q) Φ_R(p).
        """
        left_res = _frobenius(
            product.left_mult_matrix - (p.left_mult_matrix @ q.left_mult_matrix)
        )
        right_res = _frobenius(
            product.right_mult_matrix - (q.right_mult_matrix @ p.right_mult_matrix)
        )
        scale = max(1.0, p.norm * q.norm)
        return {
            "left_homomorphism_residual": float(left_res / scale),
            "right_antihomomorphism_residual": float(right_res / scale),
            "bimodule_commutation_residual": self.phase2_bimodule_commutation(p, q),
        }

    def phase2_close_and_open_phase3(
        self,
        state: QuaternionicState,
    ) -> Phase2Handoff:
        """
        Fase 2.16 — Cierre formal de Fase 2 y apertura verificada de Fase 3.

        Definición formal de frontera:

            Φ₂→₃ : q ∈ H ↦ (q, ρ₂) ∈ H × D₂

        Este es el último método de la Fase 2. Su contrato es exactamente el
        dominio de `phase3_from_phase2`. Con ello la Fase 2 queda anidada,
        como prefijo functorial, dentro de la Fase 3.
        """
        conj = self.phase2_conjugate(state)
        cstar = self.phase2_cstar_identity(state)
        bimodule = self.phase2_bimodule_commutation(state, conj)

        algebra_report: Dict[str, Any] = {
            "phase_token": state.phase_token,
            "norm": state.norm,
            "squared_norm": state.squared_norm,
            "vector_norm": state.vector_norm,
            "is_unitary": state.is_unitary,
            "unitarity_residual": state.unitarity_residual,
            "orthogonality_residual": state.orthogonality_residual,
            "condition_estimate": state.condition_estimate,
            "cstar_residual": float(cstar),
            "bimodule_self_commutation_residual": float(bimodule),
            "sha256_hash": state.sha256_hash,
        }

        handoff = Phase2Handoff(
            state=state,
            algebra_report=algebra_report,
            next_entrypoint=_PHASE2_ENTRY,
        )

        opened = self.phase3_from_phase2(handoff)
        if opened.sha256_hash != state.sha256_hash:
            raise RuntimeError(
                "Invariante de anidamiento Φ₂→₃ violado: el estado admitido "
                "por Fase 3 no coincide con el estado auditado de Fase 2."
            )

        logger.debug(
            "Fase 2 cerrada. Estado %s auditado algebraicamente.",
            state.sha256_hash[:12],
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 3 — ESPECTRO, TRANSPORTE, RIEMANN Y GOBERNANZA
    # (continuación formal de phase2_close_and_open_phase3)
    # ═════════════════════════════════════════════════════════════════════════

    def phase3_from_phase2(self, handoff: Phase2Handoff) -> QuaternionicState:
        """
        Fase 3.0 — Entrada formal desde Fase 2.

        Continuación directa de `phase2_close_and_open_phase3`.
        """
        if not isinstance(handoff, Phase2Handoff):
            raise TypeError("Se esperaba Phase2Handoff como frontera Φ₂→₃.")
        if handoff.next_entrypoint != _PHASE2_ENTRY:
            raise ValueError(
                "Phase2Handoff inválido: el punto de entrada esperado es "
                f"{_PHASE2_ENTRY!r}."
            )
        return handoff.state

    def phase3_characteristic_polynomial(
        self,
        state: QuaternionicState,
    ) -> Dict[str, Any]:
        """
        Fase 3.1 — Polinomio característico de L(q).

            χ(λ) = (λ² − 2 q₀ λ + ||q||²)²
                 = λ⁴ − 4 q₀ λ³ + (4 q₀² + 2 ||q||²) λ²
                   − 4 q₀ ||q||² λ + ||q||⁴
        """
        q0 = state.scalar_part
        n2 = state.squared_norm
        theoretical = np.array(
            [
                1.0,
                -4.0 * q0,
                4.0 * q0 * q0 + 2.0 * n2,
                -4.0 * q0 * n2,
                n2 * n2,
            ],
            dtype=np.float64,
        )
        try:
            numerical = np.asarray(np.poly(la.eigvals(state.left_mult_matrix)), dtype=np.float64)
            if numerical.shape != (5,):
                numerical = theoretical.copy()
                poly_residual = math.nan
            else:
                poly_residual = float(
                    math.hypot(*(numerical - theoretical))
                    / max(1.0, float(np.max(np.abs(theoretical))))
                )
        except la.LinAlgError as exc:
            logger.warning("Fallo en polinomio característico: %s", exc)
            numerical = theoretical.copy()
            poly_residual = math.nan

        return {
            "theoretical_coefficients": theoretical.tolist(),
            "numerical_coefficients": numerical.tolist(),
            "characteristic_residual": poly_residual,
        }

    def phase3_spectral_audit(self, state: QuaternionicState) -> Dict[str, Any]:
        """
        Fase 3.2 — Auditoría espectral de L(q).

        σ(L(q)) = {q₀ + i ||v||, q₀ − i ||v||}, cada uno con multiplicidad 2.
        El radio espectral coincide con ||q||. Los valores singulares son
        todos iguales a ||q|| (isometría escalada).
        """
        M = np.array(state.left_mult_matrix, copy=True)

        try:
            eigvals = la.eigvals(M)
        except la.LinAlgError as exc:
            logger.warning("Fallo en el cálculo de autovalores: %s", exc)
            eigvals = np.empty(0, dtype=np.complex128)

        spectral_radius = (
            float(np.max(np.abs(eigvals))) if eigvals.size > 0 else math.nan
        )

        try:
            singular_values = la.svd(M, compute_uv=False)
        except la.LinAlgError as exc:
            logger.warning("Fallo en SVD: %s", exc)
            singular_values = np.empty(0, dtype=np.float64)

        try:
            determinant = float(np.real_if_close(la.det(M)))
        except la.LinAlgError as exc:
            logger.warning("Fallo en el determinante: %s", exc)
            determinant = math.nan

        expected_det = state.squared_norm * state.squared_norm
        determinant_residual = _relative_residual(determinant, expected_det)
        spectral_residual = _relative_residual(spectral_radius, state.norm)

        if singular_values.size > 0:
            singular_residual = float(
                np.max(np.abs(singular_values - state.norm))
            ) / max(1.0, state.norm)
            if state.norm >= self._reg and float(np.min(singular_values)) > 0.0:
                condition_number = float(
                    np.max(singular_values) / np.min(singular_values)
                )
            else:
                condition_number = math.inf
        else:
            singular_residual = math.nan
            condition_number = math.nan

        r = state.vector_norm
        q0 = state.scalar_part
        expected_eigenvalues = np.array(
            [
                complex(q0, r),
                complex(q0, -r),
                complex(q0, r),
                complex(q0, -r),
            ],
            dtype=np.complex128,
        )

        if eigvals.size == 4:
            distances = [
                float(np.min(np.abs(eigvals - expected)))
                for expected in expected_eigenvalues
            ]
            model_residual = float(math.fsum(distances) / 4.0)
        else:
            model_residual = math.nan

        charpoly = self.phase3_characteristic_polynomial(state)

        return {
            "phase_token": state.phase_token,
            "eigenvalues": eigvals.tolist(),
            "spectral_radius": spectral_radius,
            "singular_values": singular_values.tolist(),
            "determinant": determinant,
            "expected_determinant": expected_det,
            "determinant_residual": determinant_residual,
            "spectral_residual": spectral_residual,
            "singular_residual": singular_residual,
            "eigenvalue_model_residual": model_residual,
            "condition_number": condition_number,
            "state_norm": state.norm,
            "state_vector_norm": state.vector_norm,
            **charpoly,
        }

    def phase3_hopf_fibration(self, state: QuaternionicState) -> Dict[str, Any]:
        """
        Fase 3.3 — Fibración de Hopf π : S³ → S², generador de π₃(S²) ≅ Z.

        Identificando q ≅ (z₁, z₂) ∈ C²,
            π(q) = (2 z₁ z̄₂, |z₁|² − |z₂|²) ∈ S² ⊂ R³.

        Si q no es unitario se usa u = q/||q||. El origen se reporta como fibra nula.
        """
        if state.norm < self._reg:
            return {
                "fiber": "null",
                "base_point": [0.0, 0.0, 0.0],
                "base_norm": 0.0,
                "sphericity_residual": 0.0,
            }

        u = state.vector_rep / state.norm
        z1 = complex(float(u[0]), float(u[1]))
        z2 = complex(float(u[2]), float(u[3]))
        hopf_xy = 2.0 * z1 * z2.conjugate()
        hopf_z = (abs(z1) ** 2) - (abs(z2) ** 2)
        base = np.array([hopf_xy.real, hopf_xy.imag, hopf_z], dtype=np.float64)
        base_norm, _ = _stable_norm(base)
        sphericity = abs(base_norm - 1.0)
        return {
            "fiber": "S1",
            "base_point": base.tolist(),
            "base_norm": float(base_norm),
            "sphericity_residual": float(sphericity),
            "z1": z1,
            "z2": z2,
        }

    def phase3_riemann_sphere_chart(
        self,
        state: QuaternionicState,
    ) -> Dict[str, Any]:
        """
        Fase 3.4 — Atlas estereográfico de Im(q)/||Im(q)|| ∈ S².

        Carta norte: Z = (x + i y) / (1 − z), regular fuera del polo norte.
        Carta sur:   W = (x − i y) / (1 + z), regular fuera del polo sur.
        Factor conforme de la métrica redonda: 2 / (1 + |Z|²).
        """
        hopf = self.phase3_hopf_fibration(state)
        v = state.vector_part
        v_norm = state.vector_norm

        if v_norm < self._reg:
            return {
                "chart": "origin",
                "coordinate": complex(0.0, 0.0),
                "coordinate_real": 0.0,
                "coordinate_imag": 0.0,
                "height": 0.0,
                "metric_factor": 1.0,
                "sphericity_residual": 0.0,
                "hopf": hopf,
            }

        u = v / v_norm
        x, y, z = float(u[0]), float(u[1]), float(u[2])
        z = max(-1.0, min(1.0, z))
        sphericity_residual = abs((x * x + y * y + z * z) - 1.0)

        north_den = 1.0 - z
        south_den = 1.0 + z

        if abs(north_den) > self._tol:
            coordinate = complex(x, y) / north_den
            chart = "north"
        elif abs(south_den) > self._tol:
            coordinate = complex(x, -y) / south_den
            chart = "south"
        else:
            coordinate = complex(math.inf, math.inf)
            chart = "singular"

        abs_z = abs(coordinate)
        if math.isfinite(abs_z):
            metric_factor = 2.0 / (1.0 + abs_z * abs_z)
        else:
            metric_factor = 0.0

        return {
            "chart": chart,
            "coordinate": coordinate,
            "coordinate_real": float(coordinate.real) if math.isfinite(coordinate.real) else math.nan,
            "coordinate_imag": float(coordinate.imag) if math.isfinite(coordinate.imag) else math.nan,
            "height": float(z),
            "metric_factor": float(metric_factor),
            "sphericity_residual": float(sphericity_residual),
            "hopf": hopf,
        }

    def phase3_geodesic_distance(
        self,
        p: QuaternionicState,
        q: QuaternionicState,
        projective: bool = False,
    ) -> float:
        """
        Fase 3.5 — Distancia geodésica.

        Sobre S³: d(p, q) = arccos ⟨û, v̂⟩.
        Sobre RP³ ≅ SO(3): d(p, q) = arccos |⟨û, v̂⟩|  (identifica ±).
        """
        if p.norm < self._reg or q.norm < self._reg:
            raise ZeroDivisionError(
                "La distancia geodésica no está definida para estados nulos."
            )
        inner = self.phase2_inner_product(p, q) / (p.norm * q.norm)
        inner = _clip_unit(inner)
        if projective:
            inner = abs(inner)
        return float(math.acos(inner))

    def phase3_slerp(
        self,
        p: QuaternionicState,
        q: QuaternionicState,
        t: float,
    ) -> QuaternionicState:
        """
        Fase 3.6 — Interpolación geodésica de menor arco sobre S³.

            slerp(p, q; t) = (sin((1−t)θ) û + sin(t θ) v̂) / sin θ
        """
        if not math.isfinite(t):
            raise ValueError("El parámetro t de SLERP debe ser finito.")
        if p.norm < self._reg or q.norm < self._reg:
            raise ZeroDivisionError("SLERP no está definido para estados nulos.")

        pu = p.vector_rep / p.norm
        qu = q.vector_rep / q.norm
        dot = _clip_unit(_real_inner(pu, qu))

        if dot < 0.0:
            qu = -qu
            dot = -dot

        if dot > 0.9995:
            lerp = (1.0 - t) * pu + t * qu
            return self.phase2_normalize(self._build_state(lerp, "PHASE3/SLERP"))

        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        w_p = math.sin((1.0 - t) * theta) / sin_theta
        w_q = math.sin(t * theta) / sin_theta
        interp = w_p * pu + w_q * qu
        return self._build_state(interp, "PHASE3/SLERP")

    def phase3_parallel_transport(
        self,
        state: QuaternionicState,
        rotor: Optional[Any] = None,
        mode: str = "sandwich",
    ) -> Tuple[QuaternionicState, Dict[str, float]]:
        """
        Fase 3.7 — Transporte paralelo cuaterniónico libre de Gimbal Lock.

        Modos:
          - "left":     q' = r q
          - "right":    q' = q r
          - "sandwich": q' = r q r*   (acción Ad : Sp(1) → SO(3) sobre Im H)

        El modo sandwich preserva la parte real cuando r es unitario y realiza
        el recubrimiento 2-a-1 Spin(3) → SO(3).
        """
        if rotor is None:
            rotor_state = self._build_state(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "PHASE3/IDENTITY_ROTOR",
            )
        elif isinstance(rotor, QuaternionicState):
            rotor_state = rotor
        else:
            rotor_state = self._build_state(
                np.asarray(rotor, dtype=np.float64),
                "PHASE3/EXTERNAL_ROTOR",
            )

        if rotor_state.norm < self._reg:
            raise ZeroDivisionError(
                "El rotor tiene norma nula o sub-regular; el transporte no está definido."
            )

        rotor_unitarity_residual = abs(rotor_state.norm - 1.0)
        if rotor_unitarity_residual > self._tolerance_of(1.0):
            if self._auto_renormalize:
                rotor_state = self.phase2_normalize(rotor_state)
                rotor_unitarity_residual = abs(rotor_state.norm - 1.0)
            else:
                raise ValueError(
                    "El rotor no es unitario dentro de la tolerancia activa."
                )

        if mode == "left":
            transported = self.phase2_hamilton_product(rotor_state, state, verify=True)
        elif mode == "right":
            transported = self.phase2_hamilton_product(state, rotor_state, verify=True)
        elif mode == "sandwich":
            rotor_conj = self.phase2_conjugate(rotor_state)
            tmp = self.phase2_hamilton_product(rotor_state, state, verify=True)
            transported = self.phase2_hamilton_product(tmp, rotor_conj, verify=True)
        else:
            raise ValueError(
                "Modo de transporte inválido. Use 'left', 'right' o 'sandwich'."
            )

        norm_drift = _relative_residual(transported.norm, state.norm)
        if mode == "sandwich":
            scalar_invariance_residual = abs(
                transported.scalar_part - state.scalar_part
            )
        else:
            scalar_invariance_residual = math.nan

        metrics: Dict[str, float] = {
            "norm_drift": float(norm_drift),
            "scalar_invariance_residual": float(scalar_invariance_residual),
            "rotor_norm": float(rotor_state.norm),
            "rotor_unitarity_residual": float(rotor_unitarity_residual),
            "transport_mode_is_unitary": 1.0 if transported.is_unitary else 0.0,
        }
        return transported, metrics

    def phase3_liouville_de_rham_audit(
        self,
        state: QuaternionicState,
    ) -> Dict[str, Any]:
        """
        Fase 3.8 — Volumen de Liouville y generadores de de Rham/Hamilton.

        det L(q) = ||q||⁴. Si ||q|| = 1, L(q) ∈ SO(4) y preserva el volumen
        de Haar de S³. Si q₀ = 0, L(q) es antisimétrica (tr = 0): generador
        infinitesimal de su(2) ⊂ so(4), divergencia nula.
        """
        M = np.array(state.left_mult_matrix, copy=True)
        try:
            determinant = float(np.real_if_close(la.det(M)))
        except la.LinAlgError as exc:
            logger.warning("Fallo en determinante Liouville: %s", exc)
            determinant = math.nan

        expected_det = state.squared_norm * state.squared_norm
        if state.norm < self._reg:
            liouville_residual = math.nan
            volume_ratio = math.nan
        else:
            liouville_residual = _relative_residual(determinant, expected_det)
            if math.isfinite(determinant) and expected_det != 0.0:
                volume_ratio = determinant / expected_det
            else:
                volume_ratio = math.nan

        fro_M = _frobenius(M)
        skew_residual = _frobenius(M + M.T) / max(1.0, fro_M)
        trace = float(np.trace(M))
        infinitesimal_volume_preserving = abs(state.scalar_part) <= self._tolerance_of(1.0)

        return {
            "phase_token": state.phase_token,
            "determinant": determinant,
            "expected_determinant": expected_det,
            "liouville_residual": float(liouville_residual),
            "volume_ratio": float(volume_ratio),
            "skew_symmetric_residual": float(skew_residual),
            "trace": trace,
            "infinitesimal_divergence": trace,
            "infinitesimal_volume_preserving": bool(infinitesimal_volume_preserving),
        }

    def phase3_governance_seal(
        self,
        state: QuaternionicState,
        spectral: Mapping[str, Any],
        transport_metrics: Mapping[str, float],
        riemann: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Fase 3.9 — Sello de gobernanza de lazo cerrado.

        La carga útil se serializa en binario canónico (SHA-256), evitando
        `repr` no portable entre versiones de CPython.
        """
        h = hashlib.sha256()
        h.update(b"QSS/GOVERNANCE/v3")
        h.update(state.sha256_hash.encode("utf-8"))
        h.update(state.phase_token.encode("utf-8"))

        numeric_keys = (
            spectral.get("spectral_radius"),
            spectral.get("spectral_residual"),
            spectral.get("determinant"),
            spectral.get("determinant_residual"),
            spectral.get("eigenvalue_model_residual"),
            transport_metrics.get("norm_drift"),
            transport_metrics.get("scalar_invariance_residual"),
            transport_metrics.get("rotor_unitarity_residual"),
            riemann.get("height"),
            riemann.get("coordinate_real"),
            riemann.get("coordinate_imag"),
            riemann.get("metric_factor"),
        )
        for value in numeric_keys:
            try:
                h.update(_pack_f64(float(value)))
            except (TypeError, ValueError):
                h.update(_pack_f64(math.nan))

        chart = str(riemann.get("chart", ""))
        h.update(len(chart).to_bytes(8, "little"))
        h.update(chart.encode("utf-8"))

        seal = h.hexdigest()
        return {
            "seal": seal,
            "hash_algorithm": "sha256",
            "encoding": "canonical-binary-v3",
            "state_hash": state.sha256_hash,
        }

    def phase3_close_loop(
        self,
        S: np.ndarray,
        rotor: Optional[Any] = None,
        transport_mode: str = "sandwich",
    ) -> Phase3Report:
        """
        Fase 3.10 — Orquestación completa de lazo cerrado.

        Ejecuta el funtor compuesto Φ₃ ∘ Φ₂ ∘ Φ₁:

            Fase 1 → Fase 2 → Fase 3.

        Retorna `Phase3Report` con estado transportado, estado fuente,
        auditorías espectrales, carta de Riemann/Hopf, métricas de transporte,
        Liouville y sello de gobernanza.
        """
        handoff_1 = self.phase1_close_and_open_phase2(S)
        state_phase2 = self.phase2_from_phase1(handoff_1)

        handoff_2 = self.phase2_close_and_open_phase3(state_phase2)
        state_phase3 = self.phase3_from_phase2(handoff_2)

        spectral = self.phase3_spectral_audit(state_phase3)
        riemann = self.phase3_riemann_sphere_chart(state_phase3)
        transported, transport_metrics = self.phase3_parallel_transport(
            state_phase3,
            rotor=rotor,
            mode=transport_mode,
        )
        liouville = self.phase3_liouville_de_rham_audit(state_phase3)
        governance = self.phase3_governance_seal(
            state_phase3,
            spectral,
            transport_metrics,
            riemann,
        )

        phase_chain = (
            handoff_1.state.phase_token,
            handoff_2.state.phase_token,
            state_phase3.phase_token,
            transported.phase_token,
        )

        return Phase3Report(
            state=transported,
            source_state=state_phase3,
            spectral_audit=spectral,
            riemann_chart=riemann,
            transport_metrics=transport_metrics,
            liouville_audit=liouville,
            governance_seal=governance,
            phase_chain=phase_chain,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # COMPATIBILIDAD CON API LEGADA
    # ═════════════════════════════════════════════════════════════════════════

    def build_state(self, S: np.ndarray) -> QuaternionicState:
        """API legada — equivalente a Fase 1.4."""
        return self.phase1_construct_canonical_state(S)

    def quaternionic_multiply(
        self,
        p: QuaternionicState,
        q: QuaternionicState,
    ) -> QuaternionicState:
        """API legada — equivalente a Fase 2.3."""
        return self.phase2_hamilton_product(p, q, verify=True)

    def quaternionic_inverse(self, state: QuaternionicState) -> QuaternionicState:
        """API legada — equivalente a Fase 2.10."""
        return self.phase2_inverse(state)

    def project_to_riemann_sphere(
        self,
        state: QuaternionicState,
    ) -> Tuple[complex, float]:
        """
        API legada — equivalente a Fase 3.4.

        Retorna
        -------
        Tuple[complex, float]
            (coordenada compleja estereográfica, altura z).
        """
        chart = self.phase3_riemann_sphere_chart(state)
        return complex(chart["coordinate"]), float(chart["height"])
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : FPU Tensor Fock Collider                                                  ║
║ Ubicación: app/wisdom/fpu_tensor_fock_collider.py                                    ║
║ Versión  : 12.0.0-SOVEREIGN-FPU-FOCK-GKSL-POINCARE-L3-PhD                            ║
║ Función  : Colisionador cuántico tensorial de Fock Fermi-Pasta-Ulam, álgebra CAR/CCR ║
║            de Jordan-Wigner, dinámicas disipativas GKSL de Lindblad con entropía de  ║
║            Spohn, mecánica celeste de Poincaré, teoría KAM y automejora RSI³.        ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL
─────────────────────────────────────────────────
El FPU Tensor Fock Collider opera como el reactor cuántico tensorial y colisionador de Fock
en el Estrato Wisdom (V_𝕎) de la arquitectura cognitiva APU Filter. Su función es modelar y
auditar representaciones algebraicas fermiónicas y bosónicas sobre el espacio de Fock discretizado
ℱ(ℋ), cuantificando la aniquilación relativista e⁻ + e⁺ → 2γ (equivalencia de 1022.0 keV por par),
evaluando la acción del generador infinitesimal de Lindblad-Gorini-Kossakowski-Sudarshan-Lindblad (GKSL)
sobre operadores de densidad ρ ∈ 𝔇(ℋ_N), certificando la producción de entropía no-negativa de Spohn
σ(ρ) = -Tr((ln ρ - ln σ) ℒ(ρ)) ≥ 0, sondeando la preservación de toros diofánticos KAM en secciones
transversales de Poincaré y gobernando la automejora recursiva de nivel 3 (RSI³).

Estructuración operativa trifásica anidada (OODA):
  • FASE 1 (OBSERVE) : Auditoría de álgebras cuánticas: relaciones de anticonmutación canónicas (CAR)
                       {c_i, c_j†} = δ_{ij}·I₄ para dos modos de Jordan-Wigner, relaciones de conmutación
                       canónicas (CCR) [b, b†] = I_d - d|d-1⟩⟨d-1| para el modo bosónico truncado en d = N+1,
                       evaluación del generador disipativo GKSL ℒ(ρ) = -i[H, ρ] + ∑ (L_k ρ L_k† - ½{L_k†L_k, ρ}),
                       autenticación criptográfica HMAC-SHA256 con consumo de nonces contra ataques de replay,
                       y emisión de la semilla canónica FockStateSeed mediante `export_fock_state_seed`.
  • FASE 2 (ORIENT)  : Construcción de la sección transversal discreta de Poincaré Σ = {ρ : Re Tr(H_probe ρ) - c = 0},
                       cálculo de cruces transversales e interpolación de Hénon, estimación diofántica KAM
                       |⟨k, ω⟩| ≥ γ / ‖k‖₁^τ sobre frecuencias espectrales de modulación áurea FPU, análisis de
                       forma normal de Birkhoff, y ejecución del controlador adaptativo de 3 niveles RSI³ con
                       búsqueda de Armijo sobre el potencial de Lyapunov V(ρ) = ‖ρ - ρ*‖_F² y amortiguamiento
                       de jerk discreto d³θ/dt³. Cierra emitiendo la trayectoria FockTrajectory mediante `iterate_level_3`.
  • FASE 3 (DECIDE)  : Orquestación soberana mediante `SovereignFockAuditor.full_audit`, clasificación de
                       estabilidad en islas elípticas vs. caos homoclínico de Melnikov, cálculo de la raíz de
                       Merkle binaria SHA-256 sobre el reporte unificado y sellado con firma de procedencia HMAC.

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. Álgebra de Fock Fermiónica CAR y Transformada de Jordan-Wigner:
   Para dos modos fermiónicos sobre ℋ = ℂ⁴ ≅ ℂ² ⊗ ℂ²:
       c₀ = σ₋ ⊗ I₂,   c₁ = σ_z ⊗ σ₋,   donde σ₋ = [[0, 1], [0, 0]],   σ_z = [[1, 0], [0, -1]].
   Relaciones de anticonmutación canónicas de Dirac:
       {c_i, c_j†} = c_i c_j† + c_j† c_i = δ_{ij} · I₄,   {c_i, c_j} = 0,   c_i² = 0.
   Modo bosónico truncado CCR de escalera con dimensión de corte d = min(n_max + 1, N):
       b |n⟩ = √n |n-1⟩,   b† |n⟩ = √(n+1) |n+1⟩ (para n < d-1),   b† |d-1⟩ = 0.
       [b, b†] = b b† - b† b = I_d - d · |d-1⟩⟨d-1|.

2. Semigrupos Cuánticos Dinámicos y Teorema de Spohn:
   Sea ℒ : 𝔇(ℋ) → 𝔗(ℋ) el generador infinitesimal GKSL en la formulación de Lindblad:
       ℒ(ρ) = -i [H, ρ] + ∑_k γ_k ( L_k ρ L_k† - ½ {L_k† L_k, ρ} ).
   Preserva la traza Tr(ℒ(ρ)) = 0 y la hermiticidad ℒ(ρ)† = ℒ(ρ).
   Para un estado de referencia estacionario tracial o fiel σ con ℒ(σ) = 0:
   La producción de entropía de Spohn cuantifica la irreversibilidad termodinámica del proceso:
       σ_Spohn(ρ) = -Tr( (ln ρ - ln σ) · ℒ(ρ) ) ≥ 0.
   La tasa de variación de la entropía de von Neumann S(ρ) = -Tr(ρ ln ρ) satisface:
       dS/dt = -Tr( ln ρ · ℒ(ρ) ).

3. Mecánica Celeste de Poincaré y Resonancias Diofánticas KAM:
   Sección transversal discreta de Poincaré:
       Σ_c = { ρ ∈ 𝔇(ℋ_N) | g(ρ) = Re Tr(H_probe · ρ) - c = 0 },   con ‖H_probe‖_F = 1.
   Transversalidad estricta de cruce:
       |dg[Δρ]| = |Re Tr(H_probe · (ρ_{t+1} - ρ_t))| > ε_trans.
   Condición diofantina KAM sobre el toro invariante 𝕋^m con frecuencias modales FPU:
       |⟨k, ω⟩| ≥ γ / ‖k‖₁^τ,   ∀ k ∈ ℤ^m \ {0},   γ > 0,   τ ≥ m - 1.
   Aproximación empírica de rotación de Birkhoff con desenrollamiento de fase:
       ρ_rot = (θ_N - θ_0) / (2π · N).

4. Automejora Recursiva Nivel 3 (RSI³):
   Sobre el espacio métrico de operadores de densidad (𝔇(ℋ_N), ‖·‖_F):
   • Nivel 0: I₀(ρ) = ρ (identidad).
   • Nivel 1: I₁(ρ) = (1 - θ) ρ + θ ρ*, con θ ajustado por condición de Armijo sobre V(ρ) = ‖ρ - ρ*‖_F²:
       V((1 - α)ρ + α ρ*) ≤ V(ρ) - c · α · V(ρ).
   • Nivel 2: Adaptación del paso θ_{t+1} = θ_t · exp(η · (ratio_t - (1 - θ_t))).
   • Nivel 3: Control adaptativo de la tasa de meta-aprendizaje η_{t+1} amortiguando la 3ª diferencia finita
     (jerk numérico) Δ³θ = θ_t - 3θ_{t-1} + 3θ_{t-2} - θ_{t-3}.

AXIOMAS FORMALES
────────────────
[Axioma 1] Densidad Cuántica: ρ ∈ 𝔇(ℋ_N) ⟹ ρ = ρ†, spec(ρ) ⊂ [0, 1], Tr(ρ) = 1.
[Axioma 2] Álgebra CAR Exacta: En el subespacio fermiónico de 4 dimensiones, {c_i, c_j†} = δ_{ij}·I₄
           y c_i² = 0 con residuo nulo a precisión de máquina (≤ 10⁻¹⁰).
[Axioma 3] No-Negatividad de Spohn: Para todo generador GKSL con estado estacionario σ de rango completo,
           σ_Spohn(ρ) ≥ -10⁻⁹ dentro de tolerancias numéricas.
[Axioma 4] Conservación y No-Repudio Criptográfico: Cada informe emitido sella sus componentes
           sustantivos en un árbol de Merkle binario determinista validable vía HMAC-SHA256.

INVARIANTES DE ESTADO
─────────────────────
1. Pureza cuántica Π(ρ) = Tr(ρ²) ∈ [1/N, 1] y entropía de von Neumann S(ρ) ≥ 0.
2. Residuo de conservación de traza del generador GKSL |Tr(ℒ(ρ))| ≤ 10⁻¹⁰.
3. Inviolabilidad de nonces: cada solicitud de aniquilación consume un nonce único en el proceso.
4. Raíz de Merkle binaria SHA-256 invariante ante serialización canónica.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, fields, replace
from enum import IntEnum
from itertools import combinations
from typing import Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.FPUTensorFockCollider")

# ──────────────────────────────────────────────────────────────────────────────
# Constantes y dominios criptográficos
# ──────────────────────────────────────────────────────────────────────────────
_MODULE_VERSION: Final[str] = "12.0.0-VERIFIED-SCOPE-FAIL-CLOSED"

_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_ALGEBRA_RESIDUAL_TOL: Final[float] = 1.0e-10
_DENSITY_VALIDATION_TOL: Final[float] = 1.0e-9

_POSITRON_REST_ENERGY_KEV: Final[float] = 1022.0
_FERMIONIC_SUBSPACE_DIM: Final[int] = 4
_BOSONIC_TRUNCATION_N: Final[int] = 6

_KAM_MAX_K: Final[int] = 8
_KAM_EFFECTIVE_DIM: Final[int] = 6
_RESONANCE_THRESHOLD: Final[float] = 1.0e-3

_L3_DEFAULT_ITERS: Final[int] = 6
_L3_THETA_MIN: Final[float] = 0.05
_L3_THETA_MAX: Final[float] = 0.95
_L3_META_MIN: Final[float] = 0.05
_L3_META_MAX: Final[float] = 0.90
_LYAPUNOV_ARMIJO_ITERS: Final[int] = 8
_LYAPUNOV_ARMIJO_C: Final[float] = 1.0e-4
_L3_CONVERGENCE_TOL: Final[float] = 1.0e-8
_L3_JERK_TOL: Final[float] = 0.05

_HOMOCLINIC_HS_TOL: Final[float] = 1.0e-4
_SECTION_TOL: Final[float] = 1.0e-12

_AUTH_DOMAIN: Final[bytes] = b"APU/FPU-TENSOR-FOCK/COLLIDER/AUTH/v12\x00"
_REPORT_HMAC_DOMAIN: Final[bytes] = b"APU/FPU-TENSOR-FOCK/COLLIDER/REPORT/v12\x00"
_NONCE_DOMAIN: Final[bytes] = b"APU/FPU-TENSOR-FOCK/COLLIDER/NONCE/v12\x00"


# ──────────────────────────────────────────────────────────────────────────────
# Validación y utilidades numéricas
# ──────────────────────────────────────────────────────────────────────────────
def _require_integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} debe ser un entero.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} debe ser >= {minimum}.")
    return result


def _as_square_matrix(matrix: NDArray, name: str) -> NDArray[np.complex128]:
    try:
        a = np.asarray(matrix, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} no se pudo convertir a matriz compleja.") from exc

    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError(f"{name} debe ser una matriz cuadrada no vacía.")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contiene valores NaN o infinitos.")
    return a


@dataclass(frozen=True, slots=True)
class DensityProjectionDiagnostics:
    """Diagnóstico de la reparación espectral de una matriz candidata."""

    input_hermiticity_residual: float
    input_trace_real: float
    input_trace_imag: float
    min_eigenvalue_before: float
    correction_frobenius_norm: float


def _project_eigenvalues_to_simplex(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Proyección euclídea de un vector real sobre el simplex {x >= 0, sum(x)=1}.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Se esperaba un vector real finito no vacío.")

    sorted_x = np.sort(x)[::-1]
    cssv = np.cumsum(sorted_x) - 1.0
    indices = np.arange(1, x.size + 1, dtype=np.float64)
    active = np.flatnonzero(sorted_x - cssv / indices > 0.0)

    if active.size == 0:
        return np.full(x.size, 1.0 / float(x.size), dtype=np.float64)

    rho = int(active[-1])
    tau = float(cssv[rho] / float(rho + 1))
    projected = np.maximum(x - tau, 0.0)

    total = float(projected.sum())
    if not math.isfinite(total) or total <= _WILKINSON_EPSILON:
        return np.full(x.size, 1.0 / float(x.size), dtype=np.float64)
    return projected / total


def _project_density_matrix_with_diagnostics(
    matrix: NDArray,
) -> Tuple[NDArray[np.complex128], DensityProjectionDiagnostics]:
    r"""
    Proyección métrica, en norma de Frobenius, de una matriz finita sobre el
    conjunto de matrices de densidad:

        D_n = {ρ = ρ†, ρ >= 0, Tr(ρ) = 1}.

    Se hermitiza la entrada y se proyecta su espectro sobre el simplex.
    Matemáticamente, la proyección métrica sobre un conjunto convexo cerrado
    es no expansiva; la implementación está sujeta a error de redondeo.
    """
    a = _as_square_matrix(matrix, "matrix")
    herm = 0.5 * (a + a.conj().T)
    eigvals, eigvecs = la.eigh(herm)

    projected_eigs = _project_eigenvalues_to_simplex(
        np.asarray(np.real(eigvals), dtype=np.float64)
    )
    rho = (eigvecs * projected_eigs) @ eigvecs.conj().T
    rho = 0.5 * (rho + rho.conj().T)

    trace = np.trace(rho)
    trace_real = float(np.real(trace))
    if not math.isfinite(trace_real) or trace_real <= _WILKINSON_EPSILON:
        rho = np.eye(a.shape[0], dtype=np.complex128) / float(a.shape[0])
    else:
        rho = rho / trace_real

    input_trace = np.trace(a)
    diagnostics = DensityProjectionDiagnostics(
        input_hermiticity_residual=float(np.linalg.norm(a - a.conj().T, ord="fro")),
        input_trace_real=float(np.real(input_trace)),
        input_trace_imag=float(np.imag(input_trace)),
        min_eigenvalue_before=float(np.min(np.real(eigvals))),
        correction_frobenius_norm=float(np.linalg.norm(rho - a, ord="fro")),
    )
    return np.asarray(rho, dtype=np.complex128), diagnostics


def _higham_density_projection(matrix: NDArray) -> NDArray[np.complex128]:
    """
    Alias privado de compatibilidad.

    No es el algoritmo de Higham para matrices de correlación: implementa la
    proyección espectral sobre el simplex de estados de densidad.
    """
    rho, _ = _project_density_matrix_with_diagnostics(matrix)
    return rho


def _assert_density_matrix(
    matrix: NDArray,
    *,
    name: str = "rho",
    tol: float = _DENSITY_VALIDATION_TOL,
) -> NDArray[np.complex128]:
    """Valida hermiticidad, positividad semidefinida y traza unitaria."""
    a = _as_square_matrix(matrix, name)
    scale = max(1.0, float(np.linalg.norm(a, ord="fro")))
    herm_residual = float(np.linalg.norm(a - a.conj().T, ord="fro"))
    if herm_residual > tol * scale:
        raise ValueError(f"{name} no es hermítica dentro de tolerancia.")

    herm = 0.5 * (a + a.conj().T)
    eigvals = la.eigvalsh(herm)
    if float(np.min(np.real(eigvals))) < -tol * scale:
        raise ValueError(f"{name} no es semidefinida positiva.")

    trace = np.trace(a)
    if abs(float(np.imag(trace))) > tol:
        raise ValueError(f"Tr({name}) tiene parte imaginaria significativa.")
    if abs(float(np.real(trace)) - 1.0) > tol:
        raise ValueError(f"{name} debe tener traza unitaria.")
    return a


def _matrix_log_if_full_rank(
    matrix: NDArray[np.complex128],
    floor: float = _SPECTRAL_PSD_FLOOR,
) -> Optional[NDArray[np.complex128]]:
    """Logaritmo matricial espectral sólo si el estado es numéricamente fiel."""
    herm = 0.5 * (matrix + matrix.conj().T)
    eigvals, eigvecs = la.eigh(herm)
    eigvals = np.real(eigvals)
    if eigvals.size == 0 or float(np.min(eigvals)) <= floor:
        return None
    log_eigs = np.log(eigvals)
    return (eigvecs * log_eigs) @ eigvecs.conj().T


def _von_neumann_entropy(rho: NDArray[np.complex128]) -> float:
    r"""S(ρ) = -Tr(ρ log ρ), con 0 log 0 := 0."""
    a = _as_square_matrix(rho, "rho")
    eigvals = np.real(la.eigvalsh(0.5 * (a + a.conj().T)))
    if float(np.min(eigvals)) < -_DENSITY_VALIDATION_TOL:
        raise ValueError("La entropía requiere una matriz semidefinida positiva.")
    positive = eigvals[eigvals > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def _purity(rho: NDArray[np.complex128]) -> float:
    r"""Π(ρ) = Tr(ρ²), evaluada en la matriz proporcionada."""
    a = _as_square_matrix(rho, "rho")
    return float(np.real(np.trace(a @ a)))


def _density_sha256(rho: NDArray[np.complex128]) -> str:
    """Hash SHA-256 canónico: forma y bytes complex128 little-endian C-order."""
    a = _as_square_matrix(rho, "rho")
    canonical = np.ascontiguousarray(np.asarray(a, dtype=np.dtype("<c16")))
    shape = np.asarray(a.shape, dtype=np.dtype("<u8")).tobytes()
    return hashlib.sha256(shape + canonical.tobytes(order="C")).hexdigest()


def _canonical_value(value: object) -> object:
    """
    Conversión determinista a JSON-compatible.

    Los flotantes finitos se representan por su hexadecimal IEEE-754 para no
    depender de redondeos de formato decimal.
    """
    if isinstance(value, IntEnum):
        return {"$enum": value.name, "$value": int(value)}
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("No se admite NaN/Inf en la serialización canónica.")
        return {"$float64": value.hex()}
    if isinstance(value, complex):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise ValueError("No se admite un complejo NaN/Inf.")
        return {"$complex128": [value.real.hex(), value.imag.hex()]}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    raise TypeError(f"Tipo no soportado en la serialización canónica: {type(value)!r}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _merkle_root_named_fields(named_fields: Sequence[Tuple[str, object]]) -> str:
    """
    Construye una raíz Merkle binaria con separación de dominios:

      leaf   = SHA256(0x00 || nombre || 0x00 || valor_canónico)
      parent = SHA256(0x01 || hijo_izq || hijo_der)
      odd    = SHA256(0x02 || último_hijo)
    """
    leaves: List[bytes] = []
    for name, value in named_fields:
        leaf = hashlib.sha256(
            b"\x00"
            + name.encode("utf-8")
            + b"\x00"
            + _canonical_json_bytes(value)
        ).digest()
        leaves.append(leaf)

    if not leaves:
        return hashlib.sha256(b"\x03EMPTY-MERKLE-TREE").hexdigest()

    level = leaves
    while len(level) > 1:
        if len(level) % 2:
            level = level + [hashlib.sha256(b"\x02" + level[-1]).digest()]
        level = [
            hashlib.sha256(b"\x01" + level[i] + level[i + 1]).digest()
            for i in range(0, len(level), 2)
        ]
    return level[0].hex()


# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1: CAR, CCR, GKSL, autenticación y exportación de la semilla                    ║
# ║ La última definición de método es export_fock_state_seed; su resultado inicia FASE 2.║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

class ParticleStatisticsCategory(IntEnum):
    """Categoría estadística usada por los diagnósticos del módulo."""

    FERMION_CAR = 0x01
    BOSON_CCR = 0x02
    ANTIMATTER = 0x03


@dataclass(frozen=True, slots=True)
class FockAlgebraAudit:
    """Residuals de CAR/CCR distinguiendo bloque físico y espacio ambiente."""

    dimension: int
    fermionic_block_dimension: int
    bosonic_block_dimension: int
    car_residual: float
    car_nilpotent_residual: float
    ccr_residual: float
    car_valid_on_block: bool
    ccr_valid_on_block: bool
    car_ambient_residual: float
    ccr_ambient_residual: float
    car_valid_on_ambient_space: bool
    ccr_valid_on_ambient_space: bool


@dataclass(frozen=True, slots=True)
class GKSLDiagnostics:
    """Resultados numéricos de una aplicación del generador GKSL."""

    generator_action: NDArray[np.complex128]
    dissipator: NDArray[np.complex128]
    trace_preserving_residual: float
    hermiticity_preserving_residual: float
    dissipator_norm: float
    entropy_rate: Optional[float]
    spohn_entropy_production: Optional[float]
    spohn_status: str
    spohn_nonnegative_within_tolerance: Optional[bool]
    stationary_state_residual: Optional[float]


@dataclass(frozen=True, slots=True)
class AnnihilationReactionReport:
    """
    Informe inmutable de software.

    `released_energy_kev` es la equivalencia de masa en reposo por el número de
    solicitudes autorizadas; no es una medición de energía ni una simulación de
    la cinemática e⁻e⁺ → 2γ.
    """

    reaction_id: str
    requested_anomalies: int
    electron_anomalies_purged: int
    positron_auth_valid: bool
    released_energy_kev: float
    back_action_db: float
    spohn_entropy_production: Optional[float]
    gksl_dissipation_rate: float
    sha256_merkle_digest: str
    timestamp_utc: float

    auth_status: str = "NOT_REQUESTED"
    transaction_digest: str = ""
    nonce_digest: str = ""
    report_hmac: str = ""
    signing_key_id: str = "UNSIGNED"
    security_key_configured: bool = False
    energy_interpretation: str = (
        "REST_MASS_EQUIVALENT_ONLY; no kinetic, recoil or detector model"
    )
    back_action_measured: bool = False
    model_scope: str = (
        "Algebraic blocks and software accounting; no physical particle simulation"
    )
    source_state_sha256: str = ""
    state_projection_correction_norm: float = 0.0

    car_residual: float = 0.0
    ccr_residual: float = 0.0
    car_valid: bool = False
    ccr_valid: bool = False
    car_nilpotent_residual: float = 0.0
    car_ambient_residual: float = 0.0
    ccr_ambient_residual: float = 0.0
    car_valid_on_ambient_space: bool = False
    ccr_valid_on_ambient_space: bool = False

    gksl_trace_residual: float = 0.0
    gksl_hermiticity_residual: float = 0.0
    gksl_spohn_status: str = "NOT_EVALUATED"
    von_neumann_entropy_rate: Optional[float] = None

    lyapunov_exponent: Optional[float] = None
    lyapunov_exponent_semantics: str = (
        "UNAVAILABLE; a state-residual decay proxy is not a tangent Lyapunov exponent"
    )
    spectral_radius: Optional[float] = None
    birkhoff_classification: str = "UNASSESSED"
    kam_diophantine_gamma: Optional[float] = None
    kam_stable: Optional[bool] = None
    kam_scan_status: str = "NOT_ASSESSED_NO_FREQUENCIES"
    poincare_crossings: int = 0
    fock_winding_numbers: Tuple[float, ...] = field(default_factory=tuple)
    contraction_final: float = 0.0
    meta_learning_final: float = 0.0
    l3_certified: bool = False
    l3_numerically_converged: bool = False
    l3_certification_status: str = "NOT_FORMALLY_CERTIFIED"

    von_neumann_entropy: float = 0.0
    purity_final: float = 1.0
    poincare_rotation_number: Optional[float] = None
    poincare_cartan_invariant: Optional[float] = None
    banach_kappa: Optional[float] = None
    banach_certified: bool = False
    fixed_step_contraction_bound: Optional[float] = None
    lyapunov_function_final: float = 0.0
    twist_coefficient: Optional[float] = None
    homoclinic_detected: Optional[bool] = None
    recurrence_candidate: Optional[bool] = None
    recurrence_status: str = "NOT_ASSESSED"
    mean_motion_resonances: int = 0
    characteristic_exponent_max: Optional[float] = None
    audit_status: str = "BASE_REPORT"


def _report_merkle_root(report: AnnihilationReactionReport) -> str:
    excluded = {"sha256_merkle_digest", "report_hmac"}
    named = [
        (item.name, getattr(report, item.name))
        for item in fields(report)
        if item.name not in excluded
    ]
    return _merkle_root_named_fields(named)


class FPUTensorFockCollider:
    """
    Motor de auditoría de bloques CAR/CCR, generador GKSL y contabilidad HMAC.

    `dimension` es la dimensión de la matriz de estado usada por el motor; no
    implica por sí misma una factorización física conjunta fermión-fotón.
    """

    def __init__(
        self,
        dimension: int = 56,
        secret_key: Optional[bytes] = None,
        gamma: float = 0.0025,
    ) -> None:
        self.dimension: Final[int] = _require_integer(dimension, "dimension", 4)

        gamma_value = float(gamma)
        if not math.isfinite(gamma_value) or gamma_value < 0.0:
            raise ValueError("gamma debe ser finito y no negativo.")
        self.gamma: Final[float] = gamma_value

        if secret_key is None:
            self._secret_key: Optional[bytes] = None
        else:
            key = bytes(secret_key)
            if len(key) < 32:
                raise ValueError("La clave HMAC debe tener al menos 32 bytes.")
            self._secret_key = key

        self._consumed_nonce_digests: set[str] = set()
        self._nonce_lock = threading.Lock()

    @property
    def security_key_configured(self) -> bool:
        """Indica si se configuró una clave explícita; no expone la clave."""
        return self._secret_key is not None

    @property
    def signing_key_id(self) -> str:
        if self._secret_key is None:
            return "UNSIGNED"
        return hashlib.sha256(self._secret_key).hexdigest()[:16]

    def project_density_matrix_with_diagnostics(
        self, matrix: NDArray
    ) -> Tuple[NDArray[np.complex128], DensityProjectionDiagnostics]:
        a = _as_square_matrix(matrix, "density_matrix")
        if a.shape != (self.dimension, self.dimension):
            raise ValueError(
                f"Se esperaba una matriz {(self.dimension, self.dimension)}; "
                f"se recibió {a.shape}. No se embeben ni truncan estados en silencio."
            )
        return _project_density_matrix_with_diagnostics(a)

    def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
        """Compatibilidad: devuelve la proyección métrica sobre estados de densidad."""
        rho, _ = self.project_density_matrix_with_diagnostics(matrix)
        return rho

    # ─── CAR: dos modos de Jordan–Wigner ──────────────────────────────────────
    def _construct_fermionic_operators(
        self,
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        r"""
        Construye los operadores de aniquilación:

            c₀ = σ₋ ⊗ I₂
            c₁ = σ_z ⊗ σ₋.

        CAR se verifica en el bloque superior izquierdo de dimensión 4. Para
        dimension > 4, el complemento es un sector desacoplado, no otra copia
        de la representación irreducible de dos modos.
        """
        sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
        i2 = np.eye(2, dtype=np.complex128)

        c0_block = np.kron(sm, i2)
        c1_block = np.kron(sz, sm)

        c0 = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        c1 = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        c0[:_FERMIONIC_SUBSPACE_DIM, :_FERMIONIC_SUBSPACE_DIM] = c0_block
        c1[:_FERMIONIC_SUBSPACE_DIM, :_FERMIONIC_SUBSPACE_DIM] = c1_block
        return c0, c1

    def verify_car_anti_commutation(
        self,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> Tuple[bool, float]:
        """Verifica {cᵢ,cⱼ†}=δᵢⱼI₄ en el bloque fermiónico; los estados se ignoran."""
        del state_a, state_b
        c0, c1 = self._construct_fermionic_operators()
        c0 = c0[:_FERMIONIC_SUBSPACE_DIM, :_FERMIONIC_SUBSPACE_DIM]
        c1 = c1[:_FERMIONIC_SUBSPACE_DIM, :_FERMIONIC_SUBSPACE_DIM]
        identity = np.eye(_FERMIONIC_SUBSPACE_DIM, dtype=np.complex128)

        residual = max(
            float(np.linalg.norm(c0 @ c0.conj().T + c0.conj().T @ c0 - identity, ord="fro")),
            float(np.linalg.norm(c0 @ c1.conj().T + c1.conj().T @ c0, ord="fro")),
            float(np.linalg.norm(c1 @ c0.conj().T + c0.conj().T @ c1, ord="fro")),
            float(np.linalg.norm(c1 @ c1.conj().T + c1.conj().T @ c1 - identity, ord="fro")),
        )
        return residual <= _ALGEBRA_RESIDUAL_TOL, residual

    def verify_car_nilpotency_and_mutual_anticommutator(self) -> Tuple[bool, float]:
        """Verifica cᵢ²=0 y {c₀,c₁}=0 en el bloque fermiónico."""
        c0, c1 = self._construct_fermionic_operators()
        c0 = c0[:_FERMIONIC_SUBSPACE_DIM, :_FERMIONIC_SUBSPACE_DIM]
        c1 = c1[:_FERMIONIC_SUBSPACE_DIM, :_FERMIONIC_SUBSPACE_DIM]

        residual = max(
            float(np.linalg.norm(c0 @ c0, ord="fro")),
            float(np.linalg.norm(c1 @ c1, ord="fro")),
            float(np.linalg.norm(c0 @ c1 + c1 @ c0, ord="fro")),
        )
        return residual <= _ALGEBRA_RESIDUAL_TOL, residual

    # ─── CCR: escalera bosónica truncada ──────────────────────────────────────
    def _construct_bosonic_operators(
        self, n_max: int = _BOSONIC_TRUNCATION_N
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        r"""
        Construye b y b† embebidos en el espacio ambiente.

        El bloque efectivo tiene d=min(n_max+1, dimension) niveles. En ese
        bloque, [b,b†]=I_d-d|d-1><d-1| es exacta. Si el ambiente es menor que
        la truncación solicitada, se audita explícitamente la truncación
        efectiva, no el nivel solicitado que no cabe.
        """
        n_max = _require_integer(n_max, "n_max", 0)
        d = min(n_max + 1, self.dimension)
        b = np.zeros((self.dimension, self.dimension), dtype=np.complex128)

        for n in range(1, d):
            b[n - 1, n] = math.sqrt(float(n))
        return b, b.conj().T

    def verify_ccr_commutation(
        self, n_max: int = _BOSONIC_TRUNCATION_N
    ) -> Tuple[bool, float]:
        """Verifica la CCR truncada en el bloque bosónico efectivo."""
        n_max = _require_integer(n_max, "n_max", 0)
        b, b_dag = self._construct_bosonic_operators(n_max)
        d = min(n_max + 1, self.dimension)

        b_block = b[:d, :d]
        b_dag_block = b_dag[:d, :d]
        commutator = b_block @ b_dag_block - b_dag_block @ b_block

        expected = np.eye(d, dtype=np.complex128)
        expected[-1, -1] -= float(d)
        residual = float(np.linalg.norm(commutator - expected, ord="fro"))
        return residual <= _ALGEBRA_RESIDUAL_TOL, residual

    def audit_algebra(self, n_max: int = _BOSONIC_TRUNCATION_N) -> FockAlgebraAudit:
        """Audita CAR/CCR en sus bloques y mide también el defecto ambiente."""
        n_max = _require_integer(n_max, "n_max", 0)

        c0, c1 = self._construct_fermionic_operators()
        d_f = _FERMIONIC_SUBSPACE_DIM
        c0_f, c1_f = c0[:d_f, :d_f], c1[:d_f, :d_f]
        i_f = np.eye(d_f, dtype=np.complex128)

        car_residual = max(
            float(np.linalg.norm(c0_f @ c0_f.conj().T + c0_f.conj().T @ c0_f - i_f, ord="fro")),
            float(np.linalg.norm(c0_f @ c1_f.conj().T + c1_f.conj().T @ c0_f, ord="fro")),
            float(np.linalg.norm(c1_f @ c0_f.conj().T + c0_f.conj().T @ c1_f, ord="fro")),
            float(np.linalg.norm(c1_f @ c1_f.conj().T + c1_f.conj().T @ c1_f - i_f, ord="fro")),
        )
        nilpotent_residual = max(
            float(np.linalg.norm(c0_f @ c0_f, ord="fro")),
            float(np.linalg.norm(c1_f @ c1_f, ord="fro")),
            float(np.linalg.norm(c0_f @ c1_f + c1_f @ c0_f, ord="fro")),
        )

        i_n = np.eye(self.dimension, dtype=np.complex128)
        car_ambient_residual = max(
            float(np.linalg.norm(c0 @ c0.conj().T + c0.conj().T @ c0 - i_n, ord="fro")),
            float(np.linalg.norm(c0 @ c1.conj().T + c1.conj().T @ c0, ord="fro")),
            float(np.linalg.norm(c1 @ c0.conj().T + c0.conj().T @ c1, ord="fro")),
            float(np.linalg.norm(c1 @ c1.conj().T + c1.conj().T @ c1 - i_n, ord="fro")),
        )

        b, b_dag = self._construct_bosonic_operators(n_max)
        d_b = min(n_max + 1, self.dimension)
        comm = b @ b_dag - b_dag @ b

        expected_block = np.eye(d_b, dtype=np.complex128)
        expected_block[-1, -1] -= float(d_b)
        ccr_residual = float(
            np.linalg.norm(comm[:d_b, :d_b] - expected_block, ord="fro")
        )

        expected_ambient = np.eye(self.dimension, dtype=np.complex128)
        expected_ambient[d_b - 1, d_b - 1] -= float(d_b)
        ccr_ambient_residual = float(
            np.linalg.norm(comm - expected_ambient, ord="fro")
        )

        return FockAlgebraAudit(
            dimension=self.dimension,
            fermionic_block_dimension=d_f,
            bosonic_block_dimension=d_b,
            car_residual=car_residual,
            car_nilpotent_residual=nilpotent_residual,
            ccr_residual=ccr_residual,
            car_valid_on_block=car_residual <= _ALGEBRA_RESIDUAL_TOL,
            ccr_valid_on_block=ccr_residual <= _ALGEBRA_RESIDUAL_TOL,
            car_ambient_residual=car_ambient_residual,
            ccr_ambient_residual=ccr_ambient_residual,
            car_valid_on_ambient_space=car_ambient_residual <= _ALGEBRA_RESIDUAL_TOL,
            ccr_valid_on_ambient_space=ccr_ambient_residual <= _ALGEBRA_RESIDUAL_TOL,
        )

    # ─── Generador GKSL y diagnósticos de entropía ────────────────────────────
    def _prepare_hamiltonian(self, hamiltonian_eff: Optional[NDArray]) -> NDArray[np.complex128]:
        if hamiltonian_eff is None:
            return np.zeros((self.dimension, self.dimension), dtype=np.complex128)

        h = _as_square_matrix(hamiltonian_eff, "hamiltonian_eff")
        if h.shape != (self.dimension, self.dimension):
            raise ValueError("hamiltonian_eff tiene dimensión incompatible.")

        residual = float(np.linalg.norm(h - h.conj().T, ord="fro"))
        scale = max(1.0, float(np.linalg.norm(h, ord="fro")))
        if residual > _DENSITY_VALIDATION_TOL * scale:
            raise ValueError("El Hamiltoniano efectivo debe ser hermítico.")
        return 0.5 * (h + h.conj().T)

    def _prepare_jump_operators(
        self,
        jump_operators: Optional[Sequence[NDArray]],
        gamma: Optional[float],
    ) -> Tuple[NDArray[np.complex128], ...]:
        if jump_operators is not None:
            prepared: List[NDArray[np.complex128]] = []
            for index, operator in enumerate(jump_operators):
                jump = _as_square_matrix(operator, f"jump_operators[{index}]")
                if jump.shape != (self.dimension, self.dimension):
                    raise ValueError(f"jump_operators[{index}] tiene dimensión incompatible.")
                prepared.append(jump)
            return tuple(prepared)

        rate = self.gamma if gamma is None else float(gamma)
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError("gamma debe ser finito y no negativo.")

        c0, _ = self._construct_fermionic_operators()
        return (math.sqrt(rate) * c0,)

    @staticmethod
    def _apply_gksl_raw(
        rho: NDArray[np.complex128],
        hamiltonian: NDArray[np.complex128],
        jumps: Sequence[NDArray[np.complex128]],
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Devuelve (L(ρ), D(ρ)) para un Hamiltoniano y saltos ya validados."""
        commutator_term = -1j * (hamiltonian @ rho - rho @ hamiltonian)
        dissipator = np.zeros_like(rho, dtype=np.complex128)

        for jump in jumps:
            jump_dag = jump.conj().T
            k = jump_dag @ jump
            dissipator += (
                jump @ rho @ jump_dag
                - 0.5 * (k @ rho + rho @ k)
            )
        return commutator_term + dissipator, dissipator

    def evaluate_gksl(
        self,
        density_matrix: NDArray,
        hamiltonian_eff: Optional[NDArray] = None,
        *,
        stationary_state: Optional[NDArray] = None,
        jump_operators: Optional[Sequence[NDArray]] = None,
        gamma: Optional[float] = None,
    ) -> GKSLDiagnostics:
        r"""
        Evalúa L(ρ)=-i[H,ρ]+D(ρ).

        Si se suministra σ, Spohn sólo se evalúa cuando:
          1. σ es un estado de densidad de rango numéricamente completo;
          2. ||L(σ)|| es pequeño dentro de la tolerancia;
          3. ρ también tiene rango numéricamente completo.

        No se recorta ni se oculta una producción de Spohn negativa.
        """
        rho, _ = self.project_density_matrix_with_diagnostics(density_matrix)
        h = self._prepare_hamiltonian(hamiltonian_eff)
        jumps = self._prepare_jump_operators(jump_operators, gamma)

        generator_action, dissipator = self._apply_gksl_raw(rho, h, jumps)
        trace_residual = abs(complex(np.trace(generator_action)))
        hermiticity_residual = float(
            np.linalg.norm(generator_action - generator_action.conj().T, ord="fro")
        )
        entropy_rate: Optional[float] = None
        spohn: Optional[float] = None
        spohn_valid: Optional[bool] = None
        stationary_residual: Optional[float] = None

        log_rho = _matrix_log_if_full_rank(rho)
        if log_rho is not None:
            entropy_rate = float(
                -np.real(np.trace(log_rho @ generator_action))
            )

        if stationary_state is None:
            spohn_status = "NO_STATIONARY_REFERENCE"
        else:
            sigma = _assert_density_matrix(
                stationary_state, name="stationary_state"
            )
            if sigma.shape != rho.shape:
                raise ValueError("stationary_state tiene dimensión incompatible.")

            sigma_action, _ = self._apply_gksl_raw(sigma, h, jumps)
            stationary_residual = float(
                np.linalg.norm(sigma_action, ord="fro")
            )

            if stationary_residual > _DENSITY_VALIDATION_TOL:
                spohn_status = "REFERENCE_NOT_STATIONARY"
            else:
                log_sigma = _matrix_log_if_full_rank(sigma)
                if log_sigma is None:
                    spohn_status = "REFERENCE_NOT_FULL_RANK"
                elif log_rho is None:
                    spohn_status = "STATE_NOT_FULL_RANK"
                else:
                    spohn = float(
                        -np.real(
                            np.trace((log_rho - log_sigma) @ generator_action)
                        )
                    )
                    spohn_valid = spohn >= -1.0e-9
                    spohn_status = (
                        "NONNEGATIVE_WITHIN_NUMERICAL_TOLERANCE"
                        if spohn_valid
                        else "NEGATIVE_CHECK_ASSUMPTIONS_AND_NUMERICS"
                    )

        return GKSLDiagnostics(
            generator_action=np.asarray(generator_action, dtype=np.complex128),
            dissipator=np.asarray(dissipator, dtype=np.complex128),
            trace_preserving_residual=float(trace_residual),
            hermiticity_preserving_residual=hermiticity_residual,
            dissipator_norm=float(np.linalg.norm(dissipator, ord="fro")),
            entropy_rate=entropy_rate,
            spohn_entropy_production=spohn,
            spohn_status=spohn_status,
            spohn_nonnegative_within_tolerance=spohn_valid,
            stationary_state_residual=stationary_residual,
        )

    def compute_lindblad_gksl_dissipation_and_spohn_entropy(
        self,
        density_matrix: NDArray,
        hamiltonian_eff: Optional[NDArray] = None,
        *,
        stationary_state: Optional[NDArray] = None,
        jump_operators: Optional[Sequence[NDArray]] = None,
        gamma: Optional[float] = None,
    ) -> Tuple[NDArray[np.complex128], Optional[float], float]:
        """Interfaz de compatibilidad: (L(ρ), σ_Spohn opcional, ||D(ρ)||₂)."""
        result = self.evaluate_gksl(
            density_matrix,
            hamiltonian_eff,
            stationary_state=stationary_state,
            jump_operators=jump_operators,
            gamma=gamma,
        )
        return (
            result.generator_action,
            result.spohn_entropy_production,
            result.dissipator_norm,
        )

    # ─── Autorización HMAC y sellado del informe ──────────────────────────────
    @staticmethod
    def _validate_auth_request(
        electron_anomalies: int,
        transaction_payload: str,
        nonce: str,
    ) -> Tuple[int, str, str]:
        count = _require_integer(electron_anomalies, "electron_anomalies", 1)
        if not isinstance(transaction_payload, str) or not transaction_payload.strip():
            raise ValueError("transaction_payload debe ser una cadena no vacía.")
        if not isinstance(nonce, str) or not nonce.strip():
            raise ValueError("Se requiere un nonce no vacío para autorizar una reacción.")
        return count, transaction_payload, nonce

    @staticmethod
    def _auth_message(
        electron_anomalies: int,
        transaction_payload: str,
        nonce: str,
    ) -> bytes:
        body = {
            "version": _MODULE_VERSION,
            "electron_anomalies": int(electron_anomalies),
            "transaction_payload": transaction_payload,
            "nonce": nonce,
        }
        return _AUTH_DOMAIN + _canonical_json_bytes(body)

    def issue_positron_auth(
        self,
        electron_anomalies: int,
        transaction_payload: str,
        nonce: str,
    ) -> str:
        """Firma una solicitud; úsese sólo en el sistema emisor de autorizaciones."""
        count, payload, request_nonce = self._validate_auth_request(
            electron_anomalies, transaction_payload, nonce
        )
        if self._secret_key is None:
            raise RuntimeError("No hay clave HMAC configurada.")
        message = self._auth_message(count, payload, request_nonce)
        return hmac.new(self._secret_key, message, hashlib.sha256).hexdigest()

    def _verify_and_consume_positron_auth(
        self,
        signature: str,
        electron_anomalies: int,
        transaction_payload: str,
        nonce: str,
    ) -> Tuple[bool, str, str]:
        nonce_digest = ""
        if isinstance(nonce, str) and nonce:
            nonce_digest = hashlib.sha256(
                _NONCE_DOMAIN + nonce.encode("utf-8")
            ).hexdigest()

        if self._secret_key is None:
            return False, "NO_HMAC_KEY_CONFIGURED", nonce_digest
        if not isinstance(signature, str):
            return False, "MALFORMED_SIGNATURE", nonce_digest

        try:
            count, payload, request_nonce = self._validate_auth_request(
                electron_anomalies, transaction_payload, nonce
            )
            received = bytes.fromhex(signature)
        except (TypeError, ValueError):
            return False, "MALFORMED_REQUEST_OR_SIGNATURE", nonce_digest

        if len(received) != hashlib.sha256().digest_size:
            return False, "MALFORMED_SIGNATURE", nonce_digest

        expected = hmac.new(
            self._secret_key,
            self._auth_message(count, payload, request_nonce),
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(received, expected):
            return False, "INVALID_SIGNATURE", nonce_digest

        with self._nonce_lock:
            if nonce_digest in self._consumed_nonce_digests:
                return False, "REPLAYED_NONCE_IN_THIS_PROCESS", nonce_digest
            self._consumed_nonce_digests.add(nonce_digest)

        return True, "AUTHORIZED_ONCE_IN_THIS_PROCESS", nonce_digest

    def _seal_report(
        self, report: AnnihilationReactionReport
    ) -> AnnihilationReactionReport:
        """
        Sella todos los campos semánticos del informe en un árbol Merkle.
        El HMAC firma la raíz; no se incluyen en el árbol ni la raíz ni la firma.
        """
        provisional = replace(
            report,
            sha256_merkle_digest="",
            report_hmac="",
            signing_key_id=self.signing_key_id,
            security_key_configured=self.security_key_configured,
        )
        root = _report_merkle_root(provisional)

        signature = ""
        if self._secret_key is not None:
            signature = hmac.new(
                self._secret_key,
                _REPORT_HMAC_DOMAIN + bytes.fromhex(root),
                hashlib.sha256,
            ).hexdigest()

        return replace(
            provisional,
            sha256_merkle_digest=root,
            report_hmac=signature,
        )

    def verify_report_signature(self, report: AnnihilationReactionReport) -> bool:
        """Verifica la raíz de contenido y el HMAC con la clave configurada."""
        if self._secret_key is None or not report.report_hmac:
            return False
        if report.signing_key_id != self.signing_key_id:
            return False

        expected_root = _report_merkle_root(report)
        if not hmac.compare_digest(expected_root, report.sha256_merkle_digest):
            return False

        expected_hmac = hmac.new(
            self._secret_key,
            _REPORT_HMAC_DOMAIN + bytes.fromhex(expected_root),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected_hmac, report.report_hmac)

    # ─── Contabilidad de reacción: no simula emisión física ──────────────────
    def collide_and_annihilate_dirac(
        self,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
        density_matrix: NDArray,
        *,
        nonce: str = "",
        hamiltonian_eff: Optional[NDArray] = None,
        stationary_state: Optional[NDArray] = None,
        jump_operators: Optional[Sequence[NDArray]] = None,
    ) -> AnnihilationReactionReport:
        """
        Emite un informe de contabilidad autenticada.

        Para una solicitud positiva, sólo un HMAC válido y un nonce no repetido
        autorizan el recuento. El nonce se recuerda sólo en memoria del proceso;
        producción requiere almacenamiento persistente y transaccional.
        """
        count = _require_integer(electron_anomalies, "electron_anomalies", 0)
        rho, projection = self.project_density_matrix_with_diagnostics(density_matrix)
        gksl = self.evaluate_gksl(
            rho,
            hamiltonian_eff,
            stationary_state=stationary_state,
            jump_operators=jump_operators,
        )

        now = float(time.time())
        reaction_id = f"DIRAC-{uuid.uuid4().hex}"

        if count == 0:
            auth_valid = False
            auth_status = "NO_REACTION_REQUESTED"
            nonce_digest = ""
            purged = 0
        else:
            auth_valid, auth_status, nonce_digest = (
                self._verify_and_consume_positron_auth(
                    positron_auth_signature,
                    count,
                    transaction_payload,
                    nonce,
                )
            )
            purged = count if auth_valid else 0

        rest_energy_equivalent = (
            float(purged) * _POSITRON_REST_ENERGY_KEV
        )
        if not math.isfinite(rest_energy_equivalent):
            raise OverflowError("La energía equivalente excede el rango float64.")

        payload_digest = (
            hashlib.sha256(transaction_payload.encode("utf-8")).hexdigest()
            if isinstance(transaction_payload, str)
            else ""
        )

        report = AnnihilationReactionReport(
            reaction_id=reaction_id,
            requested_anomalies=count,
            electron_anomalies_purged=purged,
            positron_auth_valid=auth_valid,
            released_energy_kev=rest_energy_equivalent,
            back_action_db=0.0,
            spohn_entropy_production=gksl.spohn_entropy_production,
            gksl_dissipation_rate=gksl.dissipator_norm,
            sha256_merkle_digest="",
            timestamp_utc=now,
            auth_status=auth_status,
            transaction_digest=payload_digest,
            nonce_digest=nonce_digest,
            security_key_configured=self.security_key_configured,
            source_state_sha256=_density_sha256(rho),
            state_projection_correction_norm=projection.correction_frobenius_norm,
            gksl_trace_residual=gksl.trace_preserving_residual,
            gksl_hermiticity_residual=gksl.hermiticity_preserving_residual,
            gksl_spohn_status=gksl.spohn_status,
            von_neumann_entropy_rate=gksl.entropy_rate,
            von_neumann_entropy=_von_neumann_entropy(rho),
            purity_final=_purity(rho),
            audit_status="REACTION_ACCOUNTING_COMPLETE",
        )
        return self._seal_report(report)

    # ╔══════════════════════════════════════════════════════════════════════════╗
    # ║ FRONTERA FASE 1 → FASE 2: export_fock_state_seed → FockStateSeed        ║
    # ╚══════════════════════════════════════════════════════════════════════════╝
    def export_fock_state_seed(
        self,
        density_matrix: NDArray,
        car_ok: bool,
        car_residual: float,
        ccr_ok: bool,
        ccr_residual: float,
        spohn_entropy: Optional[float],
        gksl_rate: float,
        *,
        target_density_matrix: Optional[NDArray] = None,
        section_probe: Optional[NDArray] = None,
        phase_observable: Optional[NDArray] = None,
        frequencies: Optional[Sequence[float]] = None,
        car_nilpotent_residual: float = 0.0,
    ) -> "FockStateSeed":
        r"""
        Serializa una semilla para el controlador numérico de FASE 2.

        `action_vector` contiene autovalores ordenados —pesos espectrales—;
        `angle_vector` contiene fases de un autovector con fijación de gauge
        dependiente de la base. Ninguno se declara acción o ángulo de Delaunay.

        Si no se proporciona un objetivo, I/d se usa sólo como referencia
        algorítmica. No se afirma que sea un estado estacionario del GKSL.
        Las frecuencias KAM no se inventan: sólo se incorporan si el usuario
        las suministra explícitamente.
        """
        rho = self.sanitize_density_matrix(density_matrix)
        dim = rho.shape[0]

        if target_density_matrix is None:
            target = np.eye(dim, dtype=np.complex128) / float(dim)
            target_semantics = "UNIFORM_ALGORITHMIC_REFERENCE_NOT_GKSL_STATIONARY"
        else:
            target, _ = self.project_density_matrix_with_diagnostics(
                target_density_matrix
            )
            target_semantics = "USER_SUPPLIED_ALGORITHMIC_REFERENCE"

        herm = 0.5 * (rho + rho.conj().T)
        eigvals, eigvecs = la.eigh(herm)
        weights = np.maximum(np.real(eigvals), 0.0)
        weights = np.sort(weights)[::-1]
        total = float(weights.sum())
        if total <= _WILKINSON_EPSILON:
            weights = np.full(dim, 1.0 / float(dim), dtype=np.float64)
        else:
            weights = weights / total

        dominant = eigvecs[:, -1].copy()
        gap = (
            float(np.real(eigvals[-1] - eigvals[-2]))
            if dim > 1
            else math.inf
        )
        gauge_defined = gap > 1.0e-10
        if gauge_defined:
            pivot = int(np.argmax(np.abs(dominant)))
            if abs(dominant[pivot]) > _WILKINSON_EPSILON:
                dominant *= np.exp(-1j * np.angle(dominant[pivot]))
                angle_vector = np.angle(dominant)
            else:
                gauge_defined = False
                angle_vector = np.zeros(dim, dtype=np.float64)
        else:
            angle_vector = np.zeros(dim, dtype=np.float64)

        source_hash = _density_sha256(rho)

        if section_probe is None:
            # Observable diagonal, hermítico, de traza cero y determinista.
            seed_bytes = bytes.fromhex(source_hash)
            diagonal = np.empty(dim, dtype=np.float64)
            for index in range(dim):
                block = hashlib.sha256(
                    seed_bytes + int(index).to_bytes(8, "big")
                ).digest()
                unit = int.from_bytes(block[:8], "big") / float(2**64)
                diagonal[index] = unit - 0.5
            diagonal -= float(np.mean(diagonal))
            norm = float(np.linalg.norm(diagonal))
            if norm <= _WILKINSON_EPSILON:
                diagonal = np.linspace(-1.0, 1.0, dim, dtype=np.float64)
                diagonal -= float(np.mean(diagonal))
                norm = float(np.linalg.norm(diagonal))
            probe = np.diag(diagonal / max(norm, _WILKINSON_EPSILON)).astype(
                np.complex128
            )
        else:
            probe = _as_square_matrix(section_probe, "section_probe")
            if probe.shape != (dim, dim):
                raise ValueError("section_probe tiene dimensión incompatible.")
            residual = float(np.linalg.norm(probe - probe.conj().T, ord="fro"))
            if residual > _DENSITY_VALIDATION_TOL * max(
                1.0, float(np.linalg.norm(probe, ord="fro"))
            ):
                raise ValueError("section_probe debe ser hermítico.")
            probe = 0.5 * (probe + probe.conj().T)

        if phase_observable is None:
            phase_op = np.zeros((dim, dim), dtype=np.complex128)
            if dim > 1:
                phase_op[0, 1] = 1.0
        else:
            phase_op = _as_square_matrix(phase_observable, "phase_observable")
            if phase_op.shape != (dim, dim):
                raise ValueError("phase_observable tiene dimensión incompatible.")

        freq = tuple(float(value) for value in (frequencies or ()))
        if any(not math.isfinite(value) for value in freq):
            raise ValueError("Las frecuencias deben ser reales y finitas.")

        return FockStateSeed(
            source_merkle=source_hash,
            rho_vector=tuple(complex(x) for x in rho.flatten()),
            target_vector=tuple(complex(x) for x in target.flatten()),
            probe_vector=tuple(complex(x) for x in probe.flatten()),
            action_vector=tuple(float(x) for x in weights),
            angle_vector=tuple(float(x) for x in angle_vector),
            frequency_vector=freq,
            car_residual=float(car_residual),
            ccr_residual=float(ccr_residual),
            car_valid=bool(car_ok),
            ccr_valid=bool(ccr_ok),
            spohn_entropy=spohn_entropy,
            gksl_rate=float(gksl_rate),
            dimension=dim,
            von_neumann_entropy=_von_neumann_entropy(rho),
            purity=_purity(rho),
            phase_observable_vector=tuple(complex(x) for x in phase_op.flatten()),
            target_semantics=target_semantics,
            frequency_source="USER_SUPPLIED" if freq else "NOT_SUPPLIED",
            dominant_eigenvector_gauge_defined=gauge_defined,
            car_nilpotent_residual=float(car_nilpotent_residual),
        )


# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2: Semilla, sección discreta, sondeos KAM/Poincaré y controlador RSI³           ║
# ║ La última definición de método es iterate_level_3; su resultado inicia FASE 3.      ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class FockStateSeed:
    r"""
    Semilla de entrada de FASE 2, producida por `export_fock_state_seed`.

    `source_merkle` se conserva por compatibilidad histórica; contiene un
    SHA-256 de contenido, no una raíz Merkle.
    """
    source_merkle: str
    rho_vector: Tuple[complex, ...]
    target_vector: Tuple[complex, ...]
    probe_vector: Tuple[complex, ...]
    action_vector: Tuple[float, ...]
    angle_vector: Tuple[float, ...]
    frequency_vector: Tuple[float, ...]
    car_residual: float
    ccr_residual: float
    car_valid: bool
    ccr_valid: bool
    spohn_entropy: Optional[float]
    gksl_rate: float
    dimension: int
    von_neumann_entropy: float = 0.0
    purity: float = 1.0
    phase_observable_vector: Tuple[complex, ...] = field(default_factory=tuple)
    target_semantics: str = "UNSPECIFIED_ALGORITHMIC_REFERENCE"
    frequency_source: str = "NOT_SUPPLIED"
    dominant_eigenvector_gauge_defined: bool = False
    car_nilpotent_residual: float = 0.0

    def __post_init__(self) -> None:
        dim = _require_integer(self.dimension, "dimension", 1)
        if len(self.rho_vector) != dim * dim:
            raise ValueError("rho_vector no tiene dim² componentes.")
        if len(self.target_vector) != dim * dim:
            raise ValueError("target_vector no tiene dim² componentes.")
        if len(self.probe_vector) != dim * dim:
            raise ValueError("probe_vector no tiene dim² componentes.")
        if len(self.action_vector) != dim or len(self.angle_vector) != dim:
            raise ValueError("action_vector y angle_vector deben tener longitud dim.")
        if self.phase_observable_vector and len(self.phase_observable_vector) != dim * dim:
            raise ValueError("phase_observable_vector no tiene dim² componentes.")

        rho = np.asarray(self.rho_vector, dtype=np.complex128).reshape((dim, dim))
        target = np.asarray(self.target_vector, dtype=np.complex128).reshape((dim, dim))
        probe = np.asarray(self.probe_vector, dtype=np.complex128).reshape((dim, dim))
        _assert_density_matrix(rho, name="seed.rho")
        _assert_density_matrix(target, name="seed.target")

        probe_residual = float(np.linalg.norm(probe - probe.conj().T, ord="fro"))
        if probe_residual > _DENSITY_VALIDATION_TOL * max(
            1.0, float(np.linalg.norm(probe, ord="fro"))
        ):
            raise ValueError("probe_vector debe representar un observable hermítico.")

        for name, values in (
            ("action_vector", self.action_vector),
            ("angle_vector", self.angle_vector),
            ("frequency_vector", self.frequency_vector),
        ):
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} contiene valores no finitos.")

    def get_rho(self) -> NDArray[np.complex128]:
        return np.asarray(self.rho_vector, dtype=np.complex128).reshape(
            (self.dimension, self.dimension)
        )

    def get_target(self) -> NDArray[np.complex128]:
        return np.asarray(self.target_vector, dtype=np.complex128).reshape(
            (self.dimension, self.dimension)
        )

    def get_probe(self) -> NDArray[np.complex128]:
        return np.asarray(self.probe_vector, dtype=np.complex128).reshape(
            (self.dimension, self.dimension)
        )

    def get_phase_observable(self) -> NDArray[np.complex128]:
        if self.phase_observable_vector:
            return np.asarray(
                self.phase_observable_vector, dtype=np.complex128
            ).reshape((self.dimension, self.dimension))
        return np.zeros((self.dimension, self.dimension), dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class FockSectionCrossing:
    """Intersección aproximada de una cuerda discreta con una sección."""

    start_index: int
    interpolation_fraction: float
    direction: str
    section_residual: float
    normalized_secant_slope: float
    dimension: int
    rho_vector: Tuple[complex, ...]

    def get_rho(self) -> NDArray[np.complex128]:
        return np.asarray(self.rho_vector, dtype=np.complex128).reshape(
            (self.dimension, self.dimension)
        )


class FockPoincareSection:
    r"""
    Sección discreta definida por g(ρ)=Re Tr(H_probe ρ)-c=0.

    La pendiente calculada con dos iterados es una pendiente secante, no una
    derivada de Lie de un flujo continuo.
    """

    def __init__(
        self,
        probe: NDArray[np.complex128],
        offset: float,
        transversality_tol: float = _SECTION_TOL,
    ) -> None:
        h = _as_square_matrix(probe, "probe")
        residual = float(np.linalg.norm(h - h.conj().T, ord="fro"))
        if residual > _DENSITY_VALIDATION_TOL * max(
            1.0, float(np.linalg.norm(h, ord="fro"))
        ):
            raise ValueError("La observable de sección debe ser hermítica.")

        offset_value = float(offset)
        tol_value = float(transversality_tol)
        if not math.isfinite(offset_value):
            raise ValueError("offset debe ser finito.")
        if not math.isfinite(tol_value) or tol_value < 0.0:
            raise ValueError("transversality_tol debe ser finita y no negativa.")
        if float(np.linalg.norm(h, ord="fro")) <= _WILKINSON_EPSILON:
            raise ValueError("La observable de sección no puede ser nula.")

        self.probe = 0.5 * (h + h.conj().T)
        self.offset = offset_value
        self.transversality_tol = tol_value

    def g(self, rho: NDArray[np.complex128]) -> float:
        a = _as_square_matrix(rho, "rho")
        if a.shape != self.probe.shape:
            raise ValueError("rho y probe tienen dimensiones incompatibles.")
        return float(np.real(np.trace(self.probe @ a)) - self.offset)

    def lie_derivative(self, d_rho: NDArray[np.complex128]) -> float:
        """Compatibilidad nominal: devuelve la derivada direccional dg[dρ]."""
        delta = _as_square_matrix(d_rho, "d_rho")
        if delta.shape != self.probe.shape:
            raise ValueError("d_rho y probe tienen dimensiones incompatibles.")
        return float(np.real(np.trace(self.probe @ delta)))

    def transversality(
        self,
        d_rho: NDArray[np.complex128],
        tol: Optional[float] = None,
    ) -> bool:
        delta = _as_square_matrix(d_rho, "d_rho")
        if delta.shape != self.probe.shape:
            raise ValueError("d_rho y probe tienen dimensiones incompatibles.")
        slope = abs(self.lie_derivative(delta))
        scale = max(
            _WILKINSON_EPSILON,
            float(np.linalg.norm(self.probe, ord="fro"))
            * float(np.linalg.norm(delta, ord="fro")),
        )
        threshold = self.transversality_tol if tol is None else float(tol)
        return slope / scale > threshold


class FockReturnMap:
    """
    Localizador de cruces de una secuencia discreta con una sección.

    `locate_crossings` interpola sobre la cuerda entre iterados; no calcula el
    mapa de retorno P:Σ→Σ ni su jacobiano.
    """

    def __init__(self, section: FockPoincareSection) -> None:
        self.section = section

    def locate_crossings(
        self,
        rho_sequence: Sequence[NDArray[np.complex128]],
    ) -> List[FockSectionCrossing]:
        events: List[FockSectionCrossing] = []
        if len(rho_sequence) < 2:
            return events

        states = [_as_square_matrix(rho, f"rho_sequence[{i}]")
                  for i, rho in enumerate(rho_sequence)]
        if any(state.shape != self.section.probe.shape for state in states):
            raise ValueError("La secuencia y la sección tienen dimensiones incompatibles.")

        g_values = [self.section.g(state) for state in states]
        tol = _SECTION_TOL

        for index in range(len(states) - 1):
            gi = g_values[index]
            gj = g_values[index + 1]
            fraction: Optional[float] = None

            if abs(gi) <= tol and (
                index == 0 or abs(g_values[index - 1]) > tol
            ):
                fraction = 0.0
            elif gi * gj < 0.0:
                denominator = gi - gj
                if abs(denominator) > _WILKINSON_EPSILON:
                    fraction = gi / denominator
            elif abs(gj) <= tol and abs(gi) > tol:
                fraction = 1.0

            if fraction is None or fraction < 0.0 or fraction > 1.0:
                continue

            delta = states[index + 1] - states[index]
            delta_norm = float(np.linalg.norm(delta, ord="fro"))
            if delta_norm <= _WILKINSON_EPSILON:
                continue
            if not self.section.transversality(delta):
                continue

            # La interpolación convexa preserva PSD y traza si los extremos son
            # estados de densidad; no se re-proyecta porque eso movería la sección.
            rho_cross = (1.0 - fraction) * states[index] + fraction * states[index + 1]
            residual = abs(self.section.g(rho_cross))
            dg = gj - gi
            normalized_slope = abs(dg) / max(
                _WILKINSON_EPSILON,
                float(np.linalg.norm(self.section.probe, ord="fro")) * delta_norm,
            )

            events.append(
                FockSectionCrossing(
                    start_index=index,
                    interpolation_fraction=float(fraction),
                    direction="INCREASING" if dg > 0.0 else "DECREASING",
                    section_residual=float(residual),
                    normalized_secant_slope=float(normalized_slope),
                    dimension=rho_cross.shape[0],
                    rho_vector=tuple(complex(x) for x in rho_cross.flatten()),
                )
            )
        return events

    def apply(
        self, rho_sequence: Sequence[NDArray[np.complex128]]
    ) -> List[NDArray[np.complex128]]:
        """Interfaz de compatibilidad que devuelve sólo las matrices de cruce."""
        return [event.get_rho() for event in self.locate_crossings(rho_sequence)]

    @staticmethod
    def rotation_number(angle_sequence: Sequence[float]) -> float:
        """
        Tasa de fase desenrollada por iterado.

        Es un diagnóstico de una señal angular proporcionada; no implica que
        exista una coordenada angular canónica del mapa.
        """
        if len(angle_sequence) < 2:
            return 0.0
        values = np.asarray(angle_sequence, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("angle_sequence debe ser finita.")
        unwrapped = np.unwrap(values)
        return float(
            (unwrapped[-1] - unwrapped[0])
            / (2.0 * math.pi * float(len(unwrapped) - 1))
        )


@dataclass(frozen=True, slots=True)
class KAMFiniteScan:
    """Resultado de una exploración finita de divisores pequeños."""

    status: str
    dimension_scanned: int
    tau: Optional[float]
    max_integer_order: int
    gamma_finite_scan: Optional[float]
    smallest_divisor: Optional[float]
    sampled_integer_vectors: int
    sampled_small_divisor_count: int
    mean_motion_resonances: int


class KAMFockTorus:
    r"""
    Exploración finita de combinaciones k·ω con soporte de k de tamaño <= 2.

    El mínimo de una lista finita no es una cota inferior para todos los
    k∈Zⁿ\{0}; por eso el resultado nunca certifica estabilidad KAM.
    """

    def __init__(self, frequencies: Sequence[float]) -> None:
        values = tuple(float(value) for value in frequencies)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Las frecuencias deben ser finitas.")
        self.frequencies = values
        self.n = len(values)

    def finite_order_scan(
        self,
        max_k: int = _KAM_MAX_K,
        tau: Optional[float] = None,
        small_divisor_threshold: float = _RESONANCE_THRESHOLD,
    ) -> KAMFiniteScan:
        max_k = _require_integer(max_k, "max_k", 1)
        threshold = float(small_divisor_threshold)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("small_divisor_threshold debe ser finito y no negativo.")

        if self.n == 0:
            return KAMFiniteScan(
                status="NOT_ASSESSED_NO_FREQUENCIES",
                dimension_scanned=0,
                tau=None,
                max_integer_order=max_k,
                gamma_finite_scan=None,
                smallest_divisor=None,
                sampled_integer_vectors=0,
                sampled_small_divisor_count=0,
                mean_motion_resonances=0,
            )

        n_eff = min(self.n, _KAM_EFFECTIVE_DIM)
        omega = np.asarray(self.frequencies[:n_eff], dtype=np.float64)
        tau_eff = float(max(0, n_eff - 1) if tau is None else tau)
        if not math.isfinite(tau_eff) or tau_eff < float(max(0, n_eff - 1)):
            raise ValueError("Se requiere tau >= n_eff - 1 para este sondeo.")

        vectors: List[Tuple[int, ...]] = []

        # Soporte de tamaño 1.
        for i in range(n_eff):
            for k in range(1, max_k + 1):
                vector = [0] * n_eff
                vector[i] = k
                vectors.append(tuple(vector))

        # Soporte de tamaño 2; se elimina la duplicación k ↔ -k.
        for i, j in combinations(range(n_eff), 2):
            for p in range(-max_k, max_k + 1):
                for q in range(-max_k, max_k + 1):
                    if p == 0 and q == 0:
                        continue
                    if p < 0 or (p == 0 and q < 0):
                        continue
                    vector = [0] * n_eff
                    vector[i] = p
                    vector[j] = q
                    vectors.append(tuple(vector))

        weighted_divisors: List[float] = []
        raw_divisors: List[float] = []
        small_count = 0
        for vector in vectors:
            k_vec = np.asarray(vector, dtype=np.float64)
            divisor = abs(float(np.dot(k_vec, omega)))
            norm_one = float(np.sum(np.abs(k_vec)))
            weighted = divisor / (norm_one ** tau_eff)
            raw_divisors.append(divisor)
            weighted_divisors.append(weighted)
            if divisor <= threshold:
                small_count += 1

        gamma = min(weighted_divisors) if weighted_divisors else None
        min_divisor = min(raw_divisors) if raw_divisors else None
        status = (
            "FINITE_ORDER_SCAN_ONLY"
            if self.n <= _KAM_EFFECTIVE_DIM
            else "FINITE_SCAN_PROJECTED_TO_FIRST_6_FREQUENCIES"
        )

        return KAMFiniteScan(
            status=status,
            dimension_scanned=n_eff,
            tau=tau_eff,
            max_integer_order=max_k,
            gamma_finite_scan=gamma,
            smallest_divisor=min_divisor,
            sampled_integer_vectors=len(vectors),
            sampled_small_divisor_count=small_count,
            mean_motion_resonances=self.mean_motion_resonance_count(
                max_order=max_k,
                threshold=threshold,
            ),
        )

    def diophantine_constant(
        self,
        max_k: int = _KAM_MAX_K,
        tau: Optional[float] = None,
    ) -> Optional[float]:
        """Compatibilidad: devuelve sólo el mínimo del sondeo finito."""
        return self.finite_order_scan(max_k=max_k, tau=tau).gamma_finite_scan

    def is_kam_stable(
        self,
        perturbation_norm: float,
        gamma: Optional[float] = None,
    ) -> Optional[bool]:
        """
        No certifica KAM con datos finitos; devuelve None deliberadamente.

        Los parámetros se conservan por compatibilidad. Un teorema aplicable
        requiere hipótesis del Hamiltoniano, regularidad, twist y constantes
        cuantitativas que no están representadas por esta clase.
        """
        del perturbation_norm, gamma
        return None

    def mean_motion_resonance_count(
        self,
        max_order: int = 5,
        threshold: float = _RESONANCE_THRESHOLD,
    ) -> int:
        """Cuenta pares con una relación pωᵢ≈qωⱼ en orden finito."""
        max_order = _require_integer(max_order, "max_order", 1)
        threshold = float(threshold)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("threshold debe ser finito y no negativo.")

        n_eff = min(self.n, _KAM_EFFECTIVE_DIM)
        omega = self.frequencies[:n_eff]
        count = 0
        for i, j in combinations(range(n_eff), 2):
            hit = False
            for p in range(1, max_order + 1):
                for q in range(1, max_order + 1):
                    divisor = abs(p * omega[i] - q * omega[j]) / float(p + q)
                    if divisor < threshold:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                count += 1
        return count

    @staticmethod
    def winding_numbers(
        angle_sequences: Sequence[Sequence[float]],
    ) -> Tuple[float, ...]:
        """
        Enrollamientos observados de series angulares proporcionadas por el
        usuario. No deriva enrollamientos de frecuencias sintéticas.
        """
        results: List[float] = []
        for sequence in angle_sequences:
            if len(sequence) < 2:
                results.append(0.0)
                continue
            values = np.asarray(sequence, dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError("Las series angulares deben ser finitas.")
            unwrapped = np.unwrap(values)
            results.append(
                float((unwrapped[-1] - unwrapped[0]) / (2.0 * math.pi))
            )
        return tuple(results)


class BirkhoffFockForm:
    """
    Auditoría conservadora de datos lineales suministrados externamente.

    Sin jacobiano de retorno y forma simpléctica verificada, la clasificación
    permanece no evaluada; un cociente de residuos de trayectoria no se trata
    como espectro de Floquet.
    """

    def __init__(
        self,
        fixed_rho: NDArray[np.complex128],
        jacobian_spectrum: Optional[Sequence[complex]] = None,
        twist_coefficient: Optional[float] = None,
        symplectic_residual: Optional[float] = None,
    ) -> None:
        self.fixed_rho = _as_square_matrix(fixed_rho, "fixed_rho")
        self.jacobian_spectrum = (
            np.asarray(tuple(jacobian_spectrum), dtype=np.complex128)
            if jacobian_spectrum is not None
            else np.asarray([], dtype=np.complex128)
        )
        self.twist_coefficient = (
            None if twist_coefficient is None else float(twist_coefficient)
        )
        self.symplectic_residual = (
            None if symplectic_residual is None else float(symplectic_residual)
        )

    def classify(self, symplectic_tol: float = 1.0e-8) -> str:
        if self.jacobian_spectrum.size == 0:
            return "UNASSESSED_NO_RETURN_MAP_JACOBIAN"
        if self.symplectic_residual is None:
            return "SPECTRUM_ONLY_NOT_BIRKHOFF"
        if self.symplectic_residual > symplectic_tol:
            return "SUPPLIED_MAP_NOT_VERIFIED_SYMPLECTIC"

        moduli = np.abs(self.jacobian_spectrum)
        if np.all(np.abs(moduli - 1.0) <= symplectic_tol):
            return "LINEAR_ELLIPTIC_CANDIDATE"
        return "LINEAR_HYPERBOLIC_OR_MIXED_CANDIDATE"

    def spectral_radius(self) -> Optional[float]:
        if self.jacobian_spectrum.size == 0:
            return None
        return float(np.max(np.abs(self.jacobian_spectrum)))

    def characteristic_exponents(self) -> Tuple[float, ...]:
        if self.jacobian_spectrum.size == 0:
            return tuple()
        return tuple(
            float(np.log(max(abs(value), _WILKINSON_EPSILON)))
            for value in self.jacobian_spectrum
        )

    def poincare_birkhoff_twist_holds(
        self, tol: float = 1.0e-12
    ) -> Optional[bool]:
        if self.twist_coefficient is None:
            return None
        return abs(self.twist_coefficient) > tol


class PoincareCelestialMechanics:
    """
    Funciones discretas para datos canónicos que el usuario proporcione.

    Las analogías de Poincaré-Cartan y twist sólo tienen su interpretación
    hamiltoniana usual si las entradas son realmente coordenadas canónicas y
    proceden de un flujo/mapa hamiltoniano apropiado.
    """

    @staticmethod
    def discrete_action_phase_integral(
        actions: Sequence[float],
        angles: Sequence[float],
    ) -> Optional[float]:
        """Suma discreta Σ Jₜ Δθₜ; no se declara invariante de un sistema disipativo."""
        n = min(len(actions), len(angles))
        if n < 2:
            return None
        j = np.asarray(actions[:n], dtype=np.float64)
        theta = np.unwrap(np.asarray(angles[:n], dtype=np.float64))
        if not np.all(np.isfinite(j)) or not np.all(np.isfinite(theta)):
            raise ValueError("actions y angles deben ser finitos.")
        return float(np.sum(j[:-1] * np.diff(theta)))

    @staticmethod
    def poincare_cartan_invariant(
        action_sequence: Sequence[float],
        angle_sequence: Sequence[float],
    ) -> Optional[float]:
        """Alias histórico; devuelve una suma discreta, no una invariancia probada."""
        return PoincareCelestialMechanics.discrete_action_phase_integral(
            action_sequence, angle_sequence
        )

    @staticmethod
    def generating_function_type2(
        angles: Sequence[float],
        new_actions: Sequence[float],
    ) -> float:
        """Producto bilineal Σ θᵢJ'ᵢ; no identifica por sí solo un mapa canónico."""
        n = min(len(angles), len(new_actions))
        return float(
            sum(float(angles[i]) * float(new_actions[i]) for i in range(n))
        )

    @staticmethod
    def twist_coefficient(
        actions: Sequence[float],
        angles: Sequence[float],
    ) -> Optional[float]:
        """
        Diferencia finita Δω/ΔJ para series canónicas suministradas.

        No debe aplicarse a la traza ni al autovalor dominante de una matriz de
        densidad como si fueran acciones de Delaunay.
        """
        n = min(len(actions), len(angles))
        if n < 3:
            return None

        j = np.asarray(actions[:n], dtype=np.float64)
        theta = np.unwrap(np.asarray(angles[:n], dtype=np.float64))
        if not np.all(np.isfinite(j)) or not np.all(np.isfinite(theta)):
            raise ValueError("actions y angles deben ser finitos.")

        omega = np.diff(theta)
        action_midpoints = 0.5 * (j[:-1] + j[1:])
        estimates: List[float] = []
        for index in range(len(omega) - 1):
            delta_j = action_midpoints[index + 1] - action_midpoints[index]
            if abs(float(delta_j)) > _WILKINSON_EPSILON:
                estimates.append(
                    float((omega[index + 1] - omega[index]) / delta_j)
                )
        return float(np.mean(estimates)) if estimates else None

    @staticmethod
    def recurrence_candidate(
        crossings: Sequence[NDArray[np.complex128]],
        hs_tol: float = _HOMOCLINIC_HS_TOL,
        min_index_separation: int = 2,
    ) -> Optional[bool]:
        """
        Busca retornos próximos en una muestra finita.

        Un retorno cercano no es una homoclínica ni demuestra recurrencia
        asintótica; False sólo significa que no se encontró un candidato en la
        muestra.
        """
        min_index_separation = _require_integer(
            min_index_separation, "min_index_separation", 1
        )
        tol = float(hs_tol)
        if not math.isfinite(tol) or tol <= 0.0:
            raise ValueError("hs_tol debe ser positivo y finito.")
        if len(crossings) < min_index_separation + 1:
            return None

        states = [_as_square_matrix(state, "crossing") for state in crossings]
        for first in range(len(states)):
            for second in range(first + min_index_separation, len(states)):
                if states[first].shape != states[second].shape:
                    raise ValueError("Los cruces tienen dimensiones incompatibles.")
                distance = float(
                    np.linalg.norm(states[first] - states[second], ord="fro")
                )
                if distance < tol:
                    return True
        return False

    @staticmethod
    def homoclinic_signature(
        crossings: Sequence[NDArray[np.complex128]],
        lyapunov: Optional[float],
        hs_tol: float = _HOMOCLINIC_HS_TOL,
        *,
        stable_manifold_points: Optional[Sequence[NDArray[np.complex128]]] = None,
        unstable_manifold_points: Optional[Sequence[NDArray[np.complex128]]] = None,
    ) -> Optional[bool]:
        """
        Interfaz histórica para un candidato homoclínico.

        Sin aproximaciones separadas de las variedades estable e inestable,
        devuelve None: cruces próximos y un exponente estimado no bastan.
        Incluso con ambas nubes de puntos, True significa candidato numérico,
        no prueba topológica ni transversalidad certificada.
        """
        if (
            stable_manifold_points is None
            or unstable_manifold_points is None
            or lyapunov is None
            or lyapunov <= 0.0
        ):
            return None

        stable = [_as_square_matrix(x, "stable_manifold_point")
                  for x in stable_manifold_points]
        unstable = [_as_square_matrix(x, "unstable_manifold_point")
                    for x in unstable_manifold_points]
        tol = float(hs_tol)
        for x in stable:
            for y in unstable:
                if x.shape != y.shape:
                    raise ValueError("Las variedades tienen dimensiones incompatibles.")
                if float(np.linalg.norm(x - y, ord="fro")) < tol:
                    return True
        return False


class RecursiveFockImprover:
    r"""
    Controlador adaptativo de tres niveles sobre matrices de densidad.

      L0: identidad.
      L1: relajación convexa y descenso de V(ρ)=||ρ-target||²_HS.
      L2: ajuste acotado del tamaño de paso usando el cociente de residuos.
      L3: adaptación de la tasa meta y amortiguación ante diferencias de tercer
          orden (jerk discreto) en el historial del tamaño de paso.

    Para θ fijo, el mapa Fθ(ρ)=(1-θ)ρ+θ target tiene contracción exacta
    1-θ en norma de Frobenius. La trayectoria adaptativa L2/L3 no se presenta
    como una aplicación fija global ni como certificación de Banach.
    """

    def __init__(
        self,
        contraction: float = 0.5,
        meta_learning: float = 0.3,
    ) -> None:
        theta = float(contraction)
        eta = float(meta_learning)
        if not math.isfinite(theta) or not math.isfinite(eta):
            raise ValueError("Los parámetros RSI³ deben ser finitos.")

        self._initial_contraction = float(
            np.clip(theta, _L3_THETA_MIN, _L3_THETA_MAX)
        )
        self._initial_meta_learning = float(
            np.clip(eta, _L3_META_MIN, _L3_META_MAX)
        )
        self.contraction = self._initial_contraction
        self.meta_learning = self._initial_meta_learning
        self._contraction_history: List[float] = []
        self._meta_history: List[float] = []
        self._last_accepted_step_size = self.contraction

    def _reset_run(self) -> None:
        """Aísla ejecuciones: el resultado de una auditoría no hereda otra."""
        self.contraction = self._initial_contraction
        self.meta_learning = self._initial_meta_learning
        self._contraction_history = []
        self._meta_history = []
        self._last_accepted_step_size = self.contraction

    def apply_level_0(self, rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.asarray(rho, dtype=np.complex128).copy()

    @staticmethod
    def _lyapunov_function(
        rho: NDArray[np.complex128],
        target: NDArray[np.complex128],
    ) -> float:
        delta = rho - target
        return float(np.real(np.vdot(delta.ravel(), delta.ravel())))

    def apply_level_1(
        self,
        rho: NDArray[np.complex128],
        target: NDArray[np.complex128],
        theta: Optional[float] = None,
    ) -> NDArray[np.complex128]:
        """Paso convexo con búsqueda de retroceso sobre una función de Lyapunov."""
        _assert_density_matrix(rho, name="rho")
        _assert_density_matrix(target, name="target")
        if rho.shape != target.shape:
            raise ValueError("rho y target tienen dimensiones incompatibles.")

        step = self.contraction if theta is None else float(theta)
        if not math.isfinite(step) or not (0.0 < step <= 1.0):
            raise ValueError("theta debe estar en (0,1].")

        v0 = self._lyapunov_function(rho, target)
        trial = step
        candidate = np.asarray(rho, dtype=np.complex128).copy()

        for _ in range(_LYAPUNOV_ARMIJO_ITERS):
            mixed = (1.0 - trial) * rho + trial * target
            v1 = self._lyapunov_function(mixed, target)
            required = v0 - _LYAPUNOV_ARMIJO_C * trial * v0

            if v1 <= required + 1.0e-14 * max(1.0, v0):
                candidate = mixed
                self._last_accepted_step_size = trial
                return np.asarray(candidate, dtype=np.complex128)

            trial *= 0.5

        self._last_accepted_step_size = 0.0
        return candidate

    def apply_level_2(
        self,
        rho: NDArray[np.complex128],
        target: NDArray[np.complex128],
        history: List[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        """Ajuste multiplicativo acotado usando el cociente observado de errores."""
        if len(history) >= 2:
            previous_error = float(
                np.linalg.norm(history[-2] - target, ord="fro")
            )
            current_error = float(np.linalg.norm(rho - target, ord="fro"))
            if previous_error > _WILKINSON_EPSILON:
                observed_ratio = current_error / previous_error
                predicted_ratio = 1.0 - self.contraction
                mismatch = float(
                    np.clip(observed_ratio - predicted_ratio, -0.25, 0.25)
                )
                factor = math.exp(self.meta_learning * mismatch)
                self.contraction = float(
                    np.clip(
                        self.contraction * factor,
                        _L3_THETA_MIN,
                        _L3_THETA_MAX,
                    )
                )

        result = self.apply_level_1(rho, target, self.contraction)
        if self._last_accepted_step_size > 0.0:
            self.contraction = float(
                np.clip(
                    self._last_accepted_step_size,
                    _L3_THETA_MIN,
                    _L3_THETA_MAX,
                )
            )
        return result

    def apply_level_3(
        self,
        rho: NDArray[np.complex128],
        target: NDArray[np.complex128],
        history: List[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        """Metaajuste L3 seguido de L2 y del paso convexo L1."""
        if len(history) >= 2:
            previous_error = float(
                np.linalg.norm(history[-2] - target, ord="fro")
            )
            current_error = float(np.linalg.norm(rho - target, ord="fro"))
            if previous_error > _WILKINSON_EPSILON:
                ratio = current_error / previous_error
                if ratio > 1.0 + 1.0e-8:
                    self.meta_learning = float(
                        np.clip(
                            0.90 * self.meta_learning,
                            _L3_META_MIN,
                            _L3_META_MAX,
                        )
                    )
                elif ratio < 0.25:
                    self.meta_learning = float(
                        np.clip(
                            1.05 * self.meta_learning,
                            _L3_META_MIN,
                            _L3_META_MAX,
                        )
                    )

        # Tercera diferencia finita: θ_t - 3θ_{t-1} + 3θ_{t-2} - θ_{t-3}.
        if len(self._contraction_history) >= 4:
            h = self._contraction_history
            jerk = h[-1] - 3.0 * h[-2] + 3.0 * h[-3] - h[-4]
            if abs(jerk) > _L3_JERK_TOL:
                self.meta_learning = float(
                    np.clip(
                        0.92 * self.meta_learning,
                        _L3_META_MIN,
                        _L3_META_MAX,
                    )
                )

        result = self.apply_level_2(rho, target, history)
        self._contraction_history.append(float(self.contraction))
        self._meta_history.append(float(self.meta_learning))
        return result

    @staticmethod
    def _estimate_local_spectral_radius(
        recent: Sequence[NDArray[np.complex128]],
    ) -> float:
        """
        Compatibilidad histórica: cociente de normas de incrementos.

        No es el radio espectral de un jacobiano.
        """
        if len(recent) < 3:
            return 0.0
        first = float(np.linalg.norm(recent[-2] - recent[-3], ord="fro"))
        second = float(np.linalg.norm(recent[-1] - recent[-2], ord="fro"))
        if first <= _WILKINSON_EPSILON:
            return 0.0 if second <= _WILKINSON_EPSILON else math.inf
        return second / first

    @staticmethod
    def _banach_kappa(
        rho_sequence: Sequence[NDArray[np.complex128]],
        target: NDArray[np.complex128],
    ) -> Optional[float]:
        """
        Máximo cociente observado en esta trayectoria; no una cota Lipschitz global.
        """
        ratios: List[float] = []
        for index in range(len(rho_sequence) - 1):
            d0 = float(np.linalg.norm(rho_sequence[index] - target, ord="fro"))
            d1 = float(np.linalg.norm(rho_sequence[index + 1] - target, ord="fro"))
            if d0 > _WILKINSON_EPSILON:
                ratio = d1 / d0
                if math.isfinite(ratio):
                    ratios.append(ratio)
                else:
                    return None
            elif d1 > _WILKINSON_EPSILON:
                return None
        return max(ratios) if ratios else None

    @staticmethod
    def _serialize(
        rho: NDArray[np.complex128],
    ) -> Tuple[complex, ...]:
        return tuple(complex(value) for value in rho.flatten())

    @staticmethod
    def deserialize(
        vector: Sequence[complex],
        dimension: int,
    ) -> NDArray[np.complex128]:
        dimension = _require_integer(dimension, "dimension", 1)
        if len(vector) != dimension * dimension:
            raise ValueError("El vector no tiene dimension² elementos.")
        result = np.asarray(vector, dtype=np.complex128).reshape(
            (dimension, dimension)
        )
        if not np.all(np.isfinite(result)):
            raise ValueError("El vector contiene NaN o infinitos.")
        return result

    @staticmethod
    def _leading_action(rho: NDArray[np.complex128]) -> float:
        """Compatibilidad: autovalor máximo, no acción canónica de Delaunay."""
        herm = 0.5 * (rho + rho.conj().T)
        eigvals = la.eigvalsh(herm)
        return float(np.max(np.real(eigvals))) if eigvals.size else 0.0

    # ╔══════════════════════════════════════════════════════════════════════════╗
    # ║ FRONTERA FASE 2 → FASE 3: iterate_level_3 → FockTrajectory              ║
    # ╚══════════════════════════════════════════════════════════════════════════╝
    def iterate_level_3(
        self,
        seed: FockStateSeed,
        max_iters: int = _L3_DEFAULT_ITERS,
        *,
        convergence_tolerance: float = _L3_CONVERGENCE_TOL,
    ) -> "FockTrajectory":
        """
        Produce una trayectoria numérica L3 y diagnósticos observables.

        El exponente legado `lyapunov_exponent` es la tasa media de cambio del
        error respecto al objetivo a lo largo de una órbita; no es el exponente
        de Lyapunov tangente de un flujo físico.
        """
        max_iters = _require_integer(max_iters, "max_iters", 0)
        tolerance = float(convergence_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("convergence_tolerance debe ser finita y no negativa.")

        self._reset_run()
        rho = seed.get_rho()
        target = seed.get_target()
        probe = seed.get_probe()
        phase_operator = seed.get_phase_observable()
        _assert_density_matrix(rho, name="seed.rho")
        _assert_density_matrix(target, name="seed.target")

        rho_sequence: List[NDArray[np.complex128]] = [rho.copy()]
        error_sequence: List[float] = [
            float(np.linalg.norm(rho - target, ord="fro"))
        ]
        lyapunov_values: List[float] = [
            self._lyapunov_function(rho, target)
        ]
        trace_sequence: List[float] = [float(np.real(np.trace(rho)))]
        leading_action_sequence: List[float] = [self._leading_action(rho)]
        entropy_sequence: List[float] = [_von_neumann_entropy(rho)]
        purity_sequence: List[float] = [_purity(rho)]
        history: List[NDArray[np.complex128]] = []

        for _ in range(max_iters):
            history.append(rho.copy())
            rho = self.apply_level_3(rho, target, history)
            rho_sequence.append(rho.copy())
            error_sequence.append(
                float(np.linalg.norm(rho - target, ord="fro"))
            )
            lyapunov_values.append(self._lyapunov_function(rho, target))
            trace_sequence.append(float(np.real(np.trace(rho))))
            leading_action_sequence.append(self._leading_action(rho))
            entropy_sequence.append(_von_neumann_entropy(rho))
            purity_sequence.append(_purity(rho))

        # Tasa de decaimiento del residuo de estado, no un exponente tangente.
        log_ratios: List[float] = []
        for previous, current in zip(error_sequence[:-1], error_sequence[1:]):
            if (
                previous > _WILKINSON_EPSILON
                and current > _WILKINSON_EPSILON
            ):
                ratio = current / previous
                if math.isfinite(ratio) and ratio > 0.0:
                    log_ratios.append(math.log(ratio))
        residual_exponent = (
            float(np.mean(log_ratios)) if log_ratios else None
        )

        observed_kappa = self._banach_kappa(rho_sequence, target)
        monotone_lyapunov = all(
            next_value <= value + 1.0e-12 * max(1.0, value)
            for value, next_value in zip(
                lyapunov_values[:-1], lyapunov_values[1:]
            )
        )
        final_error = error_sequence[-1]
        numerically_converged = bool(
            final_error <= tolerance and monotone_lyapunov
        )

        phase_values: List[float] = []
        phase_valid: List[bool] = []
        raw_phases: List[float] = []
        valid_indices: List[int] = []

        for index, state in enumerate(rho_sequence):
            signal = complex(np.trace(phase_operator @ state))
            is_valid = abs(signal) > _WILKINSON_EPSILON
            phase_valid.append(is_valid)
            if is_valid:
                raw_phases.append(float(np.angle(signal)))
                valid_indices.append(index)
            else:
                phase_values.append(0.0)

        if raw_phases:
            unwrapped = np.unwrap(np.asarray(raw_phases, dtype=np.float64))
            for index, angle in zip(valid_indices, unwrapped):
                while len(phase_values) <= index:
                    phase_values.append(0.0)
                phase_values[index] = float(angle)

        rotation_rate: Optional[float] = None
        if len(valid_indices) >= 2:
            first = valid_indices[0]
            last = valid_indices[-1]
            if last > first:
                first_phase = phase_values[first]
                last_phase = phase_values[last]
                rotation_rate = float(
                    (last_phase - first_phase)
                    / (2.0 * math.pi * float(last - first))
                )

        kam_scan = KAMFockTorus(seed.frequency_vector).finite_order_scan()
        step_bound: Optional[float] = None
        if self._contraction_history:
            step_bound = max(
                1.0 - theta for theta in self._contraction_history
            )

        return FockTrajectory(
            source_merkle=seed.source_merkle,
            dimension=seed.dimension,
            rho_sequence=tuple(self._serialize(state) for state in rho_sequence),
            action_sequence=tuple(trace_sequence),
            angle_sequence=tuple(phase_values),
            lyapunov_exponent=residual_exponent,
            convergence_residual=final_error,
            spectral_radius=None,
            is_kam_stable=None,
            winding_numbers=tuple(),
            diophantine_gamma=kam_scan.gamma_finite_scan,
            final_contraction=float(self.contraction),
            final_meta_learning=float(self.meta_learning),
            birkhoff_classification="UNASSESSED_NO_RETURN_MAP_JACOBIAN",
            leading_action_sequence=tuple(leading_action_sequence),
            von_neumann_entropy_final=entropy_sequence[-1],
            purity_final=purity_sequence[-1],
            poincare_rotation_number=rotation_rate,
            poincare_cartan_invariant=None,
            banach_kappa=observed_kappa,
            banach_certified=False,
            lyapunov_function_final=lyapunov_values[-1],
            twist_coefficient=None,
            mean_motion_resonances=kam_scan.mean_motion_resonances,
            characteristic_exponent_max=None,
            trace_sequence=tuple(trace_sequence),
            target_error_sequence=tuple(error_sequence),
            lyapunov_function_sequence=tuple(lyapunov_values),
            entropy_sequence=tuple(entropy_sequence),
            purity_sequence=tuple(purity_sequence),
            phase_valid_sequence=tuple(phase_valid),
            step_size_sequence=tuple(self._contraction_history),
            meta_learning_sequence=tuple(self._meta_history),
            monotone_lyapunov=monotone_lyapunov,
            numerically_converged=numerically_converged,
            fixed_step_contraction_bound=step_bound,
            kam_scan_status=kam_scan.status,
        )


# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3: Trayectoria, orquestación de auditoría y sellado del informe                ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class FockTrajectory:
    """Trayectoria RSI³; sus series no se interpretan automáticamente como órbita física."""

    source_merkle: str
    dimension: int
    rho_sequence: Tuple[Tuple[complex, ...], ...]
    action_sequence: Tuple[float, ...]
    angle_sequence: Tuple[float, ...]
    lyapunov_exponent: Optional[float]
    convergence_residual: float
    spectral_radius: Optional[float]
    is_kam_stable: Optional[bool]
    winding_numbers: Tuple[float, ...]
    diophantine_gamma: Optional[float]
    final_contraction: float
    final_meta_learning: float
    birkhoff_classification: str

    leading_action_sequence: Tuple[float, ...] = field(default_factory=tuple)
    von_neumann_entropy_final: float = 0.0
    purity_final: float = 1.0
    poincare_rotation_number: Optional[float] = None
    poincare_cartan_invariant: Optional[float] = None
    banach_kappa: Optional[float] = None
    banach_certified: bool = False
    lyapunov_function_final: float = 0.0
    twist_coefficient: Optional[float] = None
    mean_motion_resonances: int = 0
    characteristic_exponent_max: Optional[float] = None

    trace_sequence: Tuple[float, ...] = field(default_factory=tuple)
    target_error_sequence: Tuple[float, ...] = field(default_factory=tuple)
    lyapunov_function_sequence: Tuple[float, ...] = field(default_factory=tuple)
    entropy_sequence: Tuple[float, ...] = field(default_factory=tuple)
    purity_sequence: Tuple[float, ...] = field(default_factory=tuple)
    phase_valid_sequence: Tuple[bool, ...] = field(default_factory=tuple)
    step_size_sequence: Tuple[float, ...] = field(default_factory=tuple)
    meta_learning_sequence: Tuple[float, ...] = field(default_factory=tuple)
    monotone_lyapunov: bool = False
    numerically_converged: bool = False
    fixed_step_contraction_bound: Optional[float] = None
    kam_scan_status: str = "NOT_ASSESSED_NO_FREQUENCIES"

    def __post_init__(self) -> None:
        dim = _require_integer(self.dimension, "dimension", 1)
        if not self.rho_sequence:
            raise ValueError("rho_sequence no puede estar vacía.")
        if any(len(state) != dim * dim for state in self.rho_sequence):
            raise ValueError("Un estado serializado no tiene dimension² componentes.")
        if len(self.action_sequence) != len(self.rho_sequence):
            raise ValueError("action_sequence debe corresponder a rho_sequence.")
        if len(self.angle_sequence) != len(self.rho_sequence):
            raise ValueError("angle_sequence debe corresponder a rho_sequence.")
        if self.phase_valid_sequence and (
            len(self.phase_valid_sequence) != len(self.rho_sequence)
        ):
            raise ValueError("phase_valid_sequence tiene longitud incompatible.")

    def get_rho_sequence_arrays(self) -> List[NDArray[np.complex128]]:
        return [
            RecursiveFockImprover.deserialize(state, self.dimension)
            for state in self.rho_sequence
        ]

    def build_poincare_section(
        self,
        probe: NDArray[np.complex128],
        offset: Optional[float] = None,
    ) -> FockPoincareSection:
        states = self.get_rho_sequence_arrays()
        h = _as_square_matrix(probe, "probe")
        if h.shape != (self.dimension, self.dimension):
            raise ValueError("probe tiene dimensión incompatible.")
        herm_probe = 0.5 * (h + h.conj().T)

        values = [
            float(np.real(np.trace(herm_probe @ state)))
            for state in states
        ]
        section_offset = (
            float(np.mean(values)) if offset is None and values else
            (0.0 if offset is None else float(offset))
        )
        return FockPoincareSection(herm_probe, section_offset)

    def is_unstable_fock(self) -> bool:
        """Heurística sobre la tasa residual; no es estabilidad dinámica física."""
        return self.lyapunov_exponent is not None and self.lyapunov_exponent > 0.0

    def is_meta_stable_fock(self) -> bool:
        """Compatibilidad: comprueba convergencia numérica, no estabilidad KAM."""
        return bool(
            self.numerically_converged
            and self.monotone_lyapunov
            and self.lyapunov_exponent is not None
            and self.lyapunov_exponent < 0.0
        )


class SovereignFockAuditor:
    """
    Orquesta álgebra, contabilidad HMAC, semilla, iteración L3 y sondeos.

    Los resultados de Poincaré/KAM son diagnósticos finitos. `l3_certified`
    permanece False porque la trayectoria adaptativa no establece por sí sola
    una cota global de Banach ni un certificado físico.
    """

    def __init__(
        self,
        collider: FPUTensorFockCollider,
        improver: Optional[RecursiveFockImprover] = None,
    ) -> None:
        self.collider = collider
        self.improver = improver or RecursiveFockImprover(
            contraction=0.5,
            meta_learning=0.3,
        )

    def full_audit(
        self,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
        density_matrix: NDArray,
        max_iters: int = _L3_DEFAULT_ITERS,
        *,
        nonce: str = "",
        frequencies: Optional[Sequence[float]] = None,
        target_density_matrix: Optional[NDArray] = None,
        section_probe: Optional[NDArray] = None,
        hamiltonian_eff: Optional[NDArray] = None,
        stationary_state: Optional[NDArray] = None,
        jump_operators: Optional[Sequence[NDArray]] = None,
    ) -> AnnihilationReactionReport:
        # Validar y proyectar una sola vez la entrada que alimenta la trayectoria.
        rho, projection = self.collider.project_density_matrix_with_diagnostics(
            density_matrix
        )
        algebra = self.collider.audit_algebra()

        base_report = self.collider.collide_and_annihilate_dirac(
            electron_anomalies=electron_anomalies,
            positron_auth_signature=positron_auth_signature,
            transaction_payload=transaction_payload,
            density_matrix=rho,
            nonce=nonce,
            hamiltonian_eff=hamiltonian_eff,
            stationary_state=stationary_state,
            jump_operators=jump_operators,
        )

        # La primera frontera formal: fase 1 produce FockStateSeed.
        seed = self.collider.export_fock_state_seed(
            rho,
            algebra.car_valid_on_block,
            algebra.car_residual,
            algebra.ccr_valid_on_block,
            algebra.ccr_residual,
            base_report.spohn_entropy_production,
            base_report.gksl_dissipation_rate,
            target_density_matrix=target_density_matrix,
            section_probe=section_probe,
            frequencies=frequencies,
            car_nilpotent_residual=algebra.car_nilpotent_residual,
        )

        # La segunda frontera formal: fase 2 produce FockTrajectory.
        trajectory = self.improver.iterate_level_3(seed, max_iters=max_iters)

        section = trajectory.build_poincare_section(seed.get_probe())
        return_map = FockReturnMap(section)
        crossings = return_map.locate_crossings(
            trajectory.get_rho_sequence_arrays()
        )
        crossing_states = [event.get_rho() for event in crossings]
        recurrence_candidate = PoincareCelestialMechanics.recurrence_candidate(
            crossing_states
        )

        if recurrence_candidate is True:
            recurrence_status = "NEAR_RETURN_CANDIDATE_ONLY"
        elif recurrence_candidate is False:
            recurrence_status = "NO_NEAR_RETURN_FOUND_IN_FINITE_SAMPLE"
        else:
            recurrence_status = "INSUFFICIENT_CROSSINGS"

        # No se marca certificación Banach/KAM ni ausencia homoclínica por
        # inferencia a partir de una sola trayectoria finita.
        enriched = replace(
            base_report,
            sha256_merkle_digest="",
            report_hmac="",
            state_projection_correction_norm=projection.correction_frobenius_norm,
            car_residual=algebra.car_residual,
            ccr_residual=algebra.ccr_residual,
            car_valid=algebra.car_valid_on_block,
            ccr_valid=algebra.ccr_valid_on_block,
            car_nilpotent_residual=algebra.car_nilpotent_residual,
            car_ambient_residual=algebra.car_ambient_residual,
            ccr_ambient_residual=algebra.ccr_ambient_residual,
            car_valid_on_ambient_space=algebra.car_valid_on_ambient_space,
            ccr_valid_on_ambient_space=algebra.ccr_valid_on_ambient_space,
            lyapunov_exponent=trajectory.lyapunov_exponent,
            lyapunov_exponent_semantics=(
                "MEAN_LOG_RATIO_OF_TARGET_ERRORS; not a tangent Lyapunov exponent"
            ),
            spectral_radius=None,
            birkhoff_classification=trajectory.birkhoff_classification,
            kam_diophantine_gamma=trajectory.diophantine_gamma,
            kam_stable=None,
            kam_scan_status=trajectory.kam_scan_status,
            poincare_crossings=len(crossings),
            fock_winding_numbers=trajectory.winding_numbers,
            contraction_final=trajectory.final_contraction,
            meta_learning_final=trajectory.final_meta_learning,
            l3_certified=False,
            l3_numerically_converged=trajectory.numerically_converged,
            l3_certification_status=(
                "NOT_FORMALLY_CERTIFIED_ADAPTIVE_MAP_HAS_NO_GLOBAL_BANACH_PROOF"
            ),
            von_neumann_entropy=trajectory.von_neumann_entropy_final,
            purity_final=trajectory.purity_final,
            poincare_rotation_number=trajectory.poincare_rotation_number,
            poincare_cartan_invariant=None,
            banach_kappa=trajectory.banach_kappa,
            banach_certified=False,
            fixed_step_contraction_bound=trajectory.fixed_step_contraction_bound,
            lyapunov_function_final=trajectory.lyapunov_function_final,
            twist_coefficient=None,
            homoclinic_detected=None,
            recurrence_candidate=recurrence_candidate,
            recurrence_status=recurrence_status,
            mean_motion_resonances=trajectory.mean_motion_resonances,
            characteristic_exponent_max=None,
            audit_status="ALGEBRA_GKSL_AUTH_L3_DIAGNOSTICS_COMPLETE",
        )
        # El digest y la firma se recalculan después de incorporar todos los
        # resultados; la firma del informe base no se reutiliza.
        return self.collider._seal_report(enriched)


# ══════════════════════════════════════════════════════════════════════════════
# Pruebas de humo: verificaciones numéricas, no pruebas de teoremas físicos
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    banner = "═" * 96
    print(banner)
    print(f"FPU Tensor Fock Collider — {_MODULE_VERSION}")
    print("Smoke tests CAR/CCR + GKSL + HMAC/Merkle + trayectoria adaptativa")
    print(banner)

    # Clave sólo para demostración. No debe usarse en producción.
    test_only_key = b"DEVELOPMENT-ONLY-KEY-DO-NOT-USE-IN-PRODUCTION-32BYTES"
    collider = FPUTensorFockCollider(
        dimension=56,
        secret_key=test_only_key,
    )
    auditor = SovereignFockAuditor(collider)
    rng = np.random.default_rng(2026)

    vector = (
        rng.normal(size=(56, 1))
        + 1j * rng.normal(size=(56, 1))
    )
    density_sample = vector @ vector.conj().T
    density_sample /= np.trace(density_sample)

    algebra = collider.audit_algebra()
    print("\n── Auditoría algebraica por bloques ──")
    print(
        f"CAR bloque 4×4: {algebra.car_valid_on_block} "
        f"(residual={algebra.car_residual:.3e})"
    )
    print(
        f"CAR ambiente 56×56: {algebra.car_valid_on_ambient_space} "
        f"(residual={algebra.car_ambient_residual:.3e}; "
        "no esperado para embedding con ceros)"
    )
    print(
        f"CCR bloque efectivo {algebra.bosonic_block_dimension}×"
        f"{algebra.bosonic_block_dimension}: {algebra.ccr_valid_on_block} "
        f"(residual={algebra.ccr_residual:.3e})"
    )

    gksl = collider.evaluate_gksl(density_sample)
    print("\n── GKSL ──")
    print(f"Residual de traza: {gksl.trace_preserving_residual:.3e}")
    print(f"||D(ρ)||₂: {gksl.dissipator_norm:.6e}")
    print(f"Estado de Spohn: {gksl.spohn_status}")
    print(
        "σ_Spohn:",
        "N/A" if gksl.spohn_entropy_production is None
        else f"{gksl.spohn_entropy_production:.6e}",
    )

    payload = "TRANSACTION-DEMO-001"
    nonce = "NONCE-DEMO-001"
    signature = collider.issue_positron_auth(
        electron_anomalies=2,
        transaction_payload=payload,
        nonce=nonce,
    )

    report = auditor.full_audit(
        electron_anomalies=2,
        positron_auth_signature=signature,
        transaction_payload=payload,
        density_matrix=density_sample,
        nonce=nonce,
        max_iters=6,
    )

    print("\n── Informe de auditoría ──")
    print(f"Autorización: {report.positron_auth_valid} ({report.auth_status})")
    print(f"Pares contabilizados: {report.electron_anomalies_purged}")
    print(
        "Equivalencia de masa en reposo:",
        f"{report.released_energy_kev:.1f} keV "
        "(no es energía medida de fotones)",
    )
    print(f"Exponente residual: {report.lyapunov_exponent}")
    print(f"Estado KAM: {report.kam_scan_status}; estable={report.kam_stable}")
    print(f"Banach certificado: {report.banach_certified}")
    print(f"L3 certificado formalmente: {report.l3_certified}")
    print(f"L3 convergencia numérica: {report.l3_numerically_converged}")
    print(f"Homoclínica: {report.homoclinic_detected}")
    print(f"Candidato de retorno cercano: {report.recurrence_candidate}")
    print(f"Raíz Merkle: {report.sha256_merkle_digest}")
    print(f"HMAC de informe válido: {collider.verify_report_signature(report)}")

    assert algebra.car_valid_on_block
    assert algebra.ccr_valid_on_block
    assert gksl.trace_preserving_residual < 1.0e-10
    assert report.positron_auth_valid
    assert report.electron_anomalies_purged == 2
    assert collider.verify_report_signature(report)
    assert report.l3_certified is False

    bad_report = auditor.full_audit(
        electron_anomalies=2,
        positron_auth_signature="BAD_SIGNATURE",
        transaction_payload=payload,
        density_matrix=density_sample,
        nonce="NONCE-DEMO-BAD",
        max_iters=2,
    )
    assert not bad_report.positron_auth_valid
    assert bad_report.electron_anomalies_purged == 0

    print("\nPruebas de humo completadas; no constituyen certificación física o matemática.")
    print(banner)
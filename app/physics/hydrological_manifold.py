# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Hydrological Manifold — Evolución Doctoral en 3 Fases Anidadas      ║
║ Ruta   : app/physics/hydrological_manifold.py                                ║
║ Versión: 3.0.0-Richards-Terzaghi-Biot-DEC-KBN-Tikhonov-Spectral-Governance   ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS
========
Colector Hidrológico de de Rham–Richards para el acoplamiento hidro-geomecánico
en el Estrato Physics de APU Filter. El ciclo se formaliza como composición
de funtores de fase sobre un complejo simplicial (grafo orientado) K:

    Datos_crudos --Φ₁→ (P_f, σ', K_e, χ, σ₁)
                 --Φ₂→ (H, Q, L, i, σ₂)
                 --Φ₃→ (HydrologicalState, Σ, Γ)

Física constitutiva (Fase 1)
----------------------------
Presión de poros de Bishop–Biot, con succión ψ ≥ 0 y saturación S ∈ [0, 1]:

    u_w = −γ_w ψ,     P_f = χ u_w,     χ = α_Biot · S,
    σ'  = σ − α_Biot P_f I = σ + α_Biot S γ_w ψ I.

Conductividad relativa de Mualem–van Genuchten (m ∈ (0, 1), n = 1/(1−m)):

    K_r(S_e) = S_e^L [1 − (1 − S_e^{1/m})^m ]²,     K = K_sat K_r.

DEC / Richards (Fase 2)
-----------------------
B ∈ R^{n×m} es la matriz de incidencia (coborde d₀* ). La estrella de Hodge
primal en 1-formas se representa por W = diag(K_e) (conductancias de arista).
El Laplaciano de Hodge de 0-formas es

    Δ₀ = δ d = B W Bᵀ = L.

Estado estacionario de Richards: L H = s, con H = ψ + z. Equivale a

    L ψ = s − L z,     Q = W Bᵀ H.

Kirchhoff: B Q = s. Compatibilidad cohomológica: 1ᵀ s = 0 sobre cada
componente conexa (H⁰_dR ≅ R^{β₀}). Tikhonov L + λI fija el gauge.

Sifonamiento de Terzaghi: i_e = |ΔH_e|/L_e  vs  i_crit = (ρ_sat − ρ_w)/ρ_w.

Espectro / gobernanza (Fase 3)
------------------------------
σ(L) ⊂ [0, ∞), nulidad = β₀, conectividad algebraica λ₂ (Fiedler),
auditoría de licuación (σ'₃ ≤ 0, p' ≤ 0) y sello SHA-256 canónico.

CONTINUIDAD FORMAL
==================
    Φ₁→₂ : Phase1HydroHandoff → Fase 2
    Φ₂→₃ : Phase2HydroHandoff → Fase 3

El último método de la Fase 1 *es* la apertura verificada de la Fase 2.
El último método de la Fase 2 *es* la apertura verificada de la Fase 3.
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Final, Mapping, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "HydrologicalManifold",
    "HydrologicalState",
    "Phase1HydroHandoff",
    "Phase2HydroHandoff",
    "Phase3HydroReport",
]

__version__: Final[str] = (
    "3.0.0-Richards-Terzaghi-Biot-DEC-KBN-Tikhonov-Spectral-Governance"
)

logger = logging.getLogger("APU.Physics.HydrologicalManifold")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_SHA256_HEX_LEN: Final[int] = 64
_PHASE1_ENTRY: Final[str] = "phase2_from_phase1"
_PHASE2_ENTRY: Final[str] = "phase3_from_phase2"
_I3: Final[np.ndarray] = np.eye(3, dtype=np.float64)

IncidenceLike = Union[np.ndarray, "sp.spmatrix"]


# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES NUMÉRICAS, CANÓNICAS Y CRIPTOGRÁFICAS
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


def _canonical_bytes(arr: np.ndarray) -> bytes:
    """Bytes contiguos con prefijo de dtype y forma, libres de colisión trivial."""
    a = np.ascontiguousarray(arr)
    if np.issubdtype(a.dtype, np.floating):
        a = _canonicalize_signed_zero(np.array(a, copy=True))
    header = f"{a.dtype.str}|{a.shape}".encode("utf-8")
    return len(header).to_bytes(8, "little") + header + a.tobytes()


def _pack_f64(value: float) -> bytes:
    """Serialización little-endian de float64 con centinelas IEEE-754."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return b"\x00\x00\x00\x00\x00\x00\xf8\x7f"
    x = float(value)
    if math.isnan(x):
        return b"\x00\x00\x00\x00\x00\x00\xf8\x7f"
    if x == math.inf:
        return struct.pack("<d", math.inf)
    if x == -math.inf:
        return struct.pack("<d", -math.inf)
    return struct.pack("<d", x)


def _sha_update_str(hasher: "hashlib._Hash", text: str) -> None:
    payload = text.encode("utf-8")
    hasher.update(len(payload).to_bytes(8, "little"))
    hasher.update(payload)


def _sha_update_arr(hasher: "hashlib._Hash", arr: np.ndarray) -> None:
    payload = _canonical_bytes(arr)
    hasher.update(len(payload).to_bytes(8, "little"))
    hasher.update(payload)


def _sha256_hex_with_token(phase_token: str, *arrays: np.ndarray) -> str:
    """Firma SHA-256 canónica longitud-prefijada, invariante por fase."""
    h = hashlib.sha256()
    h.update(b"HYDRO/v3")
    _sha_update_str(h, phase_token)
    for arr in arrays:
        _sha_update_arr(h, np.asarray(arr))
    return h.hexdigest()


def _kbn_sum(arr: np.ndarray) -> float:
    """Sumación compensada Kahan–Babuška–Neumaier."""
    s = 0.0
    c = 0.0
    for x in np.asarray(arr, dtype=np.float64).ravel():
        x = float(x)
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
    return float(s + c)


def _finite_or_nan(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def _is_sparse(mat: Any) -> bool:
    return sp.issparse(mat)


# ═════════════════════════════════════════════════════════════════════════════
# OBJETOS DE ESTADO Y FRONTERAS DE FASE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True, eq=False)
class HydrologicalState:
    """
    Estado físico-espectral hidrológico verificado en la FPU.

    Invariantes
    -----------
    L = Lᵀ ⪰ 0 (Laplaciano de Hodge de 0-formas)
    B Q ≈ s     (ley de Kirchhoff / conservación de masa)
    H = ψ + z   (potencial total de Richards)
    """

    hydraulic_head: np.ndarray
    pore_pressures: np.ndarray
    effective_stresses: np.ndarray
    flow_rates: np.ndarray
    laplacian_eigenvalues: np.ndarray
    is_siphoning_active: bool
    is_liquefaction_active: bool
    sha256_hash: str

    phase_chain: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hydraulic_head", _freeze_array(self.hydraulic_head))
        object.__setattr__(self, "pore_pressures", _freeze_array(self.pore_pressures))
        object.__setattr__(
            self, "effective_stresses", _freeze_array(self.effective_stresses)
        )
        object.__setattr__(self, "flow_rates", _freeze_array(self.flow_rates))
        object.__setattr__(
            self, "laplacian_eigenvalues", _freeze_array(self.laplacian_eigenvalues)
        )

    def __hash__(self) -> int:
        return hash(self.sha256_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HydrologicalState):
            return NotImplemented
        return self.sha256_hash == other.sha256_hash

    def __repr__(self) -> str:
        return (
            f"HydrologicalState(siphoning={self.is_siphoning_active}, "
            f"liquefaction={self.is_liquefaction_active}, "
            f"seal={self.sha256_hash[:12]!r})"
        )


@dataclass(frozen=True, slots=True, eq=False)
class Phase1HydroHandoff:
    """Frontera formal Φ₁→₂: constitución local validada."""

    incidence_matrix: Any
    edge_lengths: np.ndarray
    suction: np.ndarray
    sat: np.ndarray
    K_sat_edges: np.ndarray
    total_stresses: np.ndarray
    s_pumps: np.ndarray
    node_elevations: np.ndarray
    pore_pressures: np.ndarray
    effective_stresses: np.ndarray
    is_liquefaction_active: bool
    min_principal_stresses: np.ndarray
    determinant_effective_stresses: np.ndarray
    edge_saturations: np.ndarray
    edge_conductivities: np.ndarray
    diagnostics: Dict[str, Any]
    next_entrypoint: str
    session_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_lengths", _freeze_array(self.edge_lengths))
        object.__setattr__(self, "suction", _freeze_array(self.suction))
        object.__setattr__(self, "sat", _freeze_array(self.sat))
        object.__setattr__(self, "K_sat_edges", _freeze_array(self.K_sat_edges))
        object.__setattr__(self, "total_stresses", _freeze_array(self.total_stresses))
        object.__setattr__(self, "s_pumps", _freeze_array(self.s_pumps))
        object.__setattr__(self, "node_elevations", _freeze_array(self.node_elevations))
        object.__setattr__(self, "pore_pressures", _freeze_array(self.pore_pressures))
        object.__setattr__(
            self, "effective_stresses", _freeze_array(self.effective_stresses)
        )
        object.__setattr__(
            self, "min_principal_stresses", _freeze_array(self.min_principal_stresses)
        )
        object.__setattr__(
            self,
            "determinant_effective_stresses",
            _freeze_array(self.determinant_effective_stresses),
        )
        object.__setattr__(
            self, "edge_saturations", _freeze_array(self.edge_saturations)
        )
        object.__setattr__(
            self, "edge_conductivities", _freeze_array(self.edge_conductivities)
        )

    def __hash__(self) -> int:
        return hash((self.session_sha256, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase1HydroHandoff):
            return NotImplemented
        return (
            self.session_sha256 == other.session_sha256
            and self.next_entrypoint == other.next_entrypoint
        )


@dataclass(frozen=True, slots=True, eq=False)
class Phase2HydroHandoff:
    """Frontera formal Φ₂→₃: solución DEC de Richards y auditoría de flujo."""

    incidence_matrix: Any
    edge_lengths: np.ndarray
    suction: np.ndarray
    sat: np.ndarray
    K_sat_edges: np.ndarray
    total_stresses: np.ndarray
    s_pumps: np.ndarray
    node_elevations: np.ndarray
    pore_pressures: np.ndarray
    effective_stresses: np.ndarray
    is_liquefaction_active: bool
    min_principal_stresses: np.ndarray
    determinant_effective_stresses: np.ndarray
    edge_saturations: np.ndarray
    edge_conductivities: np.ndarray
    laplacian_matrix: Any
    hydraulic_head: np.ndarray
    flow_rates: np.ndarray
    is_siphoning_active: bool
    hydraulic_gradients: np.ndarray
    diagnostics: Dict[str, Any]
    next_entrypoint: str
    session_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_lengths", _freeze_array(self.edge_lengths))
        object.__setattr__(self, "suction", _freeze_array(self.suction))
        object.__setattr__(self, "sat", _freeze_array(self.sat))
        object.__setattr__(self, "K_sat_edges", _freeze_array(self.K_sat_edges))
        object.__setattr__(self, "total_stresses", _freeze_array(self.total_stresses))
        object.__setattr__(self, "s_pumps", _freeze_array(self.s_pumps))
        object.__setattr__(self, "node_elevations", _freeze_array(self.node_elevations))
        object.__setattr__(self, "pore_pressures", _freeze_array(self.pore_pressures))
        object.__setattr__(
            self, "effective_stresses", _freeze_array(self.effective_stresses)
        )
        object.__setattr__(
            self, "min_principal_stresses", _freeze_array(self.min_principal_stresses)
        )
        object.__setattr__(
            self,
            "determinant_effective_stresses",
            _freeze_array(self.determinant_effective_stresses),
        )
        object.__setattr__(
            self, "edge_saturations", _freeze_array(self.edge_saturations)
        )
        object.__setattr__(
            self, "edge_conductivities", _freeze_array(self.edge_conductivities)
        )
        object.__setattr__(self, "hydraulic_head", _freeze_array(self.hydraulic_head))
        object.__setattr__(self, "flow_rates", _freeze_array(self.flow_rates))
        object.__setattr__(
            self, "hydraulic_gradients", _freeze_array(self.hydraulic_gradients)
        )

    def __hash__(self) -> int:
        return hash((self.session_sha256, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase2HydroHandoff):
            return NotImplemented
        return (
            self.session_sha256 == other.session_sha256
            and self.next_entrypoint == other.next_entrypoint
            and self.is_siphoning_active == other.is_siphoning_active
            and self.is_liquefaction_active == other.is_liquefaction_active
        )


@dataclass(frozen=True, slots=True, eq=False)
class Phase3HydroReport:
    """Reporte final de la Fase 3: estado, espectro, estabilidad y sello."""

    state: HydrologicalState
    spectral_audit: Dict[str, Any]
    stability_audit: Dict[str, Any]
    governance_seal: Dict[str, Any]
    phase_chain: Tuple[str, ...]

    def __hash__(self) -> int:
        return hash(self.state.sha256_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase3HydroReport):
            return NotImplemented
        return self.state.sha256_hash == other.state.sha256_hash


# ═════════════════════════════════════════════════════════════════════════════
# COLECTOR HIDROLÓGICO PRINCIPAL — TRES FASES ANIDADAS
# ═════════════════════════════════════════════════════════════════════════════

class HydrologicalManifold:
    """
    Colector Hidrológico de de Rham–Richards (FPU Secure) — 3 fases anidadas.

    FASE 1  CONSTITUTIVE : Biot–Terzaghi, Bishop, Mualem–van Genuchten.
    FASE 2  DEC/RICHARDS : Hodge Δ₀, Kirchhoff, Tikhonov, sifonamiento.
    FASE 3  SPECTRAL/GOV : espectro de Fiedler, licuación, sello SHA-256.
    """

    def __init__(
        self,
        tolerance: float = 1e-12,
        regularization: float = 1e-15,
        gravity: float = 9.81,
        max_dense_eigen_dim: int = 256,
        spectral_k: int = 20,
    ) -> None:
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance debe ser finita y estrictamente positiva.")
        if not math.isfinite(regularization) or regularization <= 0.0:
            raise ValueError(
                "regularization debe ser finita y estrictamente positiva."
            )
        if not math.isfinite(gravity) or gravity <= 0.0:
            raise ValueError("gravity debe ser finita y estrictamente positiva.")
        if max_dense_eigen_dim <= 0:
            raise ValueError("max_dense_eigen_dim debe ser positiva.")
        if spectral_k <= 0:
            raise ValueError("spectral_k debe ser positiva.")

        self._tol = float(tolerance)
        self._reg = float(regularization)
        self._gravity = float(gravity)
        self._max_dense_eigen_dim = int(max_dense_eigen_dim)
        self._spectral_k = int(spectral_k)

    def _tolerance_of(self, scale: float = 1.0) -> float:
        return max(self._tol, self._tol * abs(scale), 32.0 * _MACHINE_EPS)

    # ═════════════════════════════════════════════════════════════════════════
    # VALIDADORES CANÓNICOS (infraestructura de Fase 1)
    # ═════════════════════════════════════════════════════════════════════════

    def _validate_vector(
        self,
        name: str,
        arr: np.ndarray,
        length: Optional[int] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> np.ndarray:
        """Valida un vector real finito, con longitud opcional y cotas físicas."""
        a = np.asarray(arr, dtype=np.float64).ravel()
        if length is not None and a.size != length:
            raise ValueError(
                f"El campo '{name}' debe tener longitud {length}. Obtenido: {a.size}"
            )
        if not np.all(np.isfinite(a)):
            raise ValueError(f"El campo '{name}' contiene valores NaN o infinitos.")
        if lower is not None and np.any(a < lower):
            raise ValueError(f"El campo '{name}' viola la cota inferior {lower}.")
        if upper is not None and np.any(a > upper):
            raise ValueError(f"El campo '{name}' viola la cota superior {upper}.")
        return _canonicalize_signed_zero(a)

    def _broadcast_edge_array(
        self,
        name: str,
        values: np.ndarray,
        n_edges: int,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> np.ndarray:
        """Valida y difunde un campo de aristas (escalar o vector de longitud m)."""
        a = np.asarray(values, dtype=np.float64).ravel()
        if n_edges == 0:
            if a.size not in (0, 1):
                raise ValueError(
                    f"El campo '{name}' debe ser vacío o escalar si no hay aristas."
                )
            return np.empty(0, dtype=np.float64)
        if a.size == 1:
            a = np.full(n_edges, float(a[0]), dtype=np.float64)
        elif a.size != n_edges:
            raise ValueError(
                f"El campo '{name}' debe ser escalar o tener longitud {n_edges}. "
                f"Obtenido: {a.size}"
            )
        if not np.all(np.isfinite(a)):
            raise ValueError(f"El campo '{name}' contiene valores NaN o infinitos.")
        if lower is not None and np.any(a < lower):
            raise ValueError(f"El campo '{name}' viola la cota inferior {lower}.")
        if upper is not None and np.any(a > upper):
            raise ValueError(f"El campo '{name}' viola la cota superior {upper}.")
        return _canonicalize_signed_zero(a)

    def _validate_stress_field(
        self,
        total_stresses: np.ndarray,
        n_nodes: int,
    ) -> np.ndarray:
        """Valida σ ∈ (n, 3, 3) y lo proyecta sobre Sym³."""
        arr = np.asarray(total_stresses, dtype=np.float64)
        if arr.shape != (n_nodes, 3, 3):
            raise ValueError(
                "El tensor de esfuerzos totales debe tener forma "
                f"({n_nodes}, 3, 3). Obtenida: {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                "El tensor de esfuerzos totales contiene valores NaN o infinitos."
            )
        return 0.5 * (arr + np.swapaxes(arr, 1, 2))

    def _as_incidence_matrix(self, incidence_matrix: Any) -> Any:
        """
        Normaliza B ∈ R^{n×m} (densa float64 o CSR).

        B es el adjunto del coborde d₀ sobre 0-cocadenas: (d₀ H)_e = (Bᵀ H)_e.
        """
        if _is_sparse(incidence_matrix):
            B = incidence_matrix.tocsr().astype(np.float64)
            if B.nnz > 0 and not np.all(np.isfinite(B.data)):
                raise ValueError(
                    "La matriz de incidencia dispersa contiene valores no finitos."
                )
        else:
            arr = np.asarray(incidence_matrix, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(
                    "La matriz de incidencia debe ser bidimensional. "
                    f"Obtenido: ndim={arr.ndim}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    "La matriz de incidencia contiene valores NaN o infinitos."
                )
            B = arr
        if B.shape[0] <= 0:
            raise ValueError("La matriz de incidencia debe tener al menos un nodo.")
        return B

    def _matvec(self, A: Any, x: np.ndarray) -> np.ndarray:
        """Producto matriz-vector denso/disperso."""
        return np.asarray(A @ x, dtype=np.float64).ravel()

    def _incidence_transpose_matvec(self, B: Any, x: np.ndarray) -> np.ndarray:
        """Producto Bᵀ x (coborde discreto d₀)."""
        return np.asarray(B.T @ x, dtype=np.float64).ravel()

    def _abs_incidence(self, B: Any) -> Any:
        if _is_sparse(B):
            return abs(B.tocsr())
        return np.abs(B)

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 1 — CONSTITUTIVE / LOCAL PHYSICS
    # ═════════════════════════════════════════════════════════════════════════

    def phase1_incidence_hygiene(self, incidence_matrix: Any) -> Dict[str, float]:
        """
        Fase 1.1 — Higiene cohomológica de B.

        Un grafo orientado satisface 1ᵀ B ≈ 0 (cada arista tiene cola y cabeza).
        La desviación mide defectos de frontera o pesos no balanceados.
        El grado medio es ||B||_{0, col} promedio.
        """
        B = self._as_incidence_matrix(incidence_matrix)
        n_nodes, n_edges = B.shape
        if n_edges == 0:
            return {
                "n_nodes": float(n_nodes),
                "n_edges": 0.0,
                "column_sum_residual": 0.0,
                "mean_column_support": 0.0,
                "max_column_support": 0.0,
            }
        col_sum = np.asarray(B.sum(axis=0), dtype=np.float64).ravel()
        abs_B = self._abs_incidence(B)
        support = np.asarray(abs_B.sum(axis=0), dtype=np.float64).ravel()
        col_res = float(np.max(np.abs(col_sum))) if col_sum.size else 0.0
        return {
            "n_nodes": float(n_nodes),
            "n_edges": float(n_edges),
            "column_sum_residual": col_res,
            "mean_column_support": float(np.mean(support)) if support.size else 0.0,
            "max_column_support": float(np.max(support)) if support.size else 0.0,
        }

    def phase1_validate_constitutive_fields(
        self,
        incidence_matrix: Any,
        edge_lengths: np.ndarray,
        suction: np.ndarray,
        sat: np.ndarray,
        K_sat: np.ndarray,
        total_stresses: np.ndarray,
        s_pumps: np.ndarray,
        node_elevations: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Fase 1.2 — Validación conjunta de campos constitutivos.

        ψ ≥ 0, S ∈ [0, 1], L_e ≥ ε_reg, K_sat ≥ 0, σ ∈ Sym³.
        """
        B = self._as_incidence_matrix(incidence_matrix)
        n_nodes, n_edges = B.shape
        lengths = self._broadcast_edge_array(
            "edge_lengths", edge_lengths, n_edges, lower=self._reg
        )
        suction_v = self._validate_vector("suction", suction, n_nodes, lower=0.0)
        sat_v = self._validate_vector("saturation", sat, n_nodes, lower=0.0, upper=1.0)
        K_sat_edges = self._broadcast_edge_array("K_sat", K_sat, n_edges, lower=0.0)
        total_stresses_v = self._validate_stress_field(total_stresses, n_nodes)
        s_pumps_v = self._validate_vector("s_pumps", s_pumps, n_nodes)
        z_v = self._validate_vector("node_elevations", node_elevations, n_nodes)
        return {
            "B": B,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "edge_lengths": lengths,
            "suction": suction_v,
            "sat": sat_v,
            "K_sat_edges": K_sat_edges,
            "total_stresses": total_stresses_v,
            "s_pumps": s_pumps_v,
            "node_elevations": z_v,
        }

    def phase1_compute_pore_pressures(
        self,
        suction: np.ndarray,
        sat: np.ndarray,
        rho_w: float = 1000.0,
    ) -> np.ndarray:
        """
        Fase 1.3 — Presión de poros de Bishop–Biot.

            u_w = −γ_w ψ,     P_f = S u_w = −γ_w ψ S.

        P_f ≤ 0 en succión (tracción intersticial). γ_w = ρ_w g.
        """
        if not math.isfinite(rho_w) or rho_w <= 0.0:
            raise ValueError("rho_w debe ser finita y estrictamente positiva.")
        s = self._validate_vector("suction", np.asarray(suction, dtype=np.float64).ravel(), lower=0.0)
        sat_arr = self._validate_vector(
            "saturation", np.asarray(sat, dtype=np.float64).ravel(), lower=0.0, upper=1.0
        )
        if s.size != sat_arr.size:
            raise ValueError(
                "suction y sat deben tener la misma longitud. "
                f"Obtenido: {s.size} y {sat_arr.size}"
            )
        gamma_w = float(rho_w) * self._gravity
        pore_pressures = -gamma_w * s * sat_arr
        if not np.all(np.isfinite(pore_pressures)):
            raise ValueError(
                "El cálculo de presiones de poro produjo valores no finitos."
            )
        return _canonicalize_signed_zero(pore_pressures)

    def phase1_effective_stress_audit(
        self,
        total_stresses: np.ndarray,
        pore_pressures: np.ndarray,
        alpha_biot: float = 0.95,
    ) -> Tuple[np.ndarray, bool, np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Fase 1.4 — Esfuerzos efectivos de Biot–Terzaghi / Bishop.

            σ' = σ − α_Biot P_f I.

        Licuación (convención de suelos, compresión positiva en el tensor
        de entrada): σ'₃ ≤ ε o p' = tr(σ')/3 ≤ ε. El determinante se reporta
        como invariante I₃, no como criterio primario.
        """
        pf = self._validate_vector("pore_pressures", pore_pressures)
        n_nodes = pf.size
        total = self._validate_stress_field(total_stresses, n_nodes)
        if not math.isfinite(alpha_biot) or not (0.0 <= alpha_biot <= 1.0):
            raise ValueError("alpha_biot debe ser finito y estar en [0, 1].")

        effective = total - float(alpha_biot) * pf[:, None, None] * _I3[None, :, :]
        try:
            eigvals = np.linalg.eigvalsh(effective)
        except np.linalg.LinAlgError:
            eigvals = np.zeros((n_nodes, 3), dtype=np.float64)
            for i in range(n_nodes):
                try:
                    eigvals[i] = la.eigvalsh(0.5 * (effective[i] + effective[i].T))
                except la.LinAlgError:
                    eigvals[i] = 0.0

        min_principal = np.asarray(eigvals[:, 0], dtype=np.float64)
        det_principal = np.asarray(np.prod(eigvals, axis=1), dtype=np.float64)
        i1 = np.sum(eigvals, axis=1)
        p_mean = i1 / 3.0
        is_liquefied = bool(
            np.any(min_principal <= self._tol) or np.any(p_mean <= self._tol)
        )
        extras = {
            "alpha_biot": float(alpha_biot),
            "mean_effective_pressure": float(np.mean(p_mean)) if n_nodes else math.nan,
            "min_p_mean": float(np.min(p_mean)) if n_nodes else math.nan,
            "liquefaction_nodes": int(
                np.count_nonzero((min_principal <= self._tol) | (p_mean <= self._tol))
            ),
            "symmetry_frobenius_residual": float(
                np.sqrt(np.sum((effective - np.swapaxes(effective, 1, 2)) ** 2))
            ),
        }
        return effective, is_liquefied, min_principal, det_principal, extras

    def _effective_stress_audit(
        self,
        total_stresses: np.ndarray,
        pore_pressures: np.ndarray,
        alpha_biot: float = 0.95,
    ) -> Tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
        """Compatibilidad interna con la firma original de Fase 1.2."""
        effective, is_liq, min_p, det_p, _ = self.phase1_effective_stress_audit(
            total_stresses, pore_pressures, alpha_biot=alpha_biot
        )
        return effective, is_liq, min_p, det_p

    def phase1_compute_mualem_conductivities(
        self,
        sat: np.ndarray,
        K_sat: np.ndarray,
        L_param: float = 0.5,
        m_param: float = 0.5,
    ) -> np.ndarray:
        """
        Fase 1.5 — Conductividad hidráulica de Mualem–van Genuchten.

            K(S_e) = K_sat S_e^L [1 − (1 − S_e^{1/m})^m ]²,

        con S_e regularizado en [ε, 1], m ∈ (0, 1), n = 1/(1−m).
        Se garantiza K ∈ [0, K_sat] (módulo ruido de redondeo).
        """
        sat_arr = self._validate_vector(
            "saturation", np.asarray(sat, dtype=np.float64).ravel(), lower=0.0, upper=1.0
        )
        K_sat_arr = self._broadcast_edge_array("K_sat", K_sat, sat_arr.size, lower=0.0)
        if not math.isfinite(L_param):
            raise ValueError("L_param debe ser finita.")
        if not math.isfinite(m_param) or not (0.0 < m_param < 1.0):
            raise ValueError("m_param debe ser finito y estar en (0, 1).")

        sat_reg = np.clip(sat_arr, self._reg, 1.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sat_inv_m = np.power(sat_reg, 1.0 / m_param)
            inner = np.clip(1.0 - sat_inv_m, 0.0, 1.0)
            term = np.square(1.0 - np.power(inner, m_param))
            kr = np.power(sat_reg, L_param) * term
            kr = np.where(np.isfinite(kr), kr, 0.0)
            kr = np.clip(kr, 0.0, 1.0)
            K = K_sat_arr * kr
        K = np.where(np.isfinite(K), K, 0.0)
        K = np.clip(K, 0.0, None)
        return _canonicalize_signed_zero(K)

    def phase1_map_node_saturation_to_edges(
        self,
        incidence_matrix: Any,
        node_sat: np.ndarray,
    ) -> np.ndarray:
        """
        Fase 1.6 — Pullback de saturación nodal a aristas.

        sat_e = (Σ_{v ∈ e} sat_v) / deg_0(e)  =  (|B|ᵀ sat) / (|B|ᵀ 1).
        Es el promedio de Whitney de la 0-forma S sobre el 1-simplejo.
        """
        sat_arr = self._validate_vector("saturation", node_sat, lower=0.0, upper=1.0)
        B = self._as_incidence_matrix(incidence_matrix)
        n_edges = B.shape[1]
        if n_edges == 0:
            return np.empty(0, dtype=np.float64)
        abs_B = self._abs_incidence(B)
        sat_sum = np.asarray(abs_B.T @ sat_arr, dtype=np.float64).ravel()
        counts = np.asarray(abs_B.sum(axis=0), dtype=np.float64).ravel()
        edge_sat = sat_sum / np.maximum(counts, 1.0)
        return _canonicalize_signed_zero(np.clip(edge_sat, 0.0, 1.0))

    def _map_node_saturation_to_edges(
        self,
        incidence_matrix: Any,
        node_sat: np.ndarray,
    ) -> np.ndarray:
        """Alias interno de Fase 1.6."""
        return self.phase1_map_node_saturation_to_edges(incidence_matrix, node_sat)

    def phase1_mass_compatibility(
        self,
        s_pumps: np.ndarray,
        incidence_hygiene: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Fase 1.7 — Compatibilidad de Neumann con H⁰_dR.

        Sobre un complejo conexo sin Dirichlet, 1ᵀ s = 0 es la condición de
        solvabilidad de L H = s. El residual |Σ s| / max(1, ||s||₁) mide el
        defecto que Tikhonov deberá absorber.
        """
        s = np.asarray(s_pumps, dtype=np.float64).ravel()
        total = _kbn_sum(s)
        l1 = _kbn_sum(np.abs(s))
        residual = abs(total) / max(1.0, l1)
        return {
            "total_pump_source": float(total),
            "pump_l1": float(l1),
            "neumann_compatibility_residual": float(residual),
            "incidence_column_sum_residual": float(
                incidence_hygiene.get("column_sum_residual", math.nan)
            ),
        }

    def _phase1_session_hash(
        self,
        suction: np.ndarray,
        sat: np.ndarray,
        pore_pressures: np.ndarray,
        effective_stresses: np.ndarray,
        edge_conductivities: np.ndarray,
        s_pumps: np.ndarray,
        node_elevations: np.ndarray,
        edge_lengths: np.ndarray,
    ) -> str:
        """Fase 1.8 — Sello de sesión SHA-256 canónico longitud-prefijado."""
        digest = _sha256_hex_with_token(
            "PHASE1/CONSTITUTIVE",
            suction,
            sat,
            pore_pressures,
            effective_stresses,
            edge_conductivities,
            s_pumps,
            node_elevations,
            edge_lengths,
        )
        if len(digest) != _SHA256_HEX_LEN:
            raise RuntimeError("El sello de sesión no es un SHA-256 de 64 nibbles.")
        return digest

    def phase1_close_and_open_phase2(
        self,
        incidence_matrix: Any,
        edge_lengths: np.ndarray,
        suction: np.ndarray,
        sat: np.ndarray,
        K_sat: np.ndarray,
        total_stresses: np.ndarray,
        s_pumps: np.ndarray,
        node_elevations: np.ndarray,
        rho_w: float = 1000.0,
        alpha_biot: float = 0.95,
        L_param: float = 0.5,
        m_param: float = 0.5,
    ) -> Phase1HydroHandoff:
        """
        Fase 1.9 — Cierre formal de Fase 1 y apertura verificada de Fase 2.

        Definición formal de frontera:

            Φ₁→₂ : Datos crudos ↦ (P_f, σ', K_e, χ, σ₁)

        Este es el último método de la Fase 1. Su contrato es exactamente el
        dominio de `phase2_from_phase1`: produce `Phase1HydroHandoff` y exige
        que la Fase 2 lo admita de inmediato. Con ello la Fase 1 queda
        anidada, como prefijo functorial, dentro de la Fase 2.
        """
        fields = self.phase1_validate_constitutive_fields(
            incidence_matrix,
            edge_lengths,
            suction,
            sat,
            K_sat,
            total_stresses,
            s_pumps,
            node_elevations,
        )
        B = fields["B"]
        n_nodes = int(fields["n_nodes"])
        n_edges = int(fields["n_edges"])
        hygiene = self.phase1_incidence_hygiene(B)

        pore_pressures = self.phase1_compute_pore_pressures(
            fields["suction"], fields["sat"], rho_w=rho_w
        )
        (
            effective_stresses,
            is_liquefaction,
            min_principal,
            det_effective,
            biot_extra,
        ) = self.phase1_effective_stress_audit(
            fields["total_stresses"], pore_pressures, alpha_biot=alpha_biot
        )
        edge_sat = self.phase1_map_node_saturation_to_edges(B, fields["sat"])
        edge_conductivities = self.phase1_compute_mualem_conductivities(
            edge_sat, fields["K_sat_edges"], L_param=L_param, m_param=m_param
        )
        mass = self.phase1_mass_compatibility(fields["s_pumps"], hygiene)
        session_sha256 = self._phase1_session_hash(
            suction=fields["suction"],
            sat=fields["sat"],
            pore_pressures=pore_pressures,
            effective_stresses=effective_stresses,
            edge_conductivities=edge_conductivities,
            s_pumps=fields["s_pumps"],
            node_elevations=fields["node_elevations"],
            edge_lengths=fields["edge_lengths"],
        )

        n_vg = 1.0 / (1.0 - float(m_param)) if 0.0 < m_param < 1.0 else math.nan
        diagnostics: Dict[str, Any] = {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "rho_w": float(rho_w),
            "gamma_w": float(rho_w) * self._gravity,
            "alpha_biot": float(alpha_biot),
            "L_param": float(L_param),
            "m_param": float(m_param),
            "n_van_genuchten": float(n_vg),
            "saturation_mean": float(np.mean(fields["sat"])) if n_nodes else 0.0,
            "suction_mean": float(np.mean(fields["suction"])) if n_nodes else 0.0,
            "K_min": float(np.min(edge_conductivities)) if n_edges else 0.0,
            "K_max": float(np.max(edge_conductivities)) if n_edges else 0.0,
            "session_sha256_prefix": session_sha256[:16],
            **hygiene,
            **mass,
            **biot_extra,
        }

        handoff = Phase1HydroHandoff(
            incidence_matrix=B,
            edge_lengths=fields["edge_lengths"],
            suction=fields["suction"],
            sat=fields["sat"],
            K_sat_edges=fields["K_sat_edges"],
            total_stresses=fields["total_stresses"],
            s_pumps=fields["s_pumps"],
            node_elevations=fields["node_elevations"],
            pore_pressures=pore_pressures,
            effective_stresses=effective_stresses,
            is_liquefaction_active=bool(is_liquefaction),
            min_principal_stresses=min_principal,
            determinant_effective_stresses=det_effective,
            edge_saturations=edge_sat,
            edge_conductivities=edge_conductivities,
            diagnostics=diagnostics,
            next_entrypoint=_PHASE1_ENTRY,
            session_sha256=session_sha256,
        )

        opened = self.phase2_from_phase1(handoff)
        if opened.session_sha256 != session_sha256:
            raise RuntimeError(
                "Invariante de anidamiento Φ₁→₂ violado: el sello de sesión "
                "admitido por Fase 2 no coincide con el observado en Fase 1."
            )

        logger.debug(
            "Fase 1 hidrológica cerrada: n_nodes=%d, n_edges=%d, liquefaction=%s",
            n_nodes,
            n_edges,
            is_liquefaction,
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2 — DEC / RICHARDS SOLUTION
    # (continuación formal de phase1_close_and_open_phase2)
    # ═════════════════════════════════════════════════════════════════════════

    def phase2_from_phase1(self, handoff: Phase1HydroHandoff) -> Phase1HydroHandoff:
        """
        Fase 2.0 — Entrada formal desde Fase 1.

        Continuación directa de `phase1_close_and_open_phase2`.
        """
        if not isinstance(handoff, Phase1HydroHandoff):
            raise TypeError("Se esperaba Phase1HydroHandoff como frontera Φ₁→₂.")
        if handoff.next_entrypoint != _PHASE1_ENTRY:
            raise ValueError(
                "Phase1HydroHandoff inválido: el punto de entrada esperado es "
                f"{_PHASE1_ENTRY!r}."
            )
        if handoff.session_sha256 and len(handoff.session_sha256) != _SHA256_HEX_LEN:
            raise ValueError("El sello de sesión de Φ₁→₂ no es un SHA-256 válido.")
        if handoff.incidence_matrix is None or handoff.incidence_matrix.shape[0] <= 0:
            raise ValueError("La observación de Φ₁→₂ no contiene nudos.")
        return handoff

    def phase2_hodge_star_weights(
        self,
        edge_conductivities: np.ndarray,
        n_edges: int,
    ) -> np.ndarray:
        """
        Fase 2.1 — Estrella de Hodge primal en 1-formas.

        W = diag(K_e). K_e es conductancia DEC (ya absorbe geometría dual).
        Se exige K_e ≥ 0 para que Δ₀ sea semidefinido positivo.
        """
        return self._broadcast_edge_array(
            "edge_conductivities", edge_conductivities, n_edges, lower=0.0
        )

    def phase2_assemble_richards_laplacian(
        self,
        incidence_matrix: Any,
        edge_conductivities: np.ndarray,
    ) -> Any:
        """
        Fase 2.2 — Laplaciano de Hodge de 0-formas.

            L = B W Bᵀ,     W = ⋆₁,     L = Lᵀ ⪰ 0.

        Simetrización explícita ½(L + Lᵀ) elimina sesgo de redondeo.
        """
        B = self._as_incidence_matrix(incidence_matrix)
        n_nodes, n_edges = B.shape
        K = self.phase2_hodge_star_weights(edge_conductivities, n_edges)
        if n_edges == 0:
            if _is_sparse(B):
                return sp.csr_matrix((n_nodes, n_nodes), dtype=np.float64)
            return np.zeros((n_nodes, n_nodes), dtype=np.float64)
        if _is_sparse(B):
            B_sp = B.tocsr()
            W = sp.diags(K, format="csr")
            L = B_sp @ W @ B_sp.T
            L = 0.5 * (L + L.T)
            return L.tocsr()
        W = np.diag(K)
        L = B @ W @ B.T
        L = 0.5 * (L + L.T)
        return np.asarray(L, dtype=np.float64)

    def phase2_tikhonov_scale(self, L: Any) -> float:
        """
        Fase 2.3 — Escala de Tikhonov λ = max(ε_reg, ε_mach · mean|diag L|).

        Fija el gauge de H⁰ y regulariza componentes casi singulares sin
        dominar un sistema bien condicionado.
        """
        n = L.shape[0]
        if n == 0:
            return self._reg
        if _is_sparse(L):
            diag = np.asarray(L.diagonal(), dtype=np.float64).ravel()
        else:
            diag = np.asarray(np.diag(L), dtype=np.float64).ravel()
        mean_diag = float(np.mean(np.abs(diag))) if diag.size else 1.0
        if not math.isfinite(mean_diag) or mean_diag <= 0.0:
            mean_diag = 1.0
        return float(max(self._reg, _MACHINE_EPS * mean_diag))

    def _tikhonov_scale(self, L: Any) -> float:
        return self.phase2_tikhonov_scale(L)

    def _add_tikhonov(self, L: Any, reg: float) -> Any:
        """L_reg = L + λ I."""
        n = L.shape[0]
        if _is_sparse(L):
            return (L + reg * sp.eye(n, format="csr", dtype=np.float64)).tocsr()
        return L + reg * np.eye(n, dtype=np.float64)

    def _solve_linear_symmetric(self, A: Any, b: np.ndarray) -> np.ndarray:
        """
        Fase 2.4 — Solver SPD: spsolve / Cholesky (assume_a='pos') / LU.
        """
        if b.size == 0:
            return np.empty(0, dtype=np.float64)
        if _is_sparse(A):
            try:
                x = spla.spsolve(A.tocsc(), b)
                x = np.asarray(x, dtype=np.float64).ravel()
                if np.all(np.isfinite(x)):
                    return x
            except Exception as exc:
                logger.warning("spsolve falló; se hará fallback denso: %s", exc)
                A = A.toarray()
        try:
            x = la.solve(A, b, assume_a="pos")
        except Exception:
            x = la.solve(A, b)
        return np.asarray(x, dtype=np.float64).ravel()

    def phase2_kirchhoff_residual(
        self,
        B: Any,
        flow_rates: np.ndarray,
        s_pumps: np.ndarray,
    ) -> float:
        """
        Fase 2.5 — Residual de Kirchhoff ||B Q − s|| / max(1, ||s||).

        Es la divergencia discreta (ley de corrientes) sobre 0-cadenas.
        """
        div = self._matvec(B, flow_rates)
        num = float(la.norm(div - s_pumps))
        den = max(1.0, float(la.norm(s_pumps)))
        return float(num / den)

    def _solve_richards_with_laplacian(
        self,
        L: Any,
        B: Any,
        edge_conductivities: np.ndarray,
        s_pumps: np.ndarray,
        node_elevations: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Fase 2.6 — Solución de Richards con elevación explícita.

            (L + λI) ψ = s − L z,     H = ψ + z,     Q = K ⊙ (Bᵀ H).

        Diagnostica residual algebraico, conservación de masa y Kirchhoff.
        """
        n_nodes = L.shape[0]
        if n_nodes == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), {}

        reg = self.phase2_tikhonov_scale(L)
        L_reg = self._add_tikhonov(L, reg)
        Lz = self._matvec(L, node_elevations)
        rhs = s_pumps - Lz
        psi = self._solve_linear_symmetric(L_reg, rhs)
        hydraulic_head = psi + node_elevations
        dH = self._incidence_transpose_matvec(B, hydraulic_head)
        flow_rates = edge_conductivities * dH

        residual_vec = self._matvec(L_reg, psi) - rhs
        residual_norm = float(la.norm(residual_vec))
        rhs_norm = float(la.norm(rhs))
        relative_residual = residual_norm / max(1.0, rhs_norm)

        balance_vec = s_pumps - self._matvec(L, hydraulic_head)
        pump_abs_sum = abs(_kbn_sum(np.abs(s_pumps)))
        mass_residual = abs(_kbn_sum(balance_vec)) / max(1.0, pump_abs_sum)
        kirchhoff = self.phase2_kirchhoff_residual(B, flow_rates, s_pumps)

        mean_h = float(np.mean(hydraulic_head)) if hydraulic_head.size else 0.0
        h_oscillation = float(np.max(hydraulic_head) - np.min(hydraulic_head)) if hydraulic_head.size else 0.0

        diagnostics: Dict[str, float] = {
            "tikhonov_lambda": float(reg),
            "richards_relative_residual": float(relative_residual),
            "richards_mass_residual": float(mass_residual),
            "kirchhoff_residual": float(kirchhoff),
            "total_head_mean": mean_h,
            "total_head_oscillation": h_oscillation,
            "total_flow_abs_sum": float(_kbn_sum(np.abs(flow_rates))) if flow_rates.size else 0.0,
            "gauge_psi_mean": float(np.mean(psi)) if psi.size else 0.0,
        }
        return hydraulic_head, flow_rates, diagnostics

    def phase2_solve_richards_flow(
        self,
        handoff: Phase1HydroHandoff,
    ) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Fase 2.7 — Orquestación DEC de Richards.

        Retorna (L, H, Q, diagnósticos).
        """
        L = self.phase2_assemble_richards_laplacian(
            handoff.incidence_matrix, handoff.edge_conductivities
        )
        H, Q, diagnostics = self._solve_richards_with_laplacian(
            L,
            handoff.incidence_matrix,
            handoff.edge_conductivities,
            handoff.s_pumps,
            handoff.node_elevations,
        )
        return L, H, Q, diagnostics

    def phase2_evaluate_siphoning(
        self,
        hydraulic_head: np.ndarray,
        incidence_matrix: Any,
        edge_lengths: np.ndarray,
        rho_sat: float,
        rho_w: float,
    ) -> Tuple[bool, np.ndarray, Dict[str, float]]:
        """
        Fase 2.8 — Gradiente crítico de Terzaghi / piping.

            i_e = |ΔH_e| / L_e,     i_crit = (ρ_sat − ρ_w) / ρ_w,
            FoS_i = i_crit / max_e i_e.
        """
        if not math.isfinite(rho_w) or rho_w <= 0.0:
            raise ValueError("rho_w debe ser finita y estrictamente positiva.")
        if not math.isfinite(rho_sat) or rho_sat <= 0.0:
            raise ValueError("rho_sat debe ser finita y estrictamente positiva.")

        i_crit = max(0.0, (rho_sat - rho_w) / rho_w)
        B = self._as_incidence_matrix(incidence_matrix)
        n_nodes, n_edges = B.shape
        H = self._validate_vector("hydraulic_head", hydraulic_head, n_nodes)
        lengths = self._broadcast_edge_array(
            "edge_lengths", edge_lengths, n_edges, lower=self._reg
        )
        if n_edges == 0:
            return False, np.empty(0, dtype=np.float64), {
                "i_crit": float(i_crit),
                "siphoning_max_gradient": 0.0,
                "siphoning_fos": math.inf,
                "siphoning_edges": 0.0,
            }

        dH = self._incidence_transpose_matvec(B, H)
        gradients = np.abs(dH) / lengths
        max_i = float(np.max(gradients)) if gradients.size else 0.0
        is_siphoning = bool(np.any(gradients > i_crit + self._tol))
        n_crit = int(np.count_nonzero(gradients > i_crit + self._tol))
        fos = float(i_crit / max_i) if max_i > self._reg else math.inf
        extras = {
            "i_crit": float(i_crit),
            "siphoning_max_gradient": max_i,
            "siphoning_fos": fos,
            "siphoning_edges": float(n_crit),
        }
        return is_siphoning, gradients, extras

    def _evaluate_siphoning(
        self,
        hydraulic_head: np.ndarray,
        incidence_matrix: Any,
        edge_lengths: np.ndarray,
        rho_sat: float,
        rho_w: float,
    ) -> Tuple[bool, np.ndarray]:
        """Alias interno con la firma original de Fase 2.7."""
        is_siphoning, gradients, _ = self.phase2_evaluate_siphoning(
            hydraulic_head, incidence_matrix, edge_lengths, rho_sat, rho_w
        )
        return is_siphoning, gradients

    def phase2_close_and_open_phase3(
        self,
        phase1_handoff: Phase1HydroHandoff,
        rho_sat: float = 2000.0,
        rho_w: float = 1000.0,
    ) -> Phase2HydroHandoff:
        """
        Fase 2.9 — Cierre formal de Fase 2 y apertura verificada de Fase 3.

        Definición formal de frontera:

            Φ₂→₃ : constitución local ↦ (H, Q, L, i, σ₂)

        Este es el último método de la Fase 2. Su contrato es exactamente el
        dominio de `phase3_from_phase2`. Con ello la Fase 2 queda anidada,
        como prefijo functorial, dentro de la Fase 3.
        """
        validated = self.phase2_from_phase1(phase1_handoff)
        L, H, Q, solve_diag = self.phase2_solve_richards_flow(validated)
        is_siphoning, gradients, siphon_diag = self.phase2_evaluate_siphoning(
            H,
            validated.incidence_matrix,
            validated.edge_lengths,
            rho_sat=rho_sat,
            rho_w=rho_w,
        )

        diagnostics = dict(validated.diagnostics)
        diagnostics.update(solve_diag)
        diagnostics.update(siphon_diag)
        diagnostics["rho_sat"] = float(rho_sat)
        diagnostics["rho_w_phase2"] = float(rho_w)

        handoff = Phase2HydroHandoff(
            incidence_matrix=validated.incidence_matrix,
            edge_lengths=validated.edge_lengths,
            suction=validated.suction,
            sat=validated.sat,
            K_sat_edges=validated.K_sat_edges,
            total_stresses=validated.total_stresses,
            s_pumps=validated.s_pumps,
            node_elevations=validated.node_elevations,
            pore_pressures=validated.pore_pressures,
            effective_stresses=validated.effective_stresses,
            is_liquefaction_active=validated.is_liquefaction_active,
            min_principal_stresses=validated.min_principal_stresses,
            determinant_effective_stresses=validated.determinant_effective_stresses,
            edge_saturations=validated.edge_saturations,
            edge_conductivities=validated.edge_conductivities,
            laplacian_matrix=L,
            hydraulic_head=H,
            flow_rates=Q,
            is_siphoning_active=bool(is_siphoning),
            hydraulic_gradients=gradients,
            diagnostics=diagnostics,
            next_entrypoint=_PHASE2_ENTRY,
            session_sha256=validated.session_sha256,
        )

        opened = self.phase3_from_phase2(handoff)
        if opened.session_sha256 != handoff.session_sha256:
            raise RuntimeError(
                "Invariante de anidamiento Φ₂→₃ violado: el sello de sesión "
                "admitido por Fase 3 no coincide con el de Fase 2."
            )

        logger.debug(
            "Fase 2 hidrológica cerrada: siphoning=%s, mass_residual=%.3e",
            is_siphoning,
            solve_diag.get("richards_mass_residual", math.nan),
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 3 — SPECTRAL / GOVERNANCE
    # (continuación formal de phase2_close_and_open_phase3)
    # ═════════════════════════════════════════════════════════════════════════

    def phase3_from_phase2(self, handoff: Phase2HydroHandoff) -> Phase2HydroHandoff:
        """
        Fase 3.0 — Entrada formal desde Fase 2.

        Continuación directa de `phase2_close_and_open_phase3`.
        """
        if not isinstance(handoff, Phase2HydroHandoff):
            raise TypeError("Se esperaba Phase2HydroHandoff como frontera Φ₂→₃.")
        if handoff.next_entrypoint != _PHASE2_ENTRY:
            raise ValueError(
                "Phase2HydroHandoff inválido: el punto de entrada esperado es "
                f"{_PHASE2_ENTRY!r}."
            )
        if handoff.session_sha256 and len(handoff.session_sha256) != _SHA256_HEX_LEN:
            raise ValueError("El sello de sesión de Φ₂→₃ no es un SHA-256 válido.")
        return handoff

    def phase3_laplacian_spectrum(self, L: Any) -> np.ndarray:
        """
        Fase 3.1 — Espectro de Δ₀.

        Sistemas pequeños: eigvalsh denso. Sistemas grandes: eigsh SA
        (autovalores algebraicos menores). Autovalores negativos por debajo
        de la tolerancia se clipean a 0 (ruido de PSD).
        """
        n = L.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float64)

        vals: np.ndarray
        try:
            if _is_sparse(L):
                if n <= self._max_dense_eigen_dim:
                    vals = la.eigvalsh(L.toarray())
                else:
                    k = min(self._spectral_k, max(n - 1, 1))
                    if n <= 1:
                        vals = la.eigvalsh(L.toarray())
                    else:
                        vals = spla.eigsh(
                            L, k=k, which="SA", return_eigenvectors=False
                        )
            else:
                if n <= self._max_dense_eigen_dim:
                    vals = la.eigvalsh(L)
                else:
                    L_sp = sp.csr_matrix(L)
                    k = min(self._spectral_k, max(n - 1, 1))
                    if n <= 1:
                        vals = la.eigvalsh(L)
                    else:
                        vals = spla.eigsh(
                            L_sp, k=k, which="SA", return_eigenvectors=False
                        )
        except Exception as exc:
            logger.warning("Fallo en auditoría espectral: %s", exc)
            try:
                if _is_sparse(L) and n <= 1024:
                    vals = la.eigvalsh(L.toarray())
                elif (not _is_sparse(L)) and n <= 1024:
                    vals = la.eigvalsh(L)
                else:
                    vals = np.empty(0, dtype=np.float64)
            except Exception:
                vals = np.empty(0, dtype=np.float64)

        vals = np.sort(np.asarray(vals, dtype=np.float64).ravel())
        if vals.size > 0:
            tiny = max(self._tol, 1e-10)
            vals = np.where((vals < 0.0) & (vals > -tiny), 0.0, vals)
        return vals

    def _laplacian_spectrum(self, L: Any) -> np.ndarray:
        return self.phase3_laplacian_spectrum(L)

    def phase3_spectral_audit(
        self,
        handoff: Phase2HydroHandoff,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fase 3.2 — Auditoría espectral de Fiedler / Cheeger.

        λ₁ = 0  (si β₀ ≥ 1),  λ₂ = conectividad algebraica,
        h ≤ √(2 λ₂) cota de Cheeger, κ ≈ λ_max / λ_min⁺.
        """
        eigvals = self.phase3_laplacian_spectrum(handoff.laplacian_matrix)
        n = int(handoff.laplacian_matrix.shape[0])
        diagnostics: Dict[str, Any] = {
            "n_eigenvalues": int(eigvals.size),
            "spectral_subset": bool(eigvals.size < n),
            "ambient_dimension": n,
        }
        if eigvals.size > 0:
            tiny = max(self._tol, 1e-10)
            pos = eigvals[eigvals > tiny]
            lam_min_pos = float(pos[0]) if pos.size else math.nan
            lam_max = float(eigvals[-1])
            kappa = (
                float(lam_max / lam_min_pos)
                if pos.size and math.isfinite(lam_min_pos) and lam_min_pos > 0.0
                else math.inf
            )
            lam2 = float(eigvals[1]) if eigvals.size >= 2 else math.nan
            cheeger_upper = (
                math.sqrt(2.0 * max(lam2, 0.0)) if math.isfinite(lam2) else math.nan
            )
            diagnostics.update(
                {
                    "spectral_min": float(eigvals[0]),
                    "spectral_max": lam_max,
                    "spectral_radius": float(np.max(np.abs(eigvals))),
                    "algebraic_connectivity": lam2,
                    "nullity": int(np.count_nonzero(eigvals <= tiny)),
                    "negative_eigenvalues": int(np.count_nonzero(eigvals < -tiny)),
                    "condition_number_effective": kappa,
                    "cheeger_upper_bound": float(cheeger_upper),
                }
            )
        else:
            diagnostics.update(
                {
                    "spectral_min": math.nan,
                    "spectral_max": math.nan,
                    "spectral_radius": math.nan,
                    "algebraic_connectivity": math.nan,
                    "nullity": math.nan,
                    "negative_eigenvalues": math.nan,
                    "condition_number_effective": math.nan,
                    "cheeger_upper_bound": math.nan,
                }
            )
        return eigvals, diagnostics

    def phase3_poincare_residual(
        self,
        handoff: Phase2HydroHandoff,
        eigvals: np.ndarray,
    ) -> float:
        """
        Fase 3.3 — Residual de Poincaré discreto.

        ||H − mean(H)||²  ≲  λ₂⁻¹ ||dH||_W². Se reporta el cociente
        λ₂ ||H − H̄||² / ||Bᵀ H||_W², que debe ser ≲ 1.
        """
        H = np.asarray(handoff.hydraulic_head, dtype=np.float64).ravel()
        if H.size <= 1:
            return 0.0
        tiny = max(self._tol, 1e-10)
        lam2 = math.nan
        if eigvals.size >= 2:
            lam2 = float(eigvals[1])
        if not math.isfinite(lam2) or lam2 <= tiny:
            return math.nan
        osc = H - float(np.mean(H))
        num = float(np.dot(osc, osc))
        dH = self._incidence_transpose_matvec(handoff.incidence_matrix, H)
        w = np.asarray(handoff.edge_conductivities, dtype=np.float64).ravel()
        if dH.size == 0 or w.size != dH.size:
            return math.nan
        den = float(np.dot(w, dH * dH))
        if den <= tiny:
            return 0.0 if num <= tiny else math.inf
        return float(lam2 * num / den)

    def phase3_stability_audit(
        self,
        handoff: Phase2HydroHandoff,
    ) -> Dict[str, Any]:
        """
        Fase 3.4 — Estabilidad hidro-geomecánica.

        Licuación: σ'₃ ≤ ε o p' ≤ ε o det(σ') ≤ ε (invariante I₃).
        Sifonamiento: i_max > i_crit. Estado crítico = disyunción.
        """
        min_principal = handoff.min_principal_stresses
        det_principal = handoff.determinant_effective_stresses
        liquefaction_by_min = bool(
            np.any(min_principal <= self._tol)
        ) if min_principal.size else False
        liquefaction_by_det = bool(
            np.any(det_principal <= self._tol)
        ) if det_principal.size else False
        is_liquefaction = bool(
            handoff.is_liquefaction_active
            or liquefaction_by_min
            or liquefaction_by_det
        )
        is_siphoning = bool(handoff.is_siphoning_active)
        gradients = handoff.hydraulic_gradients
        return {
            "is_liquefaction_active": is_liquefaction,
            "liquefaction_by_min_principal": liquefaction_by_min,
            "liquefaction_by_determinant": liquefaction_by_det,
            "is_siphoning_active": is_siphoning,
            "siphoning_max_gradient": float(np.max(gradients)) if gradients.size else 0.0,
            "min_principal_stress": float(np.min(min_principal)) if min_principal.size else math.nan,
            "min_effective_determinant": float(np.min(det_principal)) if det_principal.size else math.nan,
            "critical_state": bool(is_liquefaction or is_siphoning),
            "kirchhoff_residual": _finite_or_nan(
                handoff.diagnostics.get("kirchhoff_residual")
            ),
            "richards_mass_residual": _finite_or_nan(
                handoff.diagnostics.get("richards_mass_residual")
            ),
        }

    def phase3_governance_seal(
        self,
        handoff: Phase2HydroHandoff,
        eigenvalues: np.ndarray,
        spectral: Mapping[str, Any],
        stability: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Fase 3.5 — Sello de gobernanza SHA-256 canónico (no depende de `repr`).
        """
        h = hashlib.sha256()
        h.update(b"HYDRO/CERT/v3")
        _sha_update_str(h, "PHASE3/HYDRO_GOV")
        _sha_update_arr(h, handoff.hydraulic_head)
        _sha_update_arr(h, handoff.pore_pressures)
        _sha_update_arr(h, handoff.effective_stresses)
        _sha_update_arr(h, handoff.flow_rates)
        _sha_update_arr(h, eigenvalues)
        _sha_update_arr(h, handoff.hydraulic_gradients)
        _sha_update_str(h, handoff.session_sha256)
        h.update(b"\x01" if handoff.is_siphoning_active else b"\x00")
        h.update(b"\x01" if bool(stability.get("is_liquefaction_active")) else b"\x00")
        for key in (
            "spectral_min",
            "algebraic_connectivity",
            "kirchhoff_residual",
            "siphoning_max_gradient",
        ):
            source: Mapping[str, Any] = spectral if key in spectral else stability
            h.update(_pack_f64(_finite_or_nan(source.get(key))))
        seal = h.hexdigest()
        return {
            "sha256": seal,
            "hash_algorithm": "sha256",
            "encoding": "canonical-binary-v3",
            "session_sha256": handoff.session_sha256,
        }

    def phase3_build_state(
        self,
        handoff: Phase2HydroHandoff,
        eigenvalues: np.ndarray,
        spectral_diag: Dict[str, Any],
        stability_diag: Dict[str, Any],
        seal: str,
    ) -> HydrologicalState:
        """Fase 3.6 — Consolidación del estado hidrológico inmutable."""
        phase_chain = (
            "PHASE1/CONSTITUTIVE",
            "PHASE2/DEC_RICHARDS",
            "PHASE3/SPECTRAL_GOVERNANCE",
        )
        diagnostics: Dict[str, Any] = dict(handoff.diagnostics)
        diagnostics["spectral"] = spectral_diag
        diagnostics["stability"] = stability_diag
        diagnostics["governance_seal_prefix"] = seal[:16]
        return HydrologicalState(
            hydraulic_head=handoff.hydraulic_head,
            pore_pressures=handoff.pore_pressures,
            effective_stresses=handoff.effective_stresses,
            flow_rates=handoff.flow_rates,
            laplacian_eigenvalues=eigenvalues,
            is_siphoning_active=bool(handoff.is_siphoning_active),
            is_liquefaction_active=bool(stability_diag["is_liquefaction_active"]),
            sha256_hash=seal,
            phase_chain=phase_chain,
            diagnostics=diagnostics,
        )

    def phase3_close_loop(
        self,
        phase2_handoff: Phase2HydroHandoff,
    ) -> Phase3HydroReport:
        """
        Fase 3.7 — Orquestación completa de la Fase 3.

        Ejecuta, en orden:
          1. validación de frontera Φ₂→₃;
          2. auditoría espectral y Poincaré;
          3. auditoría de estabilidad;
          4. sello de gobernanza y consolidación.
        """
        validated = self.phase3_from_phase2(phase2_handoff)
        eigenvalues, spectral_diag = self.phase3_spectral_audit(validated)
        spectral_diag["poincare_residual"] = self.phase3_poincare_residual(
            validated, eigenvalues
        )
        stability_diag = self.phase3_stability_audit(validated)
        governance = self.phase3_governance_seal(
            validated, eigenvalues, spectral_diag, stability_diag
        )
        state = self.phase3_build_state(
            validated,
            eigenvalues,
            spectral_diag,
            stability_diag,
            str(governance["sha256"]),
        )
        return Phase3HydroReport(
            state=state,
            spectral_audit=spectral_diag,
            stability_audit=stability_diag,
            governance_seal=governance,
            phase_chain=state.phase_chain,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # API PRINCIPAL COMPATIBLE
    # ═════════════════════════════════════════════════════════════════════════

    def build_state(
        self,
        incidence_matrix: Any,
        edge_lengths: np.ndarray,
        suction: np.ndarray,
        sat: np.ndarray,
        K_sat: np.ndarray,
        total_stresses: np.ndarray,
        s_pumps: np.ndarray,
        node_elevations: np.ndarray,
        rho_w: float = 1000.0,
        alpha_biot: float = 0.95,
        L_param: float = 0.5,
        m_param: float = 0.5,
        rho_sat: float = 2000.0,
    ) -> HydrologicalState:
        """
        API principal — Orquesta el funtor compuesto Φ₃ ∘ Φ₂ ∘ Φ₁.

        Equivalencia:
          FASE 1 → phase1_close_and_open_phase2
          FASE 2 → phase2_close_and_open_phase3
          FASE 3 → phase3_close_loop
        """
        phase1_handoff = self.phase1_close_and_open_phase2(
            incidence_matrix=incidence_matrix,
            edge_lengths=edge_lengths,
            suction=suction,
            sat=sat,
            K_sat=K_sat,
            total_stresses=total_stresses,
            s_pumps=s_pumps,
            node_elevations=node_elevations,
            rho_w=rho_w,
            alpha_biot=alpha_biot,
            L_param=L_param,
            m_param=m_param,
        )
        phase2_handoff = self.phase2_close_and_open_phase3(
            phase1_handoff=phase1_handoff,
            rho_sat=rho_sat,
            rho_w=rho_w,
        )
        report = self.phase3_close_loop(phase2_handoff)
        logger.info(
            "HydrologicalManifold consolidó estado. Sello: %s",
            report.state.sha256_hash[:16],
        )
        return report.state

    # ═════════════════════════════════════════════════════════════════════════
    # MÉTODOS LEGADOS (COMPATIBILIDAD)
    # ═════════════════════════════════════════════════════════════════════════

    def compute_pore_pressures(
        self,
        suction: np.ndarray,
        sat: np.ndarray,
        rho_w: float = 1000.0,
    ) -> np.ndarray:
        """API legada — equivalente a Fase 1.3."""
        return self.phase1_compute_pore_pressures(suction, sat, rho_w=rho_w)

    def compute_effective_stress_tensors(
        self,
        total_stresses: np.ndarray,
        pore_pressures: np.ndarray,
        alpha_biot: float = 0.95,
    ) -> Tuple[np.ndarray, bool]:
        """
        API legada — equivalente a Fase 1.4.

        Retorna
        -------
        Tuple[np.ndarray, bool]
            (effective_stresses, is_liquefaction_active)
        """
        effective, is_liq, _, _, _ = self.phase1_effective_stress_audit(
            total_stresses, pore_pressures, alpha_biot=alpha_biot
        )
        return effective, is_liq

    def compute_mualem_conductivities(
        self,
        sat: np.ndarray,
        K_sat: np.ndarray,
        L_param: float = 0.5,
        m_param: float = 0.5,
    ) -> np.ndarray:
        """API legada — equivalente a Fase 1.5."""
        return self.phase1_compute_mualem_conductivities(
            sat, K_sat, L_param=L_param, m_param=m_param
        )

    def solve_unsaturated_flow(
        self,
        incidence_matrix: Any,
        edge_conductivities: np.ndarray,
        s_pumps: np.ndarray,
        node_elevations: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        API legada — equivalente a Fase 2.6.

        Incluye explícitamente la elevación en el potencial total H = ψ + z.
        """
        B = self._as_incidence_matrix(incidence_matrix)
        n_nodes, n_edges = B.shape
        K = self._broadcast_edge_array(
            "edge_conductivities", edge_conductivities, n_edges, lower=0.0
        )
        s = self._validate_vector("s_pumps", s_pumps, n_nodes)
        z = self._validate_vector("node_elevations", node_elevations, n_nodes)
        L = self.phase2_assemble_richards_laplacian(B, K)
        H, Q, _ = self._solve_richards_with_laplacian(L, B, K, s, z)
        return H, Q

    def evaluate_siphoning_instability(
        self,
        H: np.ndarray,
        incidence_matrix: Any,
        edge_lengths: np.ndarray,
        rho_sat: float = 2000.0,
        rho_w: float = 1000.0,
    ) -> Tuple[bool, np.ndarray]:
        """API legada — equivalente a Fase 2.8."""
        B = self._as_incidence_matrix(incidence_matrix)
        n_nodes, n_edges = B.shape
        H_v = self._validate_vector("hydraulic_head", H, n_nodes)
        lengths = self._broadcast_edge_array(
            "edge_lengths", edge_lengths, n_edges, lower=self._reg
        )
        return self._evaluate_siphoning(
            H_v, B, lengths, rho_sat=rho_sat, rho_w=rho_w
        )
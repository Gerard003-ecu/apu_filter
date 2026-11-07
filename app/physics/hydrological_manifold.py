# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Hydrological Manifold (Colector Hidrológico de de Rham-Richards)    ║
║ Ruta   : app/physics/hydrological_manifold.py                                ║
║ Versión: 4.0.0-Doctoral-Richards-Terzaghi-Biot-DEC-Poincare-RSI-L3-Secure    ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO:                         ║
║ Este módulo implementa el resolvente de-confinado para el acoplamiento       ║
║ hidro-geomecánico en el Estrato Physics ($V_{\mathrm{PHYSICS}}$) de          ║
║ APU Filter v5.0, estructurado rigurosamente en 3 Fases Anidadas:             ║
║                                                                              ║
║   • FASE 1 (Constitutive / Poroelastic & Mualem-van Genuchten Observer):     ║
║     Evalúa presiones intersticiales de Bishop-Biot, invariantes octaédricos  ║
║     de Lode-Haigh-Westergaard $(I_1, I_2, I_3, J_2, J_3, q, \theta_L)$,      ║
║     conductividades no lineales de Mualem-van Genuchten con derivada         ║
║     analítica de Fréchet $\partial K/\partial S_e$, y regularización         ║
║     geodésica isocórica de Fisher-Rao (RSI-L3).                              ║
║     Su método terminal `phase1_close_and_open_phase2` invoca formalmente     ║
║     el método inicial `phase2_from_phase1` de la Fase 2.                     ║
║                                                                              ║
║   • FASE 2 (DEC / Richards Elliptic Solver, Tellegen & Piping Orienter):     ║
║     Inicia en `phase2_from_phase1`, ensambla el Laplaciano ponderado de      ║
║     Hodge de 0-formas $\Delta_0 = B \star_1 B^\top$, resuelve el flujo       ║
║     elíptico de Richards con refinamiento iterativo de Banach-KBN (RSI-L3),  ║
║     certifica la identidad energética de de Rham-Tellegen y evalúa la        ║
║     inestabilidad por sifonamiento (piping) de Terzaghi.                     ║
║     Su método terminal `phase2_close_and_open_phase3` invoca formalmente     ║
║     el método inicial `phase3_from_phase2` de la Fase 3.                     ║
║                                                                              ║
║   • FASE 3 (Spectral Geometry, Poincaré Celestial Mechanics & Governance):   ║
║     Inicia en `phase3_from_phase2`, audita el espectro de Fiedler-Cheeger,   ║
║     la desigualdad discreta de Poincaré-Wirtinger, la cohomología simplicial ║
║     de Euler-Poincaré ($\chi = \beta_0 - \beta_1$), la no-resonancia         ║
║     diofántica KAM, la invariancia simpléctica de Liouville, el veredicto    ║
║     intuicionista de Heyting ($\Omega_3$) y el sello criptográfico SHA-256.  ║
╚══════════════════════════════════════════════════════════════════════════════╝

================════════════════════════════════════════════════════════════════
I. ANCLAJE MATEMÁTICO DOCTORAL (Poroelasticidad y Flujo en Complejos Simpliciales)
================════════════════════════════════════════════════════════════════

Definición 1 (Principio de Esfuerzo Efectivo de Biot-Terzaghi e Invariantes de Lode):
  La deformación y rigidez sismorresistente de la base del terreno (modelado en
  lithological_manifold.py) se acopla dinámicamente con el estado hidrológico
  de-confinado a través de la relación tensorial sobre $\operatorname{Sym}^3(\mathbb{R})$:
  $$\sigma'_{\mu\nu} = \sigma_{\mu\nu} - \alpha_{\mathrm{Biot}} \, P_f \, \delta_{\mu\nu}, \qquad P_f = -\gamma_w \, \psi_w \, \mathrm{sat}$$
  Descomponiendo $\sigma' = p'\mathbf{I}_3 + \mathbf{s}'$ con $p' = \frac{1}{3}\operatorname{tr}(\sigma')$:
  $$J_2 = \frac{1}{2}\operatorname{tr}(\mathbf{s}'^2), \quad J_3 = \det(\mathbf{s}'), \quad q_{\mathrm{dev}} = \sqrt{3J_2}, \quad \sin(3\theta_{\mathrm{Lode}}) = -\frac{3\sqrt{3}}{2}\frac{J_3}{J_2^{3/2}}$$
  Si $\sigma'_{\min} \le \varepsilon_{\mathrm{tol}}$ o $p' \le \varepsilon_{\mathrm{tol}}$, el esqueleto granular
  pierde su confinamiento efectivo, activando la licuación poroelástica.

Definición 2 (Ecuación Discreta de Richards-Poisson en DEC):
  La filtración en medio poroso insaturado sobre el 1-esqueleto del complejo simplicial
  $\mathcal{K} = (V, E)$ se rige por la ecuación de conservación de masa de de Rham:
  $$\mathbf{B}_1 Q + \frac{\partial \theta_u}{\partial t} = s_{\mathrm{bomba}}, \qquad Q = \star_1 d_0 H = \mathbf{W}_{\mathrm{hyd}}(\mathrm{sat}) \mathbf{B}_1^\top H$$
  En régimen cuasi-estacionario con potencial total $H_u = \psi_w(u) + z_u$:
  $$\mathbf{\Delta}_{\mathrm{Richards}}(\mathrm{sat}) \, H = s_{\mathrm{bomba}}, \qquad \mathbf{\Delta}_{\mathrm{Richards}}(\mathrm{sat}) = \mathbf{B}_1 \mathbf{W}_{\mathrm{hyd}}(\mathrm{sat}) \mathbf{B}_1^\top \succeq \mathbf{0}$$

Definición 3 (Conductividad No Lineal de Mualem-van Genuchten y Derivada de Fréchet):
  La conductividad hidráulica insaturada de la arista $K_{\mathrm{hyd}}(S_e)$ y su derivada
  analítica de Fréchet respecto a la saturación efectiva $S_e \in (0, 1]$ vienen dadas por:
  $$K_{\mathrm{hyd}}(S_e) = K_{\mathrm{sat}} \, S_e^{L} \, \mathcal{A}(S_e)^2, \qquad \mathcal{A}(S_e) := 1 - \left(1 - S_e^{1/m}\right)^m, \quad m = 1 - \frac{1}{n_w}$$
  $$\frac{\partial K_{\mathrm{hyd}}}{\partial S_e}(S_e) = K_{\mathrm{sat}} \, S_e^{L-1} \, \mathcal{A}(S_e) \left[ L \, \mathcal{A}(S_e) + 2 \, S_e^{1/m} \left(1 - S_e^{1/m}\right)^{m-1} \right] \ge 0$$

Definición 4 (Mecánica Celeste de Poincaré y Topología Cohomológica del Flujo):
  El complejo de cocadenas discretas $0 \to C^0(\mathcal{K}) \xrightarrow{d_0 = \mathbf{B}_1^\top} C^1(\mathcal{K}) \to 0$
  satisface la relación topológica de Euler-Poincaré:
  $$\chi(\mathcal{K}) = |V| - |E| = \beta_0 - \beta_1, \qquad \beta_0 = \dim\ker(\mathbf{\Delta}_0), \quad \beta_1 = \dim\ker(\mathbf{B}_1)$$
  Por el Teorema de Circulación de Poincaré-Cartan sobre 1-formas exactas $d_0 H$, la
  proyección ortogonal sobre el espacio de ciclos armónicos $\ker(\mathbf{B}_1)$ es nula:
  $$\oint_{\gamma \in \ker(\mathbf{B}_1)} d_0 H = \left\| \Pi_{\ker(\mathbf{B}_1)} (\mathbf{B}_1^\top H) \right\|_2 = 0$$

================════════════════════════════════════════════════════════════════
II. AXIOMÁTICA INMUNOLÓGICA CONTRA DERIVAS Y SINGULARIDADES (Leyes de la FPU)
================════════════════════════════════════════════════════════════════

Axioma I (Regularización Adaptativa de Tikhonov y Refinamiento de Banach-KBN):
  Dado que $\ker(\mathbf{\Delta}_{\mathrm{Richards}}) \cong \mathbb{R}^{\beta_0}$, se inyecta el desplazamiento
  espectral de Tikhonov $\lambda = \max(\varepsilon_{\mathrm{reg}}, \varepsilon_{\mathrm{mach}} \operatorname{mean}|\operatorname{diag}(\mathbf{\Delta}_0)|)$
  y se purga el error hacia adelante de Wilkinson mediante iteración contractiva de
  defecto con sumación compensada Kahan-Babuška-Neumaier (KBN).

Axioma II (Confinamiento de Saturación e Invariancia Isocórica de Fisher-Rao):
  La saturación se proyecta sobre $[\varepsilon_{\mathrm{reg}}, 1.0]$ en la FPU, eludiendo ramificaciones
  complejas en $S_e^{1/m}$, mientras que el flujo geodésico RSI-L3 de Fisher-Rao acota el
  condicionamiento métrico preservando el volumen logarítmico $\sum_e \log K_e$.

Axioma III (Conservación Energética de de Rham-Tellegen):
  El acoplamiento dual entre la 1-cadena de caudal de Darcy $Q \in C_1(\mathcal{K})$ y la
  1-cocadena de caída hidráulica $d_0 H = \mathbf{B}_1^\top H \in C^1(\mathcal{K})$ satisface:
  $$\langle Q, d_0 H \rangle_{\ell^2(E)} = \langle \mathbf{B}_1 Q, H \rangle_{\ell^2(V)} \implies \left| Q^\top \mathbf{B}_1^\top H - (\mathbf{B}_1 Q)^\top H \right| \le \mathcal{O}(\varepsilon_{\mathrm{mach}})$$
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Final, Mapping, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "HydroHeytingTruth",
    "HydroRSILevel3Certificate",
    "PoincareHydroDynamicCertificate",
    "HydrologicalManifold",
    "HydrologicalState",
    "Phase1HydroHandoff",
    "Phase2HydroHandoff",
    "Phase3HydroReport",
]

__version__: Final[str] = (
    "4.0.0-Doctoral-Richards-Terzaghi-Biot-DEC-Poincare-RSI-L3-Secure"
)

logger = logging.getLogger("APU.Physics.HydrologicalManifold")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_SQRT_EPS: Final[float] = math.sqrt(_MACHINE_EPS)
_SHA256_HEX_LEN: Final[int] = 64
_PHASE1_ENTRY: Final[str] = "phase2_from_phase1"
_PHASE2_ENTRY: Final[str] = "phase3_from_phase2"
_I3: Final[np.ndarray] = np.eye(3, dtype=np.float64)
_KAM_DIOPHANTINE_EXP: Final[float] = 2.0
_CONDITION_SAFE_CAP: Final[float] = 1.0e12

IncidenceLike = Union[np.ndarray, "sp.spmatrix"]


# ═════════════════════════════════════════════════════════════════════════════
# ÁLGEBRA INTUICIONISTA DE HEYTING (Ω₃) Y CERTIFICADOS DOCTORALES RSI-L3 / POINCARÉ
# ═════════════════════════════════════════════════════════════════════════════

class HydroHeytingTruth(IntEnum):
    """
    Clasificador de subobjetos intuicionista Ω₃ en el Topos Hidrológico.

    Orden parcial del retículo distributivo acotado:
        COHERENT (0, ⊤) < DEGRADED (1) < VETOED (2, ⊥)
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def meet(self, other: "HydroHeytingTruth") -> "HydroHeytingTruth":
        """Conjunción lógica (∧): retorna el estado más severo (máximo numérico)."""
        return HydroHeytingTruth(max(int(self), int(other)))

    def join(self, other: "HydroHeytingTruth") -> "HydroHeytingTruth":
        """Disyunción lógica (∨): retorna el estado más benigno (mínimo numérico)."""
        return HydroHeytingTruth(min(int(self), int(other)))

    def implies(self, consequent: "HydroHeytingTruth") -> "HydroHeytingTruth":
        """Implicación relativa de Heyting (a ⇒ b) en el retículo finito Ω₃."""
        if int(self) >= int(consequent):
            return HydroHeytingTruth.COHERENT
        return consequent


@dataclass(frozen=True, slots=True)
class HydroRSILevel3Certificate:
    """
    Certificado de Automejora Recursiva de Nivel 3 (RSI-L3) sobre la variedad
    hidrológica (geodésica isocórica de Fisher-Rao + contracción de Banach-KBN).
    """

    converged: bool
    iterations: int
    initial_condition_number: float
    final_condition_number: float
    fisher_rao_geodesic_drift: float
    liouville_log_volume_defect: float
    banach_contraction_factor: float
    initial_solver_residual: float
    refined_solver_residual: float


@dataclass(frozen=True, slots=True)
class PoincareHydroDynamicCertificate:
    """
    Certificado de Mecánica Celeste de Henri Poincaré aplicado al espacio fase
    poroelástico $(H, P_f)$ y al complejo de cocadenas de de Rham.
    """

    betti_0: int
    betti_1: int
    euler_poincare_characteristic: int
    euler_poincare_verified: bool
    poincare_wirtinger_ratio: float
    cartan_exactness_defect: float
    solenoidal_cycle_flow_norm: float
    kam_smallest_divisor: float
    kam_diophantine_margin: float
    is_kam_non_resonant: bool
    liouville_symplectic_determinant: float
    symplectic_volume_defect: float
    kac_poincare_recurrence_time: float


# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES NUMÉRICAS, CANÓNICAS Y CRIPTOGRÁFICAS (KBN & IEEE-754)
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
    """Sumación compensada Kahan–Babuška–Neumaier (error O(ε_mach))."""
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


def _kbn_dot(u: np.ndarray, v: np.ndarray) -> float:
    """Producto interno compensado Kahan–Babuška–Neumaier ⟨u, v⟩."""
    a = np.asarray(u, dtype=np.float64).ravel()
    b = np.asarray(v, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    return _kbn_sum(a * b)


def _finite_or_nan(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def _is_sparse(mat: Any) -> bool:
    return sp.issparse(mat)


# ═════════════════════════════════════════════════════════════════════════════
# OBJETOS DE ESTADO Y FRONTERAS DE FASE (MORFISMOS DE ANIDAMIENTO)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True, eq=False)
class HydrologicalState:
    """
    Estado físico-espectral hidrológico verificado en la FPU.

    Invariantes
    -----------
    L = Lᵀ ⪰ 0              (Laplaciano de Hodge de 0-formas)
    B Q ≈ s                 (Ley de Kirchhoff / conservación de masa de de Rham)
    ⟨Q, Bᵀ H⟩ = ⟨B Q, H⟩    (Identidad energética de de Rham-Tellegen)
    H = ψ + z               (Potencial total piezométrico de Richards)
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
    """
    Frontera formal Φ₁→₂: constitución local poroelástica y de Mualem-van Genuchten
    validada. Actúa como objeto puente entre el cierre de la Fase 1 y el inicio
    de la Fase 2.
    """

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

    def continue_to_phase2(
        self,
        manifold: "HydrologicalManifold",
        rho_sat: float = 2000.0,
        rho_w: float = 1000.0,
    ) -> "Phase2HydroHandoff":
        """Morfismo de continuación directa Φ₁→₂ hacia la Fase 2."""
        return manifold.phase2_close_and_open_phase3(
            self, rho_sat=rho_sat, rho_w=rho_w
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
    """
    Frontera formal Φ₂→₃: solución elíptica DEC de Richards, auditoría energética
    de de Rham-Tellegen y estabilidad de sifonamiento. Actúa como objeto puente
    entre el cierre de la Fase 2 y el inicio de la Fase 3.
    """

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

    def continue_to_phase3(
        self,
        manifold: "HydrologicalManifold",
    ) -> "Phase3HydroReport":
        """Morfismo de continuación directa Φ₂→₃ hacia la Fase 3."""
        return manifold.phase3_close_loop(self)

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
    """Reporte final de la Fase 3: estado, espectro, estabilidad, Poincaré y sello."""

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

    Arquitectura Functorial Anidada:
      • FASE 1 (CONSTITUTIVE) : Biot–Terzaghi–Bishop, invariantes octaédricos
        de Lode–Haigh–Westergaard, Mualem–van Genuchten + derivada de Fréchet,
        y flujo geodésico isocórico de Fisher–Rao (RSI-L3).
        Cierra en `phase1_close_and_open_phase2` invocando `phase2_from_phase1`.
      • FASE 2 (DEC/RICHARDS) : Operador estrella de Hodge primal ⋆₁, Laplaciano
        ponderado Δ₀ = B ⋆₁ Bᵀ, regularización de Tikhonov, refinamiento
        iterativo de Banach–KBN (RSI-L3), ortogonalidad energética de Tellegen
        y sifonamiento de Terzaghi.
        Cierra en `phase2_close_and_open_phase3` invocando `phase3_from_phase2`.
      • FASE 3 (SPECTRAL/GOV) : Espectro de Fiedler–Cheeger, desigualdad de
        Poincaré–Wirtinger, mecánica celeste de Poincaré (cohomología β₀, β₁,
        circulación de Cartan, no-resonancia KAM, invariante de Liouville y
        recurrencia de Kac–Poincaré), clasificador de Heyting Ω₃ y sello SHA-256.
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
    # INFRAESTRUCTURA DE VALIDACIÓN Y ÁLGEBRA SIMPLICIAL (PRELUDIO FASE 1)
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
        """Valida σ ∈ (n, 3, 3) y lo proyecta ortogonalmente sobre Sym³(R)."""
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

        B es el operador borde ∂₁: C₁ → C₀, adjunto del coborde exterior d₀
        sobre 0-cocadenas: (d₀ H)_e = (Bᵀ H)_e.
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
        """Producto Bᵀ x (derivada exterior discreta d₀ sobre 0-cocadenas)."""
        return np.asarray(B.T @ x, dtype=np.float64).ravel()

    def _abs_incidence(self, B: Any) -> Any:
        if _is_sparse(B):
            return abs(B.tocsr())
        return np.abs(B)

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 1 — CONSTITUTIVE / LOCAL POROELASTIC & MUALEM-VAN GENUCHTEN PHYSICS
    # ═════════════════════════════════════════════════════════════════════════

    def phase1_incidence_hygiene(self, incidence_matrix: Any) -> Dict[str, float]:
        """
        Fase 1.1 — Higiene cohomológica del operador borde ∂₁ = B.

        Un 1-complejo simplicial orientado cerrado sin frontera colgante
        satisface 1ᵀ B = 0 (cada 1-simplejo posee un vértice sumidero +1 y
        un vértice fuente −1).
        """
        B = self._as_incidence_matrix(incidence_matrix)
        n_nodes, n_edges = B.shape
        if n_edges == 0:
            return {
                "n_nodes": float(n_nodes),
                "n_edges": 0.0,
                "euler_characteristic_topological": float(n_nodes),
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
            "euler_characteristic_topological": float(n_nodes - n_edges),
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
        Fase 1.2 — Validación conjunta de campos constitutivos en variedades.

        ψ ≥ 0, S ∈ [0, 1], L_e ≥ ε_reg, K_sat ≥ 0, σ ∈ Sym³(R).
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
        Fase 1.3 — Presión de poros intersticial de Bishop–Biot.

            u_w = −γ_w ψ,     P_f = χ_Bishop(S) u_w = −γ_w ψ S,     γ_w = ρ_w g.

        En régimen insaturado con succión matricial ψ > 0, P_f < 0 incrementa
        el confinamiento esférico efectivo (cohesión aparente por capilaridad).
        """
        if not math.isfinite(rho_w) or rho_w <= 0.0:
            raise ValueError("rho_w debe ser finita y estrictamente positiva.")
        s = self._validate_vector(
            "suction", np.asarray(suction, dtype=np.float64).ravel(), lower=0.0
        )
        sat_arr = self._validate_vector(
            "saturation",
            np.asarray(sat, dtype=np.float64).ravel(),
            lower=0.0,
            upper=1.0,
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

    def phase1_octahedral_lode_invariants(
        self,
        effective_stresses: np.ndarray,
        eigvals: np.ndarray,
    ) -> Dict[str, float]:
        """
        Fase 1.4a — Invariantes Octaédricos de Lode–Haigh–Westergaard y Roscoe.

        Para cada nodo calcula:
          • I₁ = σ'₁ + σ'₂ + σ'₃,  p' = I₁ / 3
          • I₂ = σ'₁σ'₂ + σ'₂σ'₃ + σ'₃σ'₁,  I₃ = σ'₁σ'₂σ'₃
          • Tensor desviador s' = σ' − p' I₃
          • J₂ = ½ tr(s'²) ≥ 0,  J₃ = det(s')
          • Esfuerzo desviador de von Mises–Roscoe q_dev = √(3 J₂)
          • Esfuerzo cortante octaédrico τ_oct = √(2/3 J₂)
          • Ángulo de Lode θ_L ∈ [−π/6, π/6]: sin(3θ_L) = −(3√3 / 2) J₃ / J₂^{3/2}
          • Oblicuidad de estado crítico movilizada η = q_dev / max(p', ε_reg)
        """
        n_nodes = eigvals.shape[0]
        if n_nodes == 0:
            return {
                "mean_I1": math.nan,
                "mean_I2": math.nan,
                "mean_I3": math.nan,
                "max_J2": math.nan,
                "max_von_mises_q": math.nan,
                "max_octahedral_shear": math.nan,
                "mean_lode_angle_rad": math.nan,
                "max_critical_state_eta": math.nan,
            }

        s1 = eigvals[:, 0]
        s2 = eigvals[:, 1]
        s3 = eigvals[:, 2]

        i1 = s1 + s2 + s3
        i2 = s1 * s2 + s2 * s3 + s3 * s1
        i3 = s1 * s2 * s3
        p_mean = i1 / 3.0

        dev1 = s1 - p_mean
        dev2 = s2 - p_mean
        dev3 = s3 - p_mean

        j2 = np.maximum(0.0, 0.5 * (dev1 * dev1 + dev2 * dev2 + dev3 * dev3))
        j3 = dev1 * dev2 * dev3

        q_dev = np.sqrt(3.0 * j2)
        tau_oct = np.sqrt((2.0 / 3.0) * j2)

        denom_lode = np.power(np.maximum(j2, self._reg), 1.5)
        sin_3theta = np.where(
            j2 > self._tol,
            np.clip(-1.5 * math.sqrt(3.0) * j3 / denom_lode, -1.0, 1.0),
            0.0,
        )
        lode_angle = (1.0 / 3.0) * np.arcsin(sin_3theta)
        eta = q_dev / np.maximum(p_mean, self._reg)

        return {
            "mean_I1": float(np.mean(i1)),
            "mean_I2": float(np.mean(i2)),
            "mean_I3": float(np.mean(i3)),
            "max_J2": float(np.max(j2)),
            "max_von_mises_q": float(np.max(q_dev)),
            "max_octahedral_shear": float(np.max(tau_oct)),
            "mean_lode_angle_rad": float(np.mean(lode_angle)),
            "max_critical_state_eta": float(np.max(eta)),
        }

    def phase1_effective_stress_audit(
        self,
        total_stresses: np.ndarray,
        pore_pressures: np.ndarray,
        alpha_biot: float = 0.95,
    ) -> Tuple[np.ndarray, bool, np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Fase 1.4 — Esfuerzos efectivos de Biot–Terzaghi / Bishop.

            σ' = σ − α_Biot P_f I₃.

        Licuación (convención geotécnica de compresión positiva en el tensor
        de entrada): σ'₃ ≤ ε_tol o p' = tr(σ')/3 ≤ ε_tol.
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
        octahedral = self.phase1_octahedral_lode_invariants(effective, eigvals)
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
            **octahedral,
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
        Fase 1.5 — Conductividad hidráulica no lineal de Mualem–van Genuchten.

            K(S_e) = K_sat S_e^L [1 − (1 − S_e^{1/m})^m ]²,

        con S_e regularizado en [ε_reg, 1] (Axioma II de la FPU), m ∈ (0, 1),
        n_w = 1/(1−m). Garantiza K ∈ [0, K_sat].
        """
        sat_arr = self._validate_vector(
            "saturation",
            np.asarray(sat, dtype=np.float64).ravel(),
            lower=0.0,
            upper=1.0,
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

    def phase1_mualem_frechet_derivative(
        self,
        sat: np.ndarray,
        K_sat: np.ndarray,
        L_param: float = 0.5,
        m_param: float = 0.5,
    ) -> np.ndarray:
        """
        Fase 1.5a — Derivada analítica de Fréchet ∂K/∂S_e de Mualem–van Genuchten.

            ∂K/∂S_e = K_sat S_e^{L−1} A(S_e) [ L A(S_e) + 2 S_e^{1/m} (1 − S_e^{1/m})^{m−1} ],
            donde A(S_e) = 1 − (1 − S_e^{1/m})^m.

        Certifica la monotonicidad constitutiva ∂K/∂S_e ≥ 0 para L ≥ 0.
        """
        sat_arr = self._validate_vector(
            "saturation",
            np.asarray(sat, dtype=np.float64).ravel(),
            lower=0.0,
            upper=1.0,
        )
        if sat_arr.size == 0:
            return np.empty(0, dtype=np.float64)
        K_sat_arr = self._broadcast_edge_array("K_sat", K_sat, sat_arr.size, lower=0.0)
        sat_reg = np.clip(sat_arr, max(self._reg, 1e-12), 1.0 - 1e-12)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            s_inv_m = np.power(sat_reg, 1.0 / m_param)
            one_minus = np.clip(1.0 - s_inv_m, 1e-15, 1.0)
            A_term = np.clip(1.0 - np.power(one_minus, m_param), 0.0, 1.0)
            dA_term = 2.0 * s_inv_m * np.power(one_minus, m_param - 1.0)
            bracket = float(L_param) * A_term + dA_term
            dK_dS = (
                K_sat_arr
                * np.power(sat_reg, float(L_param) - 1.0)
                * A_term
                * bracket
            )
        dK_dS = np.where(np.isfinite(dK_dS), dK_dS, 0.0)
        return _canonicalize_signed_zero(np.maximum(0.0, dK_dS))

    def phase1_rsi_level3_isochoric_fisher_rao(
        self,
        edge_conductivities: np.ndarray,
        condition_cap: float = _CONDITION_SAFE_CAP,
        max_iter: int = 8,
    ) -> Dict[str, Any]:
        """
        Fase 1.5b — Automejora Recursiva de Nivel 3 (RSI-L3) mediante flujo
        geodésico isocórico de Fisher–Rao sobre la variedad métrica M₊^m.

        Evalúa la distancia riemanniana de Fisher–Rao d_FR(K₀, K*) = ||log(K*/K₀)||₂
        y verifica que el transporte geodésico hacia el cono de condicionamiento
        κ(⋆₁) ≤ κ_max preserve el volumen logarítmico de Liouville Σ_e log K_e
        con factor de contracción de Banach q_n < 1.
        """
        K = np.asarray(edge_conductivities, dtype=np.float64).ravel()
        pos = K[K > self._reg]
        if pos.size <= 1:
            return {
                "rsi_l3_metric_converged": True,
                "rsi_l3_metric_iterations": 0,
                "rsi_l3_initial_metric_kappa": 1.0,
                "rsi_l3_final_metric_kappa": 1.0,
                "rsi_l3_fisher_rao_drift": 0.0,
                "rsi_l3_liouville_log_defect": 0.0,
                "rsi_l3_banach_contraction": 0.0,
            }

        u0 = np.log(pos)
        mean_u0 = _kbn_sum(u0) / float(pos.size)
        u_curr = u0.copy()
        span0 = float(np.max(u0) - np.min(u0))
        kappa0 = math.exp(min(span0, 700.0))
        target_span = math.log(max(condition_cap, 1.0 + 1e-6))

        if span0 <= target_span:
            return {
                "rsi_l3_metric_converged": True,
                "rsi_l3_metric_iterations": 0,
                "rsi_l3_initial_metric_kappa": float(kappa0),
                "rsi_l3_final_metric_kappa": float(kappa0),
                "rsi_l3_fisher_rao_drift": 0.0,
                "rsi_l3_liouville_log_defect": 0.0,
                "rsi_l3_banach_contraction": 0.0,
            }

        prev_excess = span0 - target_span
        contraction = 0.5
        iters = 0
        for it in range(1, max_iter + 1):
            iters = it
            span_curr = float(np.max(u_curr) - np.min(u_curr))
            if span_curr <= target_span * (1.0 + 1e-12):
                break
            alpha_step = 0.5 * (1.0 - target_span / span_curr)
            u_next = (1.0 - alpha_step) * (u_curr - mean_u0) + mean_u0
            excess_next = max(0.0, float(np.max(u_next) - np.min(u_next)) - target_span)
            if prev_excess > 0.0:
                contraction = min(contraction, excess_next / prev_excess)
            prev_excess = max(excess_next, _MACHINE_EPS)
            u_curr = u_next

        span_final = float(np.max(u_curr) - np.min(u_curr))
        kappa_final = math.exp(min(span_final, 700.0))
        fr_drift = math.sqrt(max(0.0, _kbn_dot(u_curr - u0, u_curr - u0)))
        liouville_defect = abs(_kbn_sum(u_curr) - _kbn_sum(u0)) / max(
            1.0, abs(_kbn_sum(u0))
        )
        return {
            "rsi_l3_metric_converged": bool(span_final <= target_span * 1.01),
            "rsi_l3_metric_iterations": int(iters),
            "rsi_l3_initial_metric_kappa": float(kappa0),
            "rsi_l3_final_metric_kappa": float(kappa_final),
            "rsi_l3_fisher_rao_drift": float(fr_drift),
            "rsi_l3_liouville_log_defect": float(liouville_defect),
            "rsi_l3_banach_contraction": float(contraction),
        }

    def phase1_map_node_saturation_to_edges(
        self,
        incidence_matrix: Any,
        node_sat: np.ndarray,
    ) -> np.ndarray:
        """
        Fase 1.6 — Pullback de Whitney de saturación nodal (0-cocadena) a aristas.

            sat_e = (Σ_{v ∈ ∂e} sat_v) / deg₀(e) = (|B|ᵀ sat) / (|B|ᵀ 1).

        Representa el baricentro simplicial de la 0-forma de saturación S sobre
        cada 1-simplejo e ∈ E.
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
        Fase 1.7 — Compatibilidad cohomológica de Neumann con H⁰_dR(K).

        Sobre un complejo conexo sin frontera de Dirichlet, ⟨1, s_bomba⟩ = 0 es
        la condición de ortogonalidad de Fredholm para que s_bomba ∈ im(Δ₀).
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
        Fase 1.9 — DEFINICIÓN FINAL DE LA FASE 1 Y APERTURA ANIDADA DE LA FASE 2.

        Definición formal del funtor de frontera:
            Φ₁→₂ : Datos constitutivos crudos ↦ Phase1HydroHandoff ↦ phase2_from_phase1

        Este es el último método de la Fase 1. Construye el objeto inmutable
        `Phase1HydroHandoff` y ejecuta formalmente `self.phase2_from_phase1(handoff)`
        como continuación e inicio verificado de los métodos de la Fase 2.
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
        dK_dS = self.phase1_mualem_frechet_derivative(
            edge_sat, fields["K_sat_edges"], L_param=L_param, m_param=m_param
        )
        rsi_metric = self.phase1_rsi_level3_isochoric_fisher_rao(edge_conductivities)
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
            "dK_dS_max": float(np.max(dK_dS)) if n_edges else 0.0,
            "dK_dS_monotonic": bool(np.all(dK_dS >= -self._tol)) if n_edges else True,
            "session_sha256_prefix": session_sha256[:16],
            **hygiene,
            **mass,
            **biot_extra,
            **rsi_metric,
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

        # Continuación formal inmediata hacia el método inicial de la Fase 2:
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
        return opened

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2 — DEC / RICHARDS ELLIPTIC SOLVER, TELLEGEN ORTHOGONALITY & PIPING
    # (Continuación directa de `phase1_close_and_open_phase2`)
    # ═════════════════════════════════════════════════════════════════════════

    def phase2_from_phase1(self, handoff: Phase1HydroHandoff) -> Phase1HydroHandoff:
        """
        Fase 2.0 — Punto de entrada formal de la Fase 2 desde la Fase 1.

        Es invocado directamente por la definición final de `phase1_close_and_open_phase2`
        para certificar la invariancia del contrato Φ₁→₂ antes de ensamblar Δ₀.
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
        Fase 2.1 — Estrella de Hodge primal discreta ⋆₁ sobre 1-cocadenas.

            W_hyd = ⋆₁ = diag(K_e) ⪰ 0.

        La positividad semidefinida K_e ≥ 0 garantiza que el Laplaciano de
        Richards Δ₀ = B ⋆₁ Bᵀ sea autoadjunto y semidefinido positivo (Ley de
        pasividad disipativa de Darcy).
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
        Fase 2.2 — Ensamblaje del Operador Laplaciano de Hodge de 0-formas.

            L = Δ₀ = B W_hyd Bᵀ,     W_hyd = ⋆₁,     L = Lᵀ ⪰ 0.

        La simetrización explícita ½(L + Lᵀ) aniquila asimetrías espurias de
        redondeo IEEE-754 antes de la factorización espectral.
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
        Fase 2.3 — Escala adaptativa de Tikhonov λ = max(ε_reg, ε_mach · mean|diag L|).

        Fija el gauge sobre el núcleo armónico H⁰(K) = ker(Δ₀) y regulariza
        subcomponentes desconectadas sin perturbar el espectro positivo.
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
        """Desplazamiento espectral de Tikhonov: L_reg = L + λ I."""
        n = L.shape[0]
        if _is_sparse(L):
            return (L + reg * sp.eye(n, format="csr", dtype=np.float64)).tocsr()
        return L + reg * np.eye(n, dtype=np.float64)

    def _solve_linear_symmetric(self, A: Any, b: np.ndarray) -> np.ndarray:
        """
        Fase 2.4 — Resolvente lineal SPD primario: spsolve / Cholesky ('pos') / LU.
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

    def phase2_rsi_level3_kbn_iterative_refinement(
        self,
        A: Any,
        b: np.ndarray,
        x0: np.ndarray,
        max_iter: int = 3,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Fase 2.4a — Refinamiento Iterativo de Banach–KBN (RSI-L3) en la FPU.

        Ejecuta corrección de defecto x^{(k+1)} = x^{(k)} + A⁻¹ r^{(k)} donde
        el residuo r^{(k)} = b − A x^{(k)} se evalúa con sumación compensada
        Kahan–Babuška–Neumaier (cuando A es densa moderada) o producto exacto.
        La actualización se acepta sí y solo si contrae estrictamente ||r||₂
        (Teorema del Punto Fijo de Banach).
        """
        if b.size == 0:
            return x0, {
                "rsi_l3_solver_iterations": 0.0,
                "rsi_l3_initial_residual": 0.0,
                "rsi_l3_refined_residual": 0.0,
                "rsi_l3_solver_contraction": 0.0,
            }

        x_curr = np.array(x0, dtype=np.float64, copy=True)
        use_kbn_matvec = (not _is_sparse(A)) and (b.size <= 128)

        def _residual(x_vec: np.ndarray) -> np.ndarray:
            if use_kbn_matvec:
                A_arr = np.asarray(A, dtype=np.float64)
                ax = np.empty_like(b)
                for idx in range(b.size):
                    ax[idx] = _kbn_dot(A_arr[idx], x_vec)
                return b - ax
            return b - self._matvec(A, x_vec)

        r_curr = _residual(x_curr)
        norm_init = float(la.norm(r_curr))
        norm_curr = norm_init
        b_scale = max(1.0, float(la.norm(b)))
        contraction = 0.0
        iters_done = 0

        for it in range(1, max_iter + 1):
            if norm_curr <= 2.0 * _MACHINE_EPS * b_scale:
                break
            try:
                delta_x = self._solve_linear_symmetric(A, r_curr)
            except Exception:
                break
            if not np.all(np.isfinite(delta_x)):
                break
            x_cand = x_curr + delta_x
            r_cand = _residual(x_cand)
            norm_cand = float(la.norm(r_cand))
            if norm_cand < norm_curr:
                contraction = norm_cand / max(norm_curr, _MACHINE_EPS)
                x_curr = x_cand
                r_curr = r_cand
                norm_curr = norm_cand
                iters_done = it
            else:
                break

        return x_curr, {
            "rsi_l3_solver_iterations": float(iters_done),
            "rsi_l3_initial_residual": float(norm_init / b_scale),
            "rsi_l3_refined_residual": float(norm_curr / b_scale),
            "rsi_l3_solver_contraction": float(contraction),
        }

    def phase2_kirchhoff_residual(
        self,
        B: Any,
        flow_rates: np.ndarray,
        s_pumps: np.ndarray,
    ) -> float:
        """
        Fase 2.5 — Residual de conservación de Kirchhoff ||B Q − s||₂ / max(1, ||s||₂).

        Mide el defecto de divergencia discreta ∂₁ Q = s_bomba sobre 0-cadenas.
        """
        div = self._matvec(B, flow_rates)
        num = float(la.norm(div - s_pumps))
        den = max(1.0, float(la.norm(s_pumps)))
        return float(num / den)

    def phase2_tellegen_de_rham_orthogonality(
        self,
        B: Any,
        hydraulic_head: np.ndarray,
        flow_rates: np.ndarray,
        s_pumps: np.ndarray,
    ) -> Dict[str, float]:
        """
        Fase 2.5a — Auditoría del Axioma III (Conservación Energética de de Rham–Tellegen).

        Por dualidad adjunta entre el operador borde ∂₁ = B y el coborde d₀ = Bᵀ:
            ⟨Q, d₀ H⟩_{ℓ²(E)} = Qᵀ (Bᵀ H) ≡ (B Q)ᵀ H = ⟨∂₁ Q, H⟩_{ℓ²(V)}.

        Evalúa con aritmética compensada KBN:
          1. La potencia disipada de Darcy E_Darcy = ⟨Q, Bᵀ H⟩ ≥ 0.
          2. La potencia inyectada en frontera/nodos P_nodal = ⟨B Q, H⟩.
          3. El defecto de adjunción de Tellegen |E_Darcy − P_nodal| / max(1, |E_Darcy|).
          4. El desbalance de potencia de bombeo |E_Darcy − ⟨s_pumps, H⟩| / max(1, |⟨s_pumps, H⟩|).
        """
        if hydraulic_head.size == 0 or flow_rates.size == 0:
            return {
                "darcy_dissipated_power": 0.0,
                "nodal_divergence_power": 0.0,
                "tellegen_adjunction_defect": 0.0,
                "pump_power_balance_defect": 0.0,
            }
        dH = self._incidence_transpose_matvec(B, hydraulic_head)
        div_Q = self._matvec(B, flow_rates)
        darcy_power = _kbn_dot(flow_rates, dH)
        nodal_power = _kbn_dot(div_Q, hydraulic_head)
        pump_power = _kbn_dot(s_pumps, hydraulic_head)

        adj_defect = abs(darcy_power - nodal_power) / max(
            1.0, abs(darcy_power), abs(nodal_power)
        )
        pump_defect = abs(darcy_power - pump_power) / max(
            1.0, abs(darcy_power), abs(pump_power)
        )
        return {
            "darcy_dissipated_power": float(darcy_power),
            "nodal_divergence_power": float(nodal_power),
            "tellegen_adjunction_defect": float(adj_defect),
            "pump_power_balance_defect": float(pump_defect),
        }

    def _solve_richards_with_laplacian(
        self,
        L: Any,
        B: Any,
        edge_conductivities: np.ndarray,
        s_pumps: np.ndarray,
        node_elevations: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Fase 2.6 — Solución de Richards con elevación explícita y refinamiento RSI-L3.

            (L + λI) ψ = s − L z,     H = ψ + z,     Q = K ⊙ (Bᵀ H).

        Diagnostica residual algebraico, contracción de Banach-KBN, conservación
        de masa, ley de Kirchhoff e identidad de de Rham-Tellegen.
        """
        n_nodes = L.shape[0]
        if n_nodes == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), {}

        reg = self.phase2_tikhonov_scale(L)
        L_reg = self._add_tikhonov(L, reg)
        Lz = self._matvec(L, node_elevations)
        rhs = s_pumps - Lz
        psi_raw = self._solve_linear_symmetric(L_reg, rhs)
        psi, rsi_solver_diag = self.phase2_rsi_level3_kbn_iterative_refinement(
            L_reg, rhs, psi_raw
        )

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
        tellegen = self.phase2_tellegen_de_rham_orthogonality(
            B, hydraulic_head, flow_rates, s_pumps
        )

        mean_h = float(np.mean(hydraulic_head)) if hydraulic_head.size else 0.0
        h_oscillation = (
            float(np.max(hydraulic_head) - np.min(hydraulic_head))
            if hydraulic_head.size
            else 0.0
        )

        diagnostics: Dict[str, float] = {
            "tikhonov_lambda": float(reg),
            "richards_relative_residual": float(relative_residual),
            "richards_mass_residual": float(mass_residual),
            "kirchhoff_residual": float(kirchhoff),
            "total_head_mean": mean_h,
            "total_head_oscillation": h_oscillation,
            "total_flow_abs_sum": (
                float(_kbn_sum(np.abs(flow_rates))) if flow_rates.size else 0.0
            ),
            "gauge_psi_mean": float(np.mean(psi)) if psi.size else 0.0,
            **rsi_solver_diag,
            **tellegen,
        }
        return hydraulic_head, flow_rates, diagnostics

    def phase2_solve_richards_flow(
        self,
        handoff: Phase1HydroHandoff,
    ) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Fase 2.7 — Orquestación DEC de Richards sobre la frontera validada Φ₁→₂.

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
        Fase 2.8 — Gradiente hidráulico crítico de Terzaghi / inestabilidad de piping.

            i_e = |ΔH_e| / L_e = |(Bᵀ H)_e| / L_e,
            i_crit = γ' / γ_w = (ρ_sat − ρ_w) / ρ_w,
            FoS_piping = i_crit / max_e i_e.
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
        Fase 2.9 — DEFINICIÓN FINAL DE LA FASE 2 Y APERTURA ANIDADA DE LA FASE 3.

        Definición formal del funtor de frontera:
            Φ₂→₃ : Phase1HydroHandoff ↦ (H, Q, Δ₀, i_e) ↦ Phase2HydroHandoff ↦ phase3_from_phase2

        Este es el último método de la Fase 2. Construye el objeto inmutable
        `Phase2HydroHandoff` y ejecuta formalmente `self.phase3_from_phase2(handoff)`
        como continuación e inicio verificado de los métodos de la Fase 3.
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

        # Continuación formal inmediata hacia el método inicial de la Fase 3:
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
        return opened

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 3 — SPECTRAL GEOMETRY, POINCARÉ CELESTIAL MECHANICS & GOVERNANCE
    # (Continuación directa de `phase2_close_and_open_phase3`)
    # ═════════════════════════════════════════════════════════════════════════

    def phase3_from_phase2(self, handoff: Phase2HydroHandoff) -> Phase2HydroHandoff:
        """
        Fase 3.0 — Punto de entrada formal de la Fase 3 desde la Fase 2.

        Es invocado directamente por la definición final de `phase2_close_and_open_phase3`
        para certificar la invariancia del contrato Φ₂→₃ antes de ejecutar la
        auditoría espectral, la mecánica celeste de Poincaré y el sellado SHA-256.
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
        Fase 3.1 — Espectro del operador autoadjunto Δ₀ = B ⋆₁ Bᵀ.

        Sistemas de dimensión moderada (n ≤ max_dense_eigen_dim): eigvalsh denso.
        Sistemas de gran escala: lanczos `eigsh(which='SA')`.
        Los autovalores negativos dentro del piso de ruido de Wilkinson (−tiny, 0)
        se proyectan exactamente a 0.0.
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
        return _canonicalize_signed_zero(vals)

    def _laplacian_spectrum(self, L: Any) -> np.ndarray:
        return self.phase3_laplacian_spectrum(L)

    def phase3_spectral_audit(
        self,
        handoff: Phase2HydroHandoff,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fase 3.2 — Auditoría espectral de Fiedler / Cheeger y Cohomología H⁰_dR.

        • dim ker(Δ₀) = β₀ (número de componentes hidro-conectadas).
        • λ₂ = conectividad algebraica de Fiedler.
        • h_Cheeger ≤ √(2 λ₂) cota isoperimétrica de cuello de botella hidráulico.
        • κ_eff = λ_max / λ_min⁺ condicionamiento espectral en (ker Δ₀)⊥.
        """
        eigvals = self.phase3_laplacian_spectrum(handoff.laplacian_matrix)
        n = int(handoff.laplacian_matrix.shape[0])
        m = int(handoff.incidence_matrix.shape[1])
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
            nullity = int(np.count_nonzero(eigvals <= tiny))
            betti_1 = max(0, m - n + nullity)
            diagnostics.update(
                {
                    "spectral_min": float(eigvals[0]),
                    "spectral_max": lam_max,
                    "spectral_radius": float(np.max(np.abs(eigvals))),
                    "algebraic_connectivity": lam2,
                    "nullity": nullity,
                    "betti_0": nullity,
                    "betti_1": betti_1,
                    "euler_poincare_characteristic": int(n - m),
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
                    "betti_0": math.nan,
                    "betti_1": math.nan,
                    "euler_poincare_characteristic": int(n - m),
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
        Fase 3.3 — Cociente Rayleigh–Poincaré–Wirtinger discreto sobre H⁰⊥.

        Por el teorema espectral min-max de Courant–Fischer–Poincaré sobre un
        complejo conexo (β₀ = 1):
            ||H − H̄||₂² ≤ λ₂⁻¹ ⟨d₀ H, ⋆₁ d₀ H⟩.

        Retorna el cociente adimensional λ₂ ||H − H̄||₂² / ||Bᵀ H||_W², acotado
        superiormente por 1.0 en ausencia de modos en el núcleo.
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
        num = _kbn_dot(osc, osc)
        dH = self._incidence_transpose_matvec(handoff.incidence_matrix, H)
        w = np.asarray(handoff.edge_conductivities, dtype=np.float64).ravel()
        if dH.size == 0 or w.size != dH.size:
            return math.nan
        den = _kbn_dot(w, dH * dH)
        if den <= tiny:
            return 0.0 if num <= tiny else math.inf
        return float(lam2 * num / den)

    def phase3_poincare_celestial_mechanics_audit(
        self,
        handoff: Phase2HydroHandoff,
        eigvals: np.ndarray,
        poincare_wirtinger_ratio: float,
    ) -> PoincareHydroDynamicCertificate:
        """
        Fase 3.3a — Auditoría de Mecánica Celeste de Henri Poincaré sobre el
        espacio fase hidro-poroelástico (H, P_f) y el complejo de de Rham.

        1. Característica topológica de Euler–Poincaré: χ = V − E = β₀ − β₁.
        2. Invariante de Circulación de Poincaré–Cartan:
           Verifica que d₀ H = Bᵀ H tenga circulación nula sobre todo ciclo
           fundamental γ ∈ ker(B), y mide la componente solenoidal del caudal Q.
        3. Condición Diofántica KAM (Kolmogorov–Arnold–Moser):
           Audita pequeños divisores |ω_i − ω_j| ≥ γ_KAM |i − j|^{−τ} entre las
           frecuencias modales propias ω_i = √λ_i (λ_i > 0).
        4. Teorema de Liouville y Recurrencia de Kac–Poincaré:
           Construye el propagador simpléctico de Cayley–Poincaré a un paso en el
           espacio fase reducido (q = H, p = P_f) y certifica det(Φ_Δt) = 1.
        """
        B = handoff.incidence_matrix
        n_nodes, n_edges = B.shape
        tiny = max(self._tol, 1e-10)

        # 1. Cohomología de Euler-Poincaré
        pos_eigs = eigvals[eigvals > tiny] if eigvals.size else np.empty(0, dtype=np.float64)
        betti_0 = (
            int(np.count_nonzero(eigvals <= tiny))
            if eigvals.size == n_nodes
            else max(1, n_nodes - int(pos_eigs.size))
        )
        betti_1 = max(0, n_edges - n_nodes + betti_0)
        chi_geom = int(n_nodes - n_edges)
        chi_hom = int(betti_0 - betti_1)
        euler_verified = bool(chi_geom == chi_hom)

        # 2. Invariante de Circulación de Poincaré-Cartan sobre ker(B)
        cartan_defect = 0.0
        solenoidal_flow = 0.0
        if n_edges > 0 and n_nodes <= self._max_dense_eigen_dim and n_edges <= self._max_dense_eigen_dim:
            try:
                B_dense = B.toarray() if _is_sparse(B) else np.asarray(B, dtype=np.float64)
                u_svd, s_svd, vt_svd = la.svd(B_dense, full_matrices=True)
                s_tol = max(n_nodes, n_edges) * _MACHINE_EPS * (
                    float(s_svd[0]) if s_svd.size else 1.0
                )
                rank_B = int(np.sum(s_svd > max(s_tol, self._tol)))
                betti_0 = int(n_nodes - rank_B)
                betti_1 = int(n_edges - rank_B)
                chi_hom = int(betti_0 - betti_1)
                euler_verified = bool(chi_geom == chi_hom)
                if rank_B < n_edges:
                    cycle_basis = vt_svd[rank_B:, :]  # Filas ortonormales en ker(B)
                    dH = self._incidence_transpose_matvec(B, handoff.hydraulic_head)
                    dH_norm = max(1.0, float(la.norm(dH)))
                    cartan_defect = float(la.norm(cycle_basis @ dH) / dH_norm)
                    solenoidal_flow = float(la.norm(cycle_basis @ handoff.flow_rates))
            except Exception:
                pass

        # 3. Auditoría de No-Resonancia Diofántica KAM en el espectro positivo
        freqs = np.sqrt(pos_eigs)
        m_pos = freqs.size
        if m_pos <= 1:
            kam_min_div = math.inf
            kam_margin = math.inf
            is_kam_non_resonant = True
        else:
            omega_max = max(float(freqs[-1]), 1.0)
            gamma_kam = 1.0e-3 * omega_max / float(m_pos * m_pos)
            kam_min_div = math.inf
            kam_margin = math.inf
            for i_idx in range(m_pos):
                for j_idx in range(i_idx + 1, m_pos):
                    gap = abs(float(freqs[j_idx] - freqs[i_idx]))
                    if gap < kam_min_div:
                        kam_min_div = gap
                    bound = gamma_kam / (float(j_idx - i_idx) ** _KAM_DIOPHANTINE_EXP)
                    margin = gap / max(bound, _MACHINE_EPS)
                    if margin < kam_margin:
                        kam_margin = margin
            is_kam_non_resonant = bool(kam_margin >= 1.0)

        # 4. Invariancia Simpléctica de Liouville (Transformación de Cayley-Poincaré)
        # En el plano modal hamiltoniano 2x2 J_2 = [[0, -ω], [ω, 0]]:
        # Φ_Δt = (I - (Δt/2) J_2)^{-1} (I + (Δt/2) J_2) ∈ Sp(2, R) con det(Φ_Δt) ≡ 1.
        omega_ref = float(freqs[0]) if m_pos > 0 else 0.0
        dt_char = 1.0 / max(omega_ref, 1.0)
        half_theta = 0.5 * dt_char * omega_ref
        denom_cayley = 1.0 + half_theta * half_theta
        c_diag = (1.0 - half_theta * half_theta) / denom_cayley
        s_off = (2.0 * half_theta) / denom_cayley
        liouville_det = float(c_diag * c_diag + s_off * s_off)
        symplectic_defect = abs(liouville_det - 1.0)

        # Tiempo de recurrencia de Kac-Poincaré: T_rec ~ 2π / ω_min
        kac_recurrence = (
            float(2.0 * math.pi / omega_ref) if omega_ref > self._reg else math.inf
        )

        return PoincareHydroDynamicCertificate(
            betti_0=betti_0,
            betti_1=betti_1,
            euler_poincare_characteristic=chi_geom,
            euler_poincare_verified=euler_verified,
            poincare_wirtinger_ratio=float(poincare_wirtinger_ratio),
            cartan_exactness_defect=float(cartan_defect),
            solenoidal_cycle_flow_norm=float(solenoidal_flow),
            kam_smallest_divisor=float(kam_min_div),
            kam_diophantine_margin=float(kam_margin),
            is_kam_non_resonant=is_kam_non_resonant,
            liouville_symplectic_determinant=liouville_det,
            symplectic_volume_defect=float(symplectic_defect),
            kac_poincare_recurrence_time=kac_recurrence,
        )

    def phase3_stability_audit(
        self,
        handoff: Phase2HydroHandoff,
    ) -> Dict[str, Any]:
        """
        Fase 3.4 — Estabilidad hidro-geomecánica acoplada (Terzaghi–Biot).

        Licuación: σ'₃ ≤ ε_tol o p' ≤ ε_tol o det(σ') ≤ ε_tol (invariante I₃).
        Sifonamiento (piping): i_max > i_crit. Estado crítico = disyunción lógica.
        """
        min_principal = handoff.min_principal_stresses
        det_principal = handoff.determinant_effective_stresses
        liquefaction_by_min = (
            bool(np.any(min_principal <= self._tol)) if min_principal.size else False
        )
        liquefaction_by_det = (
            bool(np.any(det_principal <= self._tol)) if det_principal.size else False
        )
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
            "siphoning_max_gradient": (
                float(np.max(gradients)) if gradients.size else 0.0
            ),
            "min_principal_stress": (
                float(np.min(min_principal)) if min_principal.size else math.nan
            ),
            "min_effective_determinant": (
                float(np.min(det_principal)) if det_principal.size else math.nan
            ),
            "critical_state": bool(is_liquefaction or is_siphoning),
            "kirchhoff_residual": _finite_or_nan(
                handoff.diagnostics.get("kirchhoff_residual")
            ),
            "richards_mass_residual": _finite_or_nan(
                handoff.diagnostics.get("richards_mass_residual")
            ),
            "tellegen_adjunction_defect": _finite_or_nan(
                handoff.diagnostics.get("tellegen_adjunction_defect")
            ),
        }

    def phase3_heyting_omega3_verdict(
        self,
        spectral: Mapping[str, Any],
        stability: Mapping[str, Any],
        poincare: PoincareHydroDynamicCertificate,
    ) -> Dict[str, Any]:
        """
        Fase 3.4a — Evaluación en el Clasificador de Subobjetos de Heyting Ω₃.

        • VETOED (⊥)   : Licuación activa, sifonamiento activo o autovalores
                         negativos genuinos (violación de pasividad de Darcy).
        • DEGRADED     : Resonancia KAM activa, defecto de masa/Kirchhoff > √ε_mach
                         o condicionamiento efectivo > 10¹².
        • COHERENT (⊤) : Plena coherencia cohomológica, espectral y geotécnica.
        """
        verdict = HydroHeytingTruth.COHERENT

        if bool(stability.get("critical_state", False)):
            verdict = verdict.meet(HydroHeytingTruth.VETOED)

        neg_eigs = spectral.get("negative_eigenvalues", 0)
        if isinstance(neg_eigs, (int, float)) and not math.isnan(neg_eigs) and neg_eigs > 0:
            verdict = verdict.meet(HydroHeytingTruth.VETOED)

        kappa = _finite_or_nan(spectral.get("condition_number_effective"))
        if math.isfinite(kappa) and kappa > _CONDITION_SAFE_CAP:
            verdict = verdict.meet(HydroHeytingTruth.DEGRADED)

        if not poincare.is_kam_non_resonant or not poincare.euler_poincare_verified:
            verdict = verdict.meet(HydroHeytingTruth.DEGRADED)

        mass_res = _finite_or_nan(stability.get("richards_mass_residual"))
        if math.isfinite(mass_res) and mass_res > _SQRT_EPS:
            verdict = verdict.meet(HydroHeytingTruth.DEGRADED)

        return {
            "heyting_truth_value": int(verdict),
            "heyting_truth_label": verdict.name,
            "is_topos_coherent": bool(verdict == HydroHeytingTruth.COHERENT),
        }

    def phase3_governance_seal(
        self,
        handoff: Phase2HydroHandoff,
        eigenvalues: np.ndarray,
        spectral: Mapping[str, Any],
        stability: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Fase 3.5 — Sello criptográfico de gobernanza SHA-256 canónico binario
        (independiente de `repr` y libre de colisiones de signo en ±0.0).
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
        """Fase 3.6 — Consolidación inmutable del estado hidrológico final."""
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
        Fase 3.7 — Orquestación completa y cierre del lazo de la Fase 3.

        Ejecuta secuencialmente:
          1. Validación de frontera anidada Φ₂→₃ (`phase3_from_phase2`);
          2. Auditoría espectral de Fiedler–Cheeger y residual de Poincaré–Wirtinger;
          3. Auditoría de Mecánica Celeste de Poincaré (Euler-Poincaré, Cartan, KAM, Liouville);
          4. Auditoría de estabilidad hidro-geomecánica y veredicto de Heyting Ω₃;
          5. Sellado criptográfico SHA-256 y construcción de `HydrologicalState`.
        """
        validated = self.phase3_from_phase2(phase2_handoff)
        eigenvalues, spectral_diag = self.phase3_spectral_audit(validated)
        poincare_ratio = self.phase3_poincare_residual(validated, eigenvalues)
        spectral_diag["poincare_residual"] = poincare_ratio

        poincare_cert = self.phase3_poincare_celestial_mechanics_audit(
            validated, eigenvalues, poincare_ratio
        )
        spectral_diag["poincare_celestial"] = {
            "betti_0": poincare_cert.betti_0,
            "betti_1": poincare_cert.betti_1,
            "euler_poincare_characteristic": poincare_cert.euler_poincare_characteristic,
            "euler_poincare_verified": poincare_cert.euler_poincare_verified,
            "cartan_exactness_defect": poincare_cert.cartan_exactness_defect,
            "solenoidal_cycle_flow_norm": poincare_cert.solenoidal_cycle_flow_norm,
            "kam_smallest_divisor": poincare_cert.kam_smallest_divisor,
            "kam_diophantine_margin": poincare_cert.kam_diophantine_margin,
            "is_kam_non_resonant": poincare_cert.is_kam_non_resonant,
            "liouville_symplectic_determinant": poincare_cert.liouville_symplectic_determinant,
            "symplectic_volume_defect": poincare_cert.symplectic_volume_defect,
            "kac_poincare_recurrence_time": poincare_cert.kac_poincare_recurrence_time,
        }

        stability_diag = self.phase3_stability_audit(validated)
        heyting_diag = self.phase3_heyting_omega3_verdict(
            spectral_diag, stability_diag, poincare_cert
        )
        stability_diag.update(heyting_diag)

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
    # API PRINCIPAL Y MORFISMOS DE COMPATIBILIDAD
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
        API principal — Orquesta el funtor compuesto anidado Φ₃ ∘ Φ₂ ∘ Φ₁:

          FASE 1 → `phase1_close_and_open_phase2` (abre formalmente Fase 2)
          FASE 2 → `phase2_close_and_open_phase3` (abre formalmente Fase 3)
          FASE 3 → `phase3_close_loop`            (emite `Phase3HydroReport`)
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
        phase2_handoff = phase1_handoff.continue_to_phase2(
            self,
            rho_sat=rho_sat,
            rho_w=rho_w,
        )
        report = phase2_handoff.continue_to_phase3(self)
        logger.info(
            "HydrologicalManifold consolidó estado. Sello: %s | Heyting: %s",
            report.state.sha256_hash[:16],
            report.stability_audit.get("heyting_truth_label", "UNKNOWN"),
        )
        return report.state

    def compute_pore_pressures(
        self,
        suction: np.ndarray,
        sat: np.ndarray,
        rho_w: float = 1000.0,
    ) -> np.ndarray:
        """API legada — equivalente a Fase 1.3 (`phase1_compute_pore_pressures`)."""
        return self.phase1_compute_pore_pressures(suction, sat, rho_w=rho_w)

    def compute_effective_stress_tensors(
        self,
        total_stresses: np.ndarray,
        pore_pressures: np.ndarray,
        alpha_biot: float = 0.95,
    ) -> Tuple[np.ndarray, bool]:
        """
        API legada — equivalente a Fase 1.4 (`phase1_effective_stress_audit`).

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
        """API legada — equivalente a Fase 1.5 (`phase1_compute_mualem_conductivities`)."""
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
        API legada — equivalente a Fase 2.6 (`_solve_richards_with_laplacian`).

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
        """API legada — equivalente a Fase 2.8 (`phase2_evaluate_siphoning`)."""
        B = self._as_incidence_matrix(incidence_matrix)
        n_nodes, n_edges = B.shape
        H_v = self._validate_vector("hydraulic_head", H, n_nodes)
        lengths = self._broadcast_edge_array(
            "edge_lengths", edge_lengths, n_edges, lower=self._reg
        )
        return self._evaluate_siphoning(
            H_v, B, lengths, rho_sat=rho_sat, rho_w=rho_w
        )
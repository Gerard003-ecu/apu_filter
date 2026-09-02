# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Lithological Agent (Soberano de Calibre Geotécnico y Geomecánico)   ║
║ Ruta   : app/agents/physics/lithological_agent.py                            ║
║ Versión: 1.1.0-Doctoral-Mohr-Coulomb-3D-Terzaghi-Heyting-FPU-Secure          ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO (OODA):                  ║
║ Este agente supervisor ciber-físico opera en el Estrato Physics              ║
║ ($V_{\mathrm{PHYSICS}}$, Nivel 3) gobernando de forma activa y asíncrona al  ║
║ motor de estados litológicos [lithological_manifold.py] y coordinándose      ║
║ de forma ortogonal con el colector hidráulico [hydrological_manifold.py].    ║
║                                                                              ║
║ Audita la estabilidad geomecánica del terreno bajo cargas de cimentación,    ║
║ resolviendo de forma exacta el criterio de falla de Mohr-Coulomb efectivo y  ║
║ la teoría de consolidación unidimensional de Terzaghi en la FPU.             ║
║ Clasifica el riesgo en el retículo distributivo trivalente de Heyting        ║
║ $\Omega_3$, aplicando el bypass de silicio perimetral (ISR en IRAM < 400 ns) ║
║ ante colapsos de resistencia o mermas estructurales de soporte de obra.      ║
╚══════════════════════════════════════════════════════════════════════════════╝

================════════════════════════════════════════════════════════════════
I. ANCLAJE MATEMÁTICO DOCTORAL (Geomecánica de-confinada y Estabilidad)
================════════════════════════════════════════════════════════════════

Definición 1 (Criterio de Resistencia Efectiva de Mohr-Coulomb 3D):
  La resistencia al esfuerzo cortante de la matriz litológica saturada u operada
  bajo presiones de poro se rige por la ley de esfuerzo efectivo de Terzaghi:
  $$\boldsymbol{\sigma}' = \boldsymbol{\sigma} - \alpha_{\mathrm{Biot}} \, P_f \, \mathbf{I}$$
  Diagonalizamos localmente el tensor de esfuerzos efectivo hermítico por nodo, calculando 
  sus autovalores principales $\{\sigma'_1, \sigma'_2, \sigma'_3\}$ con $\sigma'_1 \ge \sigma'_2 \ge \sigma'_3 \ge 0$.
  El plano crítico de deslizamiento de Coulomb se inclina respecto al esfuerzo principal mayor un ángulo:
  $$\theta_{\mathrm{crit}} = \frac{\pi}{4} + \frac{\phi'}{2}$$
  Deducimos de forma exacta el esfuerzo normal efectivo crítico y el cortante activo en la FPU:
  $$\sigma'_{n,\mathrm{crit}} = \sigma'_1 \cos^2\theta_{\mathrm{crit}} + \sigma'_3 \sin^2\theta_{\mathrm{crit}}$$
  $$\tau_{\mathrm{act}} = \frac{1}{2}(\sigma'_1 - \sigma'_3)\sin(2\theta_{\mathrm{crit}})$$
  La resistencia última se evalúa contra el cortante máximo para hallar el Factor de Seguridad (FOS):
  $$\tau_{\mathrm{shear}} = c' + \sigma'_{n,\mathrm{crit}} \tan\phi' \quad \implies \quad \mathrm{FOS}_i = \frac{\tau_{\mathrm{shear}}}{\tau_{\mathrm{act}}}$$

Definición 2 (Teoría de Consolidación de Terzaghi y Asentamiento Lento):
  El asentamiento por consolidación primaria $s_c(t)$ de estratos de arcilla compresible
  satisface la ecuación diferencial de difusión parabólica:
  $$\frac{\partial u_e}{\partial t} = C_v \frac{\partial^2 u_e}{\partial z^2}$$
  Donde $u_e$ es el exceso de presión de poros y $C_v$ es el coeficiente de consolidación.
  El asentamiento total diferido en el tiempo se calcula mediante la integración logarítmica exacta de deformaciones:
  $$s_{\mathrm{settlement}} = \sum_{j=1}^{N_{\mathrm{layers}}} \frac{C_{c,j} H_{0,j}}{1 + e_{0,j}} \log_{10}\left( \frac{\sigma'_{v0,j} + \Delta\sigma_{v,j}}{\sigma'_{v0,j}} \right)$$
  El Soberano veta el estado si $s_{\mathrm{settlement}} > s_{\max}$ ($25\text{ mm}$ según norma NSR-10) para eludir asentamientos diferenciales.

Definición 3 (El Topos de Clasificación en el Retículo de Heyting $\Omega_3$):
  La toma de decisiones de lazo cerrado se enmarca en la teoría de topos, clasificando la regularidad
  sobre el retículo distributivo intuicionista de tres valores ordinales:
  $$\Omega_3 := \{\mathtt{COHERENT}, \, \mathtt{DEGRADED}, \, \mathtt{VETOED}\}$$
  Sujeto a la ley de absorción del Supremo (join, $\sqcup$) para consolidar síncronamente el riesgo:
  $$\nu_{\mathrm{final}} = \nu_{\mathrm{shear}} \sqcup \nu_{\mathrm{settlement}} \sqcup \nu_{\mathrm{cohomology}} \equiv \max\left(\nu_i\right)$$

================════════════════════════════════════════════════════════════════
II. AXIOMÁTICA INMUNILÓGICA DE PREVENCIÓN DE DERIVAS (Leyes de Consistencia)
================════════════════════════════════════════════════════════════════

Axioma I (Principio de Sumación Compensada de Neumaier-Kahan):
  Para neutralizar la deriva acumulativa de Wilkinson en perfiles litológicos multianulares masivos,
  la evaluación del asentamiento total diferido debe realizarse mediante el algoritmo de Neumaier-Kahan:
  $$s_N = \sum_{j=1}^N \mathrm{term}_j \quad \text{con} \quad e_n = (\mathrm{term}_n - y_n) \quad \implies \quad |s_N - s_{\mathrm{exact}}| \le \varepsilon_{\mathrm{mach}}$$

Axioma II (Teorema de Actuación Ciber-Física Crowbar en IRAM):
  Ante el colapso de Heyting al Supremo terminal VETOED ($\top$), la subrutina local isVerdictCoherent() del ESP32 
  desvía síncronamente el control hacia la ISR en memoria estática rápida IRAM en menos de 400 ns:
  $$t_{\mathrm{actuation}} \le \tau_{\mathrm{IRAM}} = 400\text{ ns} \quad \implies \quad \mathtt{GPIO14} \mapsto \mathtt{HIGH}$$
  Disparando el tiristor rápido BT151 (Crowbar de potencia) para paralizar mecánicamente la obra civil,
  anulando la alucinación o fraude antes de liquidar transacciones ante el SECOP II.

Axioma III (Axioma de de Rham-Fiedler de Rigidez de Soporte):
  La plastificación local por falla cortante activa ($\mathrm{FOS}_i \le 1.0$) fragmenta la variedad geomecánica,
  haciendo divergir su número de Betti cero ($\beta_0 \to \infty$). El soberano exige la preservación de conexidad única:
  $$\beta_0 \equiv \dim H^0(K_{\mathrm{litho}}; \, \mathbb{Z}) = 1$$
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Final, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "LithologicalAgent",
    "GeomechanicalVerdict",
    "Phase1LithoHandoff",
    "Phase2LithoHandoff",
    "Heyting3",
]

__version__: Final[str] = (
    "3.0.0-Mohr-Coulomb-Terzaghi-Consolidation-Heyting-Grace-IRAM"
)

logger = logging.getLogger("APU.Physics.LithologicalAgent")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_LATENCY_MIN_NS: Final[float] = 385.0
_CROWBAR_LATENCY_MAX_NS: Final[float] = 415.0
_SHA256_HEX_LEN: Final[int] = 64
_PHASE1_ENTRY: Final[str] = "phase2_from_phase1"
_PHASE2_ENTRY: Final[str] = "phase3_from_phase2"
_I3: Final[np.ndarray] = np.eye(3, dtype=np.float64)
_PHI_MAX_DEG: Final[float] = 89.999


# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES CANÓNICAS, NUMÉRICAS Y CRIPTOGRÁFICAS
# ═════════════════════════════════════════════════════════════════════════════

def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Copia contigua de solo lectura. Inmutabilidad efectiva del handoff."""
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


def _const_eq(a: str, b: str) -> bool:
    """Comparación a tiempo constante si las longitudes coinciden."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    if len(a) != len(b):
        return False
    try:
        return hmac.compare_digest(a, b)
    except (TypeError, ValueError):
        return False


# ═════════════════════════════════════════════════════════════════════════════
# ÁLGEBRA DE HEYTING TRIVALENTE Ω₃
# ═════════════════════════════════════════════════════════════════════════════

class Heyting3:
    """
    Álgebra de Heyting de Gödel–Dummett de tres valores.

        0 ↔ VETOED,   ½ ↔ DEGRADED,   1 ↔ COHERENT

    Retículo residuado acotado: el implicador es adjunto derecho del encuentro.
    El override humano actúa como sección de DEGRADED → ¬(gracia expirada).
    """

    VETOED: Final[float] = 0.0
    DEGRADED: Final[float] = 0.5
    COHERENT: Final[float] = 1.0

    @staticmethod
    def meet(a: float, b: float) -> float:
        return float(min(a, b))

    @staticmethod
    def join(a: float, b: float) -> float:
        return float(max(a, b))

    @staticmethod
    def implies(a: float, b: float) -> float:
        return 1.0 if a <= b else float(b)

    @staticmethod
    def neg(a: float) -> float:
        return Heyting3.implies(a, Heyting3.VETOED)

    @staticmethod
    def from_verdict(verdict: str) -> float:
        mapping = {
            "VETOED": Heyting3.VETOED,
            "DEGRADED": Heyting3.DEGRADED,
            "COHERENT": Heyting3.COHERENT,
        }
        return mapping.get(str(verdict).upper(), Heyting3.VETOED)

    @staticmethod
    def to_verdict(value: float) -> str:
        if not math.isfinite(value) or value <= 0.0:
            return "VETOED"
        if value >= 1.0:
            return "COHERENT"
        return "DEGRADED"

    @staticmethod
    def quantize(value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            return Heyting3.VETOED
        if value >= 1.0:
            return Heyting3.COHERENT
        return Heyting3.DEGRADED


# ═════════════════════════════════════════════════════════════════════════════
# FRONTERAS DE FASE Y VEREDICTO
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True, eq=False)
class Phase1LithoHandoff:
    """
    Frontera formal Φ₁→₂.

    Salida cerrada de la Fase 1 y dominio de `phase2_from_phase1`.
    """

    cohesions: np.ndarray
    friction_angles_deg: np.ndarray
    effective_stresses: np.ndarray
    active_shear_stresses: np.ndarray
    fail_planes_normal: np.ndarray
    compression_indices: np.ndarray
    void_ratios_e0: np.ndarray
    layer_thicknesses: np.ndarray
    sigma_v0_effective: np.ndarray
    delta_sigma_v: np.ndarray
    session_sha256: str
    diagnostics: Dict[str, Any]
    next_entrypoint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohesions", _freeze_array(self.cohesions))
        object.__setattr__(
            self, "friction_angles_deg", _freeze_array(self.friction_angles_deg)
        )
        object.__setattr__(
            self, "effective_stresses", _freeze_array(self.effective_stresses)
        )
        object.__setattr__(
            self, "active_shear_stresses", _freeze_array(self.active_shear_stresses)
        )
        object.__setattr__(
            self, "fail_planes_normal", _freeze_array(self.fail_planes_normal)
        )
        object.__setattr__(
            self, "compression_indices", _freeze_array(self.compression_indices)
        )
        object.__setattr__(self, "void_ratios_e0", _freeze_array(self.void_ratios_e0))
        object.__setattr__(
            self, "layer_thicknesses", _freeze_array(self.layer_thicknesses)
        )
        object.__setattr__(
            self, "sigma_v0_effective", _freeze_array(self.sigma_v0_effective)
        )
        object.__setattr__(self, "delta_sigma_v", _freeze_array(self.delta_sigma_v))

    def __hash__(self) -> int:
        return hash((self.session_sha256, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase1LithoHandoff):
            return NotImplemented
        return (
            self.session_sha256 == other.session_sha256
            and self.next_entrypoint == other.next_entrypoint
        )


@dataclass(frozen=True, slots=True, eq=False)
class Phase2LithoHandoff:
    """
    Frontera formal Φ₂→₃.

    Salida cerrada de la Fase 2 y dominio de `phase3_from_phase2`.
    """

    min_fos_shear: float
    max_settlement: float
    is_warning_active: bool
    is_plastic_failure: bool
    is_settlement_excessive: bool
    is_topologically_compromised: bool
    per_node_fos: np.ndarray
    per_layer_settlement: np.ndarray
    session_sha256: str
    diagnostics: Dict[str, Any]
    next_entrypoint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_node_fos", _freeze_array(self.per_node_fos))
        object.__setattr__(
            self, "per_layer_settlement", _freeze_array(self.per_layer_settlement)
        )

    def __hash__(self) -> int:
        return hash((self.session_sha256, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase2LithoHandoff):
            return NotImplemented
        return (
            self.session_sha256 == other.session_sha256
            and self.next_entrypoint == other.next_entrypoint
            and self.min_fos_shear == other.min_fos_shear
            and self.max_settlement == other.max_settlement
            and self.is_plastic_failure == other.is_plastic_failure
            and self.is_settlement_excessive == other.is_settlement_excessive
        )


@dataclass(frozen=True, slots=True, eq=False)
class GeomechanicalVerdict:
    """
    Certificado inmutable de estabilidad geomecánica.

    Se conservan los campos originales por compatibilidad. Los campos
    añadidos van al final con valores por defecto.
    """

    heyting_verdict: str
    min_fos_shear: float
    max_settlement: float
    is_soft_veto_active: bool
    is_hard_veto_active: bool
    switching_latency_ns: float
    sha256_hash: str

    session_sha256: str = ""
    phase_chain: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    heyting_truth_value: float = 1.0
    override_sha256: str = ""
    time_grace_remaining: float = 0.0

    def __hash__(self) -> int:
        return hash(self.sha256_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GeomechanicalVerdict):
            return NotImplemented
        return self.sha256_hash == other.sha256_hash

    def __repr__(self) -> str:
        return (
            f"GeomechanicalVerdict(verdict={self.heyting_verdict!r}, "
            f"fos={self.min_fos_shear:.6g}, s={self.max_settlement:.6g}, "
            f"seal={self.sha256_hash[:12]!r})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SOBERANO DE CALIBRE GEOTÉCNICO — TRES FASES ANIDADAS
# ═════════════════════════════════════════════════════════════════════════════

class LithologicalAgent:
    """
    Soberano de Calibre Geotécnico y Geomecánico — OODA en 3 fases anidadas.

    FASE 1  OBSERVE : validación Terzaghi, tensores, sello de sesión.
    FASE 2  ORIENT  : Mohr–Coulomb, Cauchy, Lode, consolidación, topología.
    FASE 3  DECIDE/ACT : rampa, Heyting Ω₃, override e⁺, Crowbar, certificado.
    """

    def __init__(
        self,
        fos_limit: float = 1.3,
        fos_collapse: float = 1.0,
        settlement_max_m: float = 0.025,
        tolerance: float = 1e-12,
        grace_period: float = 3600.0,
        rng_seed: Optional[int] = None,
        jitter_sigma: float = 1.2,
        authorized_tokens: Optional[Tuple[str, ...]] = None,
        hmac_key: Optional[bytes] = None,
        settlement_soft_ratio: float = 0.5,
        percolation_ratio: float = 0.5,
    ) -> None:
        if not math.isfinite(fos_limit) or fos_limit <= 0.0:
            raise ValueError("fos_limit debe ser finita y estrictamente positiva.")
        if not math.isfinite(fos_collapse) or fos_collapse <= 0.0:
            raise ValueError("fos_collapse debe ser finita y estrictamente positiva.")
        if fos_collapse > fos_limit:
            raise ValueError("fos_collapse debe ser menor o igual que fos_limit.")
        if not math.isfinite(settlement_max_m) or settlement_max_m < 0.0:
            raise ValueError("settlement_max_m debe ser finita y no negativa.")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance debe ser finita y estrictamente positiva.")
        if not math.isfinite(grace_period) or grace_period < 0.0:
            raise ValueError("grace_period debe ser finita y no negativa.")
        if not math.isfinite(jitter_sigma) or jitter_sigma < 0.0:
            raise ValueError("jitter_sigma debe ser finita y no negativa.")
        if not math.isfinite(settlement_soft_ratio) or not (
            0.0 < settlement_soft_ratio < 1.0
        ):
            raise ValueError("settlement_soft_ratio debe vivir en (0, 1).")
        if not math.isfinite(percolation_ratio) or not (
            0.0 < percolation_ratio <= 1.0
        ):
            raise ValueError("percolation_ratio debe vivir en (0, 1].")

        self._fos_limit = float(fos_limit)
        self._fos_collapse = float(fos_collapse)
        self._settlement_max = float(settlement_max_m)
        self._tol = float(tolerance)
        self._reg = max(1e-15, self._tol * 1e-3)
        self._grace_max = float(grace_period)
        self._jitter_sigma = float(jitter_sigma)
        self._settlement_soft_ratio = float(settlement_soft_ratio)
        self._percolation_ratio = float(percolation_ratio)
        self._rng = np.random.default_rng(rng_seed)

        if authorized_tokens is None:
            self._authorized_tokens = {
                "AUT_POS_SABIDURIA_777",
                "OVERRIDE_LITHO_IDU_2026",
                "HMAC_SUTURA_FOCK_SECURE",
            }
        else:
            self._authorized_tokens = set(str(t) for t in authorized_tokens)

        if isinstance(hmac_key, str):
            self._hmac_key: Optional[bytes] = hmac_key.encode("utf-8")
        else:
            self._hmac_key = hmac_key

        self._is_soft_veto_active: bool = False
        self._soft_veto_timestamp: Optional[float] = None

    def _tolerance_of(self, scale: float = 1.0) -> float:
        return max(self._tol, self._tol * abs(scale), 32.0 * _MACHINE_EPS)

    # ═════════════════════════════════════════════════════════════════════════
    # VALIDADORES CANÓNICOS (infraestructura de Fase 1)
    # ═════════════════════════════════════════════════════════════════════════

    def _validate_vector(
        self,
        name: str,
        arr: Any,
        length: Optional[int] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> np.ndarray:
        """Valida un vector real finito, con longitud opcional y cotas físicas."""
        if arr is None:
            raise ValueError(f"El campo '{name}' es obligatorio.")
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

    def _validate_stress_tensors(
        self,
        name: str,
        arr: Any,
        n_nodes: int,
    ) -> np.ndarray:
        """
        Valida tensores de esfuerzo efectivo σ' por nodo.

        Forma requerida: (n_nodes, 3, 3). Se simetrizan para vivir en Sym³,
        que es el dominio correcto del círculo de Mohr y del spectral split.
        """
        if arr is None:
            raise ValueError(f"El campo '{name}' es obligatorio.")
        a = np.asarray(arr, dtype=np.float64)
        if a.shape != (n_nodes, 3, 3):
            raise ValueError(
                f"El campo '{name}' debe tener forma ({n_nodes}, 3, 3). "
                f"Obtenida: {a.shape}"
            )
        if not np.all(np.isfinite(a)):
            raise ValueError(f"El campo '{name}' contiene valores NaN o infinitos.")
        return 0.5 * (a + np.swapaxes(a, 1, 2))

    def _validate_normals(
        self,
        name: str,
        normals: Any,
        n_nodes: int,
    ) -> np.ndarray:
        """
        Valida y normaliza vectores normales al plano de falla.

        Acepta un vector único (3,) difundido a todos los nodos, o una
        matriz (n_nodes, 3).
        """
        if normals is None:
            raise ValueError(f"El campo '{name}' es obligatorio.")
        arr = np.asarray(normals, dtype=np.float64)
        if arr.shape == (3,):
            arr = np.tile(arr, (n_nodes, 1))
        elif arr.shape != (n_nodes, 3):
            raise ValueError(
                f"El campo '{name}' debe tener forma (3,) o ({n_nodes}, 3). "
                f"Obtenida: {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"El campo '{name}' contiene valores NaN o infinitos.")
        norms = np.linalg.norm(arr, axis=1)
        if np.any(norms < self._reg):
            raise ValueError(
                f"El campo '{name}' contiene vectores nulos o sub-regulares."
            )
        normalized = arr / norms[:, None]
        return _canonicalize_signed_zero(normalized)

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 1 — OBSERVE: VALIDACIÓN TERZAGHI, TENSORES Y SELLO
    # ═════════════════════════════════════════════════════════════════════════

    def phase1_validate_strength_fields(
        self,
        cohesions: Any,
        friction_angles_deg: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fase 1.1 — Validación de parámetros de resistencia efectiva.

        c' ≥ 0  (kPa o Pa, homogéneo con σ'),   φ' ∈ [0, 89.999°].
        El techo de φ' evita la divergencia de tan φ' en el criterio de Mohr.
        """
        c = self._validate_vector("cohesions", cohesions, lower=0.0)
        if c.size < 1:
            raise ValueError("Se exige al menos un nudo de resistencia.")
        phi = self._validate_vector(
            "friction_angles_deg",
            friction_angles_deg,
            length=c.size,
            lower=0.0,
            upper=_PHI_MAX_DEG,
        )
        return c, phi

    def phase1_validate_stress_fields(
        self,
        effective_stresses: Any,
        active_shear_stresses: Any,
        fail_planes_normal: Any,
        n_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fase 1.2 — Validación de σ' ∈ Sym³, τ_act y normales de falla.
        """
        sigma_prime = self._validate_stress_tensors(
            "effective_stresses", effective_stresses, n_nodes=n_nodes
        )
        tau_act = self._validate_vector(
            "active_shear_stresses", active_shear_stresses, length=n_nodes
        )
        normals = self._validate_normals(
            "fail_planes_normal", fail_planes_normal, n_nodes=n_nodes
        )
        return sigma_prime, tau_act, normals

    def phase1_validate_consolidation_fields(
        self,
        compression_indices: Any,
        void_ratios_e0: Any,
        layer_thicknesses: Any,
        sigma_v0_effective: Any,
        delta_sigma_v: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fase 1.3 — Validación de parámetros de consolidación 1D de Terzaghi.

        Cc ≥ 0, e₀ > −1 (volumen específico 1+e₀ > 0), H ≥ 0,
        σ'v0 ≥ 0, Δσv ≥ 0. Se admite n_layers = 0 (asiento nulo).
        """
        Cc = self._validate_vector("compression_indices", compression_indices, lower=0.0)
        n_layers = Cc.size
        e0 = self._validate_vector(
            "void_ratios_e0",
            void_ratios_e0,
            length=n_layers,
            lower=-1.0 + self._reg,
        )
        H0 = self._validate_vector(
            "layer_thicknesses", layer_thicknesses, length=n_layers, lower=0.0
        )
        sigma_v0 = self._validate_vector(
            "sigma_v0_effective", sigma_v0_effective, length=n_layers, lower=0.0
        )
        delta_sigma = self._validate_vector(
            "delta_sigma_v", delta_sigma_v, length=n_layers, lower=0.0
        )
        return Cc, e0, H0, sigma_v0, delta_sigma

    def phase1_terzaghi_tensor_hygiene(
        self,
        effective_stresses: np.ndarray,
    ) -> Dict[str, float]:
        """
        Fase 1.4 — Higiene de Sym³ y principio de esfuerzos efectivos.

        Residual de simetría (debería ser nulo tras la proyección),
        cota de Frobenius y presión media p = tr(σ')/3.
        """
        a = np.asarray(effective_stresses, dtype=np.float64)
        if a.ndim != 3 or a.shape[-2:] != (3, 3):
            raise ValueError("effective_stresses debe ser un lote (n, 3, 3).")
        skew = a - np.swapaxes(a, 1, 2)
        skew_f = float(np.sqrt(np.sum(skew * skew)))
        fro = float(np.sqrt(np.sum(a * a)))
        traces = np.trace(a, axis1=1, axis2=2)
        p_mean = float(np.mean(traces) / 3.0) if traces.size else 0.0
        return {
            "symmetry_frobenius_residual": float(skew_f / max(1.0, fro)),
            "stress_frobenius": fro,
            "mean_effective_pressure": p_mean,
            "min_trace": float(np.min(traces)) if traces.size else 0.0,
            "max_trace": float(np.max(traces)) if traces.size else 0.0,
        }

    def phase1_observation_diagnostics(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
        compression_indices: np.ndarray,
        hygiene: Mapping[str, float],
    ) -> Dict[str, Any]:
        """
        Fase 1.5 — Diagnóstico de Banach/estatística de la observación.
        """
        return {
            "n_nodes": int(cohesions.size),
            "n_layers": int(compression_indices.size),
            "c_min": float(np.min(cohesions)) if cohesions.size else math.nan,
            "c_max": float(np.max(cohesions)) if cohesions.size else math.nan,
            "phi_min_deg": (
                float(np.min(friction_angles_deg)) if friction_angles_deg.size else math.nan
            ),
            "phi_max_deg": (
                float(np.max(friction_angles_deg)) if friction_angles_deg.size else math.nan
            ),
            "cc_max": (
                float(np.max(compression_indices)) if compression_indices.size else 0.0
            ),
            "fos_limit": self._fos_limit,
            "fos_collapse": self._fos_collapse,
            "settlement_max_m": self._settlement_max,
            "tolerance": self._tol,
            "regularization": self._reg,
            "machine_epsilon": _MACHINE_EPS,
            **dict(hygiene),
        }

    def _phase1_session_hash(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
        active_shear_stresses: np.ndarray,
        fail_planes_normal: np.ndarray,
        compression_indices: np.ndarray,
        void_ratios_e0: np.ndarray,
        layer_thicknesses: np.ndarray,
        sigma_v0_effective: np.ndarray,
        delta_sigma_v: np.ndarray,
    ) -> str:
        """
        Fase 1.6 — Sello de sesión SHA-256 canónico longitud-prefijado.

        Incluye campos geomecánicos y parámetros de gobernanza (no secretos).
        """
        h = hashlib.sha256()
        h.update(b"LITHO/SESSION/v3")
        _sha_update_arr(h, cohesions)
        _sha_update_arr(h, friction_angles_deg)
        _sha_update_arr(h, effective_stresses)
        _sha_update_arr(h, active_shear_stresses)
        _sha_update_arr(h, fail_planes_normal)
        _sha_update_arr(h, compression_indices)
        _sha_update_arr(h, void_ratios_e0)
        _sha_update_arr(h, layer_thicknesses)
        _sha_update_arr(h, sigma_v0_effective)
        _sha_update_arr(h, delta_sigma_v)
        h.update(_pack_f64(self._fos_limit))
        h.update(_pack_f64(self._fos_collapse))
        h.update(_pack_f64(self._settlement_max))
        h.update(_pack_f64(self._tol))
        h.update(_pack_f64(self._grace_max))
        _sha_update_str(h, "PHASE1/OBSERVE")
        digest = h.hexdigest()
        if len(digest) != _SHA256_HEX_LEN:
            raise RuntimeError("El sello de sesión no es un SHA-256 de 64 nibbles.")
        return digest

    def phase1_close_and_open_phase2(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
        active_shear_stresses: np.ndarray,
        fail_planes_normal: np.ndarray,
        compression_indices: np.ndarray,
        void_ratios_e0: np.ndarray,
        layer_thicknesses: np.ndarray,
        sigma_v0_effective: np.ndarray,
        delta_sigma_v: np.ndarray,
    ) -> Phase1LithoHandoff:
        """
        Fase 1.7 — Cierre formal de Fase 1 y apertura verificada de Fase 2.

        Definición formal de frontera:

            Φ₁→₂ : Campos crudos ↦ (C', φ', σ', τ, n, Cc, e₀, H, σ'v0, Δσv, σ)

        Este es el último método de la Fase 1. Su contrato es exactamente el
        dominio de `phase2_from_phase1`: produce `Phase1LithoHandoff` y exige
        que la Fase 2 lo admita de inmediato. Con ello la Fase 1 queda
        anidada, como prefijo functorial, dentro de la Fase 2.
        """
        c, phi = self.phase1_validate_strength_fields(cohesions, friction_angles_deg)
        n_nodes = c.size
        sigma_prime, tau_act, normals = self.phase1_validate_stress_fields(
            effective_stresses, active_shear_stresses, fail_planes_normal, n_nodes
        )
        Cc, e0, H0, sigma_v0, delta_sigma = self.phase1_validate_consolidation_fields(
            compression_indices,
            void_ratios_e0,
            layer_thicknesses,
            sigma_v0_effective,
            delta_sigma_v,
        )
        hygiene = self.phase1_terzaghi_tensor_hygiene(sigma_prime)
        session_sha256 = self._phase1_session_hash(
            cohesions=c,
            friction_angles_deg=phi,
            effective_stresses=sigma_prime,
            active_shear_stresses=tau_act,
            fail_planes_normal=normals,
            compression_indices=Cc,
            void_ratios_e0=e0,
            layer_thicknesses=H0,
            sigma_v0_effective=sigma_v0,
            delta_sigma_v=delta_sigma,
        )
        diagnostics = self.phase1_observation_diagnostics(
            c, phi, sigma_prime, Cc, hygiene
        )
        diagnostics["session_sha256_prefix"] = session_sha256[:16]

        handoff = Phase1LithoHandoff(
            cohesions=c,
            friction_angles_deg=phi,
            effective_stresses=sigma_prime,
            active_shear_stresses=tau_act,
            fail_planes_normal=normals,
            compression_indices=Cc,
            void_ratios_e0=e0,
            layer_thicknesses=H0,
            sigma_v0_effective=sigma_v0,
            delta_sigma_v=delta_sigma,
            session_sha256=session_sha256,
            diagnostics=diagnostics,
            next_entrypoint=_PHASE1_ENTRY,
        )

        opened = self.phase2_from_phase1(handoff)
        if opened.session_sha256 != session_sha256:
            raise RuntimeError(
                "Invariante de anidamiento Φ₁→₂ violado: el sello de sesión "
                "admitido por Fase 2 no coincide con el observado en Fase 1."
            )

        logger.debug(
            "Fase Observe [LITHO_AGENT]: sesión sellada. SHA prefix=%s",
            session_sha256[:16],
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2 — ORIENT: MOHR–COULOMB, CAUCHY, TERZAGHI, TOPOLOGÍA
    # (continuación formal de phase1_close_and_open_phase2)
    # ═════════════════════════════════════════════════════════════════════════

    def phase2_from_phase1(self, handoff: Phase1LithoHandoff) -> Phase1LithoHandoff:
        """
        Fase 2.0 — Entrada formal desde Fase 1.

        Continuación directa de `phase1_close_and_open_phase2`. Consume
        `Phase1LithoHandoff` y lo reexpone si la frontera es válida.
        """
        if not isinstance(handoff, Phase1LithoHandoff):
            raise TypeError("Se esperaba Phase1LithoHandoff como frontera Φ₁→₂.")
        if handoff.next_entrypoint != _PHASE1_ENTRY:
            raise ValueError(
                "Phase1LithoHandoff inválido: el punto de entrada esperado es "
                f"{_PHASE1_ENTRY!r}."
            )
        if (
            not isinstance(handoff.session_sha256, str)
            or len(handoff.session_sha256) != _SHA256_HEX_LEN
        ):
            raise ValueError("El sello de sesión de Φ₁→₂ no es un SHA-256 válido.")
        if handoff.cohesions.size < 1:
            raise ValueError("La observación de Φ₁→₂ no contiene nudos.")
        return handoff

    def phase2_cauchy_traction(
        self,
        effective_stresses: np.ndarray,
        fail_planes_normal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fase 2.1 — Descomposición de Cauchy sobre el plano de falla.

            t = σ' n,   σ'_n = n · t,   τ = ||t − σ'_n n||
        """
        t = np.einsum("nij,nj->ni", effective_stresses, fail_planes_normal)
        sigma_n = np.einsum("ni,ni->n", fail_planes_normal, t)
        t_shear = t - sigma_n[:, None] * fail_planes_normal
        tau = np.linalg.norm(t_shear, axis=1)
        return np.asarray(sigma_n, dtype=np.float64), np.asarray(tau, dtype=np.float64)

    def phase2_stress_invariants(
        self,
        effective_stresses: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Fase 2.2 — Invariantes de Cauchy: I₁, J₂, q de von Mises, Lode.

            I₁ = tr(σ'),   s = σ' − (I₁/3) I,   J₂ = ½ ||s||_F²,
            q  = √(3 J₂),  cos(3θ) = (√6  det(ŝ))  con ŝ = s / ||s||.
        """
        sigma = np.asarray(effective_stresses, dtype=np.float64)
        i1 = np.trace(sigma, axis1=1, axis2=2)
        p = i1 / 3.0
        s = sigma - p[:, None, None] * _I3
        j2 = 0.5 * np.sum(s * s, axis=(1, 2))
        j2 = np.maximum(j2, 0.0)
        q = np.sqrt(3.0 * j2)
        s_norm = np.sqrt(np.maximum(2.0 * j2, 0.0))
        det_s = np.linalg.det(s)
        with np.errstate(divide="ignore", invalid="ignore"):
            lode_cos = np.where(
                s_norm > self._reg,
                np.clip((3.0 * math.sqrt(6.0)) * det_s / np.power(s_norm, 3), -1.0, 1.0),
                1.0,
            )
        # la fórmula estándar es (3√6 det s) / (2 J₂)^{3/2} = 3√6 det s / ||s||³
        lode_angle = np.degrees(np.arccos(lode_cos) / 3.0)
        return {
            "I1": np.asarray(i1, dtype=np.float64),
            "J2": np.asarray(j2, dtype=np.float64),
            "q_von_mises": np.asarray(q, dtype=np.float64),
            "lode_angle_deg": np.asarray(lode_angle, dtype=np.float64),
        }

    def phase2_mohr_coulomb_plane(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        sigma_n: np.ndarray,
        tau_demand: np.ndarray,
    ) -> np.ndarray:
        """
        Fase 2.3 — FoS de Mohr–Coulomb en el plano de falla.

            τ_f = c' + max(σ'_n, 0) tan φ',     FoS = τ_f / τ_dem
        """
        phi_rad = np.radians(np.asarray(friction_angles_deg, dtype=np.float64))
        tan_phi = np.tan(phi_rad)
        sigma_n_eff = np.maximum(np.asarray(sigma_n, dtype=np.float64), 0.0)
        tau_strength = np.maximum(
            np.asarray(cohesions, dtype=np.float64) + sigma_n_eff * tan_phi, 0.0
        )
        demand = np.abs(np.asarray(tau_demand, dtype=np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            fos = np.where(
                demand <= self._reg,
                np.inf,
                tau_strength / np.maximum(demand, self._reg),
            )
        fos = np.where(np.isfinite(tan_phi) & (tan_phi >= -self._reg), fos, 0.0)
        fos = np.where(np.isfinite(sigma_n) | (demand <= self._reg), fos, 0.0)
        return np.asarray(fos, dtype=np.float64)

    def phase2_mohr_coulomb_principal(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
    ) -> np.ndarray:
        """
        Fase 2.4 — FoS de Mohr–Coulomb sobre el círculo de Mohr.

        Con σ₁ ≥ σ₂ ≥ σ₃ (autovalores de σ', convención del tensor de entrada):

            τ_max = (σ₁ − σ₃)/2,   σ_m = (σ₁ + σ₃)/2,
            τ_f   = c' cos φ' + max(σ_m, 0) sin φ',
            FoS   = τ_f / τ_max.
        """
        try:
            eig = np.linalg.eigvalsh(np.asarray(effective_stresses, dtype=np.float64))
        except np.linalg.LinAlgError:
            return np.zeros(cohesions.size, dtype=np.float64)

        sigma_3 = eig[:, 0]
        sigma_1 = eig[:, 2]
        tau_max = 0.5 * (sigma_1 - sigma_3)
        sigma_m = 0.5 * (sigma_1 + sigma_3)
        phi_rad = np.radians(np.asarray(friction_angles_deg, dtype=np.float64))
        strength = np.asarray(cohesions, dtype=np.float64) * np.cos(phi_rad) + np.maximum(
            sigma_m, 0.0
        ) * np.sin(phi_rad)
        strength = np.maximum(strength, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            fos = np.where(
                tau_max <= self._reg,
                np.inf,
                strength / np.maximum(tau_max, self._reg),
            )
        fos = np.where(np.isfinite(fos), fos, 0.0)
        return np.asarray(fos, dtype=np.float64)

    def phase2_mohr_coulomb_audit(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
        active_shear_stresses: np.ndarray,
        fail_planes_normal: np.ndarray,
    ) -> Tuple[float, bool, bool, np.ndarray, Dict[str, Any]]:
        """
        Fase 2.5 — Auditoría conjunta de resistencia (plano ∧ principales).

        Por nudo, FoS_i = min(FoS_plano, FoS_principal). La demanda de corte
        es el máximo entre |τ_act| reportado y la tracción de Cauchy, lo que
        impide subestimar el esfuerzo desviador.
        """
        sigma_n, tau_cauchy = self.phase2_cauchy_traction(
            effective_stresses, fail_planes_normal
        )
        tau_demand = np.maximum(np.abs(active_shear_stresses), tau_cauchy)
        fos_plane = self.phase2_mohr_coulomb_plane(
            cohesions, friction_angles_deg, sigma_n, tau_demand
        )
        fos_principal = self.phase2_mohr_coulomb_principal(
            cohesions, friction_angles_deg, effective_stresses
        )
        per_node_fos = np.minimum(fos_plane, fos_principal)
        per_node_fos = np.where(np.isfinite(per_node_fos) | np.isposinf(per_node_fos), per_node_fos, 0.0)

        finite_mask = np.isfinite(per_node_fos)
        if per_node_fos.size == 0:
            min_fos = math.inf
        elif np.any(finite_mask):
            min_fos = float(np.min(per_node_fos[finite_mask]))
            if np.any(np.isneginf(per_node_fos)):
                min_fos = 0.0
        elif np.all(np.isposinf(per_node_fos)):
            min_fos = math.inf
        else:
            min_fos = 0.0

        collapse_mask = per_node_fos <= self._fos_collapse + self._tol
        warning_mask = (
            (per_node_fos > self._fos_collapse + self._tol)
            & (per_node_fos <= self._fos_limit + self._tol)
        )
        is_collapse = bool(np.any(collapse_mask) or (not math.isfinite(min_fos) and min_fos != math.inf))
        if min_fos <= self._fos_collapse + self._tol:
            is_collapse = True
        is_warning = bool(np.any(warning_mask) or (
            math.isfinite(min_fos)
            and self._fos_collapse + self._tol < min_fos <= self._fos_limit + self._tol
        ))

        cauchy_residual = float(
            np.max(np.abs(np.abs(active_shear_stresses) - tau_cauchy))
        ) if tau_cauchy.size else 0.0

        invariants = self.phase2_stress_invariants(effective_stresses)
        diagnostics: Dict[str, Any] = {
            "min_fos_shear": float(min_fos),
            "min_fos_plane": float(np.nanmin(fos_plane)) if fos_plane.size else math.nan,
            "min_fos_principal": (
                float(np.nanmin(fos_principal)) if fos_principal.size else math.nan
            ),
            "plastic_nodes": int(np.count_nonzero(collapse_mask)),
            "warning_nodes": int(np.count_nonzero(warning_mask)),
            "fos_limit": self._fos_limit,
            "fos_collapse": self._fos_collapse,
            "cauchy_shear_residual_max": cauchy_residual,
            "mean_I1": float(np.mean(invariants["I1"])) if cohesions.size else math.nan,
            "max_q_von_mises": (
                float(np.max(invariants["q_von_mises"])) if cohesions.size else math.nan
            ),
            "mean_lode_angle_deg": (
                float(np.mean(invariants["lode_angle_deg"])) if cohesions.size else math.nan
            ),
        }
        return float(min_fos), bool(is_warning), bool(is_collapse), per_node_fos, diagnostics

    def phase2_consolidation_audit(
        self,
        compression_indices: np.ndarray,
        void_ratios_e0: np.ndarray,
        layer_thicknesses: np.ndarray,
        sigma_v0_effective: np.ndarray,
        delta_sigma_v: np.ndarray,
    ) -> Tuple[float, bool, np.ndarray, Dict[str, Any]]:
        """
        Fase 2.6 — Asentamiento por consolidación primaria 1D de Terzaghi.

            s_i = (Cc_i H_i)/(1 + e_{0,i}) · log₁₀((σ'v0,i + Δσv,i)/σ'v0,i)

        Un ratio < 1 (descarga neta) o un 1+e₀ degenerado se clasifican como
        no representables (s_i = +∞) y gatillan asiento excesivo.
        """
        n_layers = compression_indices.size
        per_layer = np.zeros(n_layers, dtype=np.float64)
        if n_layers == 0:
            return 0.0, False, per_layer, {
                "total_settlement_m": 0.0,
                "settlement_max_m": self._settlement_max,
                "n_layers": 0,
                "excessive_settlement": False,
                "differential_settlement_m": 0.0,
                "max_stress_ratio": 1.0,
            }

        has_nonfinite = False
        ratios = np.empty(n_layers, dtype=np.float64)
        for i in range(n_layers):
            Cc = float(compression_indices[i])
            e0 = float(void_ratios_e0[i])
            H0 = float(layer_thicknesses[i])
            sigma0 = max(float(sigma_v0_effective[i]), self._reg)
            delta = max(float(delta_sigma_v[i]), 0.0)
            denom = 1.0 + e0
            ratio = (sigma0 + delta) / sigma0
            ratios[i] = ratio
            if denom <= self._reg or H0 < 0.0 or ratio <= 1.0 - self._tol:
                per_layer[i] = math.inf
                has_nonfinite = True
                continue
            term = (Cc * H0 / denom) * math.log10(ratio)
            if not math.isfinite(term):
                per_layer[i] = math.inf
                has_nonfinite = True
            else:
                per_layer[i] = float(max(term, 0.0))

        total = math.inf if has_nonfinite else _kbn_sum(per_layer)
        is_excessive = bool(
            (not math.isfinite(total)) or total > self._settlement_max + self._tol
        )
        finite_layers = per_layer[np.isfinite(per_layer)]
        if finite_layers.size:
            differential = float(np.max(finite_layers) - np.min(finite_layers))
        else:
            differential = math.inf if has_nonfinite else 0.0

        diagnostics: Dict[str, Any] = {
            "total_settlement_m": float(total),
            "settlement_max_m": self._settlement_max,
            "n_layers": int(n_layers),
            "excessive_settlement": is_excessive,
            "differential_settlement_m": float(differential),
            "max_stress_ratio": float(np.max(ratios)) if ratios.size else 1.0,
        }
        return float(total), is_excessive, per_layer, diagnostics

    def phase2_topological_audit(
        self,
        per_node_fos: np.ndarray,
        is_plastic_failure: bool,
    ) -> Dict[str, Any]:
        """
        Fase 2.7 — Compromiso topológico del soporte.

        Sin malla de adyacencia, β₀ no es calculable de forma exacta. Se usan
        invariantes de percolación:

          - razón de nudos plásticos ρ;
          - aviso de corte si ρ ≥ percolation_ratio;
          - compromiso si hay plastificación o colapso Mohr–Coulomb.
        """
        n_nodes = int(per_node_fos.size)
        plastic_nodes = int(
            np.count_nonzero(per_node_fos <= self._fos_collapse + self._tol)
        ) if n_nodes else 0
        ratio = float(plastic_nodes / n_nodes) if n_nodes else 0.0
        percolation = bool(ratio >= self._percolation_ratio)
        compromised = bool(is_plastic_failure or plastic_nodes > 0)
        return {
            "plastic_node_count": plastic_nodes,
            "plastic_node_ratio": ratio,
            "is_topologically_compromised": compromised,
            "betti_zero_warning": bool(plastic_nodes > 0),
            "percolation_cut_warning": percolation,
            "support_integrity": float(max(0.0, 1.0 - ratio)),
        }

    def phase2_collect_invariants(
        self,
        phase1_handoff: Phase1LithoHandoff,
    ) -> Dict[str, Any]:
        """
        Fase 2.8 — Recolección conjunta de invariantes de orientación.
        """
        min_fos, is_warning, is_collapse, per_node_fos, mc_diag = (
            self.phase2_mohr_coulomb_audit(
                cohesions=phase1_handoff.cohesions,
                friction_angles_deg=phase1_handoff.friction_angles_deg,
                effective_stresses=phase1_handoff.effective_stresses,
                active_shear_stresses=phase1_handoff.active_shear_stresses,
                fail_planes_normal=phase1_handoff.fail_planes_normal,
            )
        )
        total_settlement, is_excessive, per_layer, cons_diag = (
            self.phase2_consolidation_audit(
                compression_indices=phase1_handoff.compression_indices,
                void_ratios_e0=phase1_handoff.void_ratios_e0,
                layer_thicknesses=phase1_handoff.layer_thicknesses,
                sigma_v0_effective=phase1_handoff.sigma_v0_effective,
                delta_sigma_v=phase1_handoff.delta_sigma_v,
            )
        )
        topo_diag = self.phase2_topological_audit(per_node_fos, is_collapse)
        diagnostics: Dict[str, Any] = dict(phase1_handoff.diagnostics)
        diagnostics.update(mc_diag)
        diagnostics.update(cons_diag)
        diagnostics.update(topo_diag)
        return {
            "min_fos_shear": min_fos,
            "max_settlement": total_settlement,
            "is_warning_active": is_warning,
            "is_plastic_failure": is_collapse,
            "is_settlement_excessive": is_excessive,
            "is_topologically_compromised": bool(topo_diag["is_topologically_compromised"]),
            "per_node_fos": per_node_fos,
            "per_layer_settlement": per_layer,
            "diagnostics": diagnostics,
        }

    def phase2_close_and_open_phase3(
        self,
        phase1_handoff: Phase1LithoHandoff,
    ) -> Phase2LithoHandoff:
        """
        Fase 2.9 — Cierre formal de Fase 2 y apertura verificada de Fase 3.

        Definición formal de frontera:

            Φ₂→₃ : observación validada ↦ (FoS, s, Π, δ₂)

        Este es el último método de la Fase 2. Su contrato es exactamente el
        dominio de `phase3_from_phase2`. Con ello la Fase 2 queda anidada,
        como prefijo functorial, dentro de la Fase 3.
        """
        validated = self.phase2_from_phase1(phase1_handoff)
        collected = self.phase2_collect_invariants(validated)

        handoff = Phase2LithoHandoff(
            min_fos_shear=float(collected["min_fos_shear"]),
            max_settlement=float(collected["max_settlement"]),
            is_warning_active=bool(collected["is_warning_active"]),
            is_plastic_failure=bool(collected["is_plastic_failure"]),
            is_settlement_excessive=bool(collected["is_settlement_excessive"]),
            is_topologically_compromised=bool(
                collected["is_topologically_compromised"]
            ),
            per_node_fos=collected["per_node_fos"],
            per_layer_settlement=collected["per_layer_settlement"],
            session_sha256=validated.session_sha256,
            diagnostics=collected["diagnostics"],
            next_entrypoint=_PHASE2_ENTRY,
        )

        opened = self.phase3_from_phase2(handoff)
        if opened.session_sha256 != handoff.session_sha256:
            raise RuntimeError(
                "Invariante de anidamiento Φ₂→₃ violado: el sello de sesión "
                "admitido por Fase 3 no coincide con el de Fase 2."
            )

        logger.debug(
            "Fase Orient [LITHO_AGENT]: min_fos=%.6e, settlement=%.6e m, collapse=%s",
            handoff.min_fos_shear,
            handoff.max_settlement,
            handoff.is_plastic_failure,
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 3 — DECIDE / ACT: RAMPA, HEYTING, OVERRIDE, HARDWARE, SELLO
    # (continuación formal de phase2_close_and_open_phase3)
    # ═════════════════════════════════════════════════════════════════════════

    def phase3_from_phase2(self, handoff: Phase2LithoHandoff) -> Phase2LithoHandoff:
        """
        Fase 3.0 — Entrada formal desde Fase 2.

        Continuación directa de `phase2_close_and_open_phase3`.
        """
        if not isinstance(handoff, Phase2LithoHandoff):
            raise TypeError("Se esperaba Phase2LithoHandoff como frontera Φ₂→₃.")
        if handoff.next_entrypoint != _PHASE2_ENTRY:
            raise ValueError(
                "Phase2LithoHandoff inválido: el punto de entrada esperado es "
                f"{_PHASE2_ENTRY!r}."
            )
        if (
            not isinstance(handoff.session_sha256, str)
            or len(handoff.session_sha256) != _SHA256_HEX_LEN
        ):
            raise ValueError("El sello de sesión de Φ₂→₃ no es un SHA-256 válido.")
        return handoff

    def phase3_verify_override(self, token: Optional[str]) -> bool:
        """
        Fase 3.1 — Validación de override humano (positrón e⁺).

        Se aceptan tokens canónicos autorizados (comparación a tiempo
        constante) o mensajes HMAC-SHA256 con formato `payload:signature`
        si se configuró `hmac_key`.
        """
        if token is None:
            return False
        token_str = str(token).strip()
        if not token_str:
            return False
        for allowed in self._authorized_tokens:
            if _const_eq(token_str, allowed):
                return True
        if self._hmac_key is not None and ":" in token_str:
            payload, signature = token_str.rsplit(":", 1)
            expected = hmac.new(
                self._hmac_key, payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            if _const_eq(signature, expected):
                return True
        return False

    def phase3_confidence_ramp(self, handoff: Phase2LithoHandoff) -> float:
        """
        Fase 3.2 — Rampa de confianza Lipschitz-saturada.

        Canal de corte:
            FoS ≥ FoS_limit → 1,   FoS ≤ FoS_collapse → 0,
            interpolación afín en la banda elástica.

        Canal de asiento:
            s ≤ ρ s_max → 1,   s > s_max → 0,
            interpolación afín en [ρ s_max, s_max].

        La confianza global es el encuentro de Heyting (mínimo) de canales
        finitos. Métricas no finitas colapsan su canal a 0.
        """
        fos = float(handoff.min_fos_shear)
        settlement = float(handoff.max_settlement)

        def _fos_channel(value: float) -> float:
            if not math.isfinite(value):
                return Heyting3.VETOED
            if value <= self._fos_collapse:
                return Heyting3.VETOED
            if value >= self._fos_limit:
                return Heyting3.COHERENT
            span = self._fos_limit - self._fos_collapse
            if span <= self._reg:
                return Heyting3.VETOED
            return float((value - self._fos_collapse) / span)

        def _settlement_channel(value: float) -> float:
            if not math.isfinite(value):
                return Heyting3.VETOED
            smax = self._settlement_max
            if smax <= self._reg:
                return Heyting3.COHERENT if value <= self._tol else Heyting3.VETOED
            soft = self._settlement_soft_ratio * smax
            if value <= soft:
                return Heyting3.COHERENT
            if value > smax:
                return Heyting3.VETOED
            span = smax - soft
            if span <= self._reg:
                return Heyting3.VETOED
            return float((smax - value) / span)

        confidence = Heyting3.meet(_fos_channel(fos), _settlement_channel(settlement))
        if handoff.is_plastic_failure or handoff.is_settlement_excessive:
            confidence = Heyting3.meet(confidence, Heyting3.VETOED)
        if not math.isfinite(confidence):
            return Heyting3.VETOED
        return float(min(1.0, max(0.0, confidence)))

    def phase3_decide_heyting(
        self,
        handoff: Phase2LithoHandoff,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Fase 3.3 — Clasificador de Heyting trivalente Ω₃.

        Reglas:
          - VETOED inmediato: colapso plástico, asiento excesivo, métricas
            no finitas o percolación total del soporte.
          - DEGRADED: FoS marginal en (FoS_collapse, FoS_limit].
          - El positrón e⁺ impide DEGRADED → VETOED por expiración de gracia.
          - Gracia expirada sin override: colapso a VETOED.
          - El encuentro con la rampa nunca eleva el veredicto.
        """
        curr_time = float(current_time if current_time is not None else time.time())
        hard_reasons = []
        if handoff.is_plastic_failure:
            hard_reasons.append("plastic_failure")
        if handoff.is_settlement_excessive:
            hard_reasons.append("excessive_settlement")
        if math.isnan(handoff.min_fos_shear) or math.isnan(handoff.max_settlement):
            hard_reasons.append("nonfinite_metrics")
        if handoff.is_topologically_compromised and float(
            handoff.diagnostics.get("plastic_node_ratio", 0.0)
        ) >= 1.0 - self._tol:
            hard_reasons.append("support_percolation")

        override_valid = self.phase3_verify_override(override_token)
        override_sha = None
        if override_valid and override_token is not None:
            override_sha = hashlib.sha256(str(override_token).encode("utf-8")).hexdigest()

        confidence = self.phase3_confidence_ramp(handoff)
        heyting = "COHERENT"
        is_hard_veto = False
        is_soft_veto = False
        time_remaining = 0.0

        if hard_reasons:
            heyting = "VETOED"
            is_hard_veto = True
            self._is_soft_veto_active = False
            self._soft_veto_timestamp = None
            logger.critical(
                "¡VETO DURO INSTANTÁNEO POR COLAPSO GEOMECÁNICO / ASENTAMIENTO "
                "EXCESIVO! Razones: %s",
                ", ".join(hard_reasons),
            )
        elif handoff.is_warning_active:
            is_soft_veto = True
            if elapsed_grace_seconds is not None:
                if not math.isfinite(float(elapsed_grace_seconds)):
                    time_remaining = 0.0
                else:
                    time_remaining = max(
                        0.0, self._grace_max - float(elapsed_grace_seconds)
                    )
                    if not self._is_soft_veto_active:
                        self._is_soft_veto_active = True
                        self._soft_veto_timestamp = curr_time
            elif simulate_grace_expired:
                time_remaining = 0.0
            elif not self._is_soft_veto_active:
                self._is_soft_veto_active = True
                self._soft_veto_timestamp = curr_time
                time_remaining = self._grace_max
                logger.warning(
                    "¡VETO SUAVE ACTIVO (LUZ ÁMBAR)! FoS marginal. "
                    "Ventana de gracia de %.0f segundos iniciada.",
                    self._grace_max,
                )
            else:
                elapsed = curr_time - float(self._soft_veto_timestamp or curr_time)
                time_remaining = max(0.0, self._grace_max - elapsed)

            if override_valid:
                heyting = "DEGRADED"
                is_soft_veto = False
                is_hard_veto = False
                time_remaining = 0.0
                self._is_soft_veto_active = False
                self._soft_veto_timestamp = None
                logger.info(
                    "¡POSITRÓN DE AUTORIZACIÓN HUMANA [e+] INYECTADO! "
                    "Aniquilando anomalía geomecánica en Fock. Sello: %s",
                    override_sha[:16] if override_sha else "UNKNOWN",
                )
            elif time_remaining <= self._tol or simulate_grace_expired:
                heyting = "VETOED"
                is_hard_veto = True
                is_soft_veto = False
                self._is_soft_veto_active = False
                self._soft_veto_timestamp = None
                logger.critical(
                    "¡PERÍODO DE GRACIA EXPIRADO SIN OVERRIDE VÁLIDO! "
                    "Colapsando Heyting a VETOED terminal."
                )
            else:
                heyting = "DEGRADED"
        else:
            heyting = "COHERENT"
            self._is_soft_veto_active = False
            self._soft_veto_timestamp = None

        truth = Heyting3.from_verdict(heyting)
        truth = Heyting3.meet(truth, Heyting3.quantize(confidence))
        heyting = Heyting3.to_verdict(truth)
        if heyting == "VETOED":
            is_hard_veto = True
            is_soft_veto = False
        elif heyting == "DEGRADED":
            is_hard_veto = False

        return {
            "heyting_verdict": heyting,
            "heyting_truth_value": float(truth),
            "confidence": float(confidence),
            "is_hard_veto": bool(is_hard_veto),
            "is_soft_veto": bool(is_soft_veto and heyting == "DEGRADED"),
            "hard_reasons": hard_reasons,
            "override_valid": bool(override_valid),
            "override_sha256": override_sha,
            "time_grace_remaining": float(time_remaining),
            "current_time": curr_time,
            "simulate_grace_expired": bool(simulate_grace_expired),
            "elapsed_grace_seconds": (
                float(elapsed_grace_seconds)
                if elapsed_grace_seconds is not None
                else math.nan
            ),
        }

    def phase3_actuate_crowbar_interlock(
        self,
        heyting_verdict: str,
    ) -> Tuple[bool, float]:
        """
        Fase 3.4 — Actuación de hardware Crowbar ESP32/BT151.

        Sólo el colapso a VETOED conmuta el interlock. El jitter gaussiano
        simula la dispersión IRAM y se recorta a [385, 415] ns.
        """
        if str(heyting_verdict) != "VETOED":
            return False, 0.0
        jitter = float(self._rng.normal(loc=398.5, scale=self._jitter_sigma))
        latency_ns = float(
            np.clip(jitter, _CROWBAR_LATENCY_MIN_NS, _CROWBAR_LATENCY_MAX_NS)
        )
        logger.critical(
            "¡COLA DE HEYTING COLAPSADA EN SOBERANO LITOLÓGICO!\n"
            "  - Ejecutando subrutina local isVerdictCoherent() en C++...\n"
            "  - Despachando ISR en IRAM en menos de 400 ns...\n"
            "  - Conmutando pin de hardware GPIO14 a HIGH en %.2f ns...\n"
            "  - ¡Tiristor rápido de potencia BT151 (Crowbar) gatillado!\n"
            "  - Mezcladoras y bombas hidráulicas paralizadas en el milisegundo cero.",
            latency_ns,
        )
        return True, latency_ns

    def _phase3_cryptographic_seal(
        self,
        handoff: Phase2LithoHandoff,
        decision: Mapping[str, Any],
        interlock_fired: bool,
        switching_latency_ns: float,
    ) -> str:
        """Fase 3.5 — Sello criptográfico SHA-256 canónico (no depende de `repr`)."""
        h = hashlib.sha256()
        h.update(b"LITHO/CERT/v3")
        _sha_update_str(h, "PHASE3/LITHO_AGENT_GOVERNANCE")
        _sha_update_arr(h, handoff.per_node_fos)
        _sha_update_arr(h, handoff.per_layer_settlement)
        _sha_update_str(h, str(decision.get("heyting_verdict", "")))
        h.update(_pack_f64(float(handoff.min_fos_shear)))
        h.update(_pack_f64(float(handoff.max_settlement)))
        h.update(_pack_f64(float(decision.get("confidence", math.nan))))
        h.update(_pack_f64(float(decision.get("heyting_truth_value", math.nan))))
        h.update(_pack_f64(float(decision.get("time_grace_remaining", math.nan))))
        h.update(_pack_f64(float(switching_latency_ns)))
        h.update(b"\x01" if handoff.is_warning_active else b"\x00")
        h.update(b"\x01" if handoff.is_plastic_failure else b"\x00")
        h.update(b"\x01" if handoff.is_settlement_excessive else b"\x00")
        h.update(b"\x01" if handoff.is_topologically_compromised else b"\x00")
        h.update(b"\x01" if bool(decision.get("is_hard_veto")) else b"\x00")
        h.update(b"\x01" if bool(decision.get("is_soft_veto")) else b"\x00")
        h.update(b"\x01" if interlock_fired else b"\x00")
        _sha_update_str(h, handoff.session_sha256)
        _sha_update_str(h, str(decision.get("override_sha256") or ""))
        return h.hexdigest()

    def phase3_issue_verdict(
        self,
        handoff: Phase2LithoHandoff,
        decision: Mapping[str, Any],
        interlock_fired: bool,
        switching_latency_ns: float,
    ) -> GeomechanicalVerdict:
        """
        Fase 3.6 — Emisión del veredicto certificado.

        Incluye Ω₃, FoS, asiento, rampa, gracia, hardware y sello SHA-256.
        """
        seal = self._phase3_cryptographic_seal(
            handoff=handoff,
            decision=decision,
            interlock_fired=interlock_fired,
            switching_latency_ns=switching_latency_ns,
        )
        phase_chain = ("PHASE1/OBSERVE", "PHASE2/ORIENT", "PHASE3/DECIDE/ACT")
        diagnostics = dict(handoff.diagnostics)
        diagnostics["decision"] = dict(decision)
        diagnostics["hardware"] = {
            "interlock_fired": interlock_fired,
            "switching_latency_ns": switching_latency_ns,
        }
        return GeomechanicalVerdict(
            heyting_verdict=str(decision["heyting_verdict"]),
            min_fos_shear=float(handoff.min_fos_shear),
            max_settlement=float(handoff.max_settlement),
            is_soft_veto_active=bool(decision["is_soft_veto"]),
            is_hard_veto_active=bool(decision["is_hard_veto"]),
            switching_latency_ns=float(switching_latency_ns),
            sha256_hash=seal,
            session_sha256=handoff.session_sha256,
            phase_chain=phase_chain,
            diagnostics=diagnostics,
            confidence=float(decision.get("confidence", math.nan)),
            heyting_truth_value=float(
                decision.get(
                    "heyting_truth_value",
                    Heyting3.from_verdict(str(decision["heyting_verdict"])),
                )
            ),
            override_sha256=str(decision.get("override_sha256") or ""),
            time_grace_remaining=float(decision.get("time_grace_remaining", 0.0)),
        )

    def phase3_close_loop(
        self,
        phase2_handoff: Phase2LithoHandoff,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
    ) -> GeomechanicalVerdict:
        """
        Fase 3.7 — Orquestación completa de la Fase 3.

        Ejecuta, en orden:
          1. validación de frontera Φ₂→₃;
          2. rampa de confianza y decisión Heyting;
          3. actuación de hardware;
          4. certificación final.
        """
        validated = self.phase3_from_phase2(phase2_handoff)
        decision = self.phase3_decide_heyting(
            handoff=validated,
            override_token=override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
            elapsed_grace_seconds=elapsed_grace_seconds,
        )
        interlock_fired, latency = self.phase3_actuate_crowbar_interlock(
            decision["heyting_verdict"]
        )
        verdict = self.phase3_issue_verdict(
            handoff=validated,
            decision=decision,
            interlock_fired=interlock_fired,
            switching_latency_ns=latency,
        )
        if verdict.heyting_verdict == "VETOED":
            logger.critical(
                "Soberano Litologico emitió VETO DURO. Sello: %s",
                verdict.sha256_hash[:16],
            )
        else:
            logger.info(
                "Soberano Litologico regulado síncronamente. Veredicto: %s. "
                "Confianza=%.6g. Sello: %s",
                verdict.heyting_verdict,
                verdict.confidence,
                verdict.sha256_hash[:16],
            )
        return verdict

    # ═════════════════════════════════════════════════════════════════════════
    # API PRINCIPAL COMPATIBLE OODA
    # ═════════════════════════════════════════════════════════════════════════

    def audit_lazo_cerrado(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
        active_shear_stresses: np.ndarray,
        fail_planes_normal: np.ndarray,
        compression_indices: np.ndarray,
        void_ratios_e0: np.ndarray,
        layer_thicknesses: np.ndarray,
        sigma_v0_effective: np.ndarray,
        delta_sigma_v: np.ndarray,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
    ) -> GeomechanicalVerdict:
        """
        API principal — Orquesta el funtor compuesto Φ₃ ∘ Φ₂ ∘ Φ₁.

        Equivalencia de fases:
          OBSERVE    → phase1_close_and_open_phase2
          ORIENT     → phase2_close_and_open_phase3
          DECIDE/ACT → phase3_close_loop
        """
        phase1_handoff = self.phase1_close_and_open_phase2(
            cohesions=cohesions,
            friction_angles_deg=friction_angles_deg,
            effective_stresses=effective_stresses,
            active_shear_stresses=active_shear_stresses,
            fail_planes_normal=fail_planes_normal,
            compression_indices=compression_indices,
            void_ratios_e0=void_ratios_e0,
            layer_thicknesses=layer_thicknesses,
            sigma_v0_effective=sigma_v0_effective,
            delta_sigma_v=delta_sigma_v,
        )
        phase2_handoff = self.phase2_close_and_open_phase3(
            phase1_handoff=phase1_handoff,
        )
        return self.phase3_close_loop(
            phase2_handoff=phase2_handoff,
            override_token=override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
            elapsed_grace_seconds=elapsed_grace_seconds,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # MÉTODOS LEGADOS (COMPATIBILIDAD)
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_geomechanical_stability(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
        active_shear_stresses: np.ndarray,
        fail_planes_normal: np.ndarray,
    ) -> Tuple[float, bool, bool]:
        """
        API legada — equivalente a Fase 2.5.

        Retorna
        -------
        Tuple[float, bool, bool]
            (min_fos, is_warning_active, is_plastic_failure)
        """
        c, phi = self.phase1_validate_strength_fields(cohesions, friction_angles_deg)
        sigma_prime, tau_act, normals = self.phase1_validate_stress_fields(
            effective_stresses, active_shear_stresses, fail_planes_normal, c.size
        )
        min_fos, is_warning, is_collapse, _, _ = self.phase2_mohr_coulomb_audit(
            cohesions=c,
            friction_angles_deg=phi,
            effective_stresses=sigma_prime,
            active_shear_stresses=tau_act,
            fail_planes_normal=normals,
        )
        return min_fos, is_warning, is_collapse

    def evaluate_consolidation_settlement(
        self,
        compression_indices: np.ndarray,
        void_ratios_e0: np.ndarray,
        layer_thicknesses: np.ndarray,
        sigma_v0_effective: np.ndarray,
        delta_sigma_v: np.ndarray,
    ) -> Tuple[float, bool]:
        """
        API legada — equivalente a Fase 2.6.

        Retorna
        -------
        Tuple[float, bool]
            (total_settlement, is_settlement_excessive)
        """
        Cc, e0, H0, sigma_v0, delta_sigma = self.phase1_validate_consolidation_fields(
            compression_indices,
            void_ratios_e0,
            layer_thicknesses,
            sigma_v0_effective,
            delta_sigma_v,
        )
        total_settlement, is_excessive, _, _ = self.phase2_consolidation_audit(
            compression_indices=Cc,
            void_ratios_e0=e0,
            layer_thicknesses=H0,
            sigma_v0_effective=sigma_v0,
            delta_sigma_v=delta_sigma,
        )
        return total_settlement, is_excessive
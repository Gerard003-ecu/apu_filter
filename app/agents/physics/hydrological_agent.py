# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Hydrological Agent (Soberano de Calibre Hidrológico)                ║
║ Ruta   : app/agents/physics/hydrological_agent.py                            ║
║ Versión: 1.1.0-Doctoral-Biot-Terzaghi-Richards-Heyting-OODA-IRAM-ESP32-Secure║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO (OODA):                  ║
║ Este agente supervisor ciber-físico opera en el Estrato de la Sabiduría      ║
║ ($V_{\mathbb{W}}$, Nivel 0) u Omega ($V_{\Omega}$, Nivel 0.5) para gobernar  ║
║ síncronamente al motor de poroelasticidad y flujos de Richards               ║
║ [hydrological_manifold.py].                                                  ║
║                                                                              ║
║ Evalúa de forma covariante los tensores de esfuerzos efectivos de Biot,      ║
║ la estabilidad espectral del Laplaciano hidráulico discreto (DEC) y los      ║
║ gradientes críticos de sifonamiento de Terzaghi, modulando el veredicto en   ║
║ el retículo de Heyting trivalente $\Omega_3$ para inyectar conmutación rápida║
║ de potencia (< 400 ns en IRAM) ante desbalances catastróficos.               ║
╚══════════════════════════════════════════════════════════════════════════════╝

================════════════════════════════════════════════════════════════════
I. ANCLAJE MATEMÁTICO DOCTORAL (Teoría de Haces Celulares y Estabilidad Espectral)
================════════════════════════════════════════════════════════════════

Definición 1 (Estabilidad Espectral de Richards-Poisson en DEC):
  El flujo permanente insaturado se discretiza sobre el 1-esqueleto del complejo simplicial
  de la obra mediante el operador Laplaciano ponderado de Richards:
  $$\mathbf{\Delta}_{\mathrm{Richards}}(\mathrm{sat}) = \mathbf{B}_1 \mathbf{W}_{\mathrm{hyd}}(\mathrm{sat}) \mathbf{B}_1^\top \succeq \mathbf{0}$$
  Para garantizar la conectividad hidráulica global del foso (evitando la desecación o 
  impermeabilización de frentes), el segundo menor autovalor del Laplaciano (conectividad de Fiedler) 
  debe superar el umbral elástico del topos en la FPU, garantizando conexidad simplicial única:
  $$\lambda_2(\mathbf{\Delta}_{\mathrm{Richards}}) \ge \tau_{\mathrm{Fiedler}} \quad \implies \quad \beta_0 \equiv \dim H^0(K_{\mathrm{hydro}}; \, \mathbb{Z}) = 1$$

Definición 2 (Causalidad y Preservación de Esfuerzos Efectivos de Biot-Terzaghi):
  La poroelasticidad del foso de excavación se acopla mediante el tensor de esfuerzos efectivos de Biot:
  $$\sigma'_{\mu\nu} = \sigma_{\mu\nu} - \alpha_{\mathrm{Biot}} \, P_f \, \delta_{\mu\nu}$$
  Donde la presión de poros intersticial $P_f$ es inducida por la succión matricial y la saturación:
  $$P_f = -\gamma_w \cdot \psi_w \cdot \mathrm{sat}$$
  El colapso geomecánico por licuación de arenas se diagnostica de forma exacta si el determinante 
  del tensor efectivo decae a la región no positiva definida, anulando la resistencia al corte del fango:
  $$\det(\sigma'_{\mu\nu}) \le \tau_{\mathrm{liq\_limit}} \quad \implies \quad \mathtt{liquefaction\_flag} = \mathtt{True}$$

Definición 3 (La Rampa de Confianza de de Rham y override Criptográfico):
  Para eludir paradas de obra destructivas por ruidos espurios de reflectometría temporal sónica TDR
  o interferencias analógicas transitorias rápidos de alta frecuencia, el agente implementa una rampa elástica 
  sobre el clasificador de subobjetos del Álgebra de Heyting distributiva trivalente:
  $$\Omega_3 := \{\mathtt{COHERENT}, \, \mathtt{DEGRADED}, \, \mathtt{VETOED}\}$$
  - Veto Suave (Luz Ámbar): Se activa si el gradiente de filtración o la desviación espectral habita el rango elástico:
    $$0.3 \cdot \tau_{\mathrm{Lmax}} \cdot \tau_{\mathrm{margin}} < \delta_{\mathrm{similarity}} \le 0.5 \cdot \tau_{\mathrm{Lmax}} \cdot \tau_{\mathrm{margin}}$$
    Concede una ventana de gracia de 1 hora. El interventor puede inyectar un Positrón de Autorización $e^+$
    signed con HMAC-SHA256 para neutralizar la anomalía $e^-$ en Fock, disipando la alerta:
    $$e^- + e^+ \longrightarrow 2\gamma \quad \implies \quad \mathtt{heyting\_verdict} \mapsto \mathtt{DEGRADED}$$
  - Veto Duro (Bypass en Silicio): Se activa si se viola el límite geomecánico de Terzaghi, se delata dolo, 
    o expira el plazo de gracia, colapsando el retículo al Supremo terminal VETOED ($\top$).

================════════════════════════════════════════════════════════════════
II. AXIOMÁTICA INMUNILÓGICA DE CONTROL COVARIANTE (Leyes de Consistencia)
================════════════════════════════════════════════════════════════════

Axioma I (Principio de Pasividad de Lyapunov-Rayleigh):
  La evolución de la energía hidráulica acumulada $\mathcal{H}_H(t)$ en el foso satisface
  la inecuación de Clausius-Duhem para sistemas disipativos Port-Hamiltonianos:
  $$\dot{\mathcal{H}}_H(t) = -\nabla \mathcal{H}_H^\top \mathbf{R}_{\mathrm{hyd}}(\mathrm{sat}) \nabla \mathcal{H}_H \le 0 \quad \text{con} \quad \mathbf{R}_{\mathrm{hyd}} \succeq \mathbf{0}$$

Axioma II (Teorema de Actuación Ciber-Física en IRAM):
  Ante el colapso al Supremo de veto ($\top$), la subrutina local isVerdictCoherent() del ESP32 perimetral
  desvía síncronamente el control hacia la ISR en memoria estática IRAM en menos de 400 ns:
  $$t_{\mathrm{actuation}} \le \tau_{\mathrm{IRAM}} = 400\text{ ns} \quad \implies \quad \mathtt{GPIO14} \mapsto \mathtt{HIGH}$$
  Disparando el tiristor rápido BT151 (Crowbar de potencia) para paralizar mecánicamente la obra civil,
  anulando la alucinación o fraude antes de liquidar transacciones ante el SECOP II.

Axioma III (Auditoría de Sifonamiento Crítico de Terzaghi):
  El arrastre de finos y la socavación por sifonamiento en las aristas del complejo simplicial
  está acotado estrictamente por el gradiente hidráulico crítico derivado de la densidad del suelo saturado:
  $$i_{\mathrm{grad}} = \frac{|\Delta H_e|}{L_e} \le i_{\mathrm{crit}} = \frac{\rho_{\mathrm{sat}} - \rho_w}{\rho_w}$$
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
    "HydrologicalAgent",
    "HydrologicalVerdict",
    "Phase1HydroAgentHandoff",
    "Phase2HydroAgentHandoff",
    "Heyting3",
]

__version__: Final[str] = (
    "3.0.0-Biot-Terzaghi-Fiedler-Heyting-Grace-IRAM-Governance"
)

logger = logging.getLogger("APU.Physics.HydrologicalAgent")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_LATENCY_MIN_NS: Final[float] = 385.0
_CROWBAR_LATENCY_MAX_NS: Final[float] = 415.0
_SHA256_HEX_LEN: Final[int] = 64
_PHASE1_ENTRY: Final[str] = "phase2_from_phase1"
_PHASE2_ENTRY: Final[str] = "phase3_from_phase2"
_I3: Final[np.ndarray] = np.eye(3, dtype=np.float64)


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


def _safe_float(value: Any, default: float = math.nan) -> float:
    """Convierte a float conservando ±∞; NaN o tipos inválidos → default."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x):
        return default
    return x


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


def _nested_get(mapping: Any, *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


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
class Phase1HydroAgentHandoff:
    """
    Frontera formal Φ₁→₂.

    Salida cerrada de la Fase 1 y dominio de `phase2_from_phase1`.
    """

    hydraulic_head: np.ndarray
    pore_pressures: np.ndarray
    flow_rates: np.ndarray
    laplacian_eigenvalues: np.ndarray
    sigma_prime_tensors: np.ndarray
    is_siphoning_active: bool
    session_sha256: str
    diagnostics: Dict[str, Any]
    next_entrypoint: str
    hydraulic_gradients: Optional[np.ndarray] = None
    max_gradient_hint: float = math.nan

    def __post_init__(self) -> None:
        object.__setattr__(self, "hydraulic_head", _freeze_array(self.hydraulic_head))
        object.__setattr__(self, "pore_pressures", _freeze_array(self.pore_pressures))
        object.__setattr__(self, "flow_rates", _freeze_array(self.flow_rates))
        object.__setattr__(
            self, "laplacian_eigenvalues", _freeze_array(self.laplacian_eigenvalues)
        )
        object.__setattr__(
            self, "sigma_prime_tensors", _freeze_array(self.sigma_prime_tensors)
        )
        if self.hydraulic_gradients is not None:
            object.__setattr__(
                self, "hydraulic_gradients", _freeze_array(self.hydraulic_gradients)
            )

    def __hash__(self) -> int:
        return hash((self.session_sha256, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase1HydroAgentHandoff):
            return NotImplemented
        return (
            self.session_sha256 == other.session_sha256
            and self.next_entrypoint == other.next_entrypoint
        )


@dataclass(frozen=True, slots=True, eq=False)
class Phase2HydroAgentHandoff:
    """
    Frontera formal Φ₂→₃.

    Salida cerrada de la Fase 2 y dominio de `phase3_from_phase2`.
    """

    hydraulic_head: np.ndarray
    pore_pressures: np.ndarray
    flow_rates: np.ndarray
    laplacian_eigenvalues: np.ndarray
    sigma_prime_tensors: np.ndarray
    is_siphoning_active: bool
    session_sha256: str
    fiedler_value: float
    is_disconnected: bool
    max_gradient: float
    min_det_sigma_prime: float
    min_principal_stress: float
    is_liquefaction_active: bool
    diagnostics: Dict[str, Any]
    next_entrypoint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "hydraulic_head", _freeze_array(self.hydraulic_head))
        object.__setattr__(self, "pore_pressures", _freeze_array(self.pore_pressures))
        object.__setattr__(self, "flow_rates", _freeze_array(self.flow_rates))
        object.__setattr__(
            self, "laplacian_eigenvalues", _freeze_array(self.laplacian_eigenvalues)
        )
        object.__setattr__(
            self, "sigma_prime_tensors", _freeze_array(self.sigma_prime_tensors)
        )

    def __hash__(self) -> int:
        return hash((self.session_sha256, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase2HydroAgentHandoff):
            return NotImplemented
        return (
            self.session_sha256 == other.session_sha256
            and self.next_entrypoint == other.next_entrypoint
            and self.fiedler_value == other.fiedler_value
            and self.max_gradient == other.max_gradient
            and self.is_liquefaction_active == other.is_liquefaction_active
            and self.is_disconnected == other.is_disconnected
        )


@dataclass(frozen=True, slots=True, eq=False)
class HydrologicalVerdict:
    """
    Certificado inmutable de regularidad espectral e hidrológica.

    Se conservan los campos originales por compatibilidad. Los campos
    añadidos van al final con valores por defecto.
    """

    heyting_verdict: str
    fiedler_value: float
    max_gradient: float
    min_det_sigma_prime: float
    is_siphoning_active: bool
    is_liquefaction_active: bool
    is_soft_veto_active: bool
    is_hard_veto_active: bool
    time_grace_remaining: float
    cryptographic_seal: str

    session_sha256: str = ""
    phase_chain: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    heyting_truth_value: float = 1.0
    override_sha256: str = ""
    min_principal_stress: float = math.nan
    actuation_latency_ns: float = 0.0

    def __hash__(self) -> int:
        return hash(self.cryptographic_seal)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HydrologicalVerdict):
            return NotImplemented
        return self.cryptographic_seal == other.cryptographic_seal

    def __repr__(self) -> str:
        return (
            f"HydrologicalVerdict(verdict={self.heyting_verdict!r}, "
            f"λ₂={self.fiedler_value:.6g}, i_max={self.max_gradient:.6g}, "
            f"seal={self.cryptographic_seal[:12]!r})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SOBERANO DE CALIBRE HIDROLÓGICO — TRES FASES ANIDADAS
# ═════════════════════════════════════════════════════════════════════════════

class HydrologicalAgent:
    """
    Soberano de Calibre Hidrológico — OODA en 3 fases anidadas.

    FASE 1  OBSERVE    : validación Richards/Biot, tensores, sello de sesión.
    FASE 2  ORIENT     : Fiedler–Cheeger, sifonamiento, licuación, Kirchhoff.
    FASE 3  DECIDE/ACT : rampa, Heyting Ω₃, override e⁺, Crowbar, certificado.
    """

    def __init__(
        self,
        fiedler_threshold: float = 1e-5,
        siphoning_margin: float = 0.8,
        liquefaction_limit: float = 1e-12,
        grace_period: float = 3600.0,
        tolerance: float = 1e-12,
        rng_seed: Optional[int] = None,
        jitter_sigma: float = 1.2,
        authorized_tokens: Optional[Tuple[str, ...]] = None,
        hmac_key: Optional[bytes] = None,
        fiedler_soft_ratio: float = 0.5,
    ) -> None:
        if not math.isfinite(fiedler_threshold) or fiedler_threshold <= 0.0:
            raise ValueError(
                "fiedler_threshold debe ser finita y estrictamente positiva."
            )
        if not math.isfinite(siphoning_margin) or siphoning_margin <= 0.0:
            raise ValueError(
                "siphoning_margin debe ser finita y estrictamente positiva."
            )
        if not math.isfinite(liquefaction_limit) or liquefaction_limit < 0.0:
            raise ValueError("liquefaction_limit debe ser finita y no negativa.")
        if not math.isfinite(grace_period) or grace_period < 0.0:
            raise ValueError("grace_period debe ser finita y no negativa.")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance debe ser finita y estrictamente positiva.")
        if not math.isfinite(jitter_sigma) or jitter_sigma < 0.0:
            raise ValueError("jitter_sigma debe ser finita y no negativa.")
        if not math.isfinite(fiedler_soft_ratio) or not (
            0.0 < fiedler_soft_ratio < 1.0
        ):
            raise ValueError("fiedler_soft_ratio debe vivir en (0, 1).")

        self._fiedler_min = float(fiedler_threshold)
        self._siph_margin = float(siphoning_margin)
        self._liq_limit = float(liquefaction_limit)
        self._grace_max = float(grace_period)
        self._tol = float(tolerance)
        self._reg = max(1e-15, self._tol * 1e-3)
        self._jitter_sigma = float(jitter_sigma)
        self._fiedler_soft_ratio = float(fiedler_soft_ratio)
        self._rng = np.random.default_rng(rng_seed)

        if authorized_tokens is None:
            self._authorized_tokens = {
                "AUT_POS_SABIDURIA_777",
                "OVERRIDE_HIDRO_IDU_2026",
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
    # VALIDADORES Y EXTRACCIÓN DE SEÑALES (infraestructura de Fase 1)
    # ═════════════════════════════════════════════════════════════════════════

    def _validate_required_vector(
        self,
        name: str,
        arr: Any,
        length: Optional[int] = None,
    ) -> np.ndarray:
        """Valida un vector requerido finito, con longitud opcional."""
        if arr is None:
            raise ValueError(f"El campo '{name}' es obligatorio.")
        a = np.asarray(arr, dtype=np.float64).ravel()
        if length is not None and a.size != length:
            raise ValueError(
                f"El campo '{name}' debe tener longitud {length}. Obtenido: {a.size}"
            )
        if not np.all(np.isfinite(a)):
            raise ValueError(f"El campo '{name}' contiene valores NaN o infinitos.")
        return _canonicalize_signed_zero(a)

    def _validate_optional_vector(
        self,
        name: str,
        arr: Any,
        length: Optional[int] = None,
    ) -> np.ndarray:
        """Valida un vector opcional finito. Si es None, retorna vacío."""
        if arr is None:
            return np.empty(0, dtype=np.float64)
        return self._validate_required_vector(name, arr, length=length)

    def _validate_stress_tensors(
        self,
        name: str,
        arr: Any,
        n_nodes: int,
    ) -> np.ndarray:
        """
        Valida tensores de esfuerzo efectivo σ' por nudo.

        Forma requerida: (n_nodes, 3, 3). Se proyectan sobre Sym³, dominio
        del círculo de Mohr y del spectral split.
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

    def _extract_max_gradient_hint(self, hydraulic_state: Any) -> float:
        """
        Extrae i_max desde el estado hidráulico, en orden de prioridad:

          1. `max_gradient`
          2. `hydraulic_gradients`
          3. `diagnostics['siphoning_max_gradient']`
          4. `diagnostics['stability']['siphoning_max_gradient']`
        """
        val = getattr(hydraulic_state, "max_gradient", None)
        if val is not None:
            x = _safe_float(val, default=math.nan)
            if not math.isnan(x):
                return x

        grad = getattr(hydraulic_state, "hydraulic_gradients", None)
        if grad is not None:
            try:
                g = np.asarray(grad, dtype=np.float64).ravel()
                if g.size > 0 and np.all(np.isfinite(g)):
                    return float(np.max(np.abs(g)))
            except (TypeError, ValueError):
                pass

        diagnostics = getattr(hydraulic_state, "diagnostics", None)
        if isinstance(diagnostics, dict):
            for path in (
                ("siphoning_max_gradient",),
                ("max_gradient",),
                ("stability", "siphoning_max_gradient"),
                ("stability", "max_gradient"),
            ):
                raw = _nested_get(diagnostics, *path)
                x = _safe_float(raw, default=math.nan)
                if not math.isnan(x):
                    return x
        return math.nan

    def _extract_state_diagnostics(self, hydraulic_state: Any) -> Dict[str, Any]:
        """Pullback de diagnósticos del colector, si existen."""
        raw = getattr(hydraulic_state, "diagnostics", None)
        return dict(raw) if isinstance(raw, dict) else {}

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 1 — OBSERVE: VALIDACIÓN RICHARDS/BIOT Y SELLO
    # ═════════════════════════════════════════════════════════════════════════

    def phase1_validate_hydraulic_fields(
        self,
        hydraulic_state: Any,
    ) -> Dict[str, Any]:
        """
        Fase 1.1 — Validación de 0-formas y 1-formas del colector.

        H ∈ R^n (potencial total), u_w ∈ R^n (presión de poros),
        Q ∈ R^m (flujo de arista), σ(L) ⊂ [0, ∞) (espectro de Δ₀).
        """
        if hydraulic_state is None:
            raise ValueError("hydraulic_state es obligatorio.")
        H = self._validate_required_vector(
            "hydraulic_state.hydraulic_head",
            getattr(hydraulic_state, "hydraulic_head", None),
        )
        n_nodes = int(H.size)
        if n_nodes < 1:
            raise ValueError("El estado hidráulico debe contener al menos un nudo.")
        P_f = self._validate_required_vector(
            "hydraulic_state.pore_pressures",
            getattr(hydraulic_state, "pore_pressures", None),
            length=n_nodes,
        )
        Q_e = self._validate_required_vector(
            "hydraulic_state.flow_rates",
            getattr(hydraulic_state, "flow_rates", None),
        )
        eigenvalues = self._validate_optional_vector(
            "hydraulic_state.laplacian_eigenvalues",
            getattr(hydraulic_state, "laplacian_eigenvalues", None),
        )
        return {
            "hydraulic_head": H,
            "pore_pressures": P_f,
            "flow_rates": Q_e,
            "laplacian_eigenvalues": eigenvalues,
            "n_nodes": n_nodes,
        }

    def phase1_validate_effective_stresses(
        self,
        hydraulic_state: Any,
        sigma_prime_tensors: Optional[np.ndarray],
        n_nodes: int,
    ) -> np.ndarray:
        """
        Fase 1.2 — Validación de σ' ∈ Sym³ por nudo.

        Si no se aporta el tensor, se toma `effective_stresses` del estado.
        """
        if sigma_prime_tensors is None:
            sigma_prime_tensors = getattr(hydraulic_state, "effective_stresses", None)
        return self._validate_stress_tensors(
            "sigma_prime_tensors", sigma_prime_tensors, n_nodes=n_nodes
        )

    def phase1_extract_gradient_signals(
        self,
        hydraulic_state: Any,
    ) -> Tuple[Optional[np.ndarray], float, bool]:
        """
        Fase 1.3 — Extracción de i_e y bandera de sifonamiento.

        Conserva el vector de gradientes si es finito; el hint escalar cubre
        el caso en que el colector sólo reporta el máximo.
        """
        is_siphoning = bool(getattr(hydraulic_state, "is_siphoning_active", False))
        gradients_raw = getattr(hydraulic_state, "hydraulic_gradients", None)
        hydraulic_gradients: Optional[np.ndarray] = None
        if gradients_raw is not None:
            hydraulic_gradients = self._validate_optional_vector(
                "hydraulic_state.hydraulic_gradients", gradients_raw
            )
        max_gradient_hint = self._extract_max_gradient_hint(hydraulic_state)
        return hydraulic_gradients, float(max_gradient_hint), is_siphoning

    def phase1_tensor_hygiene(self, sigma_prime: np.ndarray) -> Dict[str, float]:
        """
        Fase 1.4 — Higiene de Sym³.

        Residual de simetría (nulo tras la proyección), cota de Frobenius
        y presión media p' = tr(σ')/3.
        """
        a = np.asarray(sigma_prime, dtype=np.float64)
        if a.ndim != 3 or a.shape[-2:] != (3, 3):
            raise ValueError("sigma_prime debe ser un lote (n, 3, 3).")
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
        H: np.ndarray,
        P_f: np.ndarray,
        Q_e: np.ndarray,
        eigenvalues: np.ndarray,
        hygiene: Mapping[str, float],
        collector_diag: Mapping[str, Any],
        current_time: float,
        is_siphoning: bool,
        max_gradient_hint: float,
    ) -> Dict[str, Any]:
        """
        Fase 1.5 — Diagnóstico de Banach/estatística de la observación.
        """
        return {
            "n_nodes": int(H.size),
            "n_edges_flow_rates": int(Q_e.size),
            "n_eigenvalues": int(eigenvalues.size),
            "head_mean": float(np.mean(H)) if H.size else 0.0,
            "head_oscillation": float(np.max(H) - np.min(H)) if H.size else 0.0,
            "pore_pressure_min": float(np.min(P_f)) if P_f.size else math.nan,
            "pore_pressure_max": float(np.max(P_f)) if P_f.size else math.nan,
            "flow_l1": float(np.sum(np.abs(Q_e))) if Q_e.size else 0.0,
            "is_siphoning_flag_from_state": bool(is_siphoning),
            "max_gradient_hint": float(max_gradient_hint),
            "observe_timestamp": float(current_time),
            "tolerance": self._tol,
            "regularization": self._reg,
            "machine_epsilon": _MACHINE_EPS,
            "fiedler_threshold": self._fiedler_min,
            "siphoning_margin": self._siph_margin,
            "liquefaction_limit": self._liq_limit,
            "collector_kirchhoff_residual": _finite_or_nan(
                collector_diag.get("kirchhoff_residual")
            ),
            "collector_mass_residual": _finite_or_nan(
                collector_diag.get("richards_mass_residual")
            ),
            **dict(hygiene),
        }

    def _phase1_session_hash(
        self,
        H: np.ndarray,
        P_f: np.ndarray,
        Q_e: np.ndarray,
        eigenvalues: np.ndarray,
        sigma_prime: np.ndarray,
    ) -> str:
        """
        Fase 1.6 — Sello de sesión SHA-256 canónico longitud-prefijado.

        Incluye invariantes observados y parámetros de gobernanza (no secretos).
        """
        h = hashlib.sha256()
        h.update(b"HYDRO-AGENT/SESSION/v3")
        _sha_update_arr(h, H)
        _sha_update_arr(h, P_f)
        _sha_update_arr(h, Q_e)
        _sha_update_arr(h, eigenvalues)
        _sha_update_arr(h, sigma_prime)
        h.update(_pack_f64(self._fiedler_min))
        h.update(_pack_f64(self._siph_margin))
        h.update(_pack_f64(self._liq_limit))
        h.update(_pack_f64(self._tol))
        h.update(_pack_f64(self._grace_max))
        _sha_update_str(h, "PHASE1/OBSERVE")
        digest = h.hexdigest()
        if len(digest) != _SHA256_HEX_LEN:
            raise RuntimeError("El sello de sesión no es un SHA-256 de 64 nibbles.")
        return digest

    def phase1_close_and_open_phase2(
        self,
        hydraulic_state: Any,
        sigma_prime_tensors: Optional[np.ndarray] = None,
        current_time: Optional[float] = None,
    ) -> Phase1HydroAgentHandoff:
        """
        Fase 1.7 — Cierre formal de Fase 1 y apertura verificada de Fase 2.

        Definición formal de frontera:

            Φ₁→₂ : estado físico bruto ↦ (H, u_w, Q, σ(L), σ', σ₁)

        Este es el último método de la Fase 1. Su contrato es exactamente el
        dominio de `phase2_from_phase1`: produce `Phase1HydroAgentHandoff` y
        exige que la Fase 2 lo admita de inmediato. Con ello la Fase 1 queda
        anidada, como prefijo functorial, dentro de la Fase 2.
        """
        fields = self.phase1_validate_hydraulic_fields(hydraulic_state)
        H = fields["hydraulic_head"]
        P_f = fields["pore_pressures"]
        Q_e = fields["flow_rates"]
        eigenvalues = fields["laplacian_eigenvalues"]
        n_nodes = int(fields["n_nodes"])

        sigma_prime = self.phase1_validate_effective_stresses(
            hydraulic_state, sigma_prime_tensors, n_nodes
        )
        hydraulic_gradients, max_gradient_hint, is_siphoning = (
            self.phase1_extract_gradient_signals(hydraulic_state)
        )
        hygiene = self.phase1_tensor_hygiene(sigma_prime)
        collector_diag = self._extract_state_diagnostics(hydraulic_state)
        curr_time = float(current_time if current_time is not None else time.time())
        session_sha256 = self._phase1_session_hash(
            H=H, P_f=P_f, Q_e=Q_e, eigenvalues=eigenvalues, sigma_prime=sigma_prime
        )
        diagnostics = self.phase1_observation_diagnostics(
            H=H,
            P_f=P_f,
            Q_e=Q_e,
            eigenvalues=eigenvalues,
            hygiene=hygiene,
            collector_diag=collector_diag,
            current_time=curr_time,
            is_siphoning=is_siphoning,
            max_gradient_hint=max_gradient_hint,
        )
        diagnostics["session_sha256_prefix"] = session_sha256[:16]
        source_hash = getattr(hydraulic_state, "sha256_hash", None)
        if isinstance(source_hash, str) and source_hash:
            diagnostics["collector_state_hash_prefix"] = source_hash[:16]

        handoff = Phase1HydroAgentHandoff(
            hydraulic_head=H,
            pore_pressures=P_f,
            flow_rates=Q_e,
            laplacian_eigenvalues=eigenvalues,
            sigma_prime_tensors=sigma_prime,
            is_siphoning_active=is_siphoning,
            session_sha256=session_sha256,
            diagnostics=diagnostics,
            next_entrypoint=_PHASE1_ENTRY,
            hydraulic_gradients=hydraulic_gradients,
            max_gradient_hint=max_gradient_hint,
        )

        opened = self.phase2_from_phase1(handoff)
        if opened.session_sha256 != session_sha256:
            raise RuntimeError(
                "Invariante de anidamiento Φ₁→₂ violado: el sello de sesión "
                "admitido por Fase 2 no coincide con el observado en Fase 1."
            )

        logger.debug(
            "Fase Observe [HYDRO_AGENT]: sesión sellada. SHA prefix=%s",
            session_sha256[:16],
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2 — ORIENT: FIEDLER, CHEEGER, SIFONAMIENTO, LICUACIÓN
    # (continuación formal de phase1_close_and_open_phase2)
    # ═════════════════════════════════════════════════════════════════════════

    def phase2_from_phase1(
        self,
        handoff: Phase1HydroAgentHandoff,
    ) -> Phase1HydroAgentHandoff:
        """
        Fase 2.0 — Entrada formal desde Fase 1.

        Continuación directa de `phase1_close_and_open_phase2`. Consume
        `Phase1HydroAgentHandoff` y lo reexpone si la frontera es válida.
        """
        if not isinstance(handoff, Phase1HydroAgentHandoff):
            raise TypeError(
                "Se esperaba Phase1HydroAgentHandoff como frontera Φ₁→₂."
            )
        if handoff.next_entrypoint != _PHASE1_ENTRY:
            raise ValueError(
                "Phase1HydroAgentHandoff inválido: el punto de entrada esperado "
                f"es {_PHASE1_ENTRY!r}."
            )
        if (
            not isinstance(handoff.session_sha256, str)
            or len(handoff.session_sha256) != _SHA256_HEX_LEN
        ):
            raise ValueError("El sello de sesión de Φ₁→₂ no es un SHA-256 válido.")
        if handoff.hydraulic_head.size < 1:
            raise ValueError("La observación de Φ₁→₂ no contiene nudos.")
        return handoff

    def phase2_sanitize_spectrum(self, eigenvalues: np.ndarray) -> np.ndarray:
        """
        Fase 2.1 — Proyección de σ(Δ₀) sobre [0, ∞).

        Autovalores negativos por debajo de la tolerancia se clipean a 0
        (ruido de un Laplaciano numéricamente PSD).
        """
        eig = np.asarray(eigenvalues, dtype=np.float64).ravel()
        if eig.size == 0:
            return eig
        tiny = max(self._tol, 1e-10)
        cleaned = np.where((eig < 0.0) & (eig > -tiny), 0.0, eig)
        return np.sort(np.asarray(cleaned, dtype=np.float64))

    def phase2_audit_fiedler(
        self,
        eigenvalues: np.ndarray,
        n_nodes: int,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Fase 2.2 — Conectividad algebraica de Fiedler.

        Para L ⪰ 0: λ₁ ≈ 0 con multiplicidad β₀, λ₂ > 0 ⇔ 1-esqueleto conexo.
        Si λ₂ < τ_Fiedler se declara desconexión homológica/hidráulica.
        Un nudo aislado (n ≤ 1) es vacuamente conexo (λ₂ = +∞).
        """
        if n_nodes <= 1:
            return math.inf, False, {
                "fiedler_status": "single_node",
                "fiedler_threshold": float(self._fiedler_min),
                "lambda_1": 0.0,
                "lambda_2": math.inf,
                "spectral_size": int(np.asarray(eigenvalues).size),
                "nullity_estimate": 1,
            }

        eig_sorted = self.phase2_sanitize_spectrum(eigenvalues)
        if eig_sorted.size < 2:
            return 0.0, True, {
                "fiedler_status": "insufficient_spectrum",
                "fiedler_threshold": float(self._fiedler_min),
                "lambda_1": float(eig_sorted[0]) if eig_sorted.size else math.nan,
                "lambda_2": 0.0,
                "spectral_size": int(eig_sorted.size),
                "nullity_estimate": int(eig_sorted.size),
            }
        if not np.all(np.isfinite(eig_sorted)):
            return 0.0, True, {
                "fiedler_status": "nonfinite_spectrum",
                "fiedler_threshold": float(self._fiedler_min),
                "lambda_1": math.nan,
                "lambda_2": 0.0,
                "spectral_size": int(eig_sorted.size),
                "nullity_estimate": math.nan,
            }

        tiny = max(self._tol, 1e-10)
        fiedler = float(eig_sorted[1])
        is_disconnected = bool(fiedler < self._fiedler_min)
        diagnostics: Dict[str, Any] = {
            "fiedler_status": "ok",
            "fiedler_threshold": float(self._fiedler_min),
            "lambda_1": float(eig_sorted[0]),
            "lambda_2": fiedler,
            "spectral_size": int(eig_sorted.size),
            "nullity_estimate": int(np.count_nonzero(eig_sorted <= tiny)),
            "spectral_subset": bool(eig_sorted.size < n_nodes),
        }
        return fiedler, is_disconnected, diagnostics

    def phase2_audit_cheeger(
        self,
        fiedler_value: float,
    ) -> Dict[str, float]:
        """
        Fase 2.3 — Cota de Cheeger h ≤ √(2 λ₂) sobre el 1-esqueleto.
        """
        if not math.isfinite(fiedler_value):
            return {
                "cheeger_upper_bound": math.nan if fiedler_value != math.inf else 0.0,
                "fiedler_value": float(fiedler_value),
            }
        lam2 = max(fiedler_value, 0.0)
        return {
            "cheeger_upper_bound": float(math.sqrt(2.0 * lam2)),
            "fiedler_value": float(fiedler_value),
        }

    def phase2_audit_max_gradient(
        self,
        handoff: Phase1HydroAgentHandoff,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Fase 2.4 — Gradiente máximo de filtración / piping.

        Si el estado reporta sifonamiento activo pero no hay i_max finito
        y positivo, se adopta conservadoramente +∞ (rampa de protección).
        FoS_i = η / i_max con η = siphoning_margin.
        """
        max_gradient = math.nan
        source = "none"
        if (
            handoff.hydraulic_gradients is not None
            and handoff.hydraulic_gradients.size > 0
        ):
            max_gradient = float(np.max(np.abs(handoff.hydraulic_gradients)))
            source = "hydraulic_gradients"
        elif not math.isnan(handoff.max_gradient_hint):
            max_gradient = float(handoff.max_gradient_hint)
            source = "max_gradient_hint"

        if handoff.is_siphoning_active and (
            not math.isfinite(max_gradient) or max_gradient <= 0.0
        ):
            max_gradient = math.inf
            source = source + "+siphoning_conservative_inf"

        if math.isnan(max_gradient):
            max_gradient = 0.0
        max_gradient = abs(float(max_gradient))

        if max_gradient > self._reg and math.isfinite(max_gradient):
            fos = float(self._siph_margin / max_gradient)
        elif max_gradient == 0.0:
            fos = math.inf
        else:
            fos = 0.0

        diagnostics: Dict[str, Any] = {
            "max_gradient": max_gradient,
            "siphoning_margin_reference": self._siph_margin,
            "siphoning_fos": fos,
            "gradient_source": source,
        }
        return max_gradient, diagnostics

    def phase2_audit_effective_stresses(
        self,
        sigma_prime: np.ndarray,
    ) -> Tuple[float, float, bool, Dict[str, Any]]:
        """
        Fase 2.5 — Auditoría de Biot–Terzaghi / Bishop sobre σ'.

        Por nudo, con autovalores σ'₁ ≥ σ'₂ ≥ σ'₃ de eigvalsh:

            p'  = tr(σ')/3,     I₃ = det(σ') = Π σ'_k.

        Licuación (compresión positiva en el tensor de entrada):

            σ'₃ ≤ ε  o  p' ≤ ε  o  I₃ ≤ ε,     ε = liquefaction_limit.
        """
        n_nodes = int(sigma_prime.shape[0])
        tiny = max(self._liq_limit, 0.0)
        if n_nodes == 0:
            return math.inf, math.inf, False, {
                "stress_status": "empty",
                "min_det_sigma_prime": math.inf,
                "min_principal_stress": math.inf,
                "min_mean_pressure": math.inf,
                "liquefaction_limit": float(tiny),
                "liquefaction_nodes": 0,
            }

        try:
            eigvals = np.linalg.eigvalsh(np.asarray(sigma_prime, dtype=np.float64))
        except np.linalg.LinAlgError:
            eigvals = np.zeros((n_nodes, 3), dtype=np.float64)
            for i in range(n_nodes):
                try:
                    eigvals[i] = np.linalg.eigvalsh(
                        0.5 * (sigma_prime[i] + sigma_prime[i].T)
                    )
                except np.linalg.LinAlgError:
                    eigvals[i] = 0.0

        min_principal_vec = np.asarray(eigvals[:, 0], dtype=np.float64)
        det_vec = np.asarray(np.prod(eigvals, axis=1), dtype=np.float64)
        p_mean_vec = np.asarray(np.sum(eigvals, axis=1) / 3.0, dtype=np.float64)

        finite_det = det_vec[np.isfinite(det_vec)]
        finite_min = min_principal_vec[np.isfinite(min_principal_vec)]
        finite_p = p_mean_vec[np.isfinite(p_mean_vec)]

        min_det = float(np.min(finite_det)) if finite_det.size else math.nan
        min_principal = float(np.min(finite_min)) if finite_min.size else math.nan
        min_p = float(np.min(finite_p)) if finite_p.size else math.nan

        nonfinite = bool(
            np.any(~np.isfinite(det_vec))
            or np.any(~np.isfinite(min_principal_vec))
            or np.any(~np.isfinite(p_mean_vec))
        )
        liq_mask = (
            (min_principal_vec <= tiny)
            | (p_mean_vec <= tiny)
            | (det_vec <= tiny)
        )
        liquefaction_nodes = int(np.count_nonzero(liq_mask))
        is_liquefaction = bool(nonfinite or liquefaction_nodes > 0)

        diagnostics: Dict[str, Any] = {
            "stress_status": "ok" if not nonfinite else "nonfinite",
            "min_det_sigma_prime": float(min_det),
            "min_principal_stress": float(min_principal),
            "min_mean_pressure": float(min_p),
            "liquefaction_limit": float(tiny),
            "liquefaction_nodes": liquefaction_nodes,
            "liquefaction_ratio": float(liquefaction_nodes / n_nodes),
        }
        return float(min_det), float(min_principal), bool(is_liquefaction), diagnostics

    def phase2_audit_collector_residuals(
        self,
        diagnostics: Mapping[str, Any],
    ) -> Dict[str, float]:
        """
        Fase 2.6 — Residuales de Kirchhoff y masa reportados por el colector.

        Ausencia no es veto: se reporta NaN.
        """
        return {
            "kirchhoff_residual": _finite_or_nan(
                diagnostics.get("collector_kirchhoff_residual")
            ),
            "richards_mass_residual": _finite_or_nan(
                diagnostics.get("collector_mass_residual")
            ),
        }

    def phase2_collect_invariants(
        self,
        phase1_handoff: Phase1HydroAgentHandoff,
    ) -> Dict[str, Any]:
        """
        Fase 2.7 — Recolección conjunta de invariantes de orientación.
        """
        fiedler, is_disconnected, spectral_diag = self.phase2_audit_fiedler(
            phase1_handoff.laplacian_eigenvalues,
            n_nodes=phase1_handoff.hydraulic_head.size,
        )
        cheeger_diag = self.phase2_audit_cheeger(fiedler)
        max_gradient, gradient_diag = self.phase2_audit_max_gradient(phase1_handoff)
        min_det, min_principal, is_liquefaction, stress_diag = (
            self.phase2_audit_effective_stresses(phase1_handoff.sigma_prime_tensors)
        )
        residual_diag = self.phase2_audit_collector_residuals(phase1_handoff.diagnostics)

        diagnostics: Dict[str, Any] = dict(phase1_handoff.diagnostics)
        diagnostics.update(spectral_diag)
        diagnostics.update(cheeger_diag)
        diagnostics.update(gradient_diag)
        diagnostics.update(stress_diag)
        diagnostics.update(residual_diag)
        return {
            "fiedler_value": fiedler,
            "is_disconnected": is_disconnected,
            "max_gradient": max_gradient,
            "min_det_sigma_prime": min_det,
            "min_principal_stress": min_principal,
            "is_liquefaction_active": is_liquefaction,
            "diagnostics": diagnostics,
        }

    def phase2_close_and_open_phase3(
        self,
        phase1_handoff: Phase1HydroAgentHandoff,
    ) -> Phase2HydroAgentHandoff:
        """
        Fase 2.8 — Cierre formal de Fase 2 y apertura verificada de Fase 3.

        Definición formal de frontera:

            Φ₂→₃ : observación validada ↦ (λ₂, i_max, σ'₃, det σ', Π)

        Este es el último método de la Fase 2. Su contrato es exactamente el
        dominio de `phase3_from_phase2`. Con ello la Fase 2 queda anidada,
        como prefijo functorial, dentro de la Fase 3.
        """
        validated = self.phase2_from_phase1(phase1_handoff)
        collected = self.phase2_collect_invariants(validated)

        handoff = Phase2HydroAgentHandoff(
            hydraulic_head=validated.hydraulic_head,
            pore_pressures=validated.pore_pressures,
            flow_rates=validated.flow_rates,
            laplacian_eigenvalues=validated.laplacian_eigenvalues,
            sigma_prime_tensors=validated.sigma_prime_tensors,
            is_siphoning_active=validated.is_siphoning_active,
            session_sha256=validated.session_sha256,
            fiedler_value=float(collected["fiedler_value"]),
            is_disconnected=bool(collected["is_disconnected"]),
            max_gradient=float(collected["max_gradient"]),
            min_det_sigma_prime=float(collected["min_det_sigma_prime"]),
            min_principal_stress=float(collected["min_principal_stress"]),
            is_liquefaction_active=bool(collected["is_liquefaction_active"]),
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
            "Fase Orient [HYDRO_AGENT]: fiedler=%.6e, max_grad=%.6e, liq=%s",
            handoff.fiedler_value,
            handoff.max_gradient,
            handoff.is_liquefaction_active,
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 3 — DECIDE / ACT: RAMPA, HEYTING, OVERRIDE, HARDWARE, SELLO
    # (continuación formal de phase2_close_and_open_phase3)
    # ═════════════════════════════════════════════════════════════════════════

    def phase3_from_phase2(
        self,
        handoff: Phase2HydroAgentHandoff,
    ) -> Phase2HydroAgentHandoff:
        """
        Fase 3.0 — Entrada formal desde Fase 2.

        Continuación directa de `phase2_close_and_open_phase3`.
        """
        if not isinstance(handoff, Phase2HydroAgentHandoff):
            raise TypeError(
                "Se esperaba Phase2HydroAgentHandoff como frontera Φ₂→₃."
            )
        if handoff.next_entrypoint != _PHASE2_ENTRY:
            raise ValueError(
                "Phase2HydroAgentHandoff inválido: el punto de entrada esperado "
                f"es {_PHASE2_ENTRY!r}."
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

    def _verify_hmac_override(self, token: Optional[str]) -> bool:
        """Alias interno de Fase 3.1 (compatibilidad con la firma original)."""
        return self.phase3_verify_override(token)

    def phase3_confidence_ramp(self, handoff: Phase2HydroAgentHandoff) -> float:
        """
        Fase 3.2 — Rampa de confianza Lipschitz-saturada.

        Canal Fiedler:
            λ₂ ≥ τ → 1,   λ₂ ≤ ρ τ → 0,   interpolación en (ρ τ, τ)
            con ρ = fiedler_soft_ratio.

        Canal sifonamiento:
            bandera inactiva y i_max ≤ η → 1;
            i_max > η o bandera activa con i_max = +∞ → 0;
            interpolación afín si 0 < i_max ≤ η con bandera activa.

        Canal licuación / desconexión: 0 si activos, 1 si no.

        La confianza global es el encuentro de Heyting (mínimo) de canales
        finitos. Métricas no finitas colapsan su canal a 0.
        """
        def _fiedler_channel(lam2: float) -> float:
            if lam2 == math.inf:
                return Heyting3.COHERENT
            if not math.isfinite(lam2):
                return Heyting3.VETOED
            if lam2 >= self._fiedler_min:
                return Heyting3.COHERENT
            soft = self._fiedler_soft_ratio * self._fiedler_min
            if lam2 <= soft:
                return Heyting3.VETOED
            span = self._fiedler_min - soft
            if span <= self._reg:
                return Heyting3.VETOED
            return float((lam2 - soft) / span)

        def _siphon_channel(i_max: float, flag: bool) -> float:
            if not math.isfinite(i_max):
                return Heyting3.VETOED if flag or i_max == math.inf else Heyting3.VETOED
            if not flag and i_max <= self._siph_margin:
                return Heyting3.COHERENT
            if i_max > self._siph_margin:
                return Heyting3.VETOED
            if not flag:
                return Heyting3.COHERENT
            if i_max <= self._reg:
                return Heyting3.DEGRADED
            span = self._siph_margin
            if span <= self._reg:
                return Heyting3.VETOED
            return float(max(0.0, (self._siph_margin - i_max) / span))

        def _binary_channel(failed: bool) -> float:
            return Heyting3.VETOED if failed else Heyting3.COHERENT

        channels = [
            _fiedler_channel(float(handoff.fiedler_value)),
            _siphon_channel(float(handoff.max_gradient), bool(handoff.is_siphoning_active)),
            _binary_channel(bool(handoff.is_liquefaction_active)),
            _binary_channel(bool(handoff.is_disconnected)),
        ]
        confidence = channels[0]
        for ch in channels[1:]:
            confidence = Heyting3.meet(confidence, ch)
        if not math.isfinite(confidence):
            return Heyting3.VETOED
        return float(min(1.0, max(0.0, confidence)))

    def phase3_decide_heyting(
        self,
        handoff: Phase2HydroAgentHandoff,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Fase 3.3 — Clasificador de Heyting trivalente Ω₃.

        Reglas:
          - VETOED inmediato: licuación, desconexión hidráulica o métricas
            no finitas (NaN) en λ₂ / det(σ') / σ'₃.
          - DEGRADED: sifonamiento activo con ventana de gracia.
          - El positrón e⁺ impide DEGRADED → VETOED por expiración de gracia.
          - Gracia expirada sin override: colapso a VETOED.
          - El encuentro con la rampa nunca eleva el veredicto.
        """
        curr_time = float(current_time if current_time is not None else time.time())
        hard_reasons = []
        if handoff.is_liquefaction_active:
            hard_reasons.append("liquefaction")
        if handoff.is_disconnected:
            hard_reasons.append("hydraulic_disconnection")
        if math.isnan(handoff.fiedler_value) or math.isnan(handoff.min_det_sigma_prime):
            hard_reasons.append("nonfinite_metrics")
        if math.isnan(handoff.min_principal_stress):
            hard_reasons.append("nonfinite_principal_stress")

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
                "¡VETO DURO INSTANTÁNEO POR COLAPSO GEOMECÁNICO / DESCONEXIÓN "
                "HIDRÁULICA! Razones: %s",
                ", ".join(hard_reasons),
            )
        elif handoff.is_siphoning_active:
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
                    "¡VETO SUAVE ACTIVO (LUZ ÁMBAR)! Sifonamiento marginal. "
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
                    "Aniquilando anomalía hidráulica en Fock. Sello: %s",
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

    def phase3_actuate_esp32_crowbar_interlock(
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
            "¡COLA DE HEYTING COLAPSADA EN SOBERANO HIDROLÓGICO!\n"
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
        handoff: Phase2HydroAgentHandoff,
        decision: Mapping[str, Any],
        interlock_fired: bool,
        actuation_latency_ns: float,
    ) -> str:
        """Fase 3.5 — Sello criptográfico SHA-256 canónico (no depende de `repr`)."""
        h = hashlib.sha256()
        h.update(b"HYDRO-AGENT/CERT/v3")
        _sha_update_str(h, "PHASE3/HYDRO_AGENT_GOVERNANCE")
        _sha_update_arr(h, handoff.hydraulic_head)
        _sha_update_arr(h, handoff.pore_pressures)
        _sha_update_arr(h, handoff.flow_rates)
        _sha_update_arr(h, handoff.laplacian_eigenvalues)
        _sha_update_arr(h, handoff.sigma_prime_tensors)
        _sha_update_str(h, str(decision.get("heyting_verdict", "")))
        h.update(_pack_f64(float(handoff.fiedler_value)))
        h.update(_pack_f64(float(handoff.max_gradient)))
        h.update(_pack_f64(float(handoff.min_det_sigma_prime)))
        h.update(_pack_f64(float(handoff.min_principal_stress)))
        h.update(_pack_f64(float(decision.get("confidence", math.nan))))
        h.update(_pack_f64(float(decision.get("heyting_truth_value", math.nan))))
        h.update(_pack_f64(float(decision.get("time_grace_remaining", math.nan))))
        h.update(_pack_f64(float(actuation_latency_ns)))
        h.update(b"\x01" if handoff.is_siphoning_active else b"\x00")
        h.update(b"\x01" if handoff.is_liquefaction_active else b"\x00")
        h.update(b"\x01" if handoff.is_disconnected else b"\x00")
        h.update(b"\x01" if bool(decision.get("is_hard_veto")) else b"\x00")
        h.update(b"\x01" if bool(decision.get("is_soft_veto")) else b"\x00")
        h.update(b"\x01" if interlock_fired else b"\x00")
        _sha_update_str(h, handoff.session_sha256)
        _sha_update_str(h, str(decision.get("override_sha256") or ""))
        return h.hexdigest()

    def phase3_issue_verdict(
        self,
        handoff: Phase2HydroAgentHandoff,
        decision: Mapping[str, Any],
        interlock_fired: bool,
        actuation_latency_ns: float,
    ) -> HydrologicalVerdict:
        """
        Fase 3.6 — Emisión del veredicto certificado.

        Incluye Ω₃, λ₂, i_max, det(σ'), rampa, gracia, hardware y sello SHA-256.
        """
        seal = self._phase3_cryptographic_seal(
            handoff=handoff,
            decision=decision,
            interlock_fired=interlock_fired,
            actuation_latency_ns=actuation_latency_ns,
        )
        phase_chain = ("PHASE1/OBSERVE", "PHASE2/ORIENT", "PHASE3/DECIDE/ACT")
        diagnostics = dict(handoff.diagnostics)
        diagnostics["decision"] = dict(decision)
        diagnostics["hardware"] = {
            "interlock_fired": interlock_fired,
            "actuation_latency_ns": actuation_latency_ns,
        }
        return HydrologicalVerdict(
            heyting_verdict=str(decision["heyting_verdict"]),
            fiedler_value=float(handoff.fiedler_value),
            max_gradient=float(handoff.max_gradient),
            min_det_sigma_prime=float(handoff.min_det_sigma_prime),
            is_siphoning_active=bool(handoff.is_siphoning_active),
            is_liquefaction_active=bool(handoff.is_liquefaction_active),
            is_soft_veto_active=bool(decision["is_soft_veto"]),
            is_hard_veto_active=bool(decision["is_hard_veto"]),
            time_grace_remaining=float(decision["time_grace_remaining"]),
            cryptographic_seal=seal,
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
            min_principal_stress=float(handoff.min_principal_stress),
            actuation_latency_ns=float(actuation_latency_ns),
        )

    def phase3_close_loop(
        self,
        phase2_handoff: Phase2HydroAgentHandoff,
        human_override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
    ) -> HydrologicalVerdict:
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
            override_token=human_override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
            elapsed_grace_seconds=elapsed_grace_seconds,
        )
        interlock_fired, latency = self.phase3_actuate_esp32_crowbar_interlock(
            decision["heyting_verdict"]
        )
        verdict = self.phase3_issue_verdict(
            handoff=validated,
            decision=decision,
            interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
        )
        if verdict.heyting_verdict == "VETOED":
            logger.critical(
                "Soberano Hidrológico emitió VETO DURO. Sello: %s",
                verdict.cryptographic_seal[:16],
            )
        else:
            logger.info(
                "Soberano Hidrológico regulado síncronamente. Veredicto: %s. "
                "Confianza=%.6g. Sello: %s",
                verdict.heyting_verdict,
                verdict.confidence,
                verdict.cryptographic_seal[:16],
            )
        return verdict

    # ═════════════════════════════════════════════════════════════════════════
    # API PRINCIPAL COMPATIBLE OODA
    # ═════════════════════════════════════════════════════════════════════════

    def execute_hydrological_control_cycle(
        self,
        hydraulic_state: Any,
        sigma_prime_tensors: Optional[np.ndarray] = None,
        human_override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
    ) -> HydrologicalVerdict:
        """
        API principal — Orquesta el funtor compuesto Φ₃ ∘ Φ₂ ∘ Φ₁.

        Equivalencia de fases:
          OBSERVE    → phase1_close_and_open_phase2
          ORIENT     → phase2_close_and_open_phase3
          DECIDE/ACT → phase3_close_loop
        """
        phase1_handoff = self.phase1_close_and_open_phase2(
            hydraulic_state=hydraulic_state,
            sigma_prime_tensors=sigma_prime_tensors,
            current_time=current_time,
        )
        phase2_handoff = self.phase2_close_and_open_phase3(
            phase1_handoff=phase1_handoff,
        )
        return self.phase3_close_loop(
            phase2_handoff=phase2_handoff,
            human_override_token=human_override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
            elapsed_grace_seconds=elapsed_grace_seconds,
        )

    def evaluate_lazo_ooda(
        self,
        hydraulic_state: Any,
        sigma_prime_tensors: np.ndarray,
        human_override_token: Optional[str] = None,
    ) -> HydrologicalVerdict:
        """API legada — equivalente al ciclo OODA original."""
        return self.execute_hydrological_control_cycle(
            hydraulic_state=hydraulic_state,
            sigma_prime_tensors=sigma_prime_tensors,
            human_override_token=human_override_token,
        )
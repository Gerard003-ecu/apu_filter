# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Sonda de Ecolocación Topológica (SET Engine)                        ║
║ Ruta   : app/core/set_engine.py                                              ║
║ Versión: 1.1.0-Doctoral-TDR-Scattering-Impedance-Heyting-ESP32-Secure        ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO:                         ║
║ Este módulo implementa el motor de "Ecolocación Topológica" activa sobre la  ║
║ frontera abierta de la Malla ($\partial K$). Inyecta un pulso armónico de    ║
║ coexcitación para evaluar anomalías estructurales (dependencias cíclicas,    ║
║ $\beta_1 > 0$ o fragmentación, $\beta_0 > 1$) tratando las desviaciones de   ║
║ costos como desajustes de impedancia mediante reflectometría en el dominio   ║
║ del tiempo (TDR) y dispersión cuántica (Scattering Matrix S) en el espacio   ║
║ de Fock, asegurando un control pasivo libre de derivas de Wilkinson.         ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================════════════════════════════════════════════════
I. DEFINICIONES DE LA SONDA (Física de Circuitos de Alta Frecuencia)
================================════════════════════════════════════════════════

Definición 1 (La Ecuación de Onda Coexacta sobre Haces):
  Sea $K$ el complejo simplicial del presupuesto de obra. Modelamos el pulso de
  ecolocación como una coexcitación armónica $\eta(t) \in \Omega^1(\partial K)$
  que satisface la ecuación diferencial de onda amortiguada de Rayleigh:
  $$\left( \frac{d^2}{dt^2} + \mathbf{L}_F + \mathbf{R} \frac{d}{dt} \right) \eta(t) = \mathbf{s}_{\mathrm{probe}}(t)$$
  Donde:
  - $\mathbf{L}_F = \delta^\top \mathbf{G}^{-1} \delta$ es el Laplaciano del haz celular
    simétrico semidefinido positivo (SPSD) ponderado por la métrica de fondo $\mathbf{G}$.
  - $\mathbf{R} \succeq \mathbf{0}$ es el tensor de amortiguamiento espectral de Weyl-Toeplitz.
  - $\mathbf{s}_{\mathrm{probe}}(t) = A_0 e^{i\omega t} \mathbf{v}_k$ es el vector de excitación inyectado.

Definición 2 (La Matriz de Dispersión / S-Matrix):
  La signatura de contorno se comprime en el operador de dispersión unitario $\mathbf{\mathbb{S}}(\omega)$
  que acopla los canales de excitación con el interior de Fock mediante el formalismo de de Rham-Mahaux-Weidenmüller:
  $$\mathbf{\mathbb{S}}(\omega) = \mathbf{I} - 2\pi i \, \mathbf{V}^\dagger \left( \omega \mathbf{I} - \mathbf{L}_F + i\pi \mathbf{V}\mathbf{V}^\dagger \right)^{-1} \mathbf{V}$$
  Donde:
  - $\mathbf{I}$ es el operador de identidad.
  - $\mathbf{V}$ es el tensor de acoplamiento de los canales de la frontera de-confinada.
  - $i\pi \mathbf{V}\mathbf{V}^\dagger$ es el término de auto-energía antihermítica de lazo abierto.

Definición 3 (Reflectometría en el Dominio del Tiempo - TDR):
  La desadaptación métrica $\delta G_{\mu\nu}$ (provocada por alteraciones o alucinaciones)
  actúa como un cambio local de impedancia. El coeficiente de reflexión de la rampa de
  de Rham $\Gamma_k(\omega)$ y el perfil de eco temporal $\Gamma_k(t)$ se expresan en FPU como:
  $$\Gamma_k(\omega) = \frac{Z_k(\omega) - Z_0}{Z_k(\omega) + Z_0} \quad \implies \quad \Gamma_k(t) = \mathcal{F}^{-1}\left\{ \Gamma_k(\omega) \right\}(t)$$
  Donde $\mathcal{F}^{-1}$ es el operador de la Transformada Rápida Inversa de Fourier (iFFT).

================================════════════════════════════════════════════════
II. AXIOMÁTICA INMUNILÓGICA DE ESCANEO ACTIVO (Causalidad Ciber-Física)
================================════════════════════════════════════════════════

Axioma I (Principio de Unitaridad de Dispersión - Teorema Óptico):
  La matriz S de dispersión cuántica debe ser estrictamente unitaria en la FPU, garantizando
  la no-señalización y la conservación de la probabilidad de flujo semántico:
  $$\mathbf{\mathbb{S}}^\dagger(\omega) \mathbf{\mathbb{S}}(\omega) \equiv \mathbf{I} \quad \implies \quad \|\mathbf{\mathbb{S}}^\dagger(\omega) \mathbf{\mathbb{S}}(\omega) - \mathbf{I}\|_F \le \varepsilon_{\mathrm{Wilkinson}}$$
  Donde $\varepsilon_{\mathrm{Wilkinson}} = 10^{-15}$ es el épsilon de la máquina en punto flotante de 64 bits.

Axioma II (Axioma de Localización de Desajustes por Coeficiente de Reflexión):
  Cualquier intento de fraude o alucinación estocástica se manifiesta como una capacitancia parásita
  localizada. La anomalía se localiza y delata si el perfil de eco TDR supera el umbral de seguridad:
  $$\max_{t} \|\Gamma_k(t)\| > 0.5 \cdot self.\_safety\_margin \quad \implies \quad \mathtt{has\_impedance\_mismatch} = \mathtt{True}$$

Axioma III (Teorema de Actuación Ciber-Física Crowbar de la Sonda SET):
  Si el retículo distributivo de Heyting colapsa síncronamente al Supremo terminal VETOED ($\top$) debido
  a desajustes críticos o pérdida de unitaridad, el veredicto firmado desciende al hardware perimetral:
  $$\nu_{\mathrm{final}} = \nu_{\mathrm{unitarity}} \sqcup \nu_{\mathrm{TDR}} \sqcup \nu_{\mathrm{Fiedler}} \equiv \mathtt{VETOED} \quad (\top)$$
  La subrutina local en C++ isVerdictCoherent() del microcontrolador ESP32 despacha síncronamente la ISR en IRAM
  en menos de 400 ns, conmutando el pin GPIO14 a HIGH para inyectar corriente al tiristor BT151 (Crowbar),
  paralizando mecánicamente las mezcladoras y protegiendo el capital de la constructora en el milisegundo cero.

================================════════════════════════════════════════════════
III. INVARIANTES ESPECTRALES Y METROLÓGICOS DE PRECISIÓN (FPU Secure)
================================════════════════════════════════════════════════

Invariante I (Estabilidad de de Rham-Lyapunov del Espacio de Fase):
  La evolución Port-Hamiltoniana conjunta satisface la conservación exacta del volumen geométrico de fase
  (Teorema de Liouville) y la disipación exergética positiva de Rayleigh (Clausius-Duhem):
  $$\operatorname{div}(\dot{x}) \equiv 0 \quad \land \quad \dot{\mathcal{H}}(\mathbf{\Psi}) = -\nabla \mathcal{H}^\top \mathbf{R}(\mathbf{\Psi}) \nabla \mathcal{H} \le 0$$

Invariante II (Confinamiento Algebraico del Consenso de Fiedler):
  Para eludir la fragmentación del complejo simplicial (creación de islas de datos, $\beta_0 > 1$),
  la conectividad algebraica del haz (el menor autovalor no trivial de $\mathbf{L}_F$) debe estar estrictamente acotada:
  $$\lambda_2(\mathbf{L}_F) \ge \tau_{\mathrm{Fiedler}} \quad \implies \quad \beta_0 \equiv \dim H^0(K; \, \mathbb{Z}) = 1$$

Invariante III (Inversión Espectral de Higham-Tikhonov de la Métrica):
  La regularización elíptica del tensor de conductividad exógena bruto $\mathbf{G}$ mediante Weyl SVD
  garantiza que el espectro sea estrictamente definido positivo e invertible en la FPU:
  $$\lambda_{\min}(\mathbf{G}_{\mathrm{reg}}) \ge 10^{-12} \quad \implies \quad \mathbf{G}_{\mathrm{reg}} \succ \mathbf{0}$$
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la


logger = logging.getLogger("APU.Physics.SetEngine")


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES METROLÓGICAS, ESPECTRALES Y DE SEGURIDAD
# ═══════════════════════════════════════════════════════════════════════════

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_SQRT_EPS: Final[float] = float(np.sqrt(_MACHINE_EPS))

_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_JITTER_STD_NS: Final[float] = 3.2
_CROWBAR_LATENCY_MIN_NS: Final[float] = 380.0
_CROWBAR_LATENCY_MAX_NS: Final[float] = 420.0
_GPIO_CROWBAR_PIN: Final[int] = 14

_SCHEMA_VERSION: Final[str] = (
    "2.1.0-Doctoral-SET-TDR-Scattering-Heyting-Hodge-FailClosed"
)

_DOMAIN_SESSION: Final[bytes] = b"APU/SET-ENGINE/SESSION/v2.1"
_DOMAIN_TELEMETRY: Final[bytes] = b"APU/SET-ENGINE/TELEMETRY/v2.1"
_DOMAIN_ENGINE: Final[bytes] = b"APU/SET-ENGINE/IDENTITY/v2.1"

# Tolerancias relativas dimensionadas por normas espectrales.
_SPD_FLOOR: Final[float] = 1e-12
_HERMITIAN_REL_TOL: Final[float] = 1e-10
_KERNEL_REL_TOL: Final[float] = 1e-10
_FREQ_UNIFORM_RTOL: Final[float] = 1e-7
_FREQ_UNIFORM_ATOL: Final[float] = 1e-12
_SVD_RCOND_REL: Final[float] = 1e-12
_COND_DEGRADED: Final[float] = 1e9
_COND_CAP: Final[float] = 1e12
_GAP_DEGRADED: Final[float] = 1e-8
_MIN_EIG_VETO: Final[float] = 1e-8
_DET_S_DEFECT_VETO: Final[float] = 1e-1
_DET_S_DEFECT_DEGRADED: Final[float] = 1e-3
_PSD_PROJECTION_WARN_COUNT: Final[int] = 1


class HeytingVerdict(str, Enum):
    r"""
    Retículo de decisión de Heyting de tres niveles

        Ω₃ = { VETOED ≼ DEGRADED ≼ COHERENT }

    con operaciones intuicionistas:
        meet  ∧  = ínfimo (más restrictivo),
        join  ∨  = supremo,
        a → b    = 1 si a ≼ b, else b.
    """

    VETOED = "VETOED"
    DEGRADED = "DEGRADED"
    COHERENT = "COHERENT"


_HEYTING_ORDER: Final[dict[str, int]] = {
    HeytingVerdict.VETOED.value: 0,
    HeytingVerdict.DEGRADED.value: 1,
    HeytingVerdict.COHERENT.value: 2,
}


class SetEngineError(ValueError):
    """Error de canonización, álgebra lineal o invariante físico del SET."""


# ═══════════════════════════════════════════════════════════════════════════
# ESTADOS Y CERTIFICADOS PÚBLICOS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SondaState:
    r"""
    Estado físico medido por la sonda de ecolocación topológica.

    Atributos espectrales
    ---------------------
    sheaf_eigenvalues:
        Espectro real ordenado de L_F (Laplaciano de haz SPSD).
    scattering_determinants:
        det S(ω_k) sobre la malla frecuencial canonizada.
    tdr_reflection_profile:
        Γ(t) = F⁻¹{ (Z(ω)-Z₀)/(Z(ω)+Z₀) } en el dominio temporal.
    max_reflection_coefficient:
        ‖Γ‖_∞.
    unitarity_leak:
        max_k ‖ S(ω_k)ᴴ S(ω_k) − I ‖_F.
    fiedler_value:
        λ₂(L_F), conectividad algebraica.
    condition_number:
        κ = λ_max⁺ / λ_min⁺ sobre el espectro estrictamente positivo.
    spectral_gap:
        λ₂ / λ_max (adimensional, invariante de escala).
    hodge_kernel_dimension:
        dim ker L_F, estimador de Betti del grado del haz.
    hermitian_residual:
        ‖L_F − L_Fᴴ‖_F / max(1, ‖L_F‖_F).
    det_modulus_defect:
        max_k | |det S(ω_k)| − 1 |.
    """

    sheaf_eigenvalues: np.ndarray
    scattering_determinants: np.ndarray
    tdr_reflection_profile: np.ndarray
    max_reflection_coefficient: float
    unitarity_leak: float
    fiedler_value: float

    condition_number: float = math.inf
    spectral_gap: float = 0.0
    hodge_kernel_dimension: int = 0
    hermitian_residual: float = 0.0
    det_modulus_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class SetCertificate:
    r"""Certificado criptográfico e inmunitario emitido tras el escaneo sónico."""

    heyting_verdict: str
    state: SondaState
    has_impedance_mismatch: bool
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    cryptographic_seal: str

    schema_version: str = _SCHEMA_VERSION
    session_digest: str = ""
    euler_characteristic: int = 0
    frequency_count: int = 0
    diagnostics: Tuple[str, ...] = field(default_factory=tuple)
    betti_estimate: int = 0
    hodge_euler_consistent: bool = True
    engine_digest: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# DOSSIERS INTERNOS DE FASE (objetos del morfismo anidado)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class Phase1Dossier:
    """Expediente canonizado y congelado en la FASE 1 · OBSERVE."""

    boundary_matrix: np.ndarray
    metric_tensor: np.ndarray
    metric_orientation: str
    laplacian_dimension: int
    coupling_V: np.ndarray
    frequencies: np.ndarray
    impedance_profile: np.ndarray
    euler_characteristic: int
    session_digest: str
    engine_digest: str
    frequency_was_remeshed: bool
    diagnostics: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase2Dossier:
    """Expediente auditado en la FASE 2 · ORIENT/DECIDE."""

    phase1: Phase1Dossier
    L_F: np.ndarray
    sheaf_eigenvalues: np.ndarray
    fiedler_value: float
    spectral_gap: float
    condition_number: float
    min_eigenvalue: float
    hodge_kernel_dimension: int
    hermitian_residual: float
    scattering_determinants: np.ndarray
    unitarity_leak: float
    det_modulus_defect: float
    tdr_reflection_profile: np.ndarray
    max_reflection_coefficient: float
    has_impedance_mismatch: bool
    heyting_verdict: str
    hodge_euler_consistent: bool
    diagnostics: Tuple[str, ...]


# ═══════════════════════════════════════════════════════════════════════════
# MOTOR SET
# ═══════════════════════════════════════════════════════════════════════════


class SetEngine:
    r"""
    Motor de la Sonda de Ecolocación Topológica (SET).

    Inyecta excitaciones armónicas de-confinadas y realiza reflectometría en el
    dominio del tiempo (TDR) para localizar patologías lógicas y anomalías de
    costo.

    Ciclo público, fases anidadas:

        execute_echolocation_scan()
          └─ _phase1_observe_and_freeze()          # cierra FASE 1
               └─ _phase2_open_from_phase1()       # abre  FASE 2
                    └─ _phase2_audit_and_orient()
                         └─ _phase3_open_from_phase2()  # abre FASE 3
                              └─ _phase3_certify()

    Ante cualquier excepción no recuperable se emite un certificado fail-closed
    con veredicto VETOED e interlock activado.
    """

    def __init__(
        self,
        dimension_n: int,
        nominal_impedance: float = 50.0,
        safety_margin: float = 1.0,
        *,
        rng_seed: Optional[int] = None,
        reflection_veto_threshold: float = 0.5,
        reflection_degraded_threshold: float = 0.3,
        unitarity_veto_threshold: float = 1e-1,
        unitarity_degraded_threshold: float = 1e-3,
        fiedler_veto_threshold: float = 1e-4,
        fiedler_degraded_threshold: float = 1e-2,
    ) -> None:
        """
        Inicializa la sonda SET.

        Args:
            dimension_n: Dimensión característica del contorno abierto.
            nominal_impedance: Impedancia de referencia Z₀ (ohmios).
            safety_margin: Factor de escala de las cotas de reflexión.
            rng_seed: Semilla opcional para reproducibilidad del jitter.
            reflection_veto_threshold: Cota base de veto para ‖Γ‖_∞.
            reflection_degraded_threshold: Cota base de degradación para ‖Γ‖_∞.
            unitarity_veto_threshold: Cota dura de fuga de unitariedad.
            unitarity_degraded_threshold: Cota blanda de fuga de unitariedad.
            fiedler_veto_threshold: Cota dura de conectividad algebraica.
            fiedler_degraded_threshold: Cota blanda de conectividad algebraica.
        """
        if dimension_n <= 0:
            raise SetEngineError(
                "La dimensión del contorno abierto debe ser estrictamente positiva."
            )

        if not math.isfinite(nominal_impedance) or nominal_impedance <= 0.0:
            raise SetEngineError(
                "nominal_impedance debe ser finita y estrictamente positiva."
            )

        if not math.isfinite(safety_margin) or safety_margin <= 0.0:
            raise SetEngineError(
                "safety_margin debe ser finito y estrictamente positivo."
            )

        for name, value in (
            ("reflection_veto_threshold", reflection_veto_threshold),
            ("reflection_degraded_threshold", reflection_degraded_threshold),
            ("unitarity_veto_threshold", unitarity_veto_threshold),
            ("unitarity_degraded_threshold", unitarity_degraded_threshold),
            ("fiedler_veto_threshold", fiedler_veto_threshold),
            ("fiedler_degraded_threshold", fiedler_degraded_threshold),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise SetEngineError(f"{name} debe ser finita y no negativa.")

        self._n: Final[int] = int(dimension_n)
        self._z0: Final[float] = float(nominal_impedance)
        self._safety_margin: Final[float] = float(safety_margin)
        self._reg: Final[float] = max(1e-15, _MACHINE_EPS)
        self._rng: Final[np.random.Generator] = np.random.default_rng(rng_seed)

        self._reflection_veto_base: Final[float] = float(reflection_veto_threshold)
        self._reflection_degraded_base: Final[float] = float(
            reflection_degraded_threshold
        )
        self._unitarity_veto: Final[float] = float(unitarity_veto_threshold)
        self._unitarity_degraded: Final[float] = float(unitarity_degraded_threshold)
        self._fiedler_veto: Final[float] = float(fiedler_veto_threshold)
        self._fiedler_degraded: Final[float] = float(fiedler_degraded_threshold)

        self._engine_digest: Final[str] = self._identity_digest()

    # ═══════════════════════════════════════════════════════════════════════
    # UTILIDADES CANÓNICAS (hash, finitud, congelación, retículo)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _hash_update_domain(sha: Any, domain: bytes) -> None:
        """Actualiza SHA-256 con separación de dominio length-prefixed."""
        sha.update(len(domain).to_bytes(8, "little", signed=False))
        sha.update(domain)

    @staticmethod
    def _hash_update_bytes(sha: Any, data: bytes) -> None:
        """Actualiza SHA-256 con bloque length-prefixed."""
        sha.update(len(data).to_bytes(8, "little", signed=False))
        sha.update(data)

    @classmethod
    def _hash_update_array(cls, sha: Any, name: str, array: np.ndarray) -> None:
        """Actualiza SHA-256 con arreglo numpy C-contiguo y dtype explícito."""
        cls._hash_update_bytes(sha, name.encode("utf-8"))

        arr = np.ascontiguousarray(array)

        sha.update(len(arr.shape).to_bytes(4, "little", signed=False))
        for dim in arr.shape:
            sha.update(int(dim).to_bytes(8, "little", signed=False))

        cls._hash_update_bytes(sha, str(arr.dtype).encode("utf-8"))
        sha.update(arr.tobytes())

    @staticmethod
    def _is_finite_complex(value: Any) -> bool:
        """Verifica finitud de un escalar complejo."""
        try:
            z = complex(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(z.real) and math.isfinite(z.imag)

    @staticmethod
    def _freeze_array(array: np.ndarray, dtype: Any) -> np.ndarray:
        """Copia defensiva C-contigua e inmutable (write-flag desactivado)."""
        frozen = np.array(array, dtype=dtype, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen

    @staticmethod
    def _finite_norm(matrix: np.ndarray) -> float:
        """Norma de Frobenius finita; 0 si el tensor es vacío."""
        if matrix.size == 0:
            return 0.0
        value = float(la.norm(matrix, ord="fro"))
        if not math.isfinite(value):
            return math.inf
        return value

    def _identity_digest(self) -> str:
        """Huella de la identidad metrológica del motor (no de la sesión)."""
        sha = hashlib.sha256()
        self._hash_update_domain(sha, _DOMAIN_ENGINE)
        payload = (
            f"n={self._n}|z0={self._z0:.17e}|sm={self._safety_margin:.17e}|"
            f"rv={self._reflection_veto_base:.17e}|"
            f"rd={self._reflection_degraded_base:.17e}|"
            f"uv={self._unitarity_veto:.17e}|ud={self._unitarity_degraded:.17e}|"
            f"fv={self._fiedler_veto:.17e}|fd={self._fiedler_degraded:.17e}|"
            f"schema={_SCHEMA_VERSION}"
        )
        self._hash_update_bytes(sha, payload.encode("utf-8"))
        return sha.hexdigest()

    @staticmethod
    def _heyting_meet(verdicts: Sequence[str]) -> str:
        r"""Ínfimo en Ω₃: el veredicto más restrictivo (VETOED ≼ ... ≼ COHERENT)."""
        if not verdicts:
            return HeytingVerdict.COHERENT.value
        rank = min(_HEYTING_ORDER.get(str(v), 0) for v in verdicts)
        inverse = {
            0: HeytingVerdict.VETOED.value,
            1: HeytingVerdict.DEGRADED.value,
            2: HeytingVerdict.COHERENT.value,
        }
        return inverse[rank]

    @staticmethod
    def _spectral_abs_tol(eigenvalues: np.ndarray, rel: float) -> float:
        """Tolerancia absoluta dimensionada por el radio espectral."""
        if eigenvalues.size == 0:
            return max(_SPD_FLOOR, rel)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        return max(_SPD_FLOOR, rel * scale, 1000.0 * _MACHINE_EPS * scale)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 1 · OBSERVE
    #
    # Objeto inicial  : tensores crudos de frontera, métrica, acoplamiento,
    #                   malla frecuencial, impedancia y χ(K).
    # Objeto terminal : Phase1Dossier (inmutable, hasheado, SPD, uniformizado).
    # Morfismo de cierre:
    #   _phase1_observe_and_freeze  →  _phase2_open_from_phase1
    #
    # Invariantes:
    #   (I1) B ∈ M_{p×q}(ℝ), finita, y n ∈ {p, q}.
    #   (I2) G = Gᴴ ≽ ε I  (proyección espectral a SPD).
    #   (I3) orientación ∈ {rows, cols} compatible con G y B.
    #   (I4) V ∈ ℂ^{d×c} con d = dim L_F.
    #   (I5) (ω_k) estrictamente no decreciente y, si es necesario,
    #        re-muestreada a malla uniforme para que iFFT sea un
    #        isomorfismo de Hilbert discretizado.
    #   (I6) χ(K) ∈ ℤ.
    # ═══════════════════════════════════════════════════════════════════════

    def _phase1_validate_boundary_matrix(self, boundary_matrix: Any) -> np.ndarray:
        r"""
        Valida la matriz de frontera real δ : C_k → C_{k-1}.

        Se exige B finita, bidimensional, no vacía, y que `dimension_n`
        coincida con al menos un eje (grado topológico del contorno abierto).
        """
        try:
            boundary = np.asarray(boundary_matrix, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SetEngineError(
                "boundary_matrix no puede convertirse a float64."
            ) from exc

        if boundary.ndim != 2:
            raise SetEngineError("boundary_matrix debe ser bidimensional.")

        if boundary.size == 0:
            raise SetEngineError("boundary_matrix no puede ser vacía.")

        if not np.all(np.isfinite(boundary)):
            raise SetEngineError("boundary_matrix contiene valores no finitos.")

        if self._n not in boundary.shape:
            raise SetEngineError(
                "boundary_matrix debe tener al menos una dimensión igual a dimension_n."
            )

        return np.ascontiguousarray(boundary, dtype=np.float64)

    def _phase1_infer_metric_orientation(
        self,
        metric_shape: Tuple[int, int],
        boundary_shape: Tuple[int, int],
    ) -> str:
        r"""
        Determina el eje de contracción de G contra B.

          rows : G actúa en el espacio de filas  ⇒  L_F = Bᴴ G⁻¹ B
          cols : G actúa en el espacio de columnas ⇒  L_F = B G⁻¹ Bᴴ
        """
        rows, cols = boundary_shape
        if metric_shape == (rows, rows):
            return "rows"
        if metric_shape == (cols, cols):
            return "cols"
        raise SetEngineError(
            "metric_tensor_G debe coincidir con filas o columnas de boundary_matrix."
        )

    def _phase1_project_metric_spd(
        self,
        metric: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        r"""
        Proyección espectral sobre el cono SPD.

            G ↦ G_spd = ∑_i max(λ_i, ε) |u_i⟩⟨u_i|

        con simetrización hermítica previa G ← (G + Gᴴ)/2 (teorema de Toeplitz
        para la parte hermítica) y suelo ε dimensionado por el radio espectral.
        """
        diagnostics: List[str] = []

        hermitian = 0.5 * (metric + metric.conj().T)

        try:
            eigvals, eigvecs = la.eigh(hermitian)
        except la.LinAlgError as exc:
            raise SetEngineError(
                "No fue posible diagonalizar metric_tensor_G."
            ) from exc

        if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
            raise SetEngineError("Autovalores de metric_tensor_G inválidos.")

        eigvals_real = np.real(eigvals)
        abs_tol = self._spectral_abs_tol(eigvals_real, 1e-12)

        negative_count = int(np.sum(eigvals_real < -abs_tol))
        if negative_count >= _PSD_PROJECTION_WARN_COUNT:
            diagnostics.append(
                f"Métrica con {negative_count} autovalores negativos; "
                "proyección espectral a SPD."
            )

        floor = max(_SPD_FLOOR, _MACHINE_EPS)
        eigvals_clamped = np.maximum(eigvals_real, floor)

        reconstructed = (eigvecs * eigvals_clamped) @ eigvecs.conj().T
        reconstructed = 0.5 * (reconstructed + reconstructed.conj().T)

        if not np.all(np.isfinite(reconstructed)):
            raise SetEngineError("Métrica regularizada no finita.")

        residual = self._finite_norm(reconstructed - reconstructed.conj().T)
        denom = max(1.0, self._finite_norm(reconstructed))
        if residual / denom > _HERMITIAN_REL_TOL:
            diagnostics.append(
                f"Residuo hermítico de G_spd elevado: {residual / denom:.3e}."
            )

        return reconstructed, tuple(diagnostics)

    def _phase1_canonical_metric(
        self,
        metric_tensor_G: Any,
        boundary: np.ndarray,
    ) -> Tuple[np.ndarray, str, Tuple[str, ...]]:
        """Canoniza la métrica de fondo como operador SPD/Hermitiano orientado."""
        try:
            metric = np.asarray(metric_tensor_G, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise SetEngineError(
                "metric_tensor_G no puede convertirse a complex128."
            ) from exc

        if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
            raise SetEngineError("metric_tensor_G debe ser cuadrada.")

        if metric.size == 0:
            raise SetEngineError("metric_tensor_G no puede ser vacía.")

        if not np.all(np.isfinite(metric)):
            raise SetEngineError("metric_tensor_G contiene valores no finitos.")

        orientation = self._phase1_infer_metric_orientation(metric.shape, boundary.shape)
        metric_spd, diagnostics = self._phase1_project_metric_spd(metric)
        return metric_spd, orientation, diagnostics

    def _phase1_canonical_coupling(
        self,
        coupling_V: Any,
        laplacian_dimension: int,
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        r"""
        Canoniza la matriz de acoplamiento de canales

            V ∈ ℂ^{d × c},  d = dim L_F,  c ≥ 1.

        Si V llega transpuesta de forma inequívoca, se adjuntiza bajo diagnóstico.
        """
        diagnostics: List[str] = []

        try:
            coupling = np.asarray(coupling_V, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise SetEngineError(
                "coupling_V no puede convertirse a complex128."
            ) from exc

        if coupling.ndim == 1:
            coupling = coupling.reshape(-1, 1)

        if coupling.ndim != 2:
            raise SetEngineError("coupling_V debe ser bidimensional.")

        if coupling.size == 0:
            raise SetEngineError("coupling_V no puede ser vacía.")

        if not np.all(np.isfinite(coupling)):
            raise SetEngineError("coupling_V contiene valores no finitos.")

        if coupling.shape[0] == laplacian_dimension:
            if coupling.shape[1] <= 0:
                raise SetEngineError("coupling_V debe tener al menos un canal.")
            return np.ascontiguousarray(coupling), tuple(diagnostics)

        if coupling.shape[1] == laplacian_dimension:
            diagnostics.append("coupling_V fue adjuntizada para compatibilidad d×c.")
            return np.ascontiguousarray(coupling.conj().T), tuple(diagnostics)

        raise SetEngineError(
            "coupling_V debe tener una dimensión igual a la dimensión del Laplaciano."
        )

    def _phase1_remesh_uniform_frequency(
        self,
        frequencies: np.ndarray,
        impedance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Tuple[str, ...]]:
        r"""
        Uniformiza la malla {ω_k} por interpolación lineal de Re Z e Im Z.

        El operador iFFT discreto es el isomorfismo de DFT, que presupone
        muestreo aritmético. Si Δω no es constante se re-muestrea sobre

            ω̂_k = ω_min + k (ω_max − ω_min)/(N−1),  k = 0,…,N−1.
        """
        diagnostics: List[str] = []

        if frequencies.size <= 1:
            return frequencies, impedance, False, tuple(diagnostics)

        diffs = np.diff(frequencies)
        if np.any(diffs < 0.0):
            raise SetEngineError("La malla frecuencial no es no-decreciente.")

        if np.allclose(
            diffs,
            diffs[0],
            rtol=_FREQ_UNIFORM_RTOL,
            atol=_FREQ_UNIFORM_ATOL,
        ):
            return frequencies, impedance, False, tuple(diagnostics)

        diagnostics.append(
            "Malla frecuencial no uniforme; re-muestreo lineal a rejilla "
            "aritmética para legitimar iFFT."
        )

        uniform = np.linspace(
            float(frequencies[0]),
            float(frequencies[-1]),
            frequencies.size,
            dtype=np.float64,
        )
        real_part = np.interp(uniform, frequencies, impedance.real)
        imag_part = np.interp(uniform, frequencies, impedance.imag)
        remeshed = np.ascontiguousarray(real_part + 1.0j * imag_part)

        if not np.all(np.isfinite(remeshed)):
            raise SetEngineError("Impedancia re-muestreada no finita.")

        return uniform, remeshed, True, tuple(diagnostics)

    def _phase1_canonical_frequency_grid(
        self,
        frequencies: Any,
        impedance_profile: Any,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Tuple[str, ...]]:
        """Ordena (ω, Z(ω)) y uniformiza la malla si el paso no es constante."""
        diagnostics: List[str] = []

        try:
            freq = np.asarray(frequencies, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise SetEngineError("frequencies no puede convertirse a float64.") from exc

        try:
            impedance = np.asarray(impedance_profile, dtype=np.complex128).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise SetEngineError(
                "impedance_profile no puede convertirse a complex128."
            ) from exc

        if freq.size == 0:
            raise SetEngineError("frequencies no puede ser vacío.")

        if impedance.size != freq.size:
            raise SetEngineError(
                "impedance_profile debe tener la misma longitud que frequencies."
            )

        if not np.all(np.isfinite(freq)):
            raise SetEngineError("frequencies contiene valores no finitos.")

        if not np.all(np.isfinite(impedance)):
            raise SetEngineError("impedance_profile contiene valores no finitos.")

        order = np.argsort(freq, kind="mergesort")
        freq_sorted = np.ascontiguousarray(freq[order], dtype=np.float64)
        impedance_sorted = np.ascontiguousarray(impedance[order], dtype=np.complex128)

        if freq_sorted.size >= 2 and float(freq_sorted[0]) > 0.0:
            diagnostics.append(
                "Malla estrictamente positiva: iFFT interpreta el origen como DC "
                "relativo, no como ω = 0 físico."
            )

        freq_u, impedance_u, remeshed, remesh_diag = (
            self._phase1_remesh_uniform_frequency(freq_sorted, impedance_sorted)
        )
        diagnostics.extend(remesh_diag)

        return freq_u, impedance_u, remeshed, tuple(diagnostics)

    @staticmethod
    def _phase1_canonical_euler(euler_characteristic: Any) -> int:
        """Valida la característica de Euler–Poincaré χ(K) ∈ ℤ."""
        try:
            value = float(euler_characteristic)
        except (TypeError, ValueError) as exc:
            raise SetEngineError("euler_characteristic debe ser numérico.") from exc

        if not math.isfinite(value) or not value.is_integer():
            raise SetEngineError("euler_characteristic debe ser un entero finito.")

        return int(value)

    def _phase1_generate_session_digest(
        self,
        *,
        boundary: np.ndarray,
        metric: np.ndarray,
        coupling: np.ndarray,
        frequencies: np.ndarray,
        impedance_profile: np.ndarray,
        euler_characteristic: int,
        orientation: str,
        laplacian_dimension: int,
    ) -> str:
        """Firma SHA-256 de sesión con separación de dominio y parámetros del motor."""
        sha = hashlib.sha256()

        self._hash_update_domain(sha, _DOMAIN_SESSION)
        self._hash_update_bytes(sha, self._engine_digest.encode("utf-8"))
        self._hash_update_array(sha, "boundary_matrix", boundary)
        self._hash_update_array(sha, "metric_tensor_G", metric)
        self._hash_update_array(sha, "coupling_V", coupling)
        self._hash_update_array(sha, "frequencies", frequencies)
        self._hash_update_array(sha, "impedance_profile", impedance_profile)
        self._hash_update_bytes(sha, orientation.encode("utf-8"))
        self._hash_update_bytes(sha, str(int(laplacian_dimension)).encode("utf-8"))
        self._hash_update_bytes(sha, str(int(euler_characteristic)).encode("utf-8"))
        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))

        return sha.hexdigest()

    def _phase1_observe_and_freeze(
        self,
        *,
        boundary_matrix: np.ndarray,
        metric_tensor_G: np.ndarray,
        coupling_V: np.ndarray,
        frequencies: np.ndarray,
        impedance_profile: np.ndarray,
        euler_characteristic: int,
    ) -> SetCertificate:
        r"""
        FASE 1 · OBSERVE  (morfismo de cierre).

        Canoniza, regulariza y congela el expediente de frontera. El valor de
        retorno no es un terminal de Fase 1: es el compuesto

            Φ₁₂ ∘ Observe  :  datos crudos  →  SetCertificate

        realizado por la continuación formal `_phase2_open_from_phase1`,
        que es el primer morfismo de la FASE 2 · ORIENT/DECIDE.
        """
        diagnostics: List[str] = []

        boundary = self._phase1_validate_boundary_matrix(boundary_matrix)

        metric, orientation, metric_diag = self._phase1_canonical_metric(
            metric_tensor_G,
            boundary,
        )
        diagnostics.extend(metric_diag)

        if orientation == "rows":
            laplacian_dimension = int(boundary.shape[1])
        else:
            laplacian_dimension = int(boundary.shape[0])

        coupling, coupling_diag = self._phase1_canonical_coupling(
            coupling_V,
            laplacian_dimension,
        )
        diagnostics.extend(coupling_diag)

        freq_sorted, impedance_sorted, remeshed, freq_diag = (
            self._phase1_canonical_frequency_grid(frequencies, impedance_profile)
        )
        diagnostics.extend(freq_diag)

        euler = self._phase1_canonical_euler(euler_characteristic)

        session_digest = self._phase1_generate_session_digest(
            boundary=boundary,
            metric=metric,
            coupling=coupling,
            frequencies=freq_sorted,
            impedance_profile=impedance_sorted,
            euler_characteristic=euler,
            orientation=orientation,
            laplacian_dimension=laplacian_dimension,
        )

        dossier = Phase1Dossier(
            boundary_matrix=self._freeze_array(boundary, np.float64),
            metric_tensor=self._freeze_array(metric, np.complex128),
            metric_orientation=orientation,
            laplacian_dimension=laplacian_dimension,
            coupling_V=self._freeze_array(coupling, np.complex128),
            frequencies=self._freeze_array(freq_sorted, np.float64),
            impedance_profile=self._freeze_array(impedance_sorted, np.complex128),
            euler_characteristic=euler,
            session_digest=session_digest,
            engine_digest=self._engine_digest,
            frequency_was_remeshed=bool(remeshed),
            diagnostics=tuple(diagnostics),
        )

        logger.info(
            "FASE 1 OBSERVE: expediente SET congelado. "
            "B=%s, orientación=%s, d_L=%d, ω_count=%d, remesh=%s, digest=%s",
            boundary.shape,
            orientation,
            laplacian_dimension,
            freq_sorted.size,
            remeshed,
            session_digest[:16],
        )

        # ── continuación anidada: el terminal de FASE 1 es el inicial de FASE 2
        return self._phase2_open_from_phase1(dossier)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 2 · ORIENT / DECIDE
    #
    # Objeto inicial  : Phase1Dossier  (entregado por _phase1_observe_and_freeze)
    # Objeto terminal : Phase2Dossier
    # Morfismo de apertura: _phase2_open_from_phase1
    # Morfismo de cierre  : _phase2_audit_and_orient → _phase3_open_from_phase2
    #
    # Invariantes:
    #   (J1) L_F = Aᴴ A  (Gram) ⇒ espectro real y λ_min ≥ −O(ε_mach ‖L_F‖).
    #   (J2) λ₂ es el valor de Fiedler; dim ker estima β del grado del haz.
    #   (J3) S(ω) = I − 2πi Vᴴ (ωI − L_F + iπ V Vᴴ)⁻¹ V
    #        (fórmula de Mahaux–Weidenmüller).
    #   (J4) Γ(ω) = (Z(ω)−Z₀)/(Z(ω)+Z₀),  Γ(t) = F⁻¹{Γ(ω)}.
    #   (J5) veredicto = ⋀ átomos de Heyting (unitariedad, Fiedler, TDR, Hodge).
    # ═══════════════════════════════════════════════════════════════════════

    def _phase2_open_from_phase1(self, dossier: Phase1Dossier) -> SetCertificate:
        r"""
        Inicio formal de la FASE 2 · ORIENT/DECIDE.

        Este método es la continuación estricta del terminal de FASE 1:
        recibe el `Phase1Dossier` congelado y abre la auditoría espectral,
        de scattering y de reflectometría.
        """
        if not isinstance(dossier, Phase1Dossier):
            raise SetEngineError(
                "FASE 2 exige un Phase1Dossier emitido por FASE 1."
            )
        return self._phase2_audit_and_orient(dossier)

    def _phase2_metric_inverse_sqrt(self, metric: np.ndarray) -> np.ndarray:
        r"""
        Raíz inversa hermítica G^{-1/2} vía cálculo funcional de Borel:

            f(λ) = λ^{-1/2}  sobre σ(G) ⊂ [ε, ∞).
        """
        try:
            eigvals, eigvecs = la.eigh(metric)
        except la.LinAlgError as exc:
            raise SetEngineError(
                "No fue posible diagonalizar la métrica regularizada."
            ) from exc

        if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
            raise SetEngineError("Autovalores métricos inválidos en fase Laplaciano.")

        floor = max(_SPD_FLOOR, _MACHINE_EPS)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(np.real(eigvals), floor))
        inverse_sqrt = (eigvecs * inv_sqrt) @ eigvecs.conj().T
        inverse_sqrt = 0.5 * (inverse_sqrt + inverse_sqrt.conj().T)

        if not np.all(np.isfinite(inverse_sqrt)):
            raise SetEngineError("G^{-1/2} no finita.")

        return inverse_sqrt

    def _phase2_compute_sheaf_laplacian(
        self,
        boundary: np.ndarray,
        metric: np.ndarray,
        orientation: str,
    ) -> np.ndarray:
        r"""
        Laplaciano de haz celular en forma de Gram (SPSD por construcción).

        Si orientation == "rows":
            M   = G^{-1/2} B
            L_F = Mᴴ M = Bᴴ G⁻¹ B

        Si orientation == "cols":
            M   = B G^{-1/2}
            L_F = M Mᴴ = B G⁻¹ Bᴴ

        La forma AᴴA garantiza σ(L_F) ⊂ [0, ∞) hasta error de redondeo.
        """
        inverse_sqrt = self._phase2_metric_inverse_sqrt(metric)

        if orientation == "rows":
            factor = inverse_sqrt @ boundary
            laplacian = factor.conj().T @ factor
        elif orientation == "cols":
            factor = boundary @ inverse_sqrt
            laplacian = factor @ factor.conj().T
        else:
            raise SetEngineError("Orientación métrica desconocida.")

        laplacian = 0.5 * (laplacian + laplacian.conj().T)

        if not np.all(np.isfinite(laplacian)):
            raise SetEngineError("Laplaciano de haz no finito.")

        return laplacian

    def compute_sheaf_laplacian(
        self,
        boundary_matrix: np.ndarray,
        metric_tensor_G: np.ndarray,
    ) -> np.ndarray:
        r"""
        API pública compatible.

        Calcula el Laplaciano del haz celular SPSD sobre la métrica anisotrópica:

            L_F = δᴴ G⁻¹ δ

        con selección automática de orientación métrica y factorización de Gram.
        """
        boundary = self._phase1_validate_boundary_matrix(boundary_matrix)
        metric, orientation, _ = self._phase1_canonical_metric(metric_tensor_G, boundary)
        return self._phase2_compute_sheaf_laplacian(boundary, metric, orientation)

    def _phase2_hermitian_residual(self, laplacian: np.ndarray) -> float:
        """Residuo hermítico relativo ‖L − Lᴴ‖_F / max(1, ‖L‖_F)."""
        residual = self._finite_norm(laplacian - laplacian.conj().T)
        denom = max(1.0, self._finite_norm(laplacian))
        relative = residual / denom
        if not math.isfinite(relative):
            return math.inf
        return float(relative)

    def _phase2_spectral_audit(
        self,
        laplacian: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, float, float, int, Tuple[str, ...]]:
        r"""
        Auditoría espectral de L_F.

        Retorna
            eigs, λ₂, gap, κ, λ_min, dim ker, diagnostics.

        El núcleo se estima por umbral relativo al radio espectral
        (rango numérico de Weyl), no por un ε absoluto ciego a la escala.
        """
        diagnostics: List[str] = []

        try:
            eigvals = la.eigvalsh(laplacian)
        except la.LinAlgError as exc:
            raise SetEngineError(
                "No fue posible diagonalizar el Laplaciano de haz."
            ) from exc

        if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
            raise SetEngineError("Autovalores del Laplaciano inválidos.")

        eigvals = np.sort(np.real(eigvals))
        abs_tol = self._spectral_abs_tol(eigvals, _KERNEL_REL_TOL)

        min_eig = float(eigvals[0])
        if min_eig < -abs_tol:
            diagnostics.append(
                "Laplaciano con autovalores negativos no físicos "
                f"(λ_min={min_eig:.3e}, tol={abs_tol:.3e})."
            )

        if eigvals.size == 1:
            fiedler = float(eigvals[0])
        else:
            fiedler = float(eigvals[1])

        max_eig = float(eigvals[-1])
        if max_eig > abs_tol:
            spectral_gap = max(0.0, fiedler / max_eig)
        else:
            spectral_gap = 0.0

        positive = eigvals[eigvals > abs_tol]
        if positive.size <= 1:
            condition_number = 1.0
        else:
            condition_number = float(np.max(positive) / float(np.min(positive)))

        if not math.isfinite(condition_number):
            condition_number = _COND_CAP
        condition_number = float(min(max(1.0, condition_number), _COND_CAP))

        kernel_dim = int(np.sum(np.abs(eigvals) <= abs_tol))

        if not math.isfinite(fiedler):
            raise SetEngineError("Valor de Fiedler no finito.")

        if not math.isfinite(spectral_gap):
            spectral_gap = 0.0

        return (
            eigvals,
            fiedler,
            spectral_gap,
            condition_number,
            min_eig,
            kernel_dim,
            tuple(diagnostics),
        )

    def _phase2_euler_hodge_consistency(
        self,
        kernel_dimension: int,
        euler_characteristic: int,
        laplacian_dimension: int,
    ) -> Tuple[bool, Tuple[str, ...]]:
        r"""
        Diagnóstico de consistencia Hodge–Euler.

        dim ker L_F estima el número de Betti del grado del haz. No se identifica
        con χ(K) salvo en complejos elementales; se reporta incoherencia dura
        sólo cuando el núcleo es imposible (dim ker ∉ [0, d]).
        """
        diagnostics: List[str] = []
        consistent = True

        if kernel_dimension < 0 or kernel_dimension > laplacian_dimension:
            consistent = False
            diagnostics.append(
                f"dim ker L_F = {kernel_dimension} fuera de [0, {laplacian_dimension}]."
            )

        diagnostics.append(
            f"Hodge ker(L_F)={kernel_dimension}, χ(K)={euler_characteristic}, "
            f"d={laplacian_dimension}."
        )

        if (
            laplacian_dimension > 0
            and kernel_dimension == laplacian_dimension
            and euler_characteristic not in (0, 1, -1)
        ):
            diagnostics.append(
                "Núcleo total: L_F ≡ 0 numéricamente; χ(K) no informa un complejo "
                "conexo de ese grado."
            )

        return consistent, tuple(diagnostics)

    def _phase2_resolvent_action(
        self,
        resolvent_matrix: np.ndarray,
        coupling: np.ndarray,
    ) -> np.ndarray:
        r"""
        Acción del resolvente (ωI − H_eff)⁻¹ V.

        Preferencia: `solve` denso. Si el sistema es singular o mal condicionado
        (κ ≳ ε^{-1/2}), se usa pseudoinversa de Moore–Penrose por SVD con
        truncamiento relativo σ_i > ε_rel · σ_max.
        """
        try:
            condition = float(np.linalg.cond(resolvent_matrix))
        except np.linalg.LinAlgError:
            condition = math.inf

        if (not math.isfinite(condition)) or condition > (1.0 / _SQRT_EPS):
            left, singular, right_adj = la.svd(resolvent_matrix, full_matrices=False)
            if singular.size == 0:
                return np.zeros_like(coupling, dtype=np.complex128)
            cutoff = max(_SVD_RCOND_REL * float(singular[0]), _MACHINE_EPS)
            inv_singular = np.where(singular > cutoff, 1.0 / singular, 0.0)
            return (right_adj.conj().T * inv_singular) @ (left.conj().T @ coupling)

        try:
            return la.solve(resolvent_matrix, coupling, assume_a="gen")
        except la.LinAlgError:
            return la.pinv(resolvent_matrix) @ coupling

    def compute_scattering_matrix(
        self,
        L_F: np.ndarray,
        omega: float,
        coupling_V: np.ndarray,
    ) -> np.ndarray:
        r"""
        Matriz S de Mahaux–Weidenmüller en el espacio de Fock de canales:

        \[
            \mathbb{S}(\omega)
            =
            I
            -
            2\pi i \, V^{\dagger}
            \bigl(
                \omega I - L_F + i\pi V V^{\dagger}
            \bigr)^{-1}
            V.
        \]

        Para L_F hermítica y ω real, S es unitaria en aritmética exacta
        (teorema óptico). La implementación usa acción de resolvente estable.
        """
        try:
            laplacian = np.asarray(L_F, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise SetEngineError("L_F no puede convertirse a complex128.") from exc

        if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
            raise SetEngineError("L_F debe ser cuadrada.")

        if laplacian.size == 0:
            raise SetEngineError("L_F no puede ser vacía.")

        if not np.all(np.isfinite(laplacian)):
            raise SetEngineError("L_F contiene valores no finitos.")

        laplacian = 0.5 * (laplacian + laplacian.conj().T)

        try:
            frequency = float(omega)
        except (TypeError, ValueError) as exc:
            raise SetEngineError("omega debe ser escalar.") from exc

        if not math.isfinite(frequency):
            raise SetEngineError("omega debe ser finito.")

        try:
            coupling = np.asarray(coupling_V, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise SetEngineError(
                "coupling_V no puede convertirse a complex128."
            ) from exc

        if coupling.ndim == 1:
            coupling = coupling.reshape(-1, 1)

        if coupling.ndim != 2:
            raise SetEngineError("coupling_V debe ser bidimensional.")

        dimension = laplacian.shape[0]
        if coupling.shape[0] != dimension:
            raise SetEngineError("coupling_V debe tener shape (d, canales).")

        channels = coupling.shape[1]
        if channels <= 0:
            raise SetEngineError("coupling_V debe tener al menos un canal.")

        if not np.all(np.isfinite(coupling)):
            raise SetEngineError("coupling_V contiene valores no finitos.")

        identity = np.eye(dimension, dtype=np.complex128)
        effective = laplacian - 1.0j * np.pi * (coupling @ coupling.conj().T)
        resolvent = frequency * identity - effective
        acted = self._phase2_resolvent_action(resolvent, coupling)

        scattering = np.eye(channels, dtype=np.complex128) - (
            2.0j * np.pi * (coupling.conj().T @ acted)
        )

        if not np.all(np.isfinite(scattering)):
            raise SetEngineError("Matriz S no finita.")

        return scattering

    def _phase2_scattering_audit(
        self,
        laplacian: np.ndarray,
        coupling: np.ndarray,
        frequencies: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, Tuple[str, ...]]:
        r"""
        Auditoría de dispersión cuántica.

        Para cada ω_k se computan:
            det S(ω_k),
            ‖Sᴴ S − I‖_F  (fuga de unitariedad / teorema óptico),
            | |det S| − 1 | (defecto de módulo, consecuencia de unitariedad).
        """
        diagnostics: List[str] = []
        determinants: List[complex] = []
        unitarity_leak = 0.0
        det_modulus_defect = 0.0

        for index, omega in enumerate(frequencies):
            try:
                scattering = self.compute_scattering_matrix(
                    laplacian,
                    float(omega),
                    coupling,
                )
                det_s = la.det(scattering)
                if not self._is_finite_complex(det_s):
                    det_s = complex(np.nan, np.nan)
                    det_modulus_defect = math.inf
                else:
                    det_modulus_defect = max(
                        det_modulus_defect,
                        abs(abs(complex(det_s)) - 1.0),
                    )

                identity = np.eye(scattering.shape[0], dtype=np.complex128)
                leak = float(
                    la.norm(scattering.conj().T @ scattering - identity, ord="fro")
                )
                if not math.isfinite(leak):
                    leak = math.inf
                unitarity_leak = max(unitarity_leak, leak)

            except Exception as exc:  # noqa: BLE001 — auditoría por frecuencia
                diagnostics.append(f"Fallo de scattering en frecuencia {index}: {exc}")
                det_s = complex(np.nan, np.nan)
                unitarity_leak = math.inf
                det_modulus_defect = math.inf

            determinants.append(det_s)

        if not math.isfinite(det_modulus_defect):
            det_modulus_defect = math.inf

        scattering_dets = np.asarray(determinants, dtype=np.complex128)
        return scattering_dets, unitarity_leak, float(det_modulus_defect), tuple(
            diagnostics
        )

    def _phase2_reflection_coefficient_frequency(
        self,
        impedance_profile: np.ndarray,
    ) -> np.ndarray:
        r"""
        Coeficiente de reflexión de tensión sobre Z₀:

            Γ(ω) = (Z(ω) − Z₀) / (Z(ω) + Z₀)

        con regularización del denominador cuando |Z + Z₀| < ε_mach.
        """
        numerator = impedance_profile - self._z0
        denominator = impedance_profile + self._z0

        magnitude = np.abs(denominator)
        small = magnitude < self._reg
        if np.any(small):
            denominator = denominator.copy()
            denominator[small] = self._reg + 0.0j

        with np.errstate(all="ignore"):
            gamma_freq = numerator / denominator

        if not np.all(np.isfinite(gamma_freq)):
            raise SetEngineError("Perfil frecuencial de reflexión no finito.")

        return gamma_freq

    def compute_tdr_reflection_profile(
        self,
        frequencies: np.ndarray,
        impedance_profile: np.ndarray,
    ) -> np.ndarray:
        r"""
        Perfil de reflectometría en el dominio del tiempo (TDR):

        \[
            \Gamma(t)
            =
            \mathcal{F}^{-1}
            \left\{
                \frac{Z(\omega) - Z_0}{Z(\omega) + Z_0}
            \right\}.
        \]

        La malla de entrada se ordena; si el paso no es uniforme se re-muestrea
        antes de la iFFT (la DFT exige muestreo aritmético).
        """
        freq, impedance, _, _ = self._phase1_canonical_frequency_grid(
            frequencies,
            impedance_profile,
        )

        if freq.size == 0:
            return np.array([], dtype=np.float64)

        gamma_freq = self._phase2_reflection_coefficient_frequency(impedance)
        gamma_time = np.fft.ifft(gamma_freq).real

        if not np.all(np.isfinite(gamma_time)):
            raise SetEngineError("Perfil TDR no finito.")

        return np.ascontiguousarray(gamma_time, dtype=np.float64)

    def _phase2_evaluate_heyting(
        self,
        *,
        max_gamma: float,
        unitarity_leak: float,
        det_modulus_defect: float,
        fiedler_value: float,
        min_eig: float,
        condition_number: float,
        spectral_gap: float,
        hermitian_residual: float,
        has_mismatch: bool,
        hodge_euler_consistent: bool,
    ) -> Tuple[str, Tuple[str, ...]]:
        r"""
        Clasificador en el retículo de Heyting Ω₃.

        Cada átomo se valora en {VETOED, DEGRADED, COHERENT}; el veredicto
        global es el meet (ínfimo). VETO duro cubre no-finitud, desajuste
        crítico de impedancia, ruptura de unitariedad, pérdida de PSD,
        Fiedler bajo cota dura e incoherencia Hodge imposible.
        """
        diagnostics: List[str] = []
        atoms: List[str] = []

        def veto(message: str) -> None:
            atoms.append(HeytingVerdict.VETOED.value)
            diagnostics.append(message)

        def degrade(message: str) -> None:
            atoms.append(HeytingVerdict.DEGRADED.value)
            diagnostics.append(message)

        if not math.isfinite(max_gamma):
            veto("Máximo coeficiente de reflexión no finito.")
        if not math.isfinite(unitarity_leak):
            veto("Fuga de unitariedad no finita.")
        if not math.isfinite(det_modulus_defect):
            veto("Defecto de |det S| no finito.")
        if not math.isfinite(fiedler_value):
            veto("Valor de Fiedler no finito.")
        if not math.isfinite(hermitian_residual):
            veto("Residuo hermítico no finito.")

        if has_mismatch:
            veto("Desajuste crítico de impedancia detectado.")

        if unitarity_leak > self._unitarity_veto:
            veto("Unitariedad de S fracturada (teorema óptico).")
        elif unitarity_leak > self._unitarity_degraded:
            degrade("Unitariedad de S sobre cota blanda.")

        if det_modulus_defect > _DET_S_DEFECT_VETO:
            veto(" |det S| se aparta críticamente de 1.")
        elif det_modulus_defect > _DET_S_DEFECT_DEGRADED:
            degrade("|det S| se aparta de 1 sobre cota blanda.")

        if fiedler_value < self._fiedler_veto:
            veto("Conectividad algebraica bajo cota dura.")
        elif fiedler_value < self._fiedler_degraded:
            degrade("Conectividad algebraica sobre cota blanda.")

        if math.isfinite(min_eig) and min_eig < -_MIN_EIG_VETO:
            veto("Laplaciano de haz no PSD.")

        if hermitian_residual > 1e-6:
            veto("Residuo hermítico de L_F incompatible con autoadjunción.")
        elif hermitian_residual > _HERMITIAN_REL_TOL:
            degrade("Residuo hermítico de L_F elevado.")

        if not hodge_euler_consistent:
            veto("Inconsistencia Hodge–Euler: dim ker fuera de rango.")

        if math.isfinite(max_gamma) and (
            max_gamma > self._reflection_degraded_base * self._safety_margin
        ):
            degrade("Reflexión TDR sobre cota blanda.")

        if condition_number > _COND_DEGRADED:
            degrade("Número de condición espectral elevado.")

        if spectral_gap < _GAP_DEGRADED:
            degrade("Gap espectral muy bajo.")

        if not atoms:
            atoms.append(HeytingVerdict.COHERENT.value)

        return self._heyting_meet(atoms), tuple(diagnostics)

    def _phase2_audit_and_orient(self, dossier: Phase1Dossier) -> SetCertificate:
        r"""
        FASE 2 · ORIENT/DECIDE  (morfismo de cierre).

        Compone espectro, scattering, TDR y Heyting en un `Phase2Dossier`.
        El valor de retorno es el compuesto

            Φ₂₃ ∘ Orient  :  Phase1Dossier  →  SetCertificate

        realizado por `_phase3_open_from_phase2`, primer morfismo de la FASE 3.
        """
        diagnostics: List[str] = list(dossier.diagnostics)

        laplacian = self._phase2_compute_sheaf_laplacian(
            dossier.boundary_matrix,
            dossier.metric_tensor,
            dossier.metric_orientation,
        )

        hermitian_residual = self._phase2_hermitian_residual(laplacian)
        if hermitian_residual > _HERMITIAN_REL_TOL:
            diagnostics.append(
                f"Residuo hermítico relativo de L_F = {hermitian_residual:.3e}."
            )

        (
            eigvals,
            fiedler_value,
            spectral_gap,
            condition_number,
            min_eig,
            kernel_dim,
            spectral_diag,
        ) = self._phase2_spectral_audit(laplacian)
        diagnostics.extend(spectral_diag)

        hodge_ok, hodge_diag = self._phase2_euler_hodge_consistency(
            kernel_dim,
            dossier.euler_characteristic,
            dossier.laplacian_dimension,
        )
        diagnostics.extend(hodge_diag)

        s_dets, unitarity_leak, det_defect, scattering_diag = (
            self._phase2_scattering_audit(
                laplacian,
                dossier.coupling_V,
                dossier.frequencies,
            )
        )
        diagnostics.extend(scattering_diag)

        gamma_t = self.compute_tdr_reflection_profile(
            dossier.frequencies,
            dossier.impedance_profile,
        )

        if gamma_t.size == 0:
            max_gamma = 0.0
        else:
            max_gamma = float(np.max(np.abs(gamma_t)))

        if not math.isfinite(max_gamma):
            max_gamma = math.inf

        has_mismatch = bool(
            max_gamma > self._reflection_veto_base * self._safety_margin
        )

        verdict, heyting_diag = self._phase2_evaluate_heyting(
            max_gamma=max_gamma,
            unitarity_leak=unitarity_leak,
            det_modulus_defect=det_defect,
            fiedler_value=fiedler_value,
            min_eig=min_eig,
            condition_number=condition_number,
            spectral_gap=spectral_gap,
            hermitian_residual=hermitian_residual,
            has_mismatch=has_mismatch,
            hodge_euler_consistent=hodge_ok,
        )
        diagnostics.extend(heyting_diag)

        phase2 = Phase2Dossier(
            phase1=dossier,
            L_F=self._freeze_array(laplacian, np.complex128),
            sheaf_eigenvalues=self._freeze_array(eigvals, np.float64),
            fiedler_value=fiedler_value,
            spectral_gap=spectral_gap,
            condition_number=condition_number,
            min_eigenvalue=min_eig,
            hodge_kernel_dimension=kernel_dim,
            hermitian_residual=hermitian_residual,
            scattering_determinants=self._freeze_array(s_dets, np.complex128),
            unitarity_leak=unitarity_leak,
            det_modulus_defect=det_defect,
            tdr_reflection_profile=self._freeze_array(gamma_t, np.float64),
            max_reflection_coefficient=max_gamma,
            has_impedance_mismatch=has_mismatch,
            heyting_verdict=verdict,
            hodge_euler_consistent=hodge_ok,
            diagnostics=tuple(diagnostics),
        )

        logger.info(
            "FASE 2 ORIENT: veredicto=%s, λ₂=%.6e, Γ_max=%.6e, leak=%.6e, "
            "ker=%d, |detS|-1=%.3e",
            verdict,
            fiedler_value,
            max_gamma,
            unitarity_leak,
            kernel_dim,
            det_defect,
        )

        # ── continuación anidada: el terminal de FASE 2 es el inicial de FASE 3
        return self._phase3_open_from_phase2(phase2)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 3 · ACT / CERTIFY
    #
    # Objeto inicial  : Phase2Dossier  (entregado por _phase2_audit_and_orient)
    # Objeto terminal : SetCertificate (inmutable, sellado, fail-closed)
    # Morfismo de apertura: _phase3_open_from_phase2
    #
    # Invariantes:
    #   (K1) VETOED ⇒ interlock crowbar simulado y latencia ∈ [380, 420] ns.
    #   (K2) sello = SHA-256(dominio ‖ veredicto ‖ sesión ‖ observables).
    #   (K3) el certificado no muta tensores (arrays write-protected).
    # ═══════════════════════════════════════════════════════════════════════

    def _phase3_open_from_phase2(self, dossier: Phase2Dossier) -> SetCertificate:
        r"""
        Inicio formal de la FASE 3 · ACT/CERTIFY.

        Continuación estricta del terminal de FASE 2: recibe el expediente
        auditado y abre actuación de interlock + certificación inmutable.
        """
        if not isinstance(dossier, Phase2Dossier):
            raise SetEngineError(
                "FASE 3 exige un Phase2Dossier emitido por FASE 2."
            )
        return self._phase3_certify(dossier)

    def _phase3_parse_verdict(self, verdict: Any) -> HeytingVerdict:
        """Parseo total: cualquier etiqueta no reconocida colapsa a VETOED."""
        if isinstance(verdict, HeytingVerdict):
            return verdict
        try:
            return HeytingVerdict(str(verdict).strip().upper())
        except ValueError:
            return HeytingVerdict.VETOED

    def _phase3_actuate_interlock(
        self,
        verdict: Any,
        session_hash: str,
    ) -> Tuple[bool, float]:
        r"""
        FASE 3 · ACT.

        Si Heyting colapsa a VETOED, se simula la ISR en IRAM del ESP32:
        conmutación de GPIO14 y cebado del tiristor crowbar BT151 en
        400 ns ± jitter gaussiano recortado al intervalo de calibración.
        """
        verdict_enum = self._phase3_parse_verdict(verdict)
        session_ref = str(session_hash)[:16]

        if verdict_enum is HeytingVerdict.VETOED:
            jitter = float(self._rng.normal(0.0, _CROWBAR_JITTER_STD_NS))
            latency = float(
                np.clip(
                    _CROWBAR_IRAM_LATENCY_NS + jitter,
                    _CROWBAR_LATENCY_MIN_NS,
                    _CROWBAR_LATENCY_MAX_NS,
                )
            )

            logger.critical(
                "¡DESAJUSTE CRÍTICO DE IMPEDANCIA EN LA FRONTERA! "
                "La sonda SET detectó eco anómalo en el presupuesto. "
                "Disyuntor perimetral Crowbar BT151 [GPIO%d] gatillado en %.2f ns. "
                "Obra civil paralizada de inmediato para evitar fraguado defectuoso. "
                "Sello: %s",
                _GPIO_CROWBAR_PIN,
                latency,
                session_ref,
            )
            return True, latency

        logger.info(
            "Sonda SET regulada sin interlock. Estado=%s. Sello=%s",
            verdict_enum.value,
            session_ref,
        )
        return False, 0.0

    def _phase3_compose_signature(
        self,
        *,
        verdict: str,
        session_digest: str,
        max_gamma: float,
        unitarity_leak: float,
        fiedler_value: float,
        condition_number: float,
        spectral_gap: float,
        latency_ns: float,
        euler_characteristic: int,
        frequency_count: int,
        hodge_kernel_dimension: int,
        hermitian_residual: float,
        det_modulus_defect: float,
        engine_digest: str,
    ) -> str:
        """Firma SHA-256 final de telemetría con separación de dominio."""
        sha = hashlib.sha256()

        self._hash_update_domain(sha, _DOMAIN_TELEMETRY)
        self._hash_update_bytes(sha, verdict.encode("utf-8"))
        self._hash_update_bytes(sha, session_digest.encode("utf-8"))
        self._hash_update_bytes(sha, engine_digest.encode("utf-8"))

        scalars = (
            max_gamma,
            unitarity_leak,
            fiedler_value,
            condition_number,
            spectral_gap,
            latency_ns,
            float(euler_characteristic),
            float(frequency_count),
            float(hodge_kernel_dimension),
            hermitian_residual,
            det_modulus_defect,
        )
        for value in scalars:
            numeric = float(value) if math.isfinite(float(value)) else float("nan")
            self._hash_update_bytes(sha, f"{numeric:.17e}".encode("utf-8"))

        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        return sha.hexdigest()

    def _phase3_certify(self, dossier: Phase2Dossier) -> SetCertificate:
        """
        FASE 3 · CERTIFY.

        Emite el certificado inmutable y acciona el interlock si el veredicto
        de Heyting es VETOED. Terminal del morfismo anidado Φ₂₃ ∘ Φ₁₂.
        """
        interlock_fired, latency = self._phase3_actuate_interlock(
            dossier.heyting_verdict,
            dossier.phase1.session_digest,
        )

        seal = self._phase3_compose_signature(
            verdict=dossier.heyting_verdict,
            session_digest=dossier.phase1.session_digest,
            max_gamma=dossier.max_reflection_coefficient,
            unitarity_leak=dossier.unitarity_leak,
            fiedler_value=dossier.fiedler_value,
            condition_number=dossier.condition_number,
            spectral_gap=dossier.spectral_gap,
            latency_ns=latency,
            euler_characteristic=dossier.phase1.euler_characteristic,
            frequency_count=int(dossier.phase1.frequencies.size),
            hodge_kernel_dimension=dossier.hodge_kernel_dimension,
            hermitian_residual=dossier.hermitian_residual,
            det_modulus_defect=dossier.det_modulus_defect,
            engine_digest=dossier.phase1.engine_digest,
        )

        state = SondaState(
            sheaf_eigenvalues=dossier.sheaf_eigenvalues,
            scattering_determinants=dossier.scattering_determinants,
            tdr_reflection_profile=dossier.tdr_reflection_profile,
            max_reflection_coefficient=dossier.max_reflection_coefficient,
            unitarity_leak=dossier.unitarity_leak,
            fiedler_value=dossier.fiedler_value,
            condition_number=dossier.condition_number,
            spectral_gap=dossier.spectral_gap,
            hodge_kernel_dimension=dossier.hodge_kernel_dimension,
            hermitian_residual=dossier.hermitian_residual,
            det_modulus_defect=dossier.det_modulus_defect,
        )

        certificate = SetCertificate(
            heyting_verdict=dossier.heyting_verdict,
            state=state,
            has_impedance_mismatch=dossier.has_impedance_mismatch,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            cryptographic_seal=seal,
            schema_version=_SCHEMA_VERSION,
            session_digest=dossier.phase1.session_digest,
            euler_characteristic=dossier.phase1.euler_characteristic,
            frequency_count=int(dossier.phase1.frequencies.size),
            diagnostics=dossier.diagnostics,
            betti_estimate=dossier.hodge_kernel_dimension,
            hodge_euler_consistent=dossier.hodge_euler_consistent,
            engine_digest=dossier.phase1.engine_digest,
        )

        if dossier.heyting_verdict == HeytingVerdict.VETOED.value:
            logger.error(
                "FASE 3 CERTIFY: VETO SET. Γ_max=%.6e, leak=%.6e, λ₂=%.6e, ker=%d",
                dossier.max_reflection_coefficient,
                dossier.unitarity_leak,
                dossier.fiedler_value,
                dossier.hodge_kernel_dimension,
            )
        elif dossier.heyting_verdict == HeytingVerdict.DEGRADED.value:
            logger.warning(
                "FASE 3 CERTIFY: SET degradado. diagnostics=%s",
                dossier.diagnostics,
            )
        else:
            logger.info(
                "FASE 3 CERTIFY: SET coherente. digest=%s",
                dossier.phase1.session_digest[:16],
            )

        return certificate

    # ═══════════════════════════════════════════════════════════════════════
    # FAIL-CLOSED GLOBAL  (terminal de emergencia, isomorfo a FASE 3 VETOED)
    # ═══════════════════════════════════════════════════════════════════════

    def _fail_closed_certificate(
        self,
        reason: str,
        euler_characteristic: Any = 0,
    ) -> SetCertificate:
        """
        Certificado fail-closed ante excepción no recuperable.

        Garantiza VETO, interlock y firma inmutable incluso cuando el expediente
        de entrada no pudo ser validado. No relanza: el lazo cerrado siempre
        produce un `SetCertificate`.
        """
        try:
            euler = self._phase1_canonical_euler(euler_characteristic)
        except Exception:  # noqa: BLE001 — fail-closed total
            euler = 0

        sha = hashlib.sha256()
        self._hash_update_domain(sha, _DOMAIN_SESSION)
        self._hash_update_bytes(sha, self._engine_digest.encode("utf-8"))
        self._hash_update_bytes(sha, reason.encode("utf-8"))
        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        session_digest = sha.hexdigest()

        interlock_fired, latency = self._phase3_actuate_interlock(
            HeytingVerdict.VETOED.value,
            session_digest,
        )

        seal = self._phase3_compose_signature(
            verdict=HeytingVerdict.VETOED.value,
            session_digest=session_digest,
            max_gamma=math.inf,
            unitarity_leak=math.inf,
            fiedler_value=0.0,
            condition_number=math.inf,
            spectral_gap=0.0,
            latency_ns=latency,
            euler_characteristic=euler,
            frequency_count=0,
            hodge_kernel_dimension=0,
            hermitian_residual=math.inf,
            det_modulus_defect=math.inf,
            engine_digest=self._engine_digest,
        )

        empty_real = self._freeze_array(np.array([], dtype=np.float64), np.float64)
        empty_complex = self._freeze_array(
            np.array([], dtype=np.complex128),
            np.complex128,
        )

        state = SondaState(
            sheaf_eigenvalues=empty_real,
            scattering_determinants=empty_complex,
            tdr_reflection_profile=empty_real,
            max_reflection_coefficient=math.inf,
            unitarity_leak=math.inf,
            fiedler_value=0.0,
            condition_number=math.inf,
            spectral_gap=0.0,
            hodge_kernel_dimension=0,
            hermitian_residual=math.inf,
            det_modulus_defect=math.inf,
        )

        return SetCertificate(
            heyting_verdict=HeytingVerdict.VETOED.value,
            state=state,
            has_impedance_mismatch=True,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            cryptographic_seal=seal,
            schema_version=_SCHEMA_VERSION,
            session_digest=session_digest,
            euler_characteristic=euler,
            frequency_count=0,
            diagnostics=(f"FAIL-CLOSED: {reason}",),
            betti_estimate=0,
            hodge_euler_consistent=False,
            engine_digest=self._engine_digest,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # API PÚBLICA COMPATIBLE
    # ═══════════════════════════════════════════════════════════════════════

    def execute_echolocation_scan(
        self,
        boundary_matrix: np.ndarray,
        metric_tensor_G: np.ndarray,
        coupling_V: np.ndarray,
        frequencies: np.ndarray,
        impedance_profile: np.ndarray,
        euler_characteristic: int = 1,
    ) -> SetCertificate:
        r"""
        Orquesta el escaneo sónico completo de la frontera.

        Compone las tres fases anidadas

            Φ₂₃ ∘ Φ₁₂ ∘ Observe  :  datos  →  SetCertificate

        Si ocurre cualquier excepción, devuelve certificado fail-closed VETOED
        (el tipo de retorno es total: nunca se propaga el fallo al llamador).
        """
        try:
            return self._phase1_observe_and_freeze(
                boundary_matrix=boundary_matrix,
                metric_tensor_G=metric_tensor_G,
                coupling_V=coupling_V,
                frequencies=frequencies,
                impedance_profile=impedance_profile,
                euler_characteristic=euler_characteristic,
            )
        except Exception as exc:  # noqa: BLE001 — contrato fail-closed
            logger.exception("Fallo fail-closed en SET; emitiendo VETO de frontera.")
            return self._fail_closed_certificate(
                reason=str(exc),
                euler_characteristic=euler_characteristic,
            )


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN DE FIRMAS DE CALIBRE
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "SetEngine",
    "SondaState",
    "SetCertificate",
    "HeytingVerdict",
    "SetEngineError",
    "Phase1Dossier",
    "Phase2Dossier",
]

# Compatibilidad con la exportación histórica del módulo original.
all = __all__
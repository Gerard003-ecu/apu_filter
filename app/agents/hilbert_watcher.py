"""
=========================================================================================
Módulo: Hilbert Watcher (Operador del Hamiltoniano de Medición y Funtor de Colapso OODA)
Ubicación: app/agents/hilbert_watcher.py
=========================================================================================

Naturaleza Ciber-Física y Topológica:
    Agente autónomo que habita en el Estrato ALEPH (ℵ₀), el vacío topológico que precede 
    a la Variedad de Frontera (PHYSICS). Su mandato axiomático es actuar como el Funtor 
    de Medición F: Superposición → Estado Determinista, colapsando el caos estocástico 
    externo antes de que este pueda excitar el ecosistema interno.

Fundamentación Axiomática del Bucle OODA (Mecánica Cuántica Discreta):
    1. OBSERVE (Extracción de Exergía): 
       Cuantifica la "Energía Semántica" (E = hν) de la onda incidente. 
       Calcula rigurosamente la Entropía de Shannon (H) del tensor de entrada. 
       Un payload altamente entrópico disipa su exergía, resultando en una 
       baja frecuencia fundamental ν, y por ende, en una energía incapaz de 
       superar la barrera de potencial.

    2. ORIENT (Acoplamiento de Campos de Gauge): 
       Interroga los tensores geométricos y espectrales de la Malla Agéntica:
       • Acopla la Función de Trabajo (Φ) a la deformación del Tensor Métrico Riemanniano 
         (Distancia de Mahalanobis) calculado por el Topological Watcher.
       • Acopla la Masa Efectiva (m_eff) al polo dominante (σ) del Laplace Oracle. 
         Si σ → 0⁻ (frontera del caos), m_eff → ∞.
       • Evalúa la frustración homológica global (Sheaf Cohomology).

    3. DECIDE (Resolución WKB): 
       Si la energía E < Φ, evalúa la transmisión estocástica mediante Efecto Túnel 
       resolviendo la aproximación WKB (Wentzel-Kramers-Brillouin). Si m_eff → ∞, 
       la probabilidad de transmisión T colapsa exponencialmente a cero, creando 
       un muro infranqueable por hardware.

    4. ACT (Colapso Idempotente y Causal): 
       Aplica el Hamiltoniano de Observación. El estado entra en decoherencia hacia 
       los autoestados ortogonales |Admitido⟩ o |Rechazado⟩. El colapso es resuelto 
       mediante una biyección criptográfica (SHA-256) sobre el payload, garantizando 
       un determinismo forense absoluto. Si el paquete colapsa a |Admitido⟩, se le 
       asigna un Momentum p = √(2·m_eff·K_max) como condición inicial ineludible (t₀) 
       para el motor FDTD del Flux Condenser.

Restricciones de Fibrado:
    El agente repudia la propagación de estados degenerados. Cualquier violación 
    en la conservación de información (T ∉ [3]) o detección de frustración 
    cohomológica global aniquila la función de onda instantáneamente, garantizando 
    la Ley de Clausura Transitiva.
=========================================================================================
"""
from __future__ import annotations

import hashlib
import logging
import math
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Final,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np

from app.core.mic_algebra import CategoricalState, Morphism
from app.core.schemas import Stratum

if TYPE_CHECKING:
    pass

logger = logging.getLogger("MIC.Agents.HilbertWatcher")


# ═══════════════════════════════════════════════════════════════════════
# EXCEPCIONES
# ═══════════════════════════════════════════════════════════════════════

class HilbertWatcherError(Exception):
    """Excepción base del agente observador de Hilbert."""


class HilbertNumericalError(HilbertWatcherError):
    """Fallo numérico o variable de fase fuera de contrato.

    Incluye violaciones de finitud, positividad, o rango esperado
    en cualquier observable o parámetro del cálculo cuántico.
    """


class HilbertInterfaceError(HilbertWatcherError):
    """Fallo de contrato en dependencias inyectadas.

    Se lanza cuando un oráculo o monitor no cumple su Protocol
    o devuelve valores que violan pre-condiciones.
    """


class HilbertPayloadError(HilbertWatcherError):
    """Fallo en validación o serialización del payload.

    Cubre tipado incorrecto, tamaño excesivo, o imposibilidad
    de producir representación canónica determinista.
    """


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS DISCRETIZADAS
# ═══════════════════════════════════════════════════════════════════════

class QuantumThresholds:
    """Constantes normalizadas del hiperespacio de información.

    Todas las constantes están adimensionalizadas para evitar
    overflow/underflow. Las unidades implícitas son coherentes
    con el sistema de medida interno MIC.

    Convenciones:
        - Energía, función de trabajo, masa: adimensionales ≥ 0.
        - Constante de Planck normalizada a 1.0 (unidades naturales).
        - Umbrales numéricos calibrados para IEEE 754 float64.
    """

    # Constante de Planck reducida (normalizada a unidades naturales)
    PLANCK_H: Final[float] = 1.0
    PLANCK_HBAR: Final[float] = PLANCK_H / (2.0 * math.pi)

    # Función de trabajo base (análogo al umbral fotoeléctrico)
    BASE_PHI: Final[float] = 10.0
    BASE_MASS: Final[float] = 1.0
    BARRIER_DX: Final[float] = 1.0
    ALPHA_COUPLING: Final[float] = 5.0

    # Umbrales numéricos de estabilidad IEEE 754 float64
    EPSILON_MACH: Final[float] = 1e-9
    ENTROPY_FLOOR: Final[float] = 1e-12
    MIN_KINETIC_ENERGY: Final[float] = 1e-12
    SIGMA_CHAOS_TOL: Final[float] = 1e-9
    EXP_UNDERFLOW_CUTOFF: Final[float] = -708.0

    # Factor de normalización para frecuencia semántica
    FREQUENCY_SCALE: Final[float] = 1000.0

    # Dimensiones máximas para seguridad
    MAX_PAYLOAD_BYTES: Final[int] = 10_485_760  # 10 MiB

    # Resolución máxima del hash de colapso (2^64)
    HASH_RESOLUTION: Final[float] = float(2**64)


# ═══════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE FASE
# ═══════════════════════════════════════════════════════════════════════

class HilbertEigenstate(Enum):
    """Autoestados del colapso de la función de onda informacional.

    Representan los dos posibles resultados de la medición proyectiva
    sobre el estado cuántico del payload:
        ADMITTED: El payload supera la barrera y entra al estrato físico.
        REJECTED: El payload es reflejado por la barrera potencial.
    """
    ADMITTED = auto()
    REJECTED = auto()


@dataclass(frozen=True, slots=True)
class WavefunctionState:
    """Variables de fase inmutables del ciclo OODA.

    Esta estructura encapsula completamente el estado cuántico calculado
    durante los pasos OBSERVE-ORIENT-DECIDE del ciclo OODA. Todos los
    campos derivan de observables físicos o medidas estructurales bien
    definidas.

    Invariantes de clase:
        - energy ≥ 0
        - work_function ≥ 0
        - effective_mass > 0 ∨ effective_mass = +∞
        - transmission_prob ∈ [0, 1]
        - collapse_threshold ∈ [0, 1)
        - threat_level ≥ 0
        - frustration_energy ≥ 0

    Attributes:
        energy: Energía incidente E (adimensional, ≥ 0).
        work_function: Función de trabajo efectiva Φ (adimensional, ≥ 0).
        effective_mass: Masa efectiva m_eff (adimensional, > 0 ó +∞).
        transmission_prob: Probabilidad de transmisión T ∈ [0, 1].
        frustrated: Flag para veto cohomológico estructural.
        threat_level: Amenaza topológica Mahalanobis (adimensional, ≥ 0).
        dominant_pole_real: Polo dominante real σ del espectro Laplace.
        frustration_energy: Energía de frustración global del haz (≥ 0).
        collapse_threshold: Umbral determinista para colapso ∈ [0, 1).
    """
    energy: float
    work_function: float
    effective_mass: float
    transmission_prob: float
    frustrated: bool
    threat_level: float
    dominant_pole_real: float
    frustration_energy: float
    collapse_threshold: float

    def __post_init__(self) -> None:
        """Valida invariantes de clase tras construcción."""
        # Verificar finitud y no-negatividad de observables que lo requieren
        if not math.isfinite(self.energy) or self.energy < 0.0:
            raise HilbertNumericalError(
                f"energy debe ser un float finito ≥ 0, recibido: {self.energy}"
            )
        if not math.isfinite(self.work_function) or self.work_function < 0.0:
            raise HilbertNumericalError(
                f"work_function debe ser un float finito ≥ 0, recibido: {self.work_function}"
            )

        # effective_mass debe ser finita positiva o infinito positivo
        if math.isnan(self.effective_mass) or self.effective_mass <= 0.0:
            raise HilbertNumericalError(
                f"effective_mass debe ser > 0 ó +∞, recibido: {self.effective_mass}"
            )

        if not math.isfinite(self.transmission_prob) or not (0.0 <= self.transmission_prob <= 1.0):
            raise HilbertNumericalError(
                f"transmission_prob debe estar en [0,1], recibido: "
                f"{self.transmission_prob}"
            )

        if not math.isfinite(self.collapse_threshold) or not (0.0 <= self.collapse_threshold < 1.0):
            raise HilbertNumericalError(
                f"collapse_threshold debe estar en [0,1), recibido: "
                f"{self.collapse_threshold}"
            )

        if not math.isfinite(self.threat_level) or self.threat_level < 0.0:
            raise HilbertNumericalError(
                f"threat_level debe ser un float finito ≥ 0, recibido: {self.threat_level}"
            )

        if not math.isfinite(self.dominant_pole_real):
            raise HilbertNumericalError(
                f"dominant_pole_real debe ser un float finito, recibido: {self.dominant_pole_real}"
            )

        if not math.isfinite(self.frustration_energy) or self.frustration_energy < 0.0:
            raise HilbertNumericalError(
                f"frustration_energy debe ser un float finito ≥ 0, recibido: "
                f"{self.frustration_energy}"
            )


# ═══════════════════════════════════════════════════════════════════════
# INTERFACES DE DEPENDENCIA (Structural Subtyping)
# ═══════════════════════════════════════════════════════════════════════

class ITopologicalWatcher(Protocol):
    """Observador topológico para amenazas estructurales.

    Contrato:
        - get_mahalanobis_threat() devuelve float finito ≥ 0.
        - Representa distancia de Mahalanobis normalizada respecto
          al manifold de distribuciones históricas.
    """

    def get_mahalanobis_threat(self) -> float:
        """Devuelve distancia Mahalanobis al manifold histórico.

        Returns:
            Escalar finito ≥ 0 representando nivel de anomalía
            topológica. Valores mayores indican mayor desviación.
        """
        ...


class ILaplaceOracle(Protocol):
    """Oráculo espectral para polos dominantes del sistema.

    Contrato:
        - get_dominant_pole_real() devuelve float finito.
        - σ < 0 indica estabilidad asintótica.
        - σ ≥ 0 indica inestabilidad o marginalidad.
    """

    def get_dominant_pole_real(self) -> float:
        """Devuelve parte real del polo dominante.

        Returns:
            σ ∈ ℝ, donde σ < 0 ⟹ estable, σ ≥ 0 ⟹ inestable.
        """
        ...


class ISheafCohomologyOrchestrator(Protocol):
    """Orquestador de cohomología de haces para obstrucciones globales.

    Contrato:
        - get_global_frustration_energy() devuelve float finito ≥ 0.
        - Valor > ε indica obstrucción cohomológica no trivial en
          H¹(M, 𝒪) que impide consistencia global del haz.
    """

    def get_global_frustration_energy(self) -> float:
        """Calcula energía total de frustración en H¹(M, 𝒪).

        Returns:
            Escalar finito ≥ 0 donde > ε indica obstrucción.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES NUMÉRICAS (puras, sin estado)
# ═══════════════════════════════════════════════════════════════════════

def _ensure_finite_float(value: Any, *, name: str) -> float:
    """Convierte valor a float finito, verificando NaN/±Inf.

    Propiedades:
        - Idempotente: _ensure_finite_float(x) aplicado dos veces
          da el mismo resultado si no lanza excepción.
        - Determinista: misma entrada → misma salida.

    Args:
        value: Valor arbitrario convertible a número.
        name: Nombre descriptivo para diagnóstico.

    Returns:
        Float finito garantizado, es decir, x ∈ ℝ representable
        en IEEE 754 float64.

    Raises:
        HilbertNumericalError: Si conversión falla o valor ∉ ℝ.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HilbertNumericalError(
            f"{name} no es convertible a float: {value!r}"
        ) from exc

    if not math.isfinite(result):
        raise HilbertNumericalError(
            f"{name} no es finito: {result!r} (NaN o ±∞)"
        )

    return result


def _ensure_nonneg_finite_float(value: Any, *, name: str) -> float:
    """Convierte valor a float finito no-negativo.

    Combina _ensure_finite_float con validación de no-negatividad.
    Útil para observables físicos que representan magnitudes
    (energía, masa, distancia, probabilidad).

    Args:
        value: Valor arbitrario convertible a número.
        name: Nombre descriptivo para diagnóstico.

    Returns:
        Float finito ≥ 0.

    Raises:
        HilbertNumericalError: Si no finito o negativo.
    """
    result = _ensure_finite_float(value, name=name)
    if result < 0.0:
        raise HilbertNumericalError(
            f"{name} debe ser ≥ 0, recibido: {result}"
        )
    return result


def _clamp_probability(value: float) -> float:
    """Proyecta valor al intervalo cerrado [0, 1].

    Manejo explícito de valores especiales IEEE 754:
        - NaN → 0.0 (principio de máxima seguridad)
        - +Inf → 1.0
        - -Inf → 0.0

    Propiedades matemáticas:
        - Idempotente: clamp(clamp(x)) = clamp(x)
        - Monótona: x ≤ y ⟹ clamp(x) ≤ clamp(y) para x,y finitos
        - Proyección: clamp ∘ clamp = clamp (retracción)

    Args:
        value: Probabilidad sin restringir.

    Returns:
        Probabilidad p ∈ [0.0, 1.0].
    """
    if math.isnan(value):
        logger.warning("Probabilidad NaN detectada → 0.0")
        return 0.0
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _safe_context(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normaliza contexto de entrada a dict plano clonable.

    Garantiza que el resultado es un dict fresco (copia superficial)
    que puede ser extendido sin mutar el original.

    Args:
        context: Mapeo opcional de metadatos.

    Returns:
        Diccionario fresco listo para extensión.
    """
    if context is None:
        return {}

    if not isinstance(context, Mapping):
        warning_msg = (
            f"context no es Mapping: {type(context).__name__}"
        )
        logger.warning(warning_msg)
        return {"_context_warning": warning_msg}

    return dict(context)


# ═══════════════════════════════════════════════════════════════════════
# AGENTE OBSERVADOR DE HILBERT
# ═══════════════════════════════════════════════════════════════════════

class HilbertObserverAgent(Morphism):
    """Agente cuántico que ejecuta el ciclo OODA sobre CategoricalState.

    Semántica operacional (diagrama conmutativo):
        ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ OBSERVE  │───▶│  ORIENT  │───▶│  DECIDE  │───▶│   ACT    │
        │ E←H(ρ)   │    │ Φ,m,σ,F  │    │ T←WKB(·) │    │ |ψ⟩→|λ⟩  │
        └──────────┘    └──────────┘    └──────────┘    └──────────┘

    Propiedades categóricas:
        - Funtor covariante: preserva composición de morfismos MIC.
        - Sin estado mutable entre invocaciones (morfismo puro).
        - Dependencias inyectables para testeo e inversión de control.
        - Telemetría completa embebida en estado de salida.

    Invariantes de instancia:
        - _topo, _laplace, _sheaf son no-None y cumplen Protocol.
        - Toda invocación de __call__ es idempotente en el sentido
          de que el mismo state de entrada produce el mismo resultado
          (dado que los oráculos sean deterministas).
    """

    __slots__ = ("_topo", "_laplace", "_sheaf")

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return frozenset()

    @property
    def codomain(self) -> Stratum:
        return Stratum.PHYSICS

    def __init__(
        self,
        topo_watcher: ITopologicalWatcher,
        laplace_oracle: ILaplaceOracle,
        sheaf_orchestrator: ISheafCohomologyOrchestrator,
    ) -> None:
        """Inicializa el agente con dependencias verificadas.

        Args:
            topo_watcher: Monitor de amenazas topológicas.
            laplace_oracle: Oráculo espectral de estabilidad.
            sheaf_orchestrator: Coordinador de cohomología de haces.

        Raises:
            HilbertInterfaceError: Si alguna dependencia no cumple
                su Protocol o es None.
        """
        # Validar ANTES de asignar para garantizar construcción atómica
        self._validate_dependency(
            topo_watcher, "get_mahalanobis_threat", "topo_watcher"
        )
        self._validate_dependency(
            laplace_oracle, "get_dominant_pole_real", "laplace_oracle"
        )
        self._validate_dependency(
            sheaf_orchestrator, "get_global_frustration_energy",
            "sheaf_orchestrator"
        )

        self._topo = topo_watcher
        self._laplace = laplace_oracle
        self._sheaf = sheaf_orchestrator

        super().__init__()

    # ─────────────────────────────────────────────────────────────────
    # VALIDACIÓN DE CONTRATOS
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_dependency(
        obj: Any,
        method_name: str,
        obj_name: str,
    ) -> None:
        """Verifica que una dependencia cumple su contrato Protocol.

        Validaciones (en orden):
            1. obj no es None.
            2. obj posee atributo method_name.
            3. El atributo es invocable (callable).

        Args:
            obj: Instancia de dependencia a validar.
            method_name: Nombre del método requerido por Protocol.
            obj_name: Nombre legible para mensajes de error.

        Raises:
            HilbertInterfaceError: Si cualquier validación falla.
        """
        if obj is None:
            raise HilbertInterfaceError(
                f"{obj_name} no puede ser None"
            )

        if not hasattr(obj, method_name):
            raise HilbertInterfaceError(
                f"{obj_name} (tipo={type(obj).__name__}) carece del "
                f"método requerido '{method_name}'"
            )

        method = getattr(obj, method_name)
        if not callable(method):
            raise HilbertInterfaceError(
                f"{obj_name}.{method_name} no es invocable "
                f"(tipo={type(method).__name__})"
            )

    # ─────────────────────────────────────────────────────────────────
    # SERIALIZACIÓN DETERMINISTA CANÓNICA
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _serialize_payload(payload: Mapping[str, Any]) -> bytes:
        """Serializa payload de forma canónica determinista.

        Procedimiento:
            1. Verificar que payload es Mapping.
            2. Extraer pares (clave, valor).
            3. Normalizar claves a str, valores a repr canónico.
            4. Ordenar lexicográficamente por clave normalizada.
            5. Codificar la tupla ordenada como UTF-8.

        Determinismo:
            Dos payloads con las mismas claves (como str) y mismos
            valores (mismo repr) producen bytes idénticos. Esto es
            invariante entre ejecuciones del mismo intérprete Python.

        Nota sobre reproducibilidad inter-versión:
            repr() puede variar entre versiones de Python para
            algunos tipos (e.g., floats, sets). Para garantía
            absoluta inter-versión se recomienda serialización
            JSON canónica. La implementación actual prioriza
            captura de tipos complejos sobre portabilidad.

        Args:
            payload: Mapeo de datos incidentes.

        Returns:
            Representación binaria determinista UTF-8.

        Raises:
            HilbertPayloadError: Si payload no es Mapping, excede
                límite de tamaño, o falla la serialización.
        """
        if not isinstance(payload, Mapping):
            raise HilbertPayloadError(
                f"payload debe ser Mapping/dict; "
                f"recibido {type(payload).__name__}"
            )

        try:
            # Ordenamiento lexicográfico estricto para canonicidad
            ordered_items = tuple(
                sorted(
                    (str(k), repr(v))
                    for k, v in payload.items()
                )
            )
            serialized = repr(ordered_items).encode("utf-8")
        except Exception as exc:
            raise HilbertPayloadError(
                f"Falla en serialización determinista: {exc}"
            ) from exc

        # Validar tamaño DESPUÉS de serializar (medida precisa)
        if len(serialized) > QuantumThresholds.MAX_PAYLOAD_BYTES:
            raise HilbertPayloadError(
                f"payload serializado excede límite de "
                f"{QuantumThresholds.MAX_PAYLOAD_BYTES} bytes: "
                f"{len(serialized)} bytes"
            )

        return serialized

    # ─────────────────────────────────────────────────────────────────
    # OBSERVE: Estimación de energía incidente
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _byte_entropy_bits(data: bytes) -> float:
        """Calcula entropía de Shannon en bits de una secuencia de bytes.

        Fórmula:
            H(X) = -Σᵢ pᵢ log₂(pᵢ)

        donde pᵢ = count(byte=i) / len(data) para i ∈ {0,...,255},
        y la suma excluye términos con pᵢ = 0 (por convención 0·log₂(0) = 0,
        consistente con el límite lim_{p→0⁺} p·log₂(p) = 0).

        Propiedades:
            - H(X) = 0 sii data es constante (un solo byte distinto).
            - H(X) = 8 sii distribución uniforme sobre 256 símbolos.
            - H(X) ∈ [0, 8] para todo data no vacío.
            - H(∅) = 0 por convención.

        Args:
            data: Secuencia de bytes a analizar.

        Returns:
            Entropía H ∈ [0.0, 8.0] en bits.
        """
        if not data:
            return 0.0

        n = len(data)

        try:
            frequencies = np.bincount(
                np.frombuffer(data, dtype=np.uint8),
                minlength=256,
            )

            # Filtrar símbolos con frecuencia 0: su contribución
            # a la entropía es exactamente 0 (no requiere clamp)
            mask = frequencies > 0
            probabilities = frequencies[mask].astype(np.float64) / n

            # Cálculo directo sin clamp: probabilities > 0 garantizado
            # por mask, por lo que log₂ está bien definido.
            entropy = float(
                -np.dot(probabilities, np.log2(probabilities))
            )

        except Exception as exc:
            logger.error(
                "Falla en cálculo de entropía: %s", exc
            )
            return 0.0

        # Corrección de errores de redondeo en extremos
        if entropy < 0.0:
            # Teóricamente imposible, pero float64 puede producir
            # -0.0 o valores negativos minúsculos por cancelación
            entropy = 0.0

        return min(entropy, 8.0)

    def _observe_incident_wave(
        self, serialized_payload: bytes
    ) -> float:
        """[OBSERVE] Cuantifica energía incidente E del payload.

        Modelo fotoeléctrico discretizado:
            E = h·ν  donde  ν = N / max(H, H_floor) / SCALE

        Variables:
            - N = |serialized_payload| (información cruda, bytes)
            - H = entropía de Shannon (complejidad estructural, bits)
            - ν = "frecuencia" semántica (oscilaciones/escala)

        Interpretación física:
            Payloads grandes con baja entropía (repetitivos) tienen
            alta frecuencia → alta energía. Payloads comprimidos
            (alta entropía) distribuyen la energía.

        Args:
            serialized_payload: Bytes canónicos del payload.

        Returns:
            Energía incidente E ≥ 0 (adimensional).
        """
        size = len(serialized_payload)

        if size == 0:
            logger.debug("Payload vacío detectado, E = 0")
            return 0.0

        entropy = self._byte_entropy_bits(serialized_payload)
        entropy_safe = max(entropy, QuantumThresholds.ENTROPY_FLOOR)

        # Frecuencia semántica normalizada
        nu = size / (entropy_safe * QuantumThresholds.FREQUENCY_SCALE)
        energy = QuantumThresholds.PLANCK_H * nu

        energy = max(
            0.0, _ensure_finite_float(energy, name="incident_energy")
        )

        logger.debug(
            "[OBSERVE] E=%.6f, ν=%.4e, H=%.4f bits, N=%d bytes",
            energy, nu, entropy, size,
        )

        return energy

    # ─────────────────────────────────────────────────────────────────
    # ORIENT: Acoplamiento a oráculos estructurales
    # ─────────────────────────────────────────────────────────────────

    def _orient_gauge_fields(
        self,
    ) -> Tuple[float, float, bool, float, float, float]:
        """[ORIENT] Acopla variables de fase a oráculos estructurales.

        Interrogaciones y modulaciones:

        1. Amenaza topológica → Función de trabajo:
           Φ(threat) = Φ₀ + α · threat
           donde threat = d_Mahalanobis ≥ 0.
           Mayor amenaza ⟹ mayor barrera de potencial.

        2. Polo dominante → Masa efectiva:
           Si σ < -tol:  m_eff = m₀ / |σ|
           Si σ ≥ -tol:  m_eff = +∞ (barrera impenetrable)
           Interpretación: sistema inestable → partícula infinitamente
           pesada → tunneling imposible.

        3. Frustración cohomológica → Veto booleano:
           frustrated = (E_frustration > ε)
           Obstrucción en H¹(M, 𝒪) implica inconsistencia global.

        Returns:
            Tupla (Φ, m_eff, is_frustrated, threat, σ, E_frustration).

        Raises:
            HilbertNumericalError: Si observables violan invariantes.
            HilbertInterfaceError: Si oráculos fallan.
        """
        # --- Amenaza topológica (modula función de trabajo) ---
        try:
            threat_raw = self._topo.get_mahalanobis_threat()
        except Exception as exc:
            raise HilbertInterfaceError(
                f"topo_watcher.get_mahalanobis_threat() falló: {exc}"
            ) from exc

        threat = _ensure_nonneg_finite_float(
            threat_raw, name="mahalanobis_threat"
        )

        phi_t = (
            QuantumThresholds.BASE_PHI
            + QuantumThresholds.ALPHA_COUPLING * threat
        )
        phi_t = _ensure_nonneg_finite_float(
            phi_t, name="work_function"
        )

        # --- Polo dominante real (modula masa efectiva) ---
        try:
            sigma_raw = self._laplace.get_dominant_pole_real()
        except Exception as exc:
            raise HilbertInterfaceError(
                f"laplace_oracle.get_dominant_pole_real() falló: {exc}"
            ) from exc

        sigma = _ensure_finite_float(
            sigma_raw, name="dominant_pole_real"
        )

        if sigma >= -QuantumThresholds.SIGMA_CHAOS_TOL:
            # Marginalmente estable o inestable → barrera infranqueable
            m_eff = math.inf
        else:
            # Estable: |σ| > tol, por lo que denominador > 0
            m_eff = QuantumThresholds.BASE_MASS / abs(sigma)
            m_eff = _ensure_finite_float(
                m_eff, name="effective_mass"
            )
            if m_eff <= 0.0:
                raise HilbertNumericalError(
                    f"effective_mass no positiva: {m_eff} "
                    f"(σ={sigma}, BASE_MASS={QuantumThresholds.BASE_MASS})"
                )

        # --- Frustración cohomológica (veto estructural) ---
        try:
            frustration_raw = (
                self._sheaf.get_global_frustration_energy()
            )
        except Exception as exc:
            raise HilbertInterfaceError(
                f"sheaf_orchestrator."
                f"get_global_frustration_energy() falló: {exc}"
            ) from exc

        frustration = _ensure_nonneg_finite_float(
            frustration_raw, name="frustration_energy"
        )
        is_frustrated = frustration > QuantumThresholds.EPSILON_MACH

        logger.debug(
            "[ORIENT] Φ=%.4f, m_eff=%s, frustrated=%s, "
            "σ=%.6f, threat=%.4f, frustration=%.4e",
            phi_t,
            f"{m_eff:.4e}" if math.isfinite(m_eff) else "+∞",
            is_frustrated,
            sigma,
            threat,
            frustration,
        )

        return phi_t, m_eff, is_frustrated, threat, sigma, frustration

    # ─────────────────────────────────────────────────────────────────
    # DECIDE: Cálculo de probabilidad de transmisión
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _decide_quantum_transmission(
        E: float,
        Phi: float,
        m_eff: float,
        is_frustrated: bool,
    ) -> float:
        """[DECIDE] Calcula probabilidad de transmisión T.

        Reglas de decisión (evaluadas en orden de prioridad):

        R1. Veto cohomológico (is_frustrated = True):
            T = 0. Obstrucción topológica impide transmisión
            independientemente de la energía.

        R2. Masa infinita (sistema inestable):
            Si E ≥ Φ: T = 1 (transmisión fotoeléctrica clásica).
            Si E < Φ: T = 0 (barrera impenetrable, no hay túnel).

        R3. Transmisión fotoeléctrica clásica (E ≥ Φ, m_eff finita):
            T = 1. Energía suficiente para superar barrera.

        R4. Túnel WKB sub-umbral (E < Φ, m_eff finita):
            T = exp(-(2/ħ) · Δx · √(2·m_eff·(Φ-E)))

            Derivación: aproximación WKB para barrera rectangular
            de altura (Φ-E), ancho Δx, masa m_eff.

        Propiedades:
            - T ∈ [0, 1] garantizado por construcción.
            - Monótona creciente en E (a Φ, m_eff fijos).
            - Monótona decreciente en Φ (a E, m_eff fijos).
            - Monótona decreciente en m_eff (a E, Φ fijos).

        Args:
            E: Energía incidente (≥ 0).
            Phi: Función de trabajo efectiva (≥ 0).
            m_eff: Masa efectiva (> 0 ó +∞).
            is_frustrated: Veto cohomológico activo.

        Returns:
            T ∈ [0.0, 1.0].
        """
        # R1: Veto estructural absoluto
        if is_frustrated:
            logger.info(
                "[DECIDE] T=0: veto por frustración cohomológica"
            )
            return 0.0

        # R2: Masa infinita (barrera impenetrable al túnel)
        if math.isinf(m_eff):
            # Criterio de Routh-Hurwitz estricto: Si la masa es infinita (polo inestable),
            # el estado colapsa irrefutablemente a REJECTED (T = 0.0), sin importar E.
            T = 0.0
            logger.info(
                "[DECIDE] m_eff=+∞: T=%.1f (Colapso incondicional a REJECTED por polo inestable)",
                T,
            )
            return T

        # Validar masa efectiva positiva finita
        if m_eff <= 0.0:
            raise HilbertNumericalError(
                f"effective_mass no positiva en DECIDE: {m_eff}"
            )

        # R3: Transmisión fotoeléctrica clásica
        if E >= Phi:
            logger.info(
                "[DECIDE] T=1: modo fotoeléctrico "
                "(E=%.4f ≥ Φ=%.4f)", E, Phi,
            )
            return 1.0

        # R4: Túnel cuántico WKB
        barrier_height = Phi - E  # > 0 garantizado por E < Phi
        integrand = math.sqrt(max(0.0, 2.0 * m_eff * barrier_height))

        exponent = (
            -(2.0 / QuantumThresholds.PLANCK_HBAR)
            * QuantumThresholds.BARRIER_DX
            * integrand
        )

        # Protección contra underflow exponencial IEEE 754
        if exponent <= QuantumThresholds.EXP_UNDERFLOW_CUTOFF:
            logger.debug(
                "[DECIDE] Exponente=%.2f ≤ cutoff, T→0.0",
                exponent,
            )
            return 0.0

        T = _clamp_probability(math.exp(exponent))

        logger.debug(
            "[DECIDE] Modo túnel WKB: T=%.6e, "
            "exp=%.4f, barrier=%.4f, m_eff=%.4e",
            T, exponent, barrier_height, m_eff,
        )

        return T

    # ─────────────────────────────────────────────────────────────────
    # HASH DE COLAPSO DETERMINISTA
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_collapse_threshold(
        serialized_payload: bytes,
    ) -> float:
        """Genera umbral de colapso determinista ∈ [0, 1).

        Procedimiento:
            1. Computar SHA-256(serialized_payload).
            2. Extraer primeros 8 bytes como entero big-endian u64.
            3. Mapear linealmente a [0, 1) via n / 2^64.

        Propiedades:
            - Determinista: mismo payload → mismo umbral.
            - Uniformidad: SHA-256 produce distribución
              pseudo-uniforme sobre {0,...,2^256-1}, y la
              proyección a 64 bits preserva uniformidad.
            - Resolución: 2^64 valores distintos ≈ 5.4×10⁻²⁰
              de paso mínimo (vs 10⁻⁶ en versión anterior).
            - Rango: [0, 1) estrictamente, ya que n < 2^64
              implica n/2^64 < 1.

        Args:
            serialized_payload: Bytes canónicos del payload.

        Returns:
            Umbral τ ∈ [0.0, 1.0).
        """
        digest = hashlib.sha256(serialized_payload).digest()

        # Primeros 8 bytes → u64 (big-endian)
        n = int.from_bytes(digest[:8], byteorder="big", signed=False)

        # Mapeo lineal uniforme a [0, 1)
        # n ∈ {0, ..., 2^64 - 1} ⟹ n / 2^64 ∈ [0, 1)
        threshold = n / QuantumThresholds.HASH_RESOLUTION

        # Garantía defensiva (teóricamente innecesaria)
        if threshold >= 1.0:
            threshold = 1.0 - sys.float_info.epsilon
        if threshold < 0.0:
            threshold = 0.0

        return threshold

    # ─────────────────────────────────────────────────────────────────
    # ACT: Colapso de función de onda
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _act_collapse_wavefunction(
        state: CategoricalState,
        wave: WavefunctionState,
    ) -> CategoricalState:
        """[ACT] Ejecuta colapso determinista del estado cuántico.

        Regla de colapso (medición proyectiva):
            |ψ⟩ → |Admitido⟩  sii  T ≥ τ
            |ψ⟩ → |Rechazado⟩ sii  T < τ

        donde T = transmission_prob y τ = collapse_threshold.

        Post-condiciones:
            - El payload original se conserva intacto.
            - El contexto se enriquece con telemetría completa.
            - Si admitido: validated_strata incluye PHYSICS.
            - Si rechazado: validated_strata = ∅.

        Telemetría embebida (quantum_measurement):
            eigenstate, energy, work_function, effective_mass,
            transmission_prob, frustrated, threat_level,
            dominant_pole_real, frustration_energy,
            collapse_threshold, kinetic_energy, momentum, reason.

        Args:
            state: Estado categórico de entrada.
            wave: Estado de onda del ciclo OBSERVE-ORIENT-DECIDE.

        Returns:
            Nuevo CategoricalState con eigenstate colapsado.
        """
        context = _safe_context(getattr(state, "context", None))

        decision_admitted = (
            wave.transmission_prob >= wave.collapse_threshold
        )

        # Construir bloque de telemetría común
        measurement_base: Dict[str, Any] = {
            "energy": wave.energy,
            "work_function": wave.work_function,
            "effective_mass": (
                wave.effective_mass
                if math.isfinite(wave.effective_mass)
                else "+Inf"
            ),
            "transmission_prob": wave.transmission_prob,
            "frustrated": wave.frustrated,
            "threat_level": wave.threat_level,
            "dominant_pole_real": wave.dominant_pole_real,
            "frustration_energy": wave.frustration_energy,
            "collapse_threshold": wave.collapse_threshold,
        }

        if decision_admitted:
            return HilbertObserverAgent._collapse_to_admitted(
                state, wave, context, measurement_base,
            )
        else:
            return HilbertObserverAgent._collapse_to_rejected(
                state, wave, context, measurement_base,
            )

    @staticmethod
    def _collapse_to_admitted(
        state: CategoricalState,
        wave: WavefunctionState,
        context: Dict[str, Any],
        measurement_base: Dict[str, Any],
    ) -> CategoricalState:
        """Colapso al eigenestado |Admitido⟩.

        Cálculo de observables cinéticos:
            E_kin = max(E - Φ, ε)  si fotoeléctrico (E ≥ Φ)
            E_kin = ε              si túnel WKB (E < Φ)

            p = √(2 · m_eff_kinetic · E_kin)

        donde m_eff_kinetic es la masa efectiva apropiada para
        el cálculo de momentum (m_eff si finita, BASE_MASS si ∞).

        Justificación: cuando m_eff = +∞ y E ≥ Φ, la partícula
        se transmite clásicamente; usamos BASE_MASS como referencia
        para el momentum post-barrera (la masa infinita aplica
        solo al tunneling dentro de la barrera).
        """
        if wave.energy >= wave.work_function:
            kinetic_energy = max(
                QuantumThresholds.MIN_KINETIC_ENERGY,
                wave.energy - wave.work_function,
            )
            admission_reason = (
                f"Admisión fotoeléctrica clásica: "
                f"E={wave.energy:.6f} ≥ Φ={wave.work_function:.6f}"
            )
        else:
            kinetic_energy = QuantumThresholds.MIN_KINETIC_ENERGY
            admission_reason = (
                f"Admisión por túnel cuántico WKB: "
                f"T={wave.transmission_prob:.6e} ≥ "
                f"τ={wave.collapse_threshold:.6f}"
            )

        # Masa para cálculo de momentum post-barrera
        m_kinetic = (
            QuantumThresholds.BASE_MASS
            if math.isinf(wave.effective_mass)
            else wave.effective_mass
        )

        # Conservación de momento: p = √(2·m·E_kin)
        momentum = math.sqrt(2.0 * m_kinetic * kinetic_energy)
        momentum = _ensure_finite_float(
            momentum, name="quantum_momentum"
        )

        logger.info(
            "OODA [ACT]: |ψ⟩ → |Admitido⟩. "
            "p=%.6f, E=%.6f, Φ=%.6f, T=%.6e, "
            "τ=%.6f, σ=%.6f, threat=%.4f",
            momentum,
            wave.energy,
            wave.work_function,
            wave.transmission_prob,
            wave.collapse_threshold,
            wave.dominant_pole_real,
            wave.threat_level,
        )

        measurement = {
            **measurement_base,
            "eigenstate": HilbertEigenstate.ADMITTED.name,
            "kinetic_energy": kinetic_energy,
            "momentum": momentum,
            "reason": admission_reason,
        }

        new_context = {
            **context,
            "quantum_momentum": momentum,
            "quantum_measurement": measurement,
        }

        return CategoricalState(
            payload=state.payload,
            context=new_context,
            validated_strata=(
                state.validated_strata | {Stratum.PHYSICS}
            ),
        )

    @staticmethod
    def _collapse_to_rejected(
        state: CategoricalState,
        wave: WavefunctionState,
        context: Dict[str, Any],
        measurement_base: Dict[str, Any],
    ) -> CategoricalState:
        """Colapso al eigenestado |Rechazado⟩.

        Diagnóstico de causa raíz:
            - Frustración cohomológica: obstrucción en H¹(M, 𝒪).
            - Rechazo determinista: T < τ (barrera demasiado alta
              o payload con energía insuficiente).
        """
        if wave.frustrated:
            rejection_reason = (
                f"Veto cohomológico: obstrucción en H¹(M,𝒪), "
                f"E_frustration={wave.frustration_energy:.6e} > "
                f"ε={QuantumThresholds.EPSILON_MACH}"
            )
        else:
            rejection_reason = (
                f"Rechazo determinista: "
                f"T={wave.transmission_prob:.6e} < "
                f"τ={wave.collapse_threshold:.6f}"
            )

        logger.warning(
            "OODA [ACT]: |ψ⟩ → |Rechazado⟩. "
            "E=%.6f, Φ=%.6f, T=%.6e, τ=%.6f, "
            "frustrated=%s, σ=%.6f, threat=%.4f. "
            "Causa: %s",
            wave.energy,
            wave.work_function,
            wave.transmission_prob,
            wave.collapse_threshold,
            wave.frustrated,
            wave.dominant_pole_real,
            wave.threat_level,
            rejection_reason,
        )

        measurement = {
            **measurement_base,
            "eigenstate": HilbertEigenstate.REJECTED.name,
            "kinetic_energy": 0.0,
            "momentum": 0.0,
            "reason": rejection_reason,
        }

        new_context = {
            **context,
            "quantum_error": rejection_reason,
            "quantum_measurement": measurement,
        }

        return CategoricalState(
            payload=state.payload,
            context=new_context,
            validated_strata=frozenset(),
        )

    # ─────────────────────────────────────────────────────────────────
    # BUCLE OODA COMPLETO
    # ─────────────────────────────────────────────────────────────────

    def execute_ooda_loop(
        self, state: CategoricalState
    ) -> CategoricalState:
        """Punto de entrada principal del agente.

        Ejecuta secuencialmente el ciclo completo OODA con
        serialización única del payload.

        Idempotencia:
            Si el estado ya contiene evidencia de colapso ('quantum_measurement'
            en context) por haber sido procesado, se devuelve exactamente
            el mismo estado preservando la invarianza topológica.

            Serialize → Observe → Orient → Decide → Threshold → Act

        Pre-condiciones:
            - state es CategoricalState.
            - state.payload es Mapping.

        Post-condiciones:
            - Output conserva state.payload intacto.
            - Output.context contiene quantum_measurement completo.
            - Si admitido: Stratum.PHYSICS ∈ validated_strata.
            - Si rechazado: validated_strata = frozenset().

        Complejidad:
            O(|payload| · log(|payload|)) dominado por serialización
            y ordenamiento de claves.

        Args:
            state: Estado categórico de entrada.

        Returns:
            Estado colapsado con telemetría cuántica completa.

        Raises:
            HilbertWatcherError: Si pre-condiciones violadas.
            HilbertNumericalError: Si cálculos producen valores
                no finitos o fuera de rango.
            HilbertInterfaceError: Si oráculos fallan.
            HilbertPayloadError: Si payload no serializable.
        """
        # --- Validación de pre-condiciones ---
        if not isinstance(state, CategoricalState):
            raise HilbertWatcherError(
                f"state debe ser CategoricalState; "
                f"recibido {type(state).__name__}"
            )

        if not isinstance(state.payload, Mapping):
            raise HilbertWatcherError(
                f"state.payload debe ser Mapping/dict; "
                f"recibido {type(state.payload).__name__}"
            )

        # --- Verificación de Idempotencia de Medición ---
        # Si el estado ya colapsó bajo este operador, devolvemos el objeto inalterado.
        if "quantum_measurement" in state.context:
            return state

        # 0. SERIALIZACIÓN ÚNICA (evita doble cómputo)
        serialized = self._serialize_payload(state.payload)

        # 1. OBSERVE
        E = self._observe_incident_wave(serialized)

        # 2. ORIENT
        (Phi, m_eff, frustrated,
         threat, sigma, frustration) = self._orient_gauge_fields()

        # 3. DECIDE
        T = self._decide_quantum_transmission(
            E, Phi, m_eff, frustrated
        )

        # 4. UMBRAL DE COLAPSO
        collapse_threshold = self._compute_collapse_threshold(
            serialized
        )

        # Construir estado de onda consolidado (con validación)
        wave_state = WavefunctionState(
            energy=E,
            work_function=Phi,
            effective_mass=m_eff,
            transmission_prob=T,
            frustrated=frustrated,
            threat_level=threat,
            dominant_pole_real=sigma,
            frustration_energy=frustration,
            collapse_threshold=collapse_threshold,
        )

        # 5. ACT
        return self._act_collapse_wavefunction(state, wave_state)

    def __call__(
        self, state: CategoricalState
    ) -> CategoricalState:
        """Implementación del funtor Morphism para integración MIC.

        Este método hace al agente invocable como morfismo en la
        categoría de estados MIC:

            f: CategoricalState → CategoricalState

        Propiedades funtoriales:
            - Preserva identidad: f(id) opera sobre payload intacto.
            - Composición: puede componerse con otros Morphism.
        """
        return self.execute_ooda_loop(state)
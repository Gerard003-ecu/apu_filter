"""
=========================================================================================
Módulo: Quantum Admission Gate (Operador de Proyección de Hilbert y Barrera de Potencial)
Ubicación: app/physics/quantum_admission_gate.py
=========================================================================================

Naturaleza Ciber-Física y Mecánica Cuántica:
    Este módulo abandona la concepción de un filtro booleano para erigirse como el 
    Operador de Medición Cuántica en la frontera absoluta del sistema. 
    Precede axiomáticamente a la hidrodinámica electromagnética del `flux_condenser.py`, 
    garantizando que el flujo macroscópico jamás sea excitado por entropía sub-umbral 
    o superposiciones estocásticas del entorno exterior.

1. Efecto Fotoeléctrico Ciber-Físico y Acoplamiento de Gauge (Φ):
    Se modela la red de ingesta como un pozo de potencial gobernado por una Función 
    de Trabajo Φ(t). 
    [AXIOMA DE EXCITACIÓN]: Un cuanto de datos con energía incidente E = hν solo excita 
    el colector continuo si E ≥ Φ. Este umbral no es un escalar estático; está 
    acoplado como un campo de Gauge al Tensor Métrico Riemanniano del escudo topológico
    `Topological Watcher`. 
    Cualquier deformación geodésica en el sistema eleva exponencialmente la barrera Φ, 
    aniquilando la disipación por ruido.

2. Efecto Túnel y Masa Efectiva Espectral (Aproximación WKB):
    Para paquetes de emergencia cuya energía nominal es sub-umbral (E < Φ), el sistema 
    habilita el Efecto Túnel Cuántico computando la probabilidad de transmisión T 
    mediante la aproximación WKB.
    [RESTRICCIÓN LTI]: La masa efectiva (m_eff) de la función de onda no es constante; 
    es inversamente proporcional al polo dominante real (σ) dictaminado por el `Laplace Oracle`. 
    Si el sistema roza el caos determinista (σ → 0⁻), m_eff → ∞, aniquilando matemáticamente 
    la probabilidad de tunelamiento (T → 0) e impidiendo resonancias destructivas.

3. Veto Cohomológico y Colapso de Estado:
    La evaluación funge como el Hamiltoniano de Observación sobre el Espacio de Hilbert.
    Si el `Sheaf Cohomology Orchestrator` detecta una energía de Dirichlet degenerada 
    (frustración cohomológica global > ε), el operador interviene las condiciones de frontera 
    forzando un colapso determinista del 100% de la función de onda hacia el autoestado 
    puro |Rechazado⟩.

4. Transición Continua y Conservación de Momentum:
    Los cuantos informacionales que colapsan en el autoestado |Admitido⟩ inyectan su energía 
    cinética residual (K_max = E - Φ) como un momentum definido p = √(2m·K_max). 
    Este invariante físico provee las condiciones iniciales (t₀, v₀) ineludibles para las 
    diferencias finitas en el dominio del tiempo (FDTD) en el motor Port-Hamiltoniano posterior, 
    asegurando un isomorfismo perfecto entre la admisión discreta y la propagación continua.
=========================================================================================
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Final, Mapping, Optional

from app.core.mic_algebra import CategoricalState, Morphism
from app.core.schemas import Stratum

logger = logging.getLogger("MIC.Physics.QuantumAdmission")


# ======================================================================
# EXCEPCIONES
# ======================================================================


class QuantumAdmissionError(Exception):
    """Excepción base para fallos del operador de admisión cuántica."""


class QuantumNumericalError(QuantumAdmissionError):
    """Fallo numérico o entrada físicamente inválida en el cálculo cuántico."""


class QuantumInterfaceError(QuantumAdmissionError):
    """Fallo de contrato en dependencias inyectadas."""


# ======================================================================
# CONSTANTES Y TOLERANCIAS
# ======================================================================


class QuantumConstants:
    """
    Constantes físicas en unidades normalizadas de información.

    Todas las constantes son de clase y se tratan como inmutables.
    La anotación ``Final`` indica la intención de inmutabilidad;
    Python no la enforce en runtime para atributos de clase.
    """

    PLANCK_H: Final[float] = 1.0
    PLANCK_HBAR: Final[float] = PLANCK_H / (2.0 * math.pi)

    BASE_WORK_FUNCTION: Final[float] = 10.0
    BASE_EFFECTIVE_MASS: Final[float] = 1.0
    BARRIER_WIDTH: Final[float] = 1.0
    ALPHA_THREAT: Final[float] = 5.0

    MIN_KINETIC_ENERGY: Final[float] = 1e-12
    FRUSTRATION_VETO_TOL: Final[float] = 1e-9
    SIGMA_CHAOS_TOL: Final[float] = 1e-9
    ENTROPY_FLOOR: Final[float] = 1e-12
    EXP_UNDERFLOW_CUTOFF: Final[float] = -700.0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("QuantumConstants no debe ser subclaseada")


# ======================================================================
# ENUMERACIONES
# ======================================================================


class Eigenstate(Enum):
    """Autoestados del operador de medición."""

    ADMITIDO = auto()
    RECHAZADO = auto()


# ======================================================================
# ESTRUCTURAS DE DATOS
# ======================================================================


@dataclass(frozen=True, slots=True)
class QuantumMeasurement:
    """
    Registro inmutable del proceso de admisión cuántica.

    Attributes:
        eigenstate:
            Estado colapsado final.
        incident_energy:
            Energía incidente E ≥ 0.
        work_function:
            Función de trabajo efectiva Φ ≥ 0.
        tunneling_probability:
            Probabilidad de transmisión T ∈ [0, 1].
        kinetic_energy:
            Energía cinética residual K ≥ 0.
        momentum:
            Momentum de inyección p ≥ 0.
        frustration_veto:
            Si el rechazo se debió a veto cohomológico absoluto.
        effective_mass:
            Masa efectiva utilizada en WKB (puede ser +∞).
        dominant_pole_real:
            Polo dominante real σ usado para modular la masa.
        threat_level:
            Amenaza topológica Mahalanobis usada para modular Φ.
        collapse_threshold:
            Umbral determinista pseudoaleatorio derivado del payload ∈ [0, 1).
        admission_reason:
            Razón semántica de la decisión.
    """

    eigenstate: Eigenstate
    incident_energy: float
    work_function: float
    tunneling_probability: float
    kinetic_energy: float
    momentum: float
    frustration_veto: bool
    effective_mass: float
    dominant_pole_real: float
    threat_level: float
    collapse_threshold: float
    admission_reason: str

    def __repr__(self) -> str:
        return (
            f"QuantumMeasurement("
            f"eigenstate={self.eigenstate.name}, "
            f"E={self.incident_energy:.6f}, "
            f"Φ={self.work_function:.6f}, "
            f"T={self.tunneling_probability:.6e}, "
            f"K={self.kinetic_energy:.6e}, "
            f"p={self.momentum:.6e}, "
            f"veto={self.frustration_veto}, "
            f"reason='{self.admission_reason}')"
        )


# ======================================================================
# INTERFACES DE ACOPLAMIENTO
# ======================================================================


class ITopologicalWatcher:
    """Contrato mínimo para obtener la amenaza Mahalanobis."""

    def get_mahalanobis_threat(self) -> float: ...


class ILaplaceOracle:
    """Contrato mínimo para obtener el polo dominante real."""

    def get_dominant_pole_real(self) -> float: ...


class ISheafCohomologyOrchestrator:
    """Contrato mínimo para obtener energía global de frustración."""

    def get_global_frustration_energy(self) -> float: ...


# ======================================================================
# HELPERS NUMÉRICOS
# ======================================================================


def _ensure_finite_float(value: Any, *, name: str) -> float:
    """
    Convierte a float y verifica finitud.

    Raises:
        QuantumNumericalError: Si el valor no es convertible o no es finito.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise QuantumNumericalError(
            f"{name} no es convertible a float: {value!r}"
        ) from exc

    if not math.isfinite(result):
        raise QuantumNumericalError(
            f"{name} no es finito: {result!r}"
        )

    return result


def _clamp_probability(value: float) -> float:
    """
    Satura un valor a [0, 1].

    Valores NaN o ±inf se tratan como 0.0 (probabilidad nula por defecto
    ante error numérico), pero se registra un warning para NaN ya que
    indica un error de cálculo upstream.
    """
    if math.isnan(value):
        logger.warning(
            "Probabilidad NaN detectada, clamping a 0.0. "
            "Esto indica un error de cálculo upstream."
        )
        return 0.0
    if not math.isfinite(value):
        return 0.0
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _safe_context(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Normaliza un contexto a dict.

    Retorna dict vacío si es None, o dict con warning si no es Mapping.
    """
    if context is None:
        return {}
    if not isinstance(context, Mapping):
        return {
            "_context_warning": (
                f"context no mapeable recibido: {type(context).__name__}"
            )
        }
    return dict(context)


# ======================================================================
# OPERADOR DE ADMISIÓN CUÁNTICA
# ======================================================================


class QuantumAdmissionGate(Morphism):
    """
    Morfismo F: Ext → V_PHYSICS que aplica:
    - umbral fotoeléctrico discreto;
    - transmisión por túnel WKB;
    - veto estructural por frustración de haz.

    La decisión es determinista respecto del payload y el estado
    de los oracles inyectados.
    """

    def __init__(
        self,
        topo_watcher: ITopologicalWatcher,
        laplace_oracle: ILaplaceOracle,
        sheaf_orchestrator: ISheafCohomologyOrchestrator,
    ) -> None:
        super().__init__(name="QuantumAdmissionGate")
        self._topo_watcher = topo_watcher
        self._laplace_oracle = laplace_oracle
        self._sheaf_orchestrator = sheaf_orchestrator
        self._validate_dependencies()

    @property
    def domain(self) -> frozenset:
        return frozenset()

    @property
    def codomain(self) -> Stratum:
        return Stratum.PHYSICS

    # ------------------------------------------------------------------
    # VALIDACIÓN DE CONTRATOS
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """
        Verifica que las dependencias inyectadas cumplan sus contratos:
        - No nulas.
        - Métodos requeridos existen y son callable.
        """
        _CONTRACTS = (
            (self._topo_watcher, "get_mahalanobis_threat", "topo_watcher"),
            (self._laplace_oracle, "get_dominant_pole_real", "laplace_oracle"),
            (
                self._sheaf_orchestrator,
                "get_global_frustration_energy",
                "sheaf_orchestrator",
            ),
        )
        for obj, method_name, obj_name in _CONTRACTS:
            if obj is None:
                raise QuantumInterfaceError(
                    f"{obj_name} no puede ser None."
                )
            method = getattr(obj, method_name, None)
            if method is None or not callable(method):
                raise QuantumInterfaceError(
                    f"{obj_name} no implementa el método requerido "
                    f"'{method_name}'."
                )

    # ------------------------------------------------------------------
    # ENTROPÍA Y ENERGÍA INCIDENTE
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_payload(payload: Mapping[str, Any]) -> bytes:
        """
        Serialización determinista del payload.

        Produce una representación canónica basada en tupla ordenada
        por claves stringificadas. Esto garantiza:
        - Reproducibilidad del hash.
        - Independencia del orden de inserción del dict.

        Raises:
            QuantumAdmissionError: Si el payload no es Mapping o falla
                la serialización.
        """
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"payload debe ser un mapping/dict; "
                f"recibido {type(payload).__name__}."
            )

        try:
            ordered_items = tuple(
                sorted(
                    ((str(k), repr(v)) for k, v in payload.items()),
                    key=lambda kv: kv[0],
                )
            )
            return repr(ordered_items).encode("utf-8")
        except Exception as exc:
            raise QuantumAdmissionError(
                f"No fue posible serializar determinísticamente "
                f"el payload: {exc}"
            ) from exc

    @staticmethod
    def _byte_entropy(data: bytes) -> float:
        """
        Entropía de Shannon en base e (nats) sobre distribución de bytes.

            H = -Σ pᵢ ln pᵢ

        Propiedades garantizadas:
        - H ≥ 0 (clamped explícitamente).
        - H = 0 si ``data`` está vacío o todos los bytes son idénticos.
        - H ≤ ln(256) ≈ 5.545.
        """
        if not data:
            return 0.0

        n = len(data)
        counts = [0] * 256
        for byte_val in data:
            counts[byte_val] += 1

        entropy = 0.0
        for count in counts:
            if count == 0:
                continue
            p = count / n
            entropy -= p * math.log(p)

        # Clamp explícito por posibles errores de punto flotante.
        # No se necesita verificación intermedia: el clamp final
        # es suficiente y correcto.
        return max(0.0, entropy)

    def _calculate_incident_energy(
        self, payload: Mapping[str, Any]
    ) -> float:
        """
        Computa energía incidente E = hν.

        Modelo:
            ν = N / max(H, entropy_floor) / 1000

        donde:
        - N = tamaño en bytes de la serialización canónica
        - H = entropía de Shannon en nats

        La frecuencia efectiva ν captura:
        - Magnitud exergética (tamaño del payload).
        - Penalización por estructura informacional difusa (entropía alta).

        Returns:
            E ≥ 0.0 (finito).
        """
        data = self._serialize_payload(payload)
        payload_size = len(data)

        if payload_size == 0:
            return 0.0

        entropy = self._byte_entropy(data)
        effective_entropy = max(entropy, QuantumConstants.ENTROPY_FLOOR)
        nu = (payload_size / effective_entropy) / 1000.0

        raw_energy = QuantumConstants.PLANCK_H * nu
        return max(
            0.0, _ensure_finite_float(raw_energy, name="incident_energy")
        )

    # ------------------------------------------------------------------
    # MODULACIONES DE FASE
    # ------------------------------------------------------------------

    def _modulate_work_function(self) -> tuple[float, float]:
        """
        Calcula función de trabajo efectiva y nivel de amenaza.

            Φ = Φ₀ + α · threat

        donde ``threat`` es la amenaza Mahalanobis (clamped ≥ 0).

        Returns:
            Tupla ``(Φ, threat)`` con Φ ≥ 0 y threat ≥ 0.
        """
        threat_raw = self._topo_watcher.get_mahalanobis_threat()
        threat = max(
            0.0,
            _ensure_finite_float(threat_raw, name="mahalanobis_threat"),
        )

        phi = (
            QuantumConstants.BASE_WORK_FUNCTION
            + QuantumConstants.ALPHA_THREAT * threat
        )
        phi = max(
            0.0, _ensure_finite_float(phi, name="work_function")
        )

        return phi, threat

    def _modulate_effective_mass(self) -> tuple[float, float]:
        """
        Calcula masa efectiva modulada por el polo dominante real.

            m_eff = m₀ / |σ|,  si σ < -tol
            m_eff = +∞,         si σ ≥ -tol

        donde σ es el polo dominante real del oráculo de Laplace.

        La masa infinita se maneja explícitamente en
        ``_compute_wkb_tunneling_probability`` como barrera impenetrable.

        Returns:
            Tupla ``(m_eff, σ)`` con m_eff > 0 o m_eff = +∞.

        Raises:
            QuantumNumericalError: Si m_eff calculada es ≤ 0 (no física).
        """
        sigma_raw = self._laplace_oracle.get_dominant_pole_real()
        sigma = _ensure_finite_float(
            sigma_raw, name="dominant_pole_real"
        )

        if sigma >= -QuantumConstants.SIGMA_CHAOS_TOL:
            return float("inf"), sigma

        m_eff = QuantumConstants.BASE_EFFECTIVE_MASS / abs(sigma)
        m_eff = _ensure_finite_float(m_eff, name="effective_mass")
        if m_eff <= 0.0:
            raise QuantumNumericalError(
                f"effective_mass no positiva: {m_eff}"
            )

        return m_eff, sigma

    # ------------------------------------------------------------------
    # TÚNEL WKB
    # ------------------------------------------------------------------

    def _compute_wkb_tunneling_probability(
        self, E: float, Phi: float, m_eff: float
    ) -> float:
        """
        Aproximación WKB para barrera rectangular efectiva.

            T ≈ exp( -(2/ℏ) Δx √(2 m_eff (Φ - E)) )

        Reglas de despacho:
        1. Si ``E ≥ Φ``: transmisión clásica, ``T = 1.0``
           (independiente de la masa).
        2. Si ``m_eff = +∞`` y ``E < Φ``: barrera impenetrable, ``T = 0.0``.
        3. Si ``E < Φ`` y ``m_eff`` finita: cálculo WKB con clamp a [0, 1].
        4. Si el exponente está por debajo del cutoff de underflow: ``T = 0.0``.

        Args:
            E: Energía incidente (≥ 0).
            Phi: Función de trabajo (≥ 0).
            m_eff: Masa efectiva (> 0 o +∞).

        Returns:
            T ∈ [0, 1].

        Raises:
            QuantumNumericalError: Si ``m_eff`` es finita y ≤ 0.
        """
        E = max(0.0, _ensure_finite_float(E, name="incident_energy"))
        Phi = max(
            0.0, _ensure_finite_float(Phi, name="work_function")
        )

        # Regla 1: transmisión clásica (E ≥ Φ)
        # La transmisión clásica es independiente de la masa:
        # si la energía supera la barrera, el cuanto pasa.
        if E >= Phi:
            return 1.0

        # Regla 2: masa infinita → barrera impenetrable
        if math.isinf(m_eff):
            return 0.0

        # Validación de masa finita
        m_eff = _ensure_finite_float(m_eff, name="effective_mass")
        if m_eff <= 0.0:
            raise QuantumNumericalError(
                f"effective_mass debe ser positiva, recibido {m_eff}"
            )

        # Regla 3: cálculo WKB
        barrier_height = Phi - E
        integrand = math.sqrt(max(0.0, 2.0 * m_eff * barrier_height))
        exponent = -(
            (2.0 / QuantumConstants.PLANCK_HBAR)
            * QuantumConstants.BARRIER_WIDTH
            * integrand
        )

        # Regla 4: protección de underflow
        if exponent <= QuantumConstants.EXP_UNDERFLOW_CUTOFF:
            return 0.0

        return _clamp_probability(math.exp(exponent))

    # ------------------------------------------------------------------
    # COLAPSO DETERMINISTA
    # ------------------------------------------------------------------

    def _compute_collapse_threshold(
        self, payload: Mapping[str, Any]
    ) -> float:
        """
        Umbral determinista ∈ [0, 1) derivado del hash SHA-256 del payload.

        El hash se trunca a los primeros 8 bytes (64 bits) para generar
        un entero que se reduce módulo 10⁶ y se normaliza.

        Propiedad: el mismo payload siempre produce el mismo umbral.
        """
        data = self._serialize_payload(payload)
        digest = hashlib.sha256(data).digest()
        integer_value = int.from_bytes(
            digest[:8], byteorder="big", signed=False
        )
        threshold = (integer_value % 1_000_000) / 1_000_000.0
        return _clamp_probability(threshold)

    # ------------------------------------------------------------------
    # ADMISIÓN
    # ------------------------------------------------------------------

    def evaluate_admission(
        self, payload: Mapping[str, Any]
    ) -> QuantumMeasurement:
        """
        Evalúa el operador de admisión cuántica sobre un payload.

        Flujo secuencial:
        1. Veto cohomológico estructural (frustración global).
        2. Cálculo de energía incidente E.
        3. Cálculo de función de trabajo Φ y amenaza topológica.
        4. Cálculo de masa efectiva m_eff.
        5. Probabilidad de transmisión WKB T.
        6. Colapso determinista: T ≥ umbral → admitido.
        7. Cálculo de energía cinética K y momentum p si admitido.

        El momentum de inyección usa la masa efectiva modulada:
            p = √(2 · m_eff_injection · K)

        donde ``m_eff_injection`` es ``m_eff`` si es finita,
        o ``BASE_EFFECTIVE_MASS`` como fallback si ``m_eff = +∞``
        (caso que no debería ocurrir en admisión clásica, pero se
        maneja defensivamente).

        Args:
            payload: Mapping con los datos del cuanto.

        Returns:
            QuantumMeasurement inmutable con todos los diagnósticos.

        Raises:
            QuantumAdmissionError: Si el payload no es Mapping.
            QuantumNumericalError: Si hay fallos numéricos irrecuperables.
        """
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"payload debe ser un mapping/dict; "
                f"recibido {type(payload).__name__}."
            )

        # --- Paso 1: Veto cohomológico ---
        frustration_raw = (
            self._sheaf_orchestrator.get_global_frustration_energy()
        )
        frustration = _ensure_finite_float(
            frustration_raw, name="global_frustration_energy"
        )

        if frustration > QuantumConstants.FRUSTRATION_VETO_TOL:
            # Aun en veto, capturamos datos diagnósticos disponibles
            threat_level = self._safe_read_threat()
            sigma = self._safe_read_sigma()

            return QuantumMeasurement(
                eigenstate=Eigenstate.RECHAZADO,
                incident_energy=0.0,
                work_function=0.0,
                tunneling_probability=0.0,
                kinetic_energy=0.0,
                momentum=0.0,
                frustration_veto=True,
                effective_mass=float("inf"),
                dominant_pole_real=sigma,
                threat_level=threat_level,
                collapse_threshold=1.0,
                admission_reason=(
                    "Veto estructural por frustración cohomológica "
                    f"global ({frustration:.6e} > "
                    f"{QuantumConstants.FRUSTRATION_VETO_TOL:.6e})."
                ),
            )

        # --- Paso 2–5: Cálculos de fase ---
        E = self._calculate_incident_energy(payload)
        Phi, threat = self._modulate_work_function()
        m_eff, sigma = self._modulate_effective_mass()
        T = self._compute_wkb_tunneling_probability(E, Phi, m_eff)
        collapse_threshold = self._compute_collapse_threshold(payload)

        # --- Paso 6: Colapso determinista ---
        admitted = T >= collapse_threshold

        if not admitted:
            return QuantumMeasurement(
                eigenstate=Eigenstate.RECHAZADO,
                incident_energy=E,
                work_function=Phi,
                tunneling_probability=T,
                kinetic_energy=0.0,
                momentum=0.0,
                frustration_veto=False,
                effective_mass=m_eff,
                dominant_pole_real=sigma,
                threat_level=threat,
                collapse_threshold=collapse_threshold,
                admission_reason=(
                    "Rechazo probabilístico determinista: "
                    f"T ({T:.6e}) < umbral ({collapse_threshold:.6f})."
                ),
            )

        # --- Paso 7: Admisión → calcular K y p ---
        if E >= Phi:
            kinetic_energy = max(
                QuantumConstants.MIN_KINETIC_ENERGY, E - Phi
            )
            admission_reason = "Admisión clásica fotoeléctrica: E ≥ Φ."
        else:
            kinetic_energy = QuantumConstants.MIN_KINETIC_ENERGY
            admission_reason = "Admisión por túnel WKB sub-umbral."

        # Masa para momentum: usar m_eff si finita, BASE como fallback
        m_eff_for_momentum = (
            m_eff
            if math.isfinite(m_eff)
            else QuantumConstants.BASE_EFFECTIVE_MASS
        )
        momentum = math.sqrt(
            2.0 * m_eff_for_momentum * kinetic_energy
        )
        momentum = _ensure_finite_float(momentum, name="momentum")

        return QuantumMeasurement(
            eigenstate=Eigenstate.ADMITIDO,
            incident_energy=E,
            work_function=Phi,
            tunneling_probability=T,
            kinetic_energy=kinetic_energy,
            momentum=momentum,
            frustration_veto=False,
            effective_mass=m_eff,
            dominant_pole_real=sigma,
            threat_level=threat,
            collapse_threshold=collapse_threshold,
            admission_reason=admission_reason,
        )

    def _safe_read_threat(self) -> float:
        """Lee threat level defensivamente para diagnóstico en veto."""
        try:
            raw = self._topo_watcher.get_mahalanobis_threat()
            return max(
                0.0, _ensure_finite_float(raw, name="threat_diag")
            )
        except (QuantumNumericalError, Exception):
            return 0.0

    def _safe_read_sigma(self) -> float:
        """Lee polo dominante defensivamente para diagnóstico en veto."""
        try:
            raw = self._laplace_oracle.get_dominant_pole_real()
            return _ensure_finite_float(raw, name="sigma_diag")
        except (QuantumNumericalError, Exception):
            return 0.0

    # ------------------------------------------------------------------
    # MORFISMO CATEGÓRICO
    # ------------------------------------------------------------------

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Aplica el filtro cuántico a un ``CategoricalState``.

        Si es admitido, agrega ``Stratum.PHYSICS`` a los strata validados
        y registra el momentum en el contexto.

        Si es rechazado, resetea los strata validados a vacío y registra
        el error en el contexto.

        Args:
            state: Estado categórico con payload Mapping.

        Returns:
            Nuevo CategoricalState transformado.

        Raises:
            QuantumAdmissionError: Si state o state.payload no son
                del tipo esperado.
        """
        if not isinstance(state, CategoricalState):
            raise QuantumAdmissionError(
                f"state debe ser CategoricalState; "
                f"recibido {type(state).__name__}."
            )

        payload = getattr(state, "payload", None)
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"state.payload debe ser mapping/dict; "
                f"recibido {type(payload).__name__}."
            )

        context = _safe_context(getattr(state, "context", None))
        measurement = self.evaluate_admission(payload)

        if measurement.eigenstate == Eigenstate.RECHAZADO:
            error_msg = (
                "VETO CUÁNTICO: "
                f"E={measurement.incident_energy:.6f}, "
                f"Φ={measurement.work_function:.6f}, "
                f"T={measurement.tunneling_probability:.6e}, "
                f"veto_frustración={measurement.frustration_veto}, "
                f"razón='{measurement.admission_reason}'."
            )
            logger.error(error_msg)

            new_context = {
                **context,
                "quantum_error": error_msg,
                "quantum_admission": measurement,
            }

            return CategoricalState(
                payload=payload,
                context=new_context,
                validated_strata=frozenset(),
            )

        logger.info(
            "Colapso cuántico a |Admitido⟩. "
            "p=%.6f, E=%.6f, Φ=%.6f, T=%.6e, razón=%s",
            measurement.momentum,
            measurement.incident_energy,
            measurement.work_function,
            measurement.tunneling_probability,
            measurement.admission_reason,
        )

        new_context = {
            **context,
            "quantum_momentum": measurement.momentum,
            "quantum_admission": measurement,
        }

        return CategoricalState(
            payload=payload,
            context=new_context,
            validated_strata=state.validated_strata | {Stratum.PHYSICS},
        )
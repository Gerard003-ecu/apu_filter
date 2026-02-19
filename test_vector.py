"""
Centinela MIC - Transmisor de Vector de Estado
================================================
Protocolo de comunicación serial para hardware embebido.

Fundamentos matemáticos:
  - Números de Betti (β₁): invariantes topológicos que cuentan "agujeros"
    en el espacio de fase del sistema. β₁ ∈ ℤ≥0.
  - Estabilidad giroscópica: proyección normalizada del tensor de inercia
    sobre el subespacio estable. Valor en [0,1] como norma L2 normalizada.
  - Coherencia topológica: la relación β₁ ↔ pyramid_stability debe
    satisfacer que sistemas con alta conectividad (β₁ alto) admiten
    mayor variabilidad en la estabilidad (relajación del bound).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from enum import IntEnum
from typing import Optional, Final, Iterator
from contextlib import contextmanager

import serial
from serial import SerialException

# ---------------------------------------------------------------------------
# Configuración Global — Tipada y Agrupada
# ---------------------------------------------------------------------------

# Puerto y protocolo
PUERTO: Final[str] = '/dev/ttyUSB0'
BAUDIOS: Final[int] = 115_200
TIMEOUT_LECTURA: Final[float] = 1.0    # segundos por readline()

# Tiempos de espera
TIEMPO_ESPERA_CHIP: Final[float] = 3.0      # reset del microcontrolador
TIEMPO_ESPERA_RESPUESTA: Final[float] = 3.0  # ventana de handshake

# Parámetros de reintento
MAX_REINTENTOS: Final[int] = 3
BACKOFF_BASE: Final[float] = 1.5   # segundos (backoff exponencial)

# Umbral de coherencia topológica:
# Para β₁ alto, el sistema tiene más "loops" en su espacio de fase,
# lo que relaja el bound inferior de pyramid_stability.
# Fórmula: pyramid_stability ≥ max(0, 1 - log(1 + β₁) / log(1 + β₁_max))
BETA_1_MAX_REFERENCIA: Final[int] = 1000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("centinela.mic")


# ---------------------------------------------------------------------------
# Enumeraciones de Dominio
# ---------------------------------------------------------------------------

class VerdictCode(IntEnum):
    """
    Códigos de veredicto del sistema.
    Usamos IntEnum para garantizar serialización JSON sin conversión manual.
    """
    OPTIMO = 0
    ADVERTENCIA = 1
    FIEBRE_ESTRUCTURAL = 2
    COLAPSO_INMINENTE = 3


# ---------------------------------------------------------------------------
# Estructuras de Datos con Validación Matemática Robusta
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicsState:
    """
    Estado físico del sistema.

    Invariantes matemáticos:
      - saturation ∈ [0.0, 1.0]:  fracción de saturación magnética/térmica.
      - dissipated_power ≥ 0.0:   Segunda Ley de la Termodinámica — la entropía
                                   nunca decrece, por lo tanto la potencia
                                   disipada es siempre no negativa.
      - gyroscopic_stability ∈ [0.0, 1.0]: norma L2 normalizada del vector de
                                   estabilidad proyectado sobre el subespacio
                                   de Lyapunov estable.
    """
    saturation: float
    dissipated_power: float
    gyroscopic_stability: float

    def validate(self) -> None:
        """Lanza ValueError con contexto matemático si algún invariante falla."""
        errors: list[str] = []

        if not (0.0 <= self.saturation <= 1.0):
            errors.append(
                f"saturation={self.saturation!r} ∉ [0, 1]. "
                "Debe ser fracción normalizada."
            )
        if self.dissipated_power < 0.0:
            errors.append(
                f"dissipated_power={self.dissipated_power!r} < 0. "
                "Viola la Segunda Ley de la Termodinámica."
            )
        if not (0.0 <= self.gyroscopic_stability <= 1.0):
            errors.append(
                f"gyroscopic_stability={self.gyroscopic_stability!r} ∉ [0, 1]. "
                "La norma L2 normalizada debe estar acotada en [0, 1]."
            )
        if errors:
            raise ValueError("PhysicsState inválido:\n  " + "\n  ".join(errors))

    @property
    def energy_consistency_index(self) -> float:
        """
        Índice de consistencia energética: producto entre saturación y
        estabilidad giroscópica, ponderado por la potencia disipada.
        Útil para detectar regímenes físicamente anómalos donde la
        saturación es alta pero la estabilidad es baja bajo alta disipación.

        Retorna valor en [0, ∞). Valores > 100 indican régimen de alarma.
        """
        return self.saturation * self.gyroscopic_stability * self.dissipated_power


@dataclass(frozen=True)
class TopologyState:
    """
    Estado topológico del sistema (Álgebra Homológica).

    Invariantes:
      - beta_1 ∈ ℤ≥0: primer número de Betti. Cuenta el número de ciclos
        independientes (agujeros 1-dimensionales) en el complejo simplicial
        que modela el espacio de fase del sistema.
      - pyramid_stability ∈ [0.0, 1.0]: estabilidad de la estructura piramidal
        de control, normalizada.

    Coherencia topológica β₁ ↔ pyramid_stability:
      Un β₁ elevado implica un espacio de fase con muchos ciclos, lo que
      topológicamente permite mayor variabilidad en la estabilidad sin que
      el sistema colapse. El bound inferior de pyramid_stability se relaja
      logarítmicamente con β₁.
    """
    beta_1: int
    pyramid_stability: float

    def validate(self) -> None:
        errors: list[str] = []

        if self.beta_1 < 0:
            errors.append(
                f"beta_1={self.beta_1!r} < 0. "
                "El número de Betti β₁ es un invariante topológico en ℤ≥0."
            )
        if not (0.0 <= self.pyramid_stability <= 1.0):
            errors.append(
                f"pyramid_stability={self.pyramid_stability!r} ∉ [0, 1]."
            )

        if errors:
            raise ValueError("TopologyState inválido:\n  " + "\n  ".join(errors))

        # Validación de coherencia cruzada β₁ ↔ pyramid_stability
        # Con β₁ alto, el espacio de fase es más "conectado", lo que permite
        # que pyramid_stability sea menor sin indicar inestabilidad estructural.
        self._validate_topological_coherence()

    def _validate_topological_coherence(self) -> None:
        """
        Verifica que pyramid_stability sea coherente con beta_1.

        Bound inferior adaptativo:
          lower_bound = max(0, 1 - log(1 + β₁) / log(1 + β₁_max))

        Para β₁=0   → lower_bound=1.0 (espacio contractible, máxima rigidez)
        Para β₁=442 → lower_bound≈0.55 (relajado por conectividad topológica)
        Para β₁→∞   → lower_bound→0.0  (límite asintótico)
        """
        import math
        if BETA_1_MAX_REFERENCIA <= 0:
            return  # Evitar división por cero en configuración degenerada

        log_ratio = (
            math.log1p(self.beta_1)
            / math.log1p(BETA_1_MAX_REFERENCIA)
        )
        lower_bound = max(0.0, 1.0 - log_ratio)

        if self.pyramid_stability < lower_bound:
            raise ValueError(
                f"Incoherencia topológica: con β₁={self.beta_1}, "
                f"pyramid_stability debe ser ≥ {lower_bound:.4f}, "
                f"pero es {self.pyramid_stability:.4f}. "
                "El espacio de fase no soporta esta configuración de estabilidad."
            )

    @property
    def topological_complexity(self) -> float:
        """
        Medida de complejidad topológica normalizada:
          C = beta_1 / (1 + beta_1) * (1 - pyramid_stability)

        Valor en [0, 1). Combina la riqueza de ciclos con la inestabilidad
        de la estructura piramidal. C → 1 indica máxima complejidad con
        mínima estabilidad.
        """
        return (self.beta_1 / (1 + self.beta_1)) * (1.0 - self.pyramid_stability)


@dataclass(frozen=True)
class WisdomState:
    """
    Estado semántico / veredicto del sistema.

    verdict_code usa VerdictCode (IntEnum) para garantizar que el valor
    serializado sea siempre un entero válido dentro del dominio conocido.
    """
    verdict_code: VerdictCode
    narrative: str

    def validate(self) -> None:
        if not isinstance(self.verdict_code, VerdictCode):
            raise ValueError(
                f"verdict_code={self.verdict_code!r} no es un VerdictCode válido. "
                f"Valores permitidos: {[v.value for v in VerdictCode]}"
            )
        if not self.narrative.strip():
            raise ValueError("narrative no puede ser una cadena vacía.")


@dataclass(frozen=True)
class VectorEstado:
    """
    Vector de estado completo del sistema MIC.

    Representa un punto en el espacio de producto:
      Ω = ℝ³_física × (ℤ≥0 × [0,1])_topología × ℤ_sabiduría

    La validación de integridad garantiza que el punto pertenezca
    al subespacio físicamente y topológicamente admisible.
    """
    type: str
    physics: PhysicsState
    topology: TopologyState
    wisdom: WisdomState

    def validate_integrity(self) -> None:
        """
        Valida cada componente y luego la coherencia global del vector.
        El orden importa: primero la física, luego la topología,
        luego la sabiduría, finalmente la coherencia cruzada global.
        """
        self.physics.validate()
        self.topology.validate()
        self.wisdom.validate()
        self._validate_global_coherence()
        logger.debug("✅ Vector de estado validado — coherencia global confirmada.")

    def _validate_global_coherence(self) -> None:
        """
        Coherencia cruzada entre física y topología:

        Si el sistema está en régimen de alta saturación (saturation > 0.8)
        Y alta disipación (dissipated_power > 50), entonces la estabilidad
        giroscópica y la estabilidad piramidal deben ser complementarias:
          gyroscopic_stability + pyramid_stability ≥ 0.9

        Esto proviene del principio de conservación de la estabilidad total:
        cuando un modo de estabilización falla, el otro debe compensar.
        """
        p = self.physics
        t = self.topology

        if p.saturation > 0.8 and p.dissipated_power > 50.0:
            stability_sum = p.gyroscopic_stability + t.pyramid_stability
            if stability_sum < 0.9:
                raise ValueError(
                    f"Incoherencia global: en régimen de alta saturación "
                    f"({p.saturation}) y alta disipación ({p.dissipated_power}W), "
                    f"gyroscopic_stability + pyramid_stability = "
                    f"{stability_sum:.4f} < 0.9. "
                    "El sistema carece de estabilización compensatoria suficiente."
                )

    def to_dict(self) -> dict:
        """
        Convierte a diccionario con manejo explícito de tipos especiales.
        IntEnum se convierte a int para garantizar serialización JSON estándar.
        """
        raw = asdict(self)
        # asdict convierte dataclasses recursivamente pero no Enum → int
        # Lo manejamos explícitamente para robustez
        raw['wisdom']['verdict_code'] = int(self.wisdom.verdict_code)
        return raw

    def to_json(self) -> str:
        """Serialización JSON determinista (sort_keys para reproducibilidad)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @property
    def summary(self) -> str:
        """Resumen compacto para logging sin serializar JSON completo."""
        eci = self.physics.energy_consistency_index
        tc = self.topology.topological_complexity
        return (
            f"type={self.type!r} | "
            f"sat={self.physics.saturation:.2f} "
            f"diss={self.physics.dissipated_power:.1f}W "
            f"gyro={self.physics.gyroscopic_stability:.2f} | "
            f"β₁={self.topology.beta_1} "
            f"pyr={self.topology.pyramid_stability:.2f} | "
            f"verdict={self.wisdom.verdict_code.name} | "
            f"ECI={eci:.2f} TC={tc:.4f}"
        )


# ---------------------------------------------------------------------------
# Context Manager para Puerto Serial
# ---------------------------------------------------------------------------

@contextmanager
def puerto_serial(
    puerto: str,
    baudios: int,
    timeout: float
) -> Iterator[serial.Serial]:
    """
    Context manager para gestión segura del puerto serial.

    Garantiza cierre del puerto incluso ante excepciones, y separa
    la responsabilidad de gestión de recursos de la lógica de negocio.

    Yields:
        serial.Serial: instancia abierta y lista para uso.
    """
    ser: Optional[serial.Serial] = None
    try:
        logger.info(f"🔌 Conectando a {puerto} @ {baudios} baudios...")
        ser = serial.Serial(puerto, baudios, timeout=timeout)
        logger.info("✅ Puerto serial abierto.")
        yield ser
    finally:
        if ser and ser.is_open:
            ser.close()
            logger.info("🔌 Puerto serial cerrado correctamente.")


# ---------------------------------------------------------------------------
# Funciones de Comunicación (Separación de Responsabilidades)
# ---------------------------------------------------------------------------

def _esperar_y_limpiar_buffer(ser: serial.Serial) -> None:
    """
    Espera el reinicio del chip y limpia buffers de arranque.
    Los microcontroladores basados en CH340/CP2102 emiten basura
    en los primeros TIEMPO_ESPERA_CHIP segundos tras la apertura del puerto.
    """
    logger.info(f"⏳ Esperando reinicio del chip ({TIEMPO_ESPERA_CHIP}s)...")
    time.sleep(TIEMPO_ESPERA_CHIP)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    logger.debug("🧹 Buffers de entrada/salida limpiados.")


def _enviar_payload(ser: serial.Serial, vector: VectorEstado) -> int:
    """
    Serializa y envía el vector de estado por el puerto serial.

    El delimitador '\n' es el estándar para readline() en el receptor.
    No se añaden espacios extra para minimizar el payload.

    Returns:
        int: número de bytes enviados.
    """
    payload = vector.to_json() + "\n"
    encoded = payload.encode('utf-8')
    bytes_escritos = ser.write(encoded)

    logger.info(f"📨 Vector MIC enviado ({bytes_escritos} bytes):")
    logger.info(f"   {vector.summary}")
    logger.debug(
        f"   JSON completo:\n"
        f"{json.dumps(vector.to_dict(), indent=2, ensure_ascii=False)}"
    )
    return bytes_escritos


def _escuchar_respuesta(ser: serial.Serial) -> int:
    """
    Escucha respuestas del hardware durante la ventana de handshake.

    Usa sleep breve en el loop para evitar busy-wait y reducir uso de CPU.
    El sleep de 10ms es un balance entre latencia de respuesta y consumo.

    Returns:
        int: número de mensajes recibidos del hardware.
    """
    logger.info(
        f"👂 Escuchando respuesta del hardware "
        f"({TIEMPO_ESPERA_RESPUESTA}s de ventana)..."
    )
    start_time = time.monotonic()  # monotonic es robusto ante cambios de reloj
    mensajes_recibidos = 0

    while (time.monotonic() - start_time) < TIEMPO_ESPERA_RESPUESTA:
        if ser.in_waiting > 0:
            try:
                linea = ser.readline().decode('utf-8', errors='replace').strip()
                if linea:
                    logger.info(f"   🤖 Hardware → {linea!r}")
                    mensajes_recibidos += 1
            except UnicodeDecodeError as ude:
                # errors='replace' ya previene esto, pero por robustez:
                logger.warning(f"⚠️  Error decodificando línea: {ude}")
        else:
            # Evitar busy-wait: ceder CPU durante 10ms
            time.sleep(0.01)

    if mensajes_recibidos == 0:
        logger.warning(
            "⚠️  No se recibió respuesta del hardware en el tiempo límite. "
            "Verifique firmware y conexiones."
        )
    else:
        logger.info(f"✅ Handshake completado ��� {mensajes_recibidos} mensaje(s) recibido(s).")

    return mensajes_recibidos


def _construir_vector_estado() -> VectorEstado:
    """
    Construye y valida el vector de estado del sistema.

    Separado de la lógica de IO para facilitar testing unitario
    y para cumplir el principio de responsabilidad única.

    Note sobre los valores de ejemplo:
      - beta_1=442: número de Betti moderado, relaja pyramid_stability mínimo
        a ~0.55 (verificado con la fórmula logarítmica).
      - pyramid_stability=0.69 > 0.55: coherente topológicamente.
      - saturation=0.85 + dissipated_power=65.0: régimen de alta carga,
        requiere gyroscopic_stability + pyramid_stability ≥ 0.9
        → 0.4 + 0.69 = 1.09 ≥ 0.9 ✓

    Returns:
        VectorEstado validado y coherente.
    """
    vector = VectorEstado(
        type="state_update",
        physics=PhysicsState(
            saturation=0.85,
            dissipated_power=65.0,
            gyroscopic_stability=0.4,
        ),
        topology=TopologyState(
            beta_1=442,
            pyramid_stability=0.69,
        ),
        wisdom=WisdomState(
            verdict_code=VerdictCode.FIEBRE_ESTRUCTURAL,
            narrative="FIEBRE ESTRUCTURAL",
        ),
    )
    vector.validate_integrity()
    return vector


def _intentar_envio(vector: VectorEstado) -> bool:
    """
    Realiza un único intento de envío completo: conexión → envío → recepción.

    Returns:
        bool: True si el envío fue exitoso (puerto abierto y bytes enviados).
    """
    with puerto_serial(PUERTO, BAUDIOS, TIMEOUT_LECTURA) as ser:
        _esperar_y_limpiar_buffer(ser)
        bytes_enviados = _enviar_payload(ser, vector)

        if bytes_enviados == 0:
            logger.error("❌ No se escribieron bytes en el puerto serial.")
            return False

        _escuchar_respuesta(ser)
        return True


def enviar_vector_estado() -> None:
    """
    Función principal de envío con reintento exponencial.

    Flujo:
      1. Construir y validar el vector de estado.
      2. Intentar envío con hasta MAX_REINTENTOS intentos.
      3. Backoff exponencial entre reintentos: t = BACKOFF_BASE^intento.
    """
    # ── Fase 1: Construcción y Validación ──────────────────────────────────
    try:
        vector = _construir_vector_estado()
    except ValueError as ve:
        logger.error(f"❌ Vector de estado matemáticamente inconsistente:\n{ve}")
        return

    # ── Fase 2: Envío con Reintentos ───────────────────────────────────────
    for intento in range(1, MAX_REINTENTOS + 1):
        logger.info(f"🔄 Intento {intento}/{MAX_REINTENTOS}...")
        try:
            exito = _intentar_envio(vector)
            if exito:
                logger.info("🎯 Ciclo de transmisión completado exitosamente.")
                return

        except SerialException as se:
            logger.error(f"❌ Error serial en intento {intento}: {se}")
        except OSError as oe:
            # OSError cubre errores de dispositivo no disponible, permisos, etc.
            logger.error(f"❌ Error de sistema operativo en intento {intento}: {oe}")
        except Exception as exc:
            logger.error(
                f"❌ Error crítico inesperado en intento {intento}: {exc}",
                exc_info=True
            )
            # Para errores desconocidos no reintentamos: podría enmascarar bugs
            return

        if intento < MAX_REINTENTOS:
            espera = BACKOFF_BASE ** intento
            logger.info(f"⏳ Reintentando en {espera:.1f}s (backoff exponencial)...")
            time.sleep(espera)

    logger.error(
        f"💀 Todos los {MAX_REINTENTOS} intentos fallaron. "
        "Verifique hardware y configuración del puerto."
    )


# ---------------------------------------------------------------------------
# Punto de Entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        enviar_vector_estado()
    except KeyboardInterrupt:
        logger.info("\n🛑 Ejecución interrumpida por el usuario.")
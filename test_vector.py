"""
test_vector.py — Transmisor MIC con Protocolo de Handshake Estricto
====================================================================
Protocolo de "Llamada y Respuesta" (revisión 2):

  CAMBIOS QUIRÚRGICOS v2:
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. Auto-Reset por DTR/RTS: Python fuerza el reinicio del ESP32  │
  │    electrónicamente tras abrir el puerto. Elimina la necesidad  │
  │    de reconectar el cable manualmente.                           │
  │                                                                  │
  │ 2. Beacon Flexible: En lugar de buscar una cadena exacta        │
  │    (acoplada a la versión del firmware), se detectan palabras   │
  │    clave semánticas: "SENTINEL" o "READY". Esto desacopla el    │
  │    script de Python de la versión específica del firmware C++.  │
  └─────────────────────────────────────────────────────────────────┘

Flujo completo:
  1. Python abre el puerto.
  2. Python fuerza reset via DTR/RTS (automático, sin intervención).
  3. Python descarta basura del bootloader (74880 baudios → ruido).
  4. Python detecta "SENTINEL" o "READY" → ESP32 confirmado listo.
  5. Python envía el JSON + flush.
  6. Python escucha ACK del firmware.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Final, Iterator, Optional

import serial
from serial import SerialException

# ---------------------------------------------------------------------------
# Configuración Global
# ---------------------------------------------------------------------------

PUERTO: Final[str] = "/dev/ttyUSB0"
BAUDIOS: Final[int] = 115_200

# Timeout por readline(): bajo para que el loop de beacon sea reactivo.
# 100ms es el balance óptimo entre latencia y consumo de CPU.
TIMEOUT_LECTURA: Final[float] = 0.1

# ── Parámetros de Auto-Reset DTR/RTS ────────────────────────────────────────
# El ciclo DTR/RTS replica el comportamiento del IDE de Arduino al
# presionar "Upload": baja DTR para señalizar reset, luego restaura.
# Los tiempos están calibrados para el CH340/CP2102 del ESP32.
RESET_DTR_PULSO: Final[float] = 0.1   # segundos en estado de reset
RESET_POST_ESPERA: Final[float] = 0.5  # segundos para que el bootloader actúe

# ── Palabras Clave del Beacon (Búsqueda Flexible) ───────────────────────────
# Semánticamente, cualquier firmware del Centinela debería identificarse
# con "SENTINEL" o indicar disponibilidad con "READY".
# Usar .upper() en la comparación hace la detección case-insensitive.
BEACON_KEYWORDS: Final[tuple[str, ...]] = ("SENTINEL", "READY")

# ── Timeouts de Handshake ────────────────────────────────────────────────────
# El ESP32 tarda ~2-4s en reiniciar y ejecutar setup().
# 15s es un margen generoso para chips lentos o con setup() complejo.
TIMEOUT_BEACON: Final[float] = 15.0
TIMEOUT_ACK: Final[float] = 5.0

# ── Reintentos ───────────────────────────────────────────────────────────────
MAX_REINTENTOS: Final[int] = 3
BACKOFF_BASE: Final[float] = 2.0

# ── Referencia Topológica ────────────────────────────────────────────────────
BETA_1_MAX_REFERENCIA: Final[int] = 1000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("centinela.mic.test")


# ---------------------------------------------------------------------------
# Dominio: Enumeraciones
# ---------------------------------------------------------------------------


class VerdictCode(IntEnum):
    """
    Dominio cerrado de veredictos del sistema.
    IntEnum garantiza serialización JSON como entero sin conversión manual.
    """

    OPTIMO = 0
    ADVERTENCIA = 1
    FIEBRE_ESTRUCTURAL = 2
    COLAPSO_INMINENTE = 3


# ---------------------------------------------------------------------------
# Dominio: Dataclasses con Validación Matemática
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicsState:
    """
    Estado físico del sistema.

    Invariantes:
      - saturation ∈ [0, 1]: fracción de saturación normalizada.
      - dissipated_power ≥ 0: Segunda Ley de la Termodinámica.
      - gyroscopic_stability ∈ [0, 1]: norma L2 normalizada del vector
        de estabilidad proyectado sobre el subespacio de Lyapunov estable.
    """

    saturation: float
    dissipated_power: float
    gyroscopic_stability: float

    def validate(self) -> None:
        errors: list[str] = []
        if not (0.0 <= self.saturation <= 1.0):
            errors.append(
                f"saturation={self.saturation!r} ∉ [0, 1]."
            )
        if self.dissipated_power < 0.0:
            errors.append(
                f"dissipated_power={self.dissipated_power!r} < 0. "
                "Viola la Segunda Ley de la Termodinámica."
            )
        if not (0.0 <= self.gyroscopic_stability <= 1.0):
            errors.append(
                f"gyroscopic_stability={self.gyroscopic_stability!r} ∉ [0, 1]."
            )
        if errors:
            raise ValueError(
                "PhysicsState inválido:\n  " + "\n  ".join(errors)
            )

    @property
    def energy_consistency_index(self) -> float:
        """
        ECI = saturation × gyroscopic_stability × dissipated_power.
        Detecta regímenes anómalos. Valores > 100 → régimen de alarma.
        """
        return (
            self.saturation
            * self.gyroscopic_stability
            * self.dissipated_power
        )


@dataclass(frozen=True)
class TopologyState:
    """
    Estado topológico del sistema (Álgebra Homológica).

    Invariantes:
      - beta_1 ∈ ℤ≥0: primer número de Betti.
      - pyramid_stability ∈ [0, 1]: estabilidad piramidal normalizada.

    Coherencia β₁ ↔ pyramid_stability:
      lower_bound = max(0, 1 − log(1+β₁) / log(1+β₁_max))
    """

    beta_1: int
    pyramid_stability: float

    def validate(self) -> None:
        errors: list[str] = []
        if self.beta_1 < 0:
            errors.append(
                f"beta_1={self.beta_1!r} < 0. β₁ ∈ ℤ≥0."
            )
        if not (0.0 <= self.pyramid_stability <= 1.0):
            errors.append(
                f"pyramid_stability={self.pyramid_stability!r} ∉ [0, 1]."
            )
        if errors:
            raise ValueError(
                "TopologyState inválido:\n  " + "\n  ".join(errors)
            )
        self._validate_topological_coherence()

    def _validate_topological_coherence(self) -> None:
        """
        Bound inferior adaptativo para pyramid_stability dado β₁.
        β₁=442, β₁_max=1000 → lower_bound ≈ 0.118 → 0.69 ✓
        """
        import math

        if BETA_1_MAX_REFERENCIA <= 0:
            return
        log_ratio = math.log1p(self.beta_1) / math.log1p(
            BETA_1_MAX_REFERENCIA
        )
        lower_bound = max(0.0, 1.0 - log_ratio)
        if self.pyramid_stability < lower_bound:
            raise ValueError(
                f"Incoherencia topológica: con β₁={self.beta_1}, "
                f"pyramid_stability ≥ {lower_bound:.4f} requerido, "
                f"pero es {self.pyramid_stability:.4f}."
            )

    @property
    def topological_complexity(self) -> float:
        """C = β₁/(1+β₁) × (1 − pyramid_stability) ∈ [0, 1)."""
        return (self.beta_1 / (1 + self.beta_1)) * (
            1.0 - self.pyramid_stability
        )


@dataclass(frozen=True)
class WisdomState:
    """Veredicto semántico del sistema."""

    verdict_code: VerdictCode
    narrative: str

    def validate(self) -> None:
        if not isinstance(self.verdict_code, VerdictCode):
            raise ValueError(
                f"verdict_code={self.verdict_code!r} no es VerdictCode válido."
            )
        if not self.narrative.strip():
            raise ValueError("narrative no puede ser cadena vacía.")


@dataclass(frozen=True)
class VectorEstado:
    """
    Vector de estado completo del sistema MIC.

    Punto en el espacio producto:
      Ω = ℝ³_física × (ℤ≥0 × [0,1])_topología × ℤ_sabiduría
    """

    type: str
    physics: PhysicsState
    topology: TopologyState
    wisdom: WisdomState

    def validate_integrity(self) -> None:
        """Valida componentes y coherencia global cruzada."""
        self.physics.validate()
        self.topology.validate()
        self.wisdom.validate()
        self._validate_global_coherence()
        logger.debug("✅ Integridad del vector confirmada.")

    def _validate_global_coherence(self) -> None:
        """
        Principio de estabilización compensatoria:
          Si sat > 0.8 AND diss > 50W:
            gyro + pyramid ≥ 0.9
          Valores actuales: 0.4 + 0.69 = 1.09 ≥ 0.9 ✓
        """
        p, t = self.physics, self.topology
        if p.saturation > 0.8 and p.dissipated_power > 50.0:
            stability_sum = p.gyroscopic_stability + t.pyramid_stability
            if stability_sum < 0.9:
                raise ValueError(
                    f"Incoherencia global: sat={p.saturation}, "
                    f"diss={p.dissipated_power}W → "
                    f"gyro + pyramid = {stability_sum:.4f} < 0.9."
                )

    def to_dict(self) -> dict:
        """Convierte a dict con IntEnum → int para serialización JSON."""
        raw = asdict(self)
        raw["wisdom"]["verdict_code"] = int(self.wisdom.verdict_code)
        return raw

    def to_json(self) -> str:
        """JSON determinista con sort_keys para reproducibilidad."""
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True
        )

    @property
    def summary(self) -> str:
        """Línea de log compacta."""
        return (
            f"type={self.type!r} | "
            f"sat={self.physics.saturation:.2f} "
            f"diss={self.physics.dissipated_power:.1f}W "
            f"gyro={self.physics.gyroscopic_stability:.2f} | "
            f"β₁={self.topology.beta_1} "
            f"pyr={self.topology.pyramid_stability:.2f} | "
            f"verdict={self.wisdom.verdict_code.name} | "
            f"ECI={self.physics.energy_consistency_index:.2f} "
            f"TC={self.topology.topological_complexity:.4f}"
        )


# ---------------------------------------------------------------------------
# Context Manager: Puerto Serial
# ---------------------------------------------------------------------------


@contextmanager
def puerto_serial(
    puerto: str,
    baudios: int,
    timeout: float,
) -> Iterator[serial.Serial]:
    """
    Gestión declarativa del puerto serial.

    Abre el puerto y garantiza su cierre incluso ante excepciones.
    El timeout bajo (0.1s) mantiene el loop de beacon reactivo.
    """
    ser: Optional[serial.Serial] = None
    try:
        logger.info(f"🔌 Abriendo {puerto} @ {baudios} baudios...")
        ser = serial.Serial(puerto, baudios, timeout=timeout)
        logger.info("✅ Puerto abierto.")
        yield ser
    finally:
        if ser and ser.is_open:
            ser.close()
            logger.info("🔌 Puerto serial cerrado.")


# ---------------------------------------------------------------------------
# CAMBIO QUIRÚRGICO 1: Auto-Reset por DTR/RTS
# ---------------------------------------------------------------------------


def _forzar_reset_hardware(ser: serial.Serial) -> None:
    """
    Fuerza el reinicio del ESP32 mediante el ciclo DTR/RTS.

    Este mecanismo replica exactamente lo que hace el IDE de Arduino
    cuando presionas "Subir": manipula las líneas de control del
    puerto serial para pulsar el pin EN (Enable/Reset) del ESP32.

    Secuencia del pulso:
      ┌─────────────┬───────┬──────────────────────────────────────────┐
      │ Señal       │ Valor │ Efecto en el ESP32                       │
      ├─────────────┼───────┼──────────────────────────────────────────┤
      │ DTR=False   │  HIGH │ Pin EN del ESP32 va a LOW → reset activo │
      │ RTS=True    │  LOW  │ GPIO0 va a LOW → modo bootloader         │
      ├─────────────┼───────┼──────────────────────────────────────────┤
      │ DTR=True    │  LOW  │ Pin EN vuelve a HIGH → chip arranca      │
      │ RTS=False   │  HIGH │ GPIO0 vuelve a HIGH → modo ejecución     │
      └─────────────┴───────┴──────────────────────────────────────────┘

    Nota sobre la lógica invertida:
      El CH340/CP2102 invierte la polaridad: DTR=False en pyserial
      produce HIGH en el pin físico, lo cual activa el reset del ESP32
      (activo en bajo con pull-up interno).

    Args:
        ser: Puerto serial ya abierto sobre el que se aplica el pulso.
    """
    logger.info("⚡ Forzando reinicio de hardware (DTR/RTS)...")

    # ── Paso 1: Activar reset ────────────────────────────────────────────────
    ser.setDTR(False)   # EN del ESP32 → LOW (reset activo)
    ser.setRTS(True)    # GPIO0 → LOW (modo bootloader)
    time.sleep(RESET_DTR_PULSO)

    # ── Paso 2: Liberar reset → el chip arranca ──────────────────────────────
    ser.setDTR(True)    # EN del ESP32 → HIGH (chip corre)
    ser.setRTS(False)   # GPIO0 → HIGH (modo ejecución normal)
    time.sleep(RESET_POST_ESPERA)

    logger.info(
        f"   Pulso DTR/RTS completado "
        f"({RESET_DTR_PULSO}s reset + {RESET_POST_ESPERA}s espera). "
        "ESP32 reiniciando..."
    )


# ---------------------------------------------------------------------------
# CAMBIO QUIRÚRGICO 2: Beacon Flexible por Palabras Clave
# ---------------------------------------------------------------------------


def _es_beacon(linea: str) -> bool:
    """
    Detecta si una línea del firmware es un beacon de disponibilidad.

    Criterio semántico (case-insensitive):
      La línea contiene "SENTINEL" → el firmware se identificó.
      La línea contiene "READY"    → el firmware declaró disponibilidad.

    Por qué búsqueda flexible en lugar de coincidencia exacta:
      - Desacopla el script de la versión específica del firmware.
      - "=== APU SENTINEL V1.2 ===" y "=== APU SENTINEL V3.0 ===" son
        igualmente válidos: ambos confirman que el Centinela está activo.
      - "READY — Esperando JSON por Serial @ 115200" también es válido.
      - Futura versión V4.0 funcionará sin cambiar este script.

    Args:
        linea: Cadena ya decodificada y con strip() aplicado.

    Returns:
        True si la línea contiene alguna keyword de BEACON_KEYWORDS.
    """
    linea_upper = linea.upper()
    return any(keyword in linea_upper for keyword in BEACON_KEYWORDS)


def _esperar_beacon(ser: serial.Serial) -> bool:
    """
    FASE 1 — Espera del Beacon de Firmware (Portero Flexible).

    Lee líneas del puerto descartando basura del bootloader hasta
    detectar un beacon semántico, o hasta agotar TIMEOUT_BEACON.

    La detección se hace en dos pasos:
      1. _es_beacon(linea) evalúa las palabras clave.
      2. Si True → logueamos el beacon y retornamos inmediatamente.
      3. Si False → descartamos en DEBUG (no contaminamos log operacional).

    Returns:
        True  → beacon detectado, ESP32 listo para recibir JSON.
        False → timeout agotado sin beacon válido.
    """
    logger.info(
        f"🔍 Escuchando beacon del firmware "
        f"(keywords={BEACON_KEYWORDS}, timeout={TIMEOUT_BEACON}s)..."
    )
    start = time.monotonic()
    lineas_basura = 0

    while (time.monotonic() - start) < TIMEOUT_BEACON:
        try:
            raw = ser.readline()
        except SerialException as se:
            logger.error(f"❌ Error leyendo beacon: {se}")
            return False

        if not raw:
            # Timeout de 100ms sin datos: ESP32 aún no emite nada.
            continue

        linea = raw.decode("utf-8", errors="replace").strip()

        if not linea:
            continue

        # ── CAMBIO QUIRÚRGICO 2: Verificación Flexible ──────────────────────
        if _es_beacon(linea):
            elapsed = time.monotonic() - start
            logger.info(
                f"🎯 BEACON DETECTADO en {elapsed:.2f}s — {linea!r}"
            )
            return True

        # ── Basura del Bootloader: silenciosa en DEBUG ───────────────────────
        lineas_basura += 1
        logger.debug(
            f"   🗑️  Bootloader/basura [{lineas_basura:03d}]: {linea!r}"
        )

    logger.error(
        f"⏰ TIMEOUT: ninguna línea coincidió con keywords={BEACON_KEYWORDS} "
        f"en {TIMEOUT_BEACON}s. "
        f"({lineas_basura} líneas descartadas). "
        "Verifique que el firmware esté cargado y que el setup() imprima "
        "SENTINEL o READY por Serial."
    )
    return False


# ---------------------------------------------------------------------------
# Fases 2 y 3 del Protocolo (sin cambios respecto a v1)
# ---------------------------------------------------------------------------


def _enviar_json(ser: serial.Serial, vector: VectorEstado) -> bool:
    """
    FASE 2 — Envío del JSON.

    Solo se invoca tras confirmar el beacon. El delimitador '\\n'
    es el terminador que usa readStringUntil('\\n') en el firmware.
    ser.flush() garantiza vaciado del buffer del SO antes de escuchar ACK.

    Returns:
        True  → bytes escritos correctamente.
        False → error de escritura.
    """
    payload = vector.to_json() + "\n"
    encoded = payload.encode("utf-8")

    try:
        bytes_escritos = ser.write(encoded)
        ser.flush()
    except SerialException as se:
        logger.error(f"❌ Error escribiendo JSON: {se}")
        return False

    logger.info(f"📨 JSON enviado ({bytes_escritos} bytes):")
    logger.info(f"   {vector.summary}")
    logger.debug(
        "   JSON completo:\n"
        + json.dumps(vector.to_dict(), indent=2, ensure_ascii=False)
    )
    return bytes_escritos > 0


def _esperar_ack(ser: serial.Serial) -> bool:
    """
    FASE 3 — Espera del ACK del Firmware.

    Lee todas las líneas durante TIMEOUT_ACK. Considera ACK exitoso
    si se recibe al menos una línea no vacía. sleep(0.01) evita busy-wait.

    Returns:
        True  → al menos una respuesta recibida.
        False → timeout sin respuesta (advertencia, no error fatal).
    """
    logger.info(f"👂 Esperando ACK del firmware (timeout={TIMEOUT_ACK}s)...")
    start = time.monotonic()
    respuestas: list[str] = []

    while (time.monotonic() - start) < TIMEOUT_ACK:
        if ser.in_waiting > 0:
            try:
                raw = ser.readline()
                linea = raw.decode("utf-8", errors="replace").strip()
                if linea:
                    logger.info(f"   🤖 Firmware → {linea!r}")
                    respuestas.append(linea)
            except SerialException as se:
                logger.error(f"❌ Error leyendo ACK: {se}")
                break
        else:
            time.sleep(0.01)  # Ceder CPU: sin busy-wait

    if respuestas:
        logger.info(
            f"✅ ACK recibido — {len(respuestas)} línea(s) del firmware."
        )
        return True

    logger.warning(
        "⚠️  Sin ACK del firmware. "
        "El JSON puede haberse procesado en silencio o perdido."
    )
    return False


# ---------------------------------------------------------------------------
# Construcción del Vector de Estado
# ---------------------------------------------------------------------------


def _construir_vector() -> VectorEstado:
    """
    Construye y valida el VectorEstado.

    Verificación de coherencia de los valores:
      β₁=442, β₁_max=1000:
        lower_bound ≈ 0.118 → pyramid_stability=0.69 ✓
      Alta carga (sat=0.85, diss=65W):
        gyro + pyramid = 0.4 + 0.69 = 1.09 ≥ 0.9 ✓
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


# ---------------------------------------------------------------------------
# Ciclo de Handshake con Auto-Reset
# ---------------------------------------------------------------------------


def _ejecutar_ciclo_handshake(vector: VectorEstado) -> bool:
    """
    Ejecuta un ciclo completo de handshake estricto con auto-reset:

      [0] Abrir puerto serial.
      [1] Forzar reset del ESP32 vía DTR/RTS  ← CAMBIO QUIRÚRGICO 1
      [2] Esperar beacon flexible              ← CAMBIO QUIRÚRGICO 2
      [3] Enviar JSON.
      [4] Esperar ACK.

    Returns:
        True  → ciclo completado (beacon + envío OK).
        False → fallo en Fase 0, 1 ó 2.
    """
    with puerto_serial(PUERTO, BAUDIOS, TIMEOUT_LECTURA) as ser:

        # ── Fase 0: Auto-Reset ───────────────────────────────────────────────
        _forzar_reset_hardware(ser)

        # ── Fase 1: Beacon ───────────────────────────────────────────────────
        beacon_ok = _esperar_beacon(ser)
        if not beacon_ok:
            logger.error(
                "🚫 Abortando: ESP32 no emitió beacon reconocible. "
                "Enviar JSON ahora garantizaría corrupción de datos."
            )
            return False

        # ── Fase 2: Envío ────────────────────────────────────────────────────
        envio_ok = _enviar_json(ser, vector)
        if not envio_ok:
            logger.error("🚫 Fallo en la escritura del JSON al puerto.")
            return False

        # ── Fase 3: ACK ──────────────────────────────────────────────────────
        _esperar_ack(ser)

        return True


# ---------------------------------------------------------------------------
# Punto de Entrada Principal con Reintentos Exponenciales
# ---------------------------------------------------------------------------


def enviar_vector_estado() -> None:
    """
    Función principal.

    1. Construye y valida el vector de estado matemáticamente.
    2. Ejecuta el protocolo de handshake con hasta MAX_REINTENTOS intentos.
    3. Backoff exponencial entre intentos: t = BACKOFF_BASE^intento.
    """
    # ── Construcción del Vector ──────────────────────────────────────────────
    try:
        vector = _construir_vector()
        logger.info(f"📦 Vector construido: {vector.summary}")
    except ValueError as ve:
        logger.error(f"❌ Vector matemáticamente inconsistente:\n{ve}")
        return

    # ── Ciclo de Reintentos ──────────────────────────────────────────────────
    for intento in range(1, MAX_REINTENTOS + 1):
        logger.info(
            f"\n{'='*60}\n"
            f"🔄 INTENTO {intento}/{MAX_REINTENTOS} — Handshake Estricto\n"
            f"{'='*60}"
        )
        try:
            if _ejecutar_ciclo_handshake(vector):
                logger.info("🎯 Transmisión completada exitosamente.")
                return

        except SerialException as se:
            logger.error(f"❌ Error serial en intento {intento}: {se}")
        except OSError as oe:
            # Cubre: permisos, dispositivo no disponible, cable desconectado
            logger.error(f"❌ Error del SO en intento {intento}: {oe}")
        except Exception as exc:
            logger.error(
                f"❌ Error inesperado en intento {intento}: {exc}",
                exc_info=True,
            )
            # Error desconocido: no reintentar para no enmascarar bugs
            return

        if intento < MAX_REINTENTOS:
            espera = BACKOFF_BASE**intento
            logger.info(
                f"⏳ Esperando {espera:.1f}s antes del intento {intento + 1} "
                f"(backoff 2^{intento})..."
            )
            time.sleep(espera)

    logger.error(
        f"\n💀 FALLO DEFINITIVO: {MAX_REINTENTOS} intentos agotados.\n"
        f"Verifique: firmware cargado, cable USB, permisos del puerto,\n"
        f"keywords esperadas en setup(): {BEACON_KEYWORDS}."
    )


# ---------------------------------------------------------------------------
# Punto de Entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        enviar_vector_estado()
    except KeyboardInterrupt:
        logger.info("\n🛑 Ejecución interrumpida por el usuario.")
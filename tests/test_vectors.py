"""
test_vector.py — Transmisor MIC con Protocolo Pasivo
=====================================================
Revisión 3: Protocolo Pasivo (sin manipulación DTR/RTS).

HISTORIAL DE DECISIONES DE DISEÑO:
  v1: Espera fija (sleep 3s) → condición de carrera con bootloader.
  v2: Auto-reset DTR/RTS     → ESP32 DOIT atrapado en DOWNLOAD_BOOT
                                por capricho del circuito CH340/CP2102.
  v3: Protocolo Pasivo       → Python NO toca DTR/RTS. Abre el puerto
                                en modo silencioso y espera que el
                                usuario presione EN físicamente.
                                Robusto ante cualquier variante de
                                circuito USB-Serial.

FLUJO:
  1. Python abre el puerto SIN tocar líneas de control (dsrdtr=False,
     rtscts=False). El chip NO se reinicia automáticamente.
  2. Python solicita al usuario que presione EN físicamente.
  3. Python escucha con timeout: descarta basura del bootloader y
     detecta beacon semántico (SENTINEL o READY).
  4. Python limpia el buffer de entrada y envía el JSON.
  5. Python escucha el ACK del firmware.
"""

from __future__ import annotations

import json
import logging
import math
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

# Timeout por readline(): 500ms como en la propuesta.
# Más alto que v2 (100ms) porque en modo pasivo no necesitamos
# reactividad extrema: el usuario tiene tiempo de presionar EN.
TIMEOUT_LECTURA: Final[float] = 0.5

# ── Beacon ───────────────────────────────────────────────────────────────────
# Palabras clave semánticas, case-insensitive.
# Desacopladas de la versión específica del firmware.
BEACON_KEYWORDS: Final[tuple[str, ...]] = ("SENTINEL", "READY")

# Tiempo máximo para esperar que el usuario presione EN y el firmware
# emita su beacon. 60s es generoso para el factor humano.
TIMEOUT_BEACON: Final[float] = 60.0

# Pausa entre beacon y envío: da tiempo al firmware para estabilizarse
# y vacía cualquier línea rezagada del arranque.
PAUSA_POST_BEACON: Final[float] = 0.2

# ── ACK ──────────────────────────────────────────────────────────────────────
TIMEOUT_ACK: Final[float] = 5.0

# ── Reintentos ───────────────────────────────────────────────────────────────
MAX_REINTENTOS: Final[int] = 3
BACKOFF_BASE: Final[float] = 2.0

# ── Referencia Topológica ─────────────────────────────────────────────────────
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
    Dominio cerrado de veredictos.
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
        de estabilidad sobre el subespacio de Lyapunov estable.
    """

    saturation: float
    dissipated_power: float
    gyroscopic_stability: float

    def validate(self) -> None:
        errors: list[str] = []
        if not (0.0 <= self.saturation <= 1.0):
            errors.append(f"saturation={self.saturation!r} ∉ [0, 1].")
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
        Detecta regímenes anómalos. ECI > 100 → régimen de alarma.
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
      - beta_1 ∈ ℤ≥0: primer número de Betti (ciclos independientes).
      - pyramid_stability ∈ [0, 1]: estabilidad piramidal normalizada.

    Coherencia β₁ ↔ pyramid_stability:
      lower_bound = max(0, 1 − log(1+β₁) / log(1+β₁_max))
    """

    beta_1: int
    pyramid_stability: float

    def validate(self) -> None:
        errors: list[str] = []
        if self.beta_1 < 0:
            errors.append(f"beta_1={self.beta_1!r} < 0. β₁ ∈ ℤ≥0.")
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

        β₁=442, β₁_max=1000:
          log_ratio = log(443)/log(1001) ≈ 0.882
          lower_bound = max(0, 1 − 0.882) ≈ 0.118
          pyramid_stability = 0.69 ≥ 0.118 ✓
        """
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
# Context Manager: Puerto Serial en Modo Pasivo
# ---------------------------------------------------------------------------


@contextmanager
def puerto_serial_pasivo(
    puerto: str,
    baudios: int,
    timeout: float,
) -> Iterator[serial.Serial]:
    """
    Abre el puerto serial en MODO PASIVO.

    Parámetros clave:
      dsrdtr=False → Python NO controla DTR automáticamente.
                     Evita el pulso involuntario que reinicia el ESP32
                     al abrir el puerto (comportamiento del CH340/CP2102).
      rtscts=False → Python NO controla RTS automáticamente.
                     Evita que GPIO0 del ESP32 sea tirado a LOW,
                     lo que lo pondría en modo DOWNLOAD_BOOT.

    Sin estos flags en False, pyserial puede emitir señales de control
    en el momento de Serial() que confunden al circuito de auto-reset
    de la placa DOIT DevKit, atrapando al chip en modo de programación.
    """
    ser: Optional[serial.Serial] = None
    try:
        logger.info(
            f"🔌 Abriendo {puerto} @ {baudios} baudios "
            f"[MODO PASIVO: dsrdtr=False, rtscts=False]..."
        )
        ser = serial.Serial(
            puerto,
            baudios,
            timeout=timeout,
            dsrdtr=False,   # No tocar DTR → no pulsar EN del ESP32
            rtscts=False,   # No tocar RTS → no pulsar GPIO0 del ESP32
        )
        logger.info("✅ Puerto abierto en modo pasivo. Chip NO perturbado.")
        yield ser
    finally:
        if ser and ser.is_open:
            ser.close()
            logger.info("🔌 Puerto serial cerrado.")


# ---------------------------------------------------------------------------
# Interfaz de Usuario: Solicitud de Reset Manual
# ---------------------------------------------------------------------------


def _solicitar_reset_manual() -> None:
    """
    Informa al usuario que debe presionar el botón EN físicamente.

    Usamos logger.info() en lugar de print() para mantener coherencia
    del canal de salida y que los timestamps sean visibles.
    El separador visual ayuda a que la instrucción no se pierda
    entre el flujo de logs.
    """
    separador = "=" * 60
    logger.info(separador)
    logger.info("👉 ACCIÓN REQUERIDA:")
    logger.info("   Presiona el botón 'EN' (Reset) de tu ESP32 AHORA.")
    logger.info(f"   Tienes {TIMEOUT_BEACON:.0f} segundos.")
    logger.info(separador)


# ---------------------------------------------------------------------------
# Fase 1: Espera de Beacon (Modo Pasivo, con Timeout)
# ---------------------------------------------------------------------------


def _es_beacon(linea: str) -> bool:
    """
    Detecta si una línea es un beacon de disponibilidad del firmware.

    Criterio semántico case-insensitive:
      "SENTINEL" → el firmware se identificó como Centinela.
      "READY"    → el firmware declaró disponibilidad explícita.

    Desacoplado de versiones: V1.2, V3.0, V4.0 → todos válidos.
    """
    linea_upper = linea.upper()
    return any(kw in linea_upper for kw in BEACON_KEYWORDS)


def _esperar_beacon(ser: serial.Serial) -> bool:
    """
    FASE 1 — Espera del Beacon con Timeout y sin Busy-Wait.

    A diferencia de la propuesta original (bucle infinito), esta
    versión usa time.monotonic() para garantizar que el loop termine
    incluso si el usuario no presiona EN o el firmware falla.

    Manejo de líneas:
      - Vacías tras decode+strip → ignoradas silenciosamente.
      - No-beacon → logueadas en INFO para que el usuario vea
        el proceso de arranque del chip en tiempo real.
        (A diferencia de v2 donde eran DEBUG: en modo pasivo,
        mostrarlas en INFO ayuda al usuario a saber que el chip
        está vivo y comunicándose.)
      - Beacon → detectado, retorno inmediato.

    Returns:
        True  → beacon detectado dentro del timeout.
        False → timeout agotado sin beacon.
    """
    logger.info(
        f"🔍 Escuchando beacon "
        f"(keywords={BEACON_KEYWORDS}, timeout={TIMEOUT_BEACON}s)..."
    )
    start = time.monotonic()
    lineas_vistas = 0

    while (time.monotonic() - start) < TIMEOUT_BEACON:
        try:
            raw = ser.readline()
        except SerialException as se:
            logger.error(f"❌ Error leyendo del puerto: {se}")
            return False

        if not raw:
            # readline() agotó su timeout de 500ms sin datos.
            # El chip aún no arrancó o el usuario no presionó EN.
            # No busy-wait: el timeout de readline() ya cede el hilo.
            continue

        linea = raw.decode("utf-8", errors="replace").strip()

        if not linea:
            continue

        lineas_vistas += 1

        # ── Verificación de Beacon ───────────────────────────────────────────
        if _es_beacon(linea):
            elapsed = time.monotonic() - start
            logger.info(
                f"🎯 BEACON DETECTADO en {elapsed:.2f}s "
                f"(línea #{lineas_vistas}): {linea!r}"
            )
            return True

        # ── Arranque del Chip: visible en INFO ───────────────────────────────
        # En modo pasivo mostramos el arranque en INFO (no DEBUG)
        # para que el usuario confirme visualmente que el chip vive.
        logger.info(f"   📡 Chip arrancando [{lineas_vistas:03d}]: {linea!r}")

    elapsed = time.monotonic() - start
    logger.error(
        f"⏰ TIMEOUT tras {elapsed:.1f}s: ninguna línea coincidió con "
        f"keywords={BEACON_KEYWORDS}. "
        f"({lineas_vistas} líneas recibidas). "
        "Verifique que presionó EN y que el firmware imprime "
        "SENTINEL o READY en su setup()."
    )
    return False


# ---------------------------------------------------------------------------
# Fase 2: Envío del JSON
# ---------------------------------------------------------------------------


def _enviar_json(ser: serial.Serial, vector: VectorEstado) -> bool:
    """
    FASE 2 — Limpieza de buffer y Envío del JSON.

    reset_input_buffer() descarta líneas rezagadas del arranque
    que llegaron entre el beacon y este punto.
    sleep(PAUSA_POST_BEACON) da margen al firmware para que
    su loop() esté activo y escuchando antes de que llegue el JSON.

    Returns:
        True  → bytes escritos correctamente.
        False → error de escritura.
    """
    logger.info("🧹 Limpiando buffer de entrada post-beacon...")
    ser.reset_input_buffer()
    time.sleep(PAUSA_POST_BEACON)

    payload = vector.to_json() + "\n"
    encoded = payload.encode("utf-8")

    try:
        bytes_escritos = ser.write(encoded)
        ser.flush()  # Vacía buffer del SO → garantiza transmisión completa
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


# ---------------------------------------------------------------------------
# Fase 3: Escucha del ACK
# ---------------------------------------------------------------------------


def _esperar_ack(ser: serial.Serial) -> bool:
    """
    FASE 3 — Escucha del ACK del Firmware.

    Registra todas las respuestas del firmware durante TIMEOUT_ACK.
    Considera éxito si el firmware responde con una línea que
    contiene "ACK" (coincidencia semántica, igual que el beacon).

    time.monotonic() es robusto ante ajustes de reloj del sistema.
    sleep(0.01) en ausencia de datos evita busy-wait.

    Returns:
        True  → ACK semántico recibido.
        False → timeout sin ACK (advertencia, no error fatal:
                el firmware puede procesar en silencio).
    """
    logger.info(f"👂 Esperando ACK del firmware (timeout={TIMEOUT_ACK}s)...")
    start = time.monotonic()
    respuestas: list[str] = []
    ack_recibido = False

    while (time.monotonic() - start) < TIMEOUT_ACK:
        if ser.in_waiting > 0:
            try:
                raw = ser.readline()
                linea = raw.decode("utf-8", errors="replace").strip()
                if linea:
                    logger.info(f"   🤖 Firmware → {linea!r}")
                    respuestas.append(linea)
                    if "ACK" in linea.upper():
                        ack_recibido = True
            except SerialException as se:
                logger.error(f"❌ Error leyendo ACK: {se}")
                break
        else:
            time.sleep(0.01)  # Ceder CPU: sin busy-wait

    if ack_recibido:
        logger.info(
            f"🏆 ACK confirmado — "
            f"El hardware procesó el vector con éxito. "
            f"({len(respuestas)} línea(s) recibidas en total.)"
        )
        return True

    if respuestas:
        logger.warning(
            f"⚠️  {len(respuestas)} respuesta(s) recibidas, "
            "pero ninguna contiene 'ACK'. "
            "El firmware procesó algo, pero sin confirmación explícita."
        )
    else:
        logger.warning(
            "⚠️  Sin respuesta del firmware en el tiempo límite. "
            "El JSON puede haberse perdido o el firmware no lo procesó."
        )
    return False


# ---------------------------------------------------------------------------
# Construcción del Vector de Estado
# ---------------------------------------------------------------------------


def _construir_vector() -> VectorEstado:
    """
    Construye y valida el VectorEstado.

    Verificación de coherencia:
      β₁=442 → lower_bound ≈ 0.118 → pyramid_stability=0.69 ✓
      sat=0.85 + diss=65W (alta carga):
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
# Ciclo Principal: Protocolo Pasivo
# ---------------------------------------------------------------------------


def _ejecutar_ciclo_pasivo(vector: VectorEstado) -> bool:
    """
    Ejecuta un ciclo completo del Protocolo Pasivo:

      [0] Abrir puerto en modo pasivo (sin tocar DTR/RTS).
      [1] Solicitar al usuario que presione EN físicamente.
      [2] Esperar beacon semántico con timeout de 60s.
      [3] Limpiar buffer + enviar JSON.
      [4] Esperar ACK del firmware.

    Returns:
        True  → ciclo completado (beacon + envío OK).
        False → fallo en cualquier fase.
    """
    with puerto_serial_pasivo(PUERTO, BAUDIOS, TIMEOUT_LECTURA) as ser:

        # ── Fase 0: Instrucción al Usuario ───────────────────────────────────
        _solicitar_reset_manual()

        # ── Fase 1: Beacon ───────────────────────────────────────────────────
        if not _esperar_beacon(ser):
            logger.error(
                "🚫 Abortando: ESP32 no emitió beacon reconocible. "
                "Enviar JSON ahora garantizaría corrupción de datos."
            )
            return False

        # ── Fase 2: Envío ────────────────────────────────────────────────────
        if not _enviar_json(ser, vector):
            logger.error("🚫 Fallo en la escritura del JSON al puerto.")
            return False

        # ── Fase 3: ACK ──────────────────────────────────────────────────────
        _esperar_ack(ser)

        return True


# ---------------------------------------------------------------------------
# Punto de Entrada con Reintentos Exponenciales
# ---------------------------------------------------------------------------


def enviar_vector_estado() -> None:
    """
    Función principal.

    1. Construye y valida el vector matemáticamente.
    2. Ejecuta el protocolo pasivo con hasta MAX_REINTENTOS intentos.
    3. Backoff exponencial entre intentos: t = BACKOFF_BASE^intento.
    """
    # ── Construcción ─────────────────────────────────────────────────────────
    try:
        vector = _construir_vector()
        logger.info(f"📦 Vector construido y validado: {vector.summary}")
    except ValueError as ve:
        logger.error(f"❌ Vector matemáticamente inconsistente:\n{ve}")
        return

    # ── Ciclo de Reintentos ───────────────────────────────────────────────────
    for intento in range(1, MAX_REINTENTOS + 1):
        logger.info(
            f"\n{'='*60}\n"
            f"🔄 INTENTO {intento}/{MAX_REINTENTOS} — Protocolo Pasivo\n"
            f"{'='*60}"
        )
        try:
            if _ejecutar_ciclo_pasivo(vector):
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
            return  # No reintentar ante errores desconocidos

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
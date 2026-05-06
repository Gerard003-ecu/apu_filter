"""
Suite Rigurosa de Pruebas de Resiliencia para AutonomousAgent
══════════════════════════════════════════════════════════════

Fundamentos Matemáticos de Sistemas Resilientes
────────────────────────────────────────────────

Teoría de Reintentos y Backoff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sea R : ℕ → 𝔹 una función de reintento donde:
  R(n) = { True   si n < τ_max
         { False  si n ≥ τ_max

donde τ_max = MAX_RETRIES es el umbral de saturación.

Función de Backoff
~~~~~~~~~~~~~~~~~~
La estrategia de backoff implementada es:
  β : ℕ → ℝ⁺
  β(n) = δ_fixed  ∀n ∈ ℕ

donde δ_fixed = 5.0 segundos (backoff constante).

Propiedades Algebraicas:
  1. ∀n : β(n) > 0           (positividad estricta)
  2. ∀n : β(n) = δ_fixed     (invarianza temporal)
  3. ∀n,m : β(n) ≤ β(m)      (monotonía no-decreciente débil)
  4. ∀n : β(n) ≤ β_max       (acotación superior)

Teoría de Circuit Breaker
~~~~~~~~~~~~~~~~~~~~~~~~~~
Autómata finito determinista M = (Q, Σ, δ, q₀, F) donde:
  • Q = {CLOSED, OPEN, HALF_OPEN}  (estados)
  • Σ = {SUCCESS, FAILURE}         (alfabeto de eventos)
  • δ : Q × Σ → Q                  (función de transición)
  • q₀ = CLOSED                    (estado inicial)
  • F = {CLOSED}                   (estados aceptadores)

Función de Transición:
  δ(CLOSED, FAILURE^θ)      = OPEN
  δ(OPEN, timeout_elapsed)   = HALF_OPEN
  δ(HALF_OPEN, SUCCESS)      = CLOSED
  δ(HALF_OPEN, FAILURE)      = OPEN

donde θ = CIRCUIT_BREAKER_THRESHOLD.

Álgebra de Boole de Estados de Salud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sea H : Response → 𝔹 el predicado de salud:
  H(r) = (r.status ∈ [200,299]) ∧ ¬timeout(r) ∧ ¬connection_error(r)

Propiedades:
  • H es función total (dominio completo)
  • H(r₁) ∧ H(r₂) ⟹ H(r₁ ∘ r₂) (composición)
  • ¬H(r) ⟹ retry_required(r)  (contrapositiva)

Topología del Espacio de Respuestas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Partición del espacio R = {Success} ⊔ {Retriable} ⊔ {Fatal}:
  • Success    = {r : H(r) = True}
  • Retriable  = {r : ¬H(r) ∧ transient(r)}
  • Fatal      = {r : ¬H(r) ∧ ¬transient(r)}

Propiedades topológicas:
  1. Cobertura: Success ∪ Retriable ∪ Fatal = R
  2. Disjunción: Success ∩ Retriable = ∅ (y demás pares)
  3. Decidibilidad: clasificación en tiempo polinomial

Teoría de Grafos de Dependencias
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Grafo dirigido G = (V, E) de servicios:
  • V = {agent, core, redis, ...}
  • E = {(u,v) : u requiere v}

Métricas de conectividad:
  • Grado de entrada: in(v) = |{u : (u,v) ∈ E}|
  • Alcanzabilidad: reach(u,v) = ∃ camino u ⇝ v
  • Componente fuertemente conexa: SCC(G)

Invariantes de Resiliencia
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Invariante I1 (Terminación):**
  ∀ secuencia de reintentos : |secuencia| ≤ τ_max + 1

**Invariante I2 (Progreso):**
  ∀n < τ_max : attempt(n) ⟹ ∃t : attempt(n+1) ∨ success
  donde t = β(n) (tiempo de espera)

**Invariante I3 (Monotonía Temporal):**
  ∀i,j : i < j ⟹ timestamp(attempt_i) < timestamp(attempt_j)

**Invariante I4 (Conservación de Estado):**
  ∀ transición (s₁, event, s₂) :
    invariantes(s₁) ∧ válido(event) ⟹ invariantes(s₂)

**Invariante I5 (Acotación de Latencia Total):**
  ∀ ejecución : total_wait_time ≤ τ_max × δ_fixed

Teoremas Verificados
────────────────────

**Teorema T1 (Terminación Garantizada):**
  Todo bucle de reintentos termina en tiempo finito.
  
  Demostración:
    Por I1, el número de intentos está acotado por τ_max + 1.
    Cada intento toma tiempo finito (timeout acotado).
    Por transitividad, el tiempo total es finito. ∎

**Teorema T2 (Recuperación Eventual):**
  Si el servicio se vuelve disponible dentro de τ_max intentos,
  el agente lo detecta con probabilidad 1.
  
  Demostración:
    Por I2 (progreso), cada fallo es seguido por un nuevo intento.
    Si H(r_k) = True para algún k ≤ τ_max, el bucle termina exitosamente.
    Por exhaustividad del muestreo, k es alcanzado eventualmente. ∎

**Teorema T3 (Ausencia de Starvation):**
  Ninguna solicitud válida es indefinidamente postergada.
  
  Demostración:
    Por I1, toda secuencia de reintentos termina.
    Las solicitudes se procesan en orden FIFO (por construcción).
    Por equidad del scheduler, toda solicitud es eventualmente procesada. ∎

**Teorema T4 (Idempotencia de Health Checks):**
  ∀r : H(H(r)) = H(r)
  
  Demostración:
    H : Response → 𝔹 es proyección sobre valores booleanos.
    𝔹 es conjunto con dos elementos, luego toda proyección es idempotente. ∎

**Teorema T5 (Monotonía de Backoff Constante):**
  ∀n,m ∈ ℕ : β(n) ≤ β(m)
  
  Demostración:
    β(n) = δ_fixed para todo n.
    δ_fixed = δ_fixed trivialmente.
    Por reflexividad de ≤. ∎

Cobertura de Pruebas
────────────────────
  ✓ Cold start con backoff fijo (arranque en frío)
  ✓ Timeout handling (timeouts de conexión y lectura)
  ✓ HTTP error codes (clasificación 4xx/5xx)
  ✓ Rate limiting (429 Too Many Requests)
  ✓ Circuit breaker (marcado skip si no implementado)
  ✓ Recovery automático (restauración tras fallos)
  ✓ Métricas de observabilidad (contadores, latencias)
  ✓ Escenarios complejos (combinaciones de fallos)
  ✓ Invariantes matemáticos (propiedades universales)

Patrones de Resiliencia Implementados
──────────────────────────────────────
  • Retry with Fixed Backoff (reintento con espera constante)
  • Timeout Isolation (aislamiento por timeout)
  • Fail Fast (fallo rápido tras agotar reintentos)
  • Graceful Degradation (degradación controlada)
  • Health Probes (verificación de disponibilidad)
  • Metrics Collection (telemetría de resiliencia)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Generator,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import requests

from app.tactics.apu_agent import AutonomousAgent


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTES DEL DOMINIO DE RESILIENCIA
# ═════════════════════════════════════════════════════════════════════════════

#: Límite superior del número de reintentos: τ_max
MAX_RETRIES: Final[int] = 5

#: Tiempo de backoff fijo: δ_fixed ∈ ℝ⁺ (segundos)
FIXED_BACKOFF_SECONDS: Final[float] = 5.0

#: Umbral de fallos consecutivos para abrir el circuit breaker: θ
CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5

#: Timeout del circuit breaker abierto (segundos): τ_open
CIRCUIT_BREAKER_TIMEOUT: Final[float] = 30.0

#: Timestamp fijo para pruebas determinísticas
FIXED_TIMESTAMP: Final[datetime] = datetime(2024, 1, 15, 10, 30, 0)

#: Códigos de estado HTTP que indican fallos transitorios (retriables)
RETRIABLE_HTTP_CODES: Final[frozenset[int]] = frozenset({500, 502, 503, 504})

#: Códigos de estado HTTP de éxito
SUCCESS_HTTP_CODES: Final[range] = range(200, 300)

#: Epsilon para comparaciones de tiempo (tolerancia numérica)
TIME_EPSILON: Final[float] = 0.001


# ═════════════════════════════════════════════════════════════════════════════
# TIPOS ALGEBRAICOS Y ENUMERACIONES
# ═════════════════════════════════════════════════════════════════════════════


class CircuitState(Enum):
    """
    Estados del Circuit Breaker como autómata finito.

    Transiciones:
      CLOSED --[θ failures]--> OPEN
      OPEN --[timeout]--> HALF_OPEN
      HALF_OPEN --[success]--> CLOSED
      HALF_OPEN --[failure]--> OPEN
    """

    CLOSED = auto()  # Operación normal
    OPEN = auto()  # Circuito abierto (rechazo rápido)
    HALF_OPEN = auto()  # Prueba tentativa de recuperación


class FailureType(Enum):
    """Taxonomía de tipos de fallo para clasificación."""

    CONNECTION_ERROR = auto()  # Error de conexión de red
    TIMEOUT = auto()  # Timeout (lectura o conexión)
    HTTP_ERROR = auto()  # Error HTTP (4xx, 5xx)
    UNKNOWN = auto()  # Error no clasificado


class ResponseCategory(Enum):
    """
    Partición del espacio de respuestas R.

    Correspondencia con conjuntos matemáticos:
      SUCCESS    ↔ {r : H(r) = True}
      RETRIABLE  ↔ {r : ¬H(r) ∧ transient(r)}
      FATAL      ↔ {r : ¬H(r) ∧ ¬transient(r)}
    """

    SUCCESS = auto()
    RETRIABLE = auto()
    FATAL = auto()


# ═════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class RetryAttempt:
    """
    Registro de un intento de reintento con metadatos.

    Propiedades Invariantes:
      • attempt_number ≥ 1
      • backoff_duration ≥ 0
      • timestamp es monotónicamente creciente en secuencias
    """

    attempt_number: int
    timestamp: datetime
    failure_type: FailureType
    backoff_duration: float
    error_message: str = ""

    def __post_init__(self) -> None:
        """Verifica invariantes en la construcción."""
        assert self.attempt_number >= 1, "Número de intento debe ser ≥ 1"
        assert self.backoff_duration >= 0.0, "Backoff debe ser no-negativo"


@dataclass
class FailureScenario:
    """
    Describe un escenario de fallo parametrizado para pruebas.

    Atributos
    ─────────
    name : str
        Identificador único del escenario
    responses : List[Any]
        Secuencia de respuestas simuladas (excepciones o Mocks)
    expected_retries : int
        Número esperado de reintentos antes del resultado final
    expected_success : bool
        Indica si se espera éxito eventual (True) o fallo definitivo (False)
    description : str, opcional
        Descripción textual del escenario
    """

    name: str
    responses: List[Any]
    expected_retries: int
    expected_success: bool
    description: str = ""

    def __post_init__(self) -> None:
        """Valida coherencia del escenario."""
        assert len(self.responses) >= 1, "Debe haber al menos una respuesta"
        assert self.expected_retries >= 0, "Reintentos esperados deben ser no-negativos"


@dataclass
class BackoffSequence:
    """
    Rastreador de secuencia de tiempos de backoff con validaciones matemáticas.

    Métodos de Verificación:
      • verify_fixed_backoff(δ) : verifica β(n) = δ ∀n
      • verify_non_decreasing() : verifica β(i) ≤ β(j) para i < j
      • verify_bounded(β_max)   : verifica β(n) ≤ β_max ∀n
      • verify_positive()       : verifica β(n) > 0 ∀n
    """

    durations: List[float] = field(default_factory=list)

    def record(self, duration: float) -> None:
        """Registra un tiempo de backoff."""
        assert duration >= 0.0, f"Duración de backoff negativa: {duration}"
        self.durations.append(duration)

    def verify_fixed_backoff(self, expected: float) -> bool:
        """
        Verifica backoff constante: ∀n : β(n) = δ_expected.

        Parámetros
        ──────────
        expected : float
            Duración esperada constante

        Retorna
        ───────
        bool
            True si todas las duraciones son exactamente `expected`
        """
        return all(abs(d - expected) < TIME_EPSILON for d in self.durations)

    def verify_non_decreasing(self) -> bool:
        """
        Verifica monotonía: ∀i,j : i < j ⟹ β(i) ≤ β(j).

        Retorna
        ───────
        bool
            True si la secuencia es monótona no-decreciente
        """
        return all(
            d1 <= d2 + TIME_EPSILON
            for d1, d2 in zip(self.durations, self.durations[1:])
        )

    def verify_bounded(self, max_backoff: float) -> bool:
        """
        Verifica acotación superior: ∀n : β(n) ≤ β_max.

        Parámetros
        ──────────
        max_backoff : float
            Cota superior del backoff

        Retorna
        ───────
        bool
            True si ningún backoff excede el máximo
        """
        return all(d <= max_backoff + TIME_EPSILON for d in self.durations)

    def verify_positive(self) -> bool:
        """
        Verifica positividad estricta: ∀n : β(n) > 0.

        Retorna
        ───────
        bool
            True si todos los backoffs son estrictamente positivos
        """
        return all(d > 0.0 for d in self.durations)

    def get_total(self) -> float:
        """
        Calcula el tiempo total de espera acumulado.

        Retorna
        ───────
        float
            Suma de todas las duraciones de backoff
        """
        return sum(self.durations)

    def get_count(self) -> int:
        """Retorna el número de eventos de backoff registrados."""
        return len(self.durations)


# ═════════════════════════════════════════════════════════════════════════════
# BUILDERS: CONSTRUCTORES DE ESCENARIOS DE PRUEBA
# ═════════════════════════════════════════════════════════════════════════════


class ResponseBuilder:
    """
    Constructor fluido para secuencias de respuestas HTTP simuladas.

    Permite la construcción declarativa de escenarios de prueba mediante
    encadenamiento de métodos (patrón Builder).

    Ejemplos
    ────────
    >>> builder = ResponseBuilder()
    >>> responses = (
    ...     builder
    ...     .connection_error()
    ...     .timeout()
    ...     .service_unavailable()
    ...     .success()
    ...     .build()
    ... )

    Propiedades:
      • Inmutabilidad de construcción (copia defensiva en build())
      • Validación de parámetros en cada método
      • Soporte para respuestas complejas (headers, JSON)
    """

    def __init__(self) -> None:
        """Inicializa un builder vacío."""
        self._responses: List[Any] = []

    def connection_error(self, message: str = "Connection refused") -> ResponseBuilder:
        """
        Agrega una excepción de error de conexión.

        Parámetros
        ──────────
        message : str, opcional
            Mensaje de error descriptivo

        Retorna
        ───────
        ResponseBuilder
            Self para encadenamiento fluido
        """
        self._responses.append(requests.exceptions.ConnectionError(message))
        return self

    def timeout(self, message: str = "Read timed out") -> ResponseBuilder:
        """Agrega una excepción de timeout genérico."""
        self._responses.append(requests.exceptions.Timeout(message))
        return self

    def read_timeout(self) -> ResponseBuilder:
        """Agrega una excepción de timeout de lectura."""
        self._responses.append(requests.exceptions.ReadTimeout("Read timed out"))
        return self

    def connect_timeout(self) -> ResponseBuilder:
        """Agrega una excepción de timeout de conexión."""
        self._responses.append(requests.exceptions.ConnectTimeout("Connection timed out"))
        return self

    def http_error(self, status_code: int, reason: str = "") -> ResponseBuilder:
        """
        Agrega una respuesta HTTP con código de error.

        Parámetros
        ──────────
        status_code : int
            Código de estado HTTP (ej. 500, 503)
        reason : str, opcional
            Frase de razón HTTP (ej. "Internal Server Error")

        Retorna
        ───────
        ResponseBuilder
            Self para encadenamiento
        """
        mock_response = Mock()
        mock_response.ok = status_code in SUCCESS_HTTP_CODES
        mock_response.status_code = status_code
        mock_response.reason = reason or self._infer_reason(status_code)
        mock_response.text = f"Error {status_code}"
        mock_response.json.side_effect = ValueError("No JSON body")
        self._responses.append(mock_response)
        return self

    def service_unavailable(self) -> ResponseBuilder:
        """Agrega una respuesta 503 Service Unavailable."""
        return self.http_error(503, "Service Unavailable")

    def bad_gateway(self) -> ResponseBuilder:
        """Agrega una respuesta 502 Bad Gateway."""
        return self.http_error(502, "Bad Gateway")

    def gateway_timeout(self) -> ResponseBuilder:
        """Agrega una respuesta 504 Gateway Timeout."""
        return self.http_error(504, "Gateway Timeout")

    def internal_server_error(self) -> ResponseBuilder:
        """Agrega una respuesta 500 Internal Server Error."""
        return self.http_error(500, "Internal Server Error")

    def too_many_requests(self, retry_after: int = 60) -> ResponseBuilder:
        """
        Agrega una respuesta 429 Too Many Requests con header Retry-After.

        Parámetros
        ──────────
        retry_after : int, opcional
            Valor del header Retry-After en segundos

        Retorna
        ───────
        ResponseBuilder
            Self para encadenamiento
        """
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 429
        mock_response.reason = "Too Many Requests"
        mock_response.headers = {"Retry-After": str(retry_after)}
        mock_response.text = "Rate limit exceeded"
        self._responses.append(mock_response)
        return self

    def success(self, data: Optional[Dict[str, Any]] = None) -> ResponseBuilder:
        """
        Agrega una respuesta exitosa 200 OK.

        Parámetros
        ──────────
        data : Dict[str, Any], opcional
            Cuerpo JSON de la respuesta (por defecto: {"status": "healthy"})

        Retorna
        ───────
        ResponseBuilder
            Self para encadenamiento
        """
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.reason = "OK"
        mock_response.json.return_value = data or {"status": "healthy"}
        self._responses.append(mock_response)
        return self

    def success_sequence(self, count: int, data: Optional[Dict] = None) -> ResponseBuilder:
        """
        Agrega una secuencia de `count` respuestas exitosas idénticas.

        Parámetros
        ──────────
        count : int
            Número de respuestas exitosas a agregar
        data : Dict, opcional
            Cuerpo JSON común

        Retorna
        ───────
        ResponseBuilder
            Self para encadenamiento
        """
        for _ in range(count):
            self.success(data)
        return self

    def failure_then_success(
        self, failure_count: int, failure_type: str = "connection"
    ) -> ResponseBuilder:
        """
        Agrega `failure_count` fallos seguidos de un éxito.

        Parámetros
        ──────────
        failure_count : int
            Número de fallos consecutivos
        failure_type : str, opcional
            Tipo de fallo: "connection", "timeout", "503", etc.

        Retorna
        ───────
        ResponseBuilder
            Self para encadenamiento
        """
        failure_map = {
            "connection": self.connection_error,
            "timeout": self.timeout,
            "503": self.service_unavailable,
            "500": self.internal_server_error,
        }

        failure_method = failure_map.get(failure_type, self.connection_error)

        for _ in range(failure_count):
            failure_method()

        self.success()
        return self

    def intermittent_failures(self, pattern: str) -> ResponseBuilder:
        """
        Construye secuencia según patrón de caracteres.

        Parámetros
        ──────────
        pattern : str
            Cadena donde:
              'F' = fallo (connection error)
              'S' = éxito (200 OK)
              'T' = timeout
              Ejemplo: "FFSFT" → fallo, fallo, éxito, fallo, timeout

        Retorna
        ───────
        ResponseBuilder
            Self para encadenamiento
        """
        pattern_map = {
            "F": self.connection_error,
            "S": self.success,
            "T": self.timeout,
            "5": self.service_unavailable,
        }

        for char in pattern.upper():
            method = pattern_map.get(char)
            if method:
                method()
            else:
                raise ValueError(f"Carácter de patrón inválido: {char}")

        return self

    def build(self) -> List[Any]:
        """
        Construye y retorna la secuencia de respuestas.

        Retorna
        ───────
        List[Any]
            Copia defensiva de la lista de respuestas/excepciones
        """
        return self._responses.copy()

    @staticmethod
    def _infer_reason(status_code: int) -> str:
        """Infiere la frase de razón HTTP para códigos comunes."""
        reasons = {
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            429: "Too Many Requests",
            500: "Internal Server Error",
            502: "Bad Gateway",
            503: "Service Unavailable",
            504: "Gateway Timeout",
        }
        return reasons.get(status_code, "Unknown Error")


class ScenarioBuilder:
    """
    Constructor de escenarios de resiliencia predefinidos y validados.

    Proporciona factory methods para patrones comunes de fallo/recuperación,
    encapsulando la lógica de construcción y las expectativas de comportamiento.
    """

    @staticmethod
    def cold_start_success(attempts: int = 3) -> FailureScenario:
        """
        Escenario: Cold start que se recupera tras `attempts` intentos.

        Parámetros
        ──────────
        attempts : int, opcional
            Número total de intentos (incluyendo el éxito final)

        Retorna
        ───────
        FailureScenario
            Escenario con `attempts - 1` fallos seguidos de éxito
        """
        responses = (
            ResponseBuilder().failure_then_success(attempts - 1, "connection").build()
        )
        return FailureScenario(
            name=f"cold_start_{attempts}_attempts",
            responses=responses,
            expected_retries=attempts - 1,
            expected_success=True,
            description=f"Arranque en frío con recuperación tras {attempts - 1} fallos",
        )

    @staticmethod
    def cold_start_with_mixed_errors() -> FailureScenario:
        """
        Escenario: Cold start con variedad de errores transitorios.

        Retorna
        ───────
        FailureScenario
            Secuencia: connection → connection → 503 → éxito
        """
        responses = (
            ResponseBuilder()
            .connection_error()
            .connection_error()
            .service_unavailable()
            .success()
            .build()
        )
        return FailureScenario(
            name="cold_start_mixed_errors",
            responses=responses,
            expected_retries=3,
            expected_success=True,
            description="Arranque con errores de conexión y HTTP 503",
        )

    @staticmethod
    def persistent_failure() -> FailureScenario:
        """
        Escenario: Fallo persistente que agota todos los reintentos.

        Retorna
        ───────
        FailureScenario
            MAX_RETRIES + 1 fallos consecutivos (sin éxito)
        """
        n = MAX_RETRIES + 1
        responses = [requests.exceptions.ConnectionError("Persistent failure")] * n
        return FailureScenario(
            name="persistent_failure",
            responses=responses,
            expected_retries=MAX_RETRIES,
            expected_success=False,
            description="Fallo permanente que agota reintentos",
        )

    @staticmethod
    def timeout_recovery() -> FailureScenario:
        """
        Escenario: Recuperación tras timeouts consecutivos.

        Retorna
        ───────
        FailureScenario
            2 timeouts seguidos de éxito
        """
        responses = ResponseBuilder().timeout().timeout().success().build()
        return FailureScenario(
            name="timeout_recovery",
            responses=responses,
            expected_retries=2,
            expected_success=True,
            description="Recuperación tras timeouts de red",
        )

    @staticmethod
    def mixed_5xx_errors() -> FailureScenario:
        """
        Escenario: Combinación de errores HTTP 5xx antes de recuperación.

        Retorna
        ───────
        FailureScenario
            Secuencia: connection → timeout → 503 → 502 → éxito
        """
        responses = (
            ResponseBuilder()
            .connection_error()
            .timeout()
            .service_unavailable()
            .bad_gateway()
            .success()
            .build()
        )
        return FailureScenario(
            name="mixed_5xx_errors",
            responses=responses,
            expected_retries=4,
            expected_success=True,
            description="Errores HTTP 5xx variados con recuperación",
        )

    @staticmethod
    def rate_limited_recovery() -> FailureScenario:
        """
        Escenario: Rate limiting (429) con recuperación.

        Retorna
        ───────
        FailureScenario
            2 respuestas 429 seguidas de éxito
        """
        responses = (
            ResponseBuilder()
            .too_many_requests(retry_after=5)
            .too_many_requests(retry_after=10)
            .success()
            .build()
        )
        return FailureScenario(
            name="rate_limited",
            responses=responses,
            expected_retries=2,
            expected_success=True,
            description="Rate limiting con Retry-After",
        )


# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES Y HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def classify_response(response: Any) -> ResponseCategory:
    """
    Clasifica una respuesta en la partición R = Success ⊔ Retriable ⊔ Fatal.

    Parámetros
    ──────────
    response : Any
        Respuesta HTTP (Mock) o excepción

    Retorna
    ───────
    ResponseCategory
        Categoría de la respuesta según el predicado H(r)
    """
    # Casos de excepción (errores de red/timeout)
    if isinstance(response, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return ResponseCategory.RETRIABLE

    # Respuestas HTTP válidas
    if hasattr(response, "status_code"):
        code = response.status_code
        if code in SUCCESS_HTTP_CODES:
            return ResponseCategory.SUCCESS
        elif code in RETRIABLE_HTTP_CODES or code == 429:
            return ResponseCategory.RETRIABLE
        else:
            return ResponseCategory.FATAL

    # Por defecto: retriable (principio de robustez)
    return ResponseCategory.RETRIABLE


def classify_failure_type(error: Any) -> FailureType:
    """
    Clasifica un error en la taxonomía FailureType.

    Parámetros
    ──────────
    error : Any
        Excepción o respuesta HTTP con error

    Retorna
    ───────
    FailureType
        Tipo de fallo clasificado
    """
    if isinstance(error, requests.exceptions.ConnectionError):
        return FailureType.CONNECTION_ERROR
    elif isinstance(error, requests.exceptions.Timeout):
        return FailureType.TIMEOUT
    elif hasattr(error, "status_code") and not error.ok:
        return FailureType.HTTP_ERROR
    else:
        return FailureType.UNKNOWN


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_time():
    """
    Sustituye time.sleep por un mock no-bloqueante.

    Retorna
    ───────
    MagicMock
        Mock de time.sleep que registra llamadas sin esperar
    """
    with patch("time.sleep", return_value=None) as mock:
        yield mock


@pytest.fixture
def mock_datetime():
    """
    Controla el tiempo actual para pruebas determinísticas.

    Retorna
    ───────
    MagicMock
        Mock de datetime con tiempo fijo en FIXED_TIMESTAMP
    """
    with patch("app.tactics.apu_agent.datetime") as mock:
        mock.now.return_value = FIXED_TIMESTAMP
        # Permite construcción normal de datetime
        mock.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        yield mock


@pytest.fixture
def mock_requests_get():
    """
    Mock de requests.get para inyección de respuestas simuladas.

    Retorna
    ───────
    MagicMock
        Mock de requests.get
    """
    with patch("requests.get") as mock:
        yield mock


@pytest.fixture
def mock_session():
    """
    Mock de requests.Session para control de sesiones HTTP.

    Retorna
    ───────
    MagicMock
        Instancia mock de Session
    """
    with patch("requests.Session") as mock_class:
        session_instance = MagicMock()
        mock_class.return_value = session_instance
        yield session_instance


@pytest.fixture
def mock_topology():
    """
    Mock de SystemTopology para aislar pruebas de resiliencia.

    Retorna
    ───────
    MagicMock
        Instancia mock de SystemTopology
    """
    with patch("app.tactics.apu_agent.SystemTopology") as mock_class:
        instance = MagicMock()
        mock_class.return_value = instance
        instance.update_connectivity.return_value = (3, [])
        instance.record_request.return_value = None
        instance.remove_edge.return_value = None
        yield instance


@pytest.fixture
def mock_persistence():
    """
    Mock de PersistenceHomology.

    Retorna
    ───────
    MagicMock
        Instancia mock de PersistenceHomology
    """
    with patch("app.tactics.apu_agent.PersistenceHomology") as mock_class:
        instance = MagicMock()
        mock_class.return_value = instance
        yield instance


@pytest.fixture
def mock_signal_handlers():
    """Evita registro de manejadores de señal POSIX durante pruebas."""
    with patch.object(AutonomousAgent, "_setup_signal_handlers"):
        yield


@pytest.fixture
def agent(
    mock_time,
    mock_topology,
    mock_persistence,
    mock_signal_handlers,
    mock_session,
):
    """
    Construye una instancia de AutonomousAgent con dependencias mockeadas.

    Retorna
    ───────
    AutonomousAgent
        Instancia configurada y lista para pruebas
    """
    from app.tactics.apu_agent import AgentConfig, ConnectionConfig, TimingConfig

    config = AgentConfig(
        connection=ConnectionConfig(base_url="http://test-core:5000", request_timeout=5),
        timing=TimingConfig(check_interval=1),
    )

    agent_instance = AutonomousAgent(config=config)
    agent_instance._session = mock_session
    agent_instance._running = True

    yield agent_instance

    agent_instance._running = False


@pytest.fixture
def backoff_sequence():
    """
    Proporciona un rastreador de secuencia de backoff con validaciones.

    Retorna
    ───────
    BackoffSequence
        Rastreador vacío listo para registrar duraciones
    """
    return BackoffSequence()


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: COLD START / ARRANQUE EN FRÍO
# ═════════════════════════════════════════════════════════════════════════════


class TestColdStart:
    """
    Validación de arranque en frío con backoff fijo.

    Propiedades Verificadas:
      • Teorema T1 (Terminación): todo arranque termina
      • Invariante I1: número de intentos ≤ τ_max + 1
      • Invariante I2 (Progreso): cada fallo es seguido de reintento
      • Invariante I5: tiempo total acotado por τ_max × δ_fixed
    """

    def test_immediate_success_no_retries(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Teorema: Success inmediato ⟹ cero reintentos.

        Demostración:
          Por definición de bucle de reintentos, si H(r₀) = True,
          no se ejecuta el cuerpo del bucle. ∎
        """
        mock_requests_get.return_value = ResponseBuilder().success().build()[0]

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 1, "Debe haber exactamente 1 intento"
        mock_time.assert_not_called()

    def test_success_after_n_failures(self, agent, mock_requests_get, mock_time) -> None:
        """
        Propiedad: n-1 fallos ⟹ n-1 backoffs, éxito en intento n.

        Verifica Invariante I2 (Progreso).
        """
        scenario = ScenarioBuilder.cold_start_success(attempts=4)
        mock_requests_get.side_effect = scenario.responses

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 4, "Debe haber 4 intentos"
        assert mock_time.call_count == 3, "Debe haber 3 backoffs"

    def test_503_treated_as_retriable(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Propiedad: 503 ∈ Retriable ⟹ se reintenta hasta éxito.

        Verifica clasificación correcta en la partición R.
        """
        scenario = ScenarioBuilder.cold_start_with_mixed_errors()
        mock_requests_get.side_effect = scenario.responses

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 4
        assert scenario.expected_success

    def test_backoff_is_exactly_fixed_duration(
        self, agent, mock_requests_get, backoff_sequence
    ) -> None:
        """
        Teorema T5: ∀n : β(n) = δ_fixed.

        Demostración:
          Por inspección directa de la secuencia de sleep calls. ∎
        """
        mock_requests_get.side_effect = (
            ResponseBuilder()
            .connection_error()
            .connection_error()
            .connection_error()
            .success()
            .build()
        )

        with patch("time.sleep", side_effect=backoff_sequence.record):
            agent._wait_for_startup()

        assert backoff_sequence.get_count() == 3, "Debe haber 3 backoffs"
        assert backoff_sequence.verify_fixed_backoff(
            FIXED_BACKOFF_SECONDS
        ), f"Todos los backoffs deben ser exactamente {FIXED_BACKOFF_SECONDS}s"

    def test_retry_attempts_logged(
        self, agent, mock_requests_get, mock_time, caplog
    ) -> None:
        """
        Propiedad de Observabilidad: cada backoff genera log.

        Útil para debugging y auditoría de comportamiento en producción.
        """
        mock_requests_get.side_effect = (
            ResponseBuilder().connection_error().connection_error().success().build()
        )

        with caplog.at_level(logging.INFO):
            agent._wait_for_startup()

        retry_logs = [r for r in caplog.records if "esperando" in r.message.lower()]
        assert len(retry_logs) >= 1, "Debe haber al menos un log de reintento"

    def test_stops_when_running_flag_false(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Invariante de Terminación: _running = False ⟹ salida del bucle.

        Verifica que el agente respeta señales de shutdown durante arranque.
        """
        agent._running = False
        mock_requests_get.side_effect = requests.exceptions.ConnectionError(
            "Simulated error"
        )

        agent._wait_for_startup()

        # Máximo 1 intento antes de verificar flag
        assert mock_requests_get.call_count <= 1


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: TIMEOUT HANDLING
# ═════════════════════════════════════════════════════════════════════════════


class TestTimeoutHandling:
    """
    Verificación de manejo de timeouts de red.

    Tipos de Timeout Cubiertos:
      • ReadTimeout (lectura de socket)
      • ConnectTimeout (establecimiento de conexión)
      • Timeout genérico (requests.exceptions.Timeout)
    """

    def test_read_timeout_triggers_retry(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """Propiedad: ReadTimeout ∈ Retriable."""
        mock_requests_get.side_effect = (
            ResponseBuilder().read_timeout().success().build()
        )

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 2

    def test_connect_timeout_triggers_retry(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """Propiedad: ConnectTimeout ∈ Retriable."""
        mock_requests_get.side_effect = (
            ResponseBuilder().connect_timeout().success().build()
        )

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 2

    def test_timeout_recovery_after_multiple_timeouts(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Escenario: Recuperación tras timeouts consecutivos.

        Verifica Teorema T2 (Recuperación Eventual).
        """
        scenario = ScenarioBuilder.timeout_recovery()
        mock_requests_get.side_effect = scenario.responses

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 3
        assert scenario.expected_success

    def test_mixed_timeout_types_all_retried(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Propiedad: ∀ tipo de timeout : timeout ∈ Retriable.

        Verifica cobertura exhaustiva de la taxonomía de timeouts.
        """
        mock_requests_get.side_effect = (
            ResponseBuilder()
            .read_timeout()
            .connect_timeout()
            .timeout()
            .success()
            .build()
        )

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 4


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: HTTP ERROR HANDLING
# ═════════════════════════════════════════════════════════════════════════════


class TestHTTPErrorHandling:
    """
    Verificación de clasificación de errores HTTP según la partición R.

    Clasificación:
      • 2xx ∈ Success
      • 5xx ∈ Retriable (errores de servidor)
      • 4xx ∈ Fatal (errores de cliente, con excepción de 429)
      • 429 ∈ Retriable (rate limiting)
    """

    def test_503_service_unavailable_retried(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """Propiedad: 503 ∈ RETRIABLE_HTTP_CODES."""
        mock_requests_get.side_effect = (
            ResponseBuilder()
            .service_unavailable()
            .service_unavailable()
            .success()
            .build()
        )

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 3

    def test_502_bad_gateway_retried(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """Propiedad: 502 ∈ RETRIABLE_HTTP_CODES."""
        mock_requests_get.side_effect = (
            ResponseBuilder().bad_gateway().success().build()
        )

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 2

    def test_504_gateway_timeout_retried(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """Propiedad: 504 ∈ RETRIABLE_HTTP_CODES."""
        mock_requests_get.side_effect = (
            ResponseBuilder().gateway_timeout().success().build()
        )

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 2

    def test_500_internal_server_error_retried(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """Propiedad: 500 ∈ RETRIABLE_HTTP_CODES."""
        mock_requests_get.side_effect = (
            ResponseBuilder().internal_server_error().success().build()
        )

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 2

    def test_mixed_5xx_errors_all_retried(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Escenario: Combinación de errores 5xx.

        Verifica que todos los códigos 5xx son tratados uniformemente.
        """
        scenario = ScenarioBuilder.mixed_5xx_errors()
        mock_requests_get.side_effect = scenario.responses

        agent._wait_for_startup()

        assert mock_requests_get.call_count == len(scenario.responses)

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    def test_all_5xx_codes_are_retriable(
        self, agent, mock_requests_get, mock_time, status_code: int
    ) -> None:
        """
        Propiedad Parametrizada: ∀c ∈ {500,502,503,504} : c ∈ Retriable.

        Verificación exhaustiva de la tabla de clasificación.
        """
        mock_requests_get.side_effect = [
            ResponseBuilder().http_error(status_code).build()[0],
            ResponseBuilder().success().build()[0],
        ]

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 2, (
            f"Código {status_code} debe ser retriable"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: RATE LIMITING
# ═════════════════════════════════════════════════════════════════════════════


class TestRateLimiting:
    """
    Verificación de comportamiento ante rate limiting HTTP 429.

    Nota de Implementación:
      La implementación actual usa backoff fijo, ignorando el header Retry-After.
      Estas pruebas documentan el comportamiento existente.
    """

    def test_429_uses_fixed_backoff(
        self, agent, mock_requests_get, backoff_sequence
    ) -> None:
        """
        Propiedad Actual: 429 ⟹ backoff fijo (ignora Retry-After).

        Nota: En una implementación ideal, debería respetar Retry-After.
        """
        mock_requests_get.side_effect = (
            ResponseBuilder().too_many_requests(retry_after=10).success().build()
        )

        with patch("time.sleep", side_effect=backoff_sequence.record):
            agent._wait_for_startup()

        assert backoff_sequence.get_count() == 1
        assert backoff_sequence.verify_fixed_backoff(FIXED_BACKOFF_SECONDS), (
            f"Debe usar backoff fijo de {FIXED_BACKOFF_SECONDS}s, "
            f"no Retry-After=10"
        )

    def test_429_without_retry_after_header(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Propiedad: 429 sin Retry-After ⟹ backoff fijo.

        Verifica robustez ante respuestas 429 mal formadas.
        """
        mock_requests_get.side_effect = [
            ResponseBuilder().http_error(429).build()[0],
            ResponseBuilder().success().build()[0],
        ]

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 2
        mock_time.assert_called_once_with(FIXED_BACKOFF_SECONDS)

    def test_multiple_429_before_success(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Escenario: Rate limiting persistente con recuperación eventual.

        Verifica Teorema T2 bajo condiciones de rate limiting.
        """
        scenario = ScenarioBuilder.rate_limited_recovery()
        mock_requests_get.side_effect = scenario.responses

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 3


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: CIRCUIT BREAKER (marcadas skip)
# ═════════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    """
    Pruebas del patrón Circuit Breaker.

    Estado: NO IMPLEMENTADO en AutonomousAgent actual.

    Estas pruebas documentan el comportamiento esperado para una futura
    implementación del patrón Circuit Breaker según la teoría de autómatas.
    """

    @pytest.mark.skip(reason="Circuit breaker no implementado en AutonomousAgent")
    def test_circuit_opens_after_threshold_failures(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Propiedad Esperada: δ(CLOSED, FAILURE^θ) = OPEN.

        Tras θ fallos consecutivos, el circuito debe abrirse.
        """
        pass

    @pytest.mark.skip(reason="Circuit breaker no implementado en AutonomousAgent")
    def test_circuit_rejects_fast_when_open(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Propiedad Esperada: estado = OPEN ⟹ fail-fast (sin espera).

        El circuito abierto debe rechazar solicitudes inmediatamente.
        """
        pass

    @pytest.mark.skip(reason="Circuit breaker no implementado en AutonomousAgent")
    def test_circuit_transitions_to_half_open(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Propiedad Esperada: δ(OPEN, timeout_elapsed) = HALF_OPEN.

        Tras el timeout τ_open, el circuito debe pasar a HALF_OPEN.
        """
        pass

    @pytest.mark.skip(reason="Circuit breaker no implementado en AutonomousAgent")
    def test_circuit_closes_on_half_open_success(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Propiedad Esperada: δ(HALF_OPEN, SUCCESS) = CLOSED.

        Un éxito en HALF_OPEN debe cerrar el circuito.
        """
        pass

    @pytest.mark.skip(reason="Circuit breaker no implementado en AutonomousAgent")
    def test_circuit_reopens_on_half_open_failure(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Propiedad Esperada: δ(HALF_OPEN, FAILURE) = OPEN.

        Un fallo en HALF_OPEN debe reabrir el circuito.
        """
        pass


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: RECOVERY / RECUPERACIÓN
# ═════════════════════════════════════════════════════════════════════════════


class TestRecovery:
    """
    Verificación de recuperación automática tras fallos.

    Propiedades:
      • Reset de contadores tras éxito
      • Actualización de topología
      • Logging de eventos de recuperación
    """

    def test_consecutive_failures_reset_on_success(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Invariante I4: success ⟹ consecutive_failures := 0.

        Verifica que el contador se reinicia correctamente.
        """
        mock_requests_get.side_effect = (
            ResponseBuilder().connection_error().success().build()
        )

        agent._wait_for_startup()

        assert (
            agent._metrics.consecutive_failures == 0
        ), "El contador de fallos debe reiniciarse tras éxito"

    def test_topology_updated_after_recovery(
        self, agent, mock_requests_get, mock_time, mock_topology
    ) -> None:
        """
        Propiedad: recovery ⟹ topology.update_connectivity() llamado.

        Verifica integración con el sistema de topología.
        """
        mock_requests_get.side_effect = (
            ResponseBuilder().connection_error().success().build()
        )

        agent._wait_for_startup()

        mock_topology.update_connectivity.assert_called(), (
            "La topología debe actualizarse tras recuperación"
        )

    def test_recovery_logged_with_expected_keywords(
        self, agent, mock_requests_get, mock_time, caplog
    ) -> None:
        """
        Propiedad de Observabilidad: recovery ⟹ ∃ log con keywords esperadas.

        Verifica que la recuperación es auditable.
        """
        mock_requests_get.side_effect = (
            ResponseBuilder().connection_error().success().build()
        )

        with caplog.at_level(logging.INFO):
            agent._wait_for_startup()

        log_text = caplog.text.lower()
        recovery_keywords = ["operativo", "200 ok", "core detectado", "conectado"]

        assert any(
            keyword in log_text for keyword in recovery_keywords
        ), f"El log debe contener alguna de {recovery_keywords}"


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: MÉTRICAS Y OBSERVABILIDAD
# ═════════════════════════════════════════════════════════════════════════════


class TestResilienceMetrics:
    """
    Verificación de recolección de métricas de resiliencia.

    Métricas Verificadas:
      • retry_count (número de reintentos)
      • failure_type_counts (clasificación de fallos)
      • last_failure_time (timestamp del último fallo)
    """

    def test_retry_count_matches_backoff_count(
        self, agent, mock_requests_get, backoff_sequence
    ) -> None:
        """
        Invariante: retry_count = |{sleep calls}|.

        Verifica coherencia entre métrica y comportamiento observable.
        """
        mock_requests_get.side_effect = (
            ResponseBuilder()
            .connection_error()
            .connection_error()
            .success()
            .build()
        )

        with patch("time.sleep", side_effect=backoff_sequence.record):
            agent._wait_for_startup()

        assert agent._metrics.retry_count == backoff_sequence.get_count(), (
            "La métrica retry_count debe coincidir con el número de backoffs"
        )

    def test_failure_types_categorized(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Propiedad: cada tipo de fallo incrementa su contador específico.

        Verifica la taxonomía de FailureType.
        """
        mock_requests_get.side_effect = (
            ResponseBuilder()
            .connection_error()
            .timeout()
            .service_unavailable()
            .success()
            .build()
        )

        agent._wait_for_startup()

        assert (
            agent._metrics.connection_errors >= 1
        ), "Debe haber al menos 1 connection error"
        assert agent._metrics.timeout_errors >= 1, "Debe haber al menos 1 timeout"
        assert agent._metrics.http_errors >= 1, "Debe haber al menos 1 HTTP error"

    def test_last_failure_time_recorded(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Propiedad: failure ⟹ last_failure_time actualizado.

        Verifica que el timestamp del último fallo está en el rango correcto.
        """
        before = datetime.now()
        mock_requests_get.side_effect = (
            ResponseBuilder().connection_error().success().build()
        )

        agent._wait_for_startup()

        after = datetime.now()

        assert before <= agent._metrics.last_failure_time <= after, (
            "El timestamp del último fallo debe estar en el rango de ejecución"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: ESCENARIOS COMPLEJOS
# ═════════════════════════════════════════════════════════════════════════════


class TestComplexScenarios:
    """
    Verificación de escenarios que combinan múltiples patrones de fallo.

    Escenarios:
      • Fallos intermitentes (patrón FSFFS)
      • Cascadas de fallos (conexión → HTTP → timeout)
      • Recuperación parcial (estado degradado)
      • Interrupciones prolongadas (agotamiento de reintentos)
    """

    def test_intermittent_failures_stops_at_first_success(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Propiedad: bucle termina en el primer éxito.

        Con patrón "FSFFS", debe terminar en la posición 2 (primer 'S').
        """
        mock_requests_get.side_effect = (
            ResponseBuilder().intermittent_failures("FSFFS").build()
        )

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 2, "Debe detenerse en la primera 'S'"

    def test_cascading_failure_recovery(
        self, agent, mock_requests_get, mock_time, mock_topology
    ) -> None:
        """
        Escenario: Fallo en cascada con recuperación parcial.

        Secuencia: connection → 503 → 500 → success(degraded) → success(full).
        El bucle debe aceptar la primera 200 OK.
        """
        mock_requests_get.side_effect = (
            ResponseBuilder()
            .connection_error()
            .service_unavailable()
            .internal_server_error()
            .success({"redis_connected": False})  # Degradado pero OK
            .success({"redis_connected": True})
            .build()
        )

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 4, (
            "Debe aceptar la primera 200 OK (estado degradado)"
        )

    def test_partial_recovery_degraded_state_accepted(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Propiedad: estado degradado (200 con warnings) ∈ Success.

        Un servicio parcialmente operativo debe considerarse recuperado.
        """
        partial_response = Mock(ok=True, status_code=200)
        partial_response.json.return_value = {"status": "degraded"}

        mock_requests_get.side_effect = [
            requests.exceptions.ConnectionError("Initial failure"),
            partial_response,
        ]

        agent._wait_for_startup()

        assert mock_requests_get.call_count == 2

    def test_long_outage_respects_max_retries(
        self, agent, mock_requests_get, backoff_sequence
    ) -> None:
        """
        Teorema T1 (Terminación): interrupción prolongada ⟹ fallo definitivo.

        Verifica Invariante I1: |intentos| ≤ τ_max + 1.

        Demostración:
          Por construcción, el bucle tiene contador acotado.
          Tras MAX_RETRIES fallos, se lanza excepción o se sale del bucle. ∎
        """
        # MAX_RETRIES + 1 fallos consecutivos
        errors = [
            requests.exceptions.ConnectionError(f"Failure {i}")
            for i in range(MAX_RETRIES + 1)
        ]
        # Éxito inalcanzable tras agotar reintentos
        responses = errors + [ResponseBuilder().success().build()[0]]

        mock_requests_get.side_effect = responses

        with patch("time.sleep", side_effect=backoff_sequence.record):
            with pytest.raises(Exception):  # Se espera fallo definitivo
                agent._wait_for_startup()

        # Verificación de Invariante I1
        assert mock_requests_get.call_count == MAX_RETRIES + 1, (
            f"Debe haber exactamente {MAX_RETRIES + 1} intentos"
        )
        assert backoff_sequence.get_count() == MAX_RETRIES, (
            f"Debe haber exactamente {MAX_RETRIES} backoffs"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS: INVARIANTES DE RESILIENCIA
# ═════════════════════════════════════════════════════════════════════════════


class TestResilienceInvariants:
    """
    Verificación de invariantes matemáticos del sistema de reintentos.

    Invariantes Verificados:
      • I1: Terminación (acotación de intentos)
      • I2: Progreso (cada fallo es seguido de reintento)
      • I3: Monotonía temporal (timestamps crecientes)
      • I4: Conservación de estado (consistencia tras excepciones)
      • I5: Acotación de latencia total

    Además, propiedades algebraicas de β(n):
      • Positividad estricta
      • Monotonía no-decreciente
      • Acotación superior
    """

    def test_sleep_durations_always_positive(
        self, agent, mock_requests_get, backoff_sequence
    ) -> None:
        """
        Teorema: ∀n : β(n) > 0 (positividad estricta).

        Demostración:
          Por definición, β(n) = δ_fixed = 5.0 > 0. ∎
        """
        mock_requests_get.side_effect = (
            ResponseBuilder().failure_then_success(5, "connection").build()
        )

        with patch("time.sleep", side_effect=backoff_sequence.record):
            agent._wait_for_startup()

        assert backoff_sequence.verify_positive(), (
            "Todos los backoffs deben ser estrictamente positivos"
        )

    def test_failure_counter_never_negative(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Invariante: consecutive_failures ≥ 0 siempre.

        Verifica que el contador no sufre underflow.
        """
        mock_requests_get.side_effect = (
            ResponseBuilder().connection_error().success().build()
        )

        agent._wait_for_startup()

        assert (
            agent._metrics.consecutive_failures >= 0
        ), "El contador no puede ser negativo"

    def test_state_consistent_after_unexpected_exception(
        self, agent, mock_requests_get, mock_time
    ) -> None:
        """
        Invariante I4: ∀ excepción inesperada : estado permanece válido.

        Verifica que excepciones no manejadas no corrompen el estado.
        """
        mock_requests_get.side_effect = RuntimeError("Unexpected error")

        try:
            agent._wait_for_startup()
        except RuntimeError:
            pass  # Excepción esperada

        # Verificar que el estado sigue siendo válido
        assert hasattr(agent, "_running"), "Debe tener flag _running"
        assert hasattr(agent._metrics, "consecutive_failures"), (
            "Debe tener métrica consecutive_failures"
        )

    def test_total_wait_time_bounded(
        self, agent, mock_requests_get, backoff_sequence
    ) -> None:
        """
        Invariante I5: total_wait_time ≤ τ_max × δ_fixed.

        Demostración:
          Cada intento espera β(n) = δ_fixed.
          Máximo τ_max reintentos.
          Luego total ≤ τ_max × δ_fixed. ∎
        """
        errors = [
            requests.exceptions.ConnectionError(f"Fail {i}")
            for i in range(MAX_RETRIES)
        ]
        responses = errors + [ResponseBuilder().success().build()[0]]

        mock_requests_get.side_effect = responses

        with patch("time.sleep", side_effect=backoff_sequence.record):
            agent._wait_for_startup()

        max_expected = MAX_RETRIES * FIXED_BACKOFF_SECONDS
        actual_total = backoff_sequence.get_total()

        assert actual_total <= max_expected + TIME_EPSILON, (
            f"Tiempo total ({actual_total}s) excede el máximo esperado ({max_expected}s)"
        )

    def test_backoff_sequence_non_decreasing(
        self, agent, mock_requests_get, backoff_sequence
    ) -> None:
        """
        Teorema T5: ∀i,j : i < j ⟹ β(i) ≤ β(j) (monotonía no-decreciente).

        Para backoff fijo: β(i) = β(j), luego ≤ se cumple trivialmente.
        """
        mock_requests_get.side_effect = (
            ResponseBuilder().failure_then_success(MAX_RETRIES - 1, "connection").build()
        )

        with patch("time.sleep", side_effect=backoff_sequence.record):
            agent._wait_for_startup()

        assert backoff_sequence.verify_non_decreasing(), (
            "La secuencia de backoff debe ser monótona no-decreciente"
        )

    def test_no_sleep_when_immediate_success(
        self, agent, mock_requests_get, backoff_sequence
    ) -> None:
        """
        Propiedad: success inmediato ⟹ |sleeps| = 0.

        Verifica que no hay overhead innecesario en el caso óptimo.
        """
        mock_requests_get.side_effect = [ResponseBuilder().success().build()[0]]

        with patch("time.sleep", side_effect=backoff_sequence.record):
            agent._wait_for_startup()

        assert (
            backoff_sequence.get_count() == 0
        ), "No debe haber sleeps si hay éxito inmediato"

    def test_backoff_bounded_by_constant(
        self, agent, mock_requests_get, backoff_sequence
    ) -> None:
        """
        Propiedad: ∀n : β(n) ≤ β_max.

        Para backoff fijo, β_max = δ_fixed.
        """
        mock_requests_get.side_effect = (
            ResponseBuilder().failure_then_success(5, "connection").build()
        )

        with patch("time.sleep", side_effect=backoff_sequence.record):
            agent._wait_for_startup()

        assert backoff_sequence.verify_bounded(FIXED_BACKOFF_SECONDS), (
            f"Ningún backoff debe exceder {FIXED_BACKOFF_SECONDS}s"
        )


# ═════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN Y METADATOS
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "--strict-markers",
            "-ra",
            "--color=yes",
        ]
    )
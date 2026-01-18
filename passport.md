# 🛂 PASSPORT.md: El Pasaporte de Telemetría (Vector de Estado)
"En APU Filter, ningún dato viaja anónimo. Cada byte que ingresa al sistema recibe una identidad y un historial clínico inmutable. El Pasaporte no es un log de errores; es la Caja Negra del vuelo del dato, registrando la física, la topología y la economía de su viaje."
**1. Concepto: El Viaje del Dato**
En la arquitectura tradicional, los datos se mueven "a ciegas" entre servicios. En APU Filter, implementamos el patrón Context Propagation. Desde el momento de la ingesta (upload), se instancia un objeto TelemetryContext (el Pasaporte).
Este objeto viaja adjunto al payload de datos a través de la Malla Agéntica, acumulando "sellos" (validaciones) y métricas físicas en cada aduana (microservicio). Al final del ciclo, el Pasaporte se sella criptográficamente y se almacena como evidencia de auditoría.

--------------------------------------------------------------------------------
**2. Anatomía del Pasaporte (Estructura de Datos)**
El pasaporte es un objeto JSON jerárquico diseñado bajo el modelo DIKW (Datos, Información, Conocimiento, Sabiduría).
2.1. Identidad y Cronometría (Header)
• request_id (UUID): Identificador único universal de la transacción. Permite la trazabilidad distribuida (Distributed Tracing).
• session_id: Enlace a la sesión de usuario para persistencia temporal.
• timestamp_entry: Momento exacto de la ingesta (t0​).
• data_hash: Firma SHA-256 del contenido original. Garantiza que los datos no han sido alterados durante el viaje (Integridad).
2.2. Visado de Física (Emitido por: El Guardián)
Registra las variables de estado termodinámico del flujo de datos.
• flyback_voltage (Vfb​): "Picos de Inestabilidad". Mide cambios bruscos en la calidad del dato.
    ◦ Umbral: Si Vfb​>5.0, se marca como "Flujo Turbulento".
• saturation (S): Nivel de llenado de la memoria operativa.
• entropy (H): Medida de desorden en la estructura del texto (Shannon/Tsallis).
    ◦ Significado: Una entropía alta indica datos "sucios" o mal estructurados que disipan energía computacional.
2.3. Sello Espectral (Emitido por: El Oráculo de Laplace)
Certifica la controlabilidad del sistema antes de procesar.
• poles_location (σ): Ubicación de los polos en el plano complejo (s=σ+jω).
    ◦ Veto: Si σ>0 (Polos en RHP), el pasaporte recibe un sello de "INESTABLE" y el proceso se detiene.
• damping_ratio (ζ): Factor de amortiguamiento. Define si el proyecto es "Nervioso" (subamortiguado) o "Burocrático" (sobreamortiguado).
2.4. Mapa Topológico (Emitido por: El Arquitecto)
Describe la geometría del grafo de presupuesto.
• betti_numbers (βn​):
    ◦ β0​ (Islas): Fragmentación del grafo.
    ◦ β1​ (Ciclos): Dependencias circulares.
• pyramid_stability (Ψ): Índice de robustez logística.
    ◦ Alerta: Si Ψ<1.0, se estampa el sello "PIRÁMIDE INVERTIDA".
2.5. Veredicto Ejecutivo (Emitido por: El Intérprete)
La síntesis final en lenguaje natural, generada por el TelemetryNarrator.
• risk_score: Puntuación agregada de riesgo (0-100).
• strategic_narrative: Explicación causal de los hallazgos ("El proyecto se rechaza por resonancia inflacionaria en el acero").

--------------------------------------------------------------------------------
**3. Protocolo de Control Fronterizo (Policy-as-Code)** 
El sistema aplica Gobernanza Computacional en tiempo real. El pasaporte es verificado en cada "Punto de Control" (Step del Pipeline).

| Punto de Control | Agente Inspector | Criterio de Rechazo (Revocación de Pasaporte) | Acción |
| :--- | :--- | :--- | :--- |
| Ingesta | FluxCondenser | Entropía Térmica: Datos ilegibles o encoding corrupto. | Rechazo Inmediato (400 Bad Request). |
| Estabilidad | LaplaceOracle | Divergencia (σ>0): El sistema es incontrolable. | Veto Técnico (422 Unprocessable Entity). |
Architect | Ciclos Críticos (β1​>0): Paradojas lógicas irresolubles. | Alerta Bloqueante (Requiere intervención humana). |
| Finanzas | RiskChallenge | Inconsistencia: Rentabilidad alta con Estabilidad baja. | Degradación de Score (Marca de "Falso Positivo"). |

**4. Implementación Técnica (telemetry.py)**
El Pasaporte vive en el código como la clase TelemetryContext.

@dataclass
class TelemetryContext:
    """
    El 'Pasaporte' de la solicitud. Thread-safe y centralizado.
    """
    request_id: str
    steps: List[Dict]       # Historial de aduanas (Trace)
    metrics: Dict[str, Any] # Variables físicas (Energy, Entropy)
    errors: List[Dict]      # Órdenes de deportación
    metadata: Dict[str, Any]# Sellos de los agentes (Laplace, Betti)
    
    def record_metric(self, component: str, name: str, value: float):
        """Estampa una métrica física en el pasaporte."""
        # ... lógica de registro thread-safe ...

    def get_business_report(self) -> Dict:
        """Genera el resumen ejecutivo para el usuario."""
        # ... traducción de métricas a narrativa ...

Este objeto es inyectado en el contexto global de Flask (g.telemetry) y pasado explícitamente a los trabajadores asíncronos, asegurando que la "historia" del dato nunca se pierda.

--------------------------------------------------------------------------------
**5. Ejemplo de Pasaporte Sellado (Artefacto JSON)**
Este es el documento final que recibe el auditor o el sistema cliente.

{
  "passport_id": "req_8f15c0a2-bcee-4b12",
  "timestamp": "2025-10-27T10:00:00Z",
  "status": "APPROVED_WITH_WARNINGS",
  "physics_stamp": {
    "agent": "FluxCondenser",
    "metrics": {
      "saturation": 0.35,          // Flujo Laminar
      "flyback_voltage": 0.02,     // Estable
      "system_temperature": 68.5   // ALERTA: Fiebre Inflacionaria
    },
    "status": "STABLE"
  },
  "spectral_stamp": {
    "agent": "LaplaceOracle",
    "metrics": {
      "poles": "LHP",              // Estable (Left Half Plane)
      "damping_ratio": 0.4,        // Subamortiguado (Nervioso)
      "phase_margin": 45.0
    },
    "verdict": "CONTROLLABLE"
  },
  "topology_stamp": {
    "agent": "BusinessTopologicalAnalyzer",
    "metrics": {
      "betti_0": 1,                // Conexo (OK)
      "betti_1": 0,                // Acíclico (OK)
      "pyramid_stability": 0.69    // CRÍTICO: Pirámide Invertida
    },
    "anomalies": ["High stress on node: CEMENTO_GRIS"]
  },
  "final_narrative": {
    "verdict": "APROBACIÓN CONDICIONADA",
    "summary": "El proyecto es viable físicamente, pero presenta una estructura de Pirámide Invertida (Psi=0.69). Se requiere ampliar la base de proveedores de Cemento para mitigar el riesgo de colapso logístico."
  }
}


--------------------------------------------------------------------------------
Este documento define el estándar de interoperabilidad y trazabilidad del ecosistema APU Filter v3.0.
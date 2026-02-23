# 🛂 PASSPORT.md: El Pasabordo Ciberfísico (Identidad del Dato)

"En APU Filter, ningún dato viaja anónimo. Cada byte que ingresa al sistema recibe una identidad y un historial clínico inmutable. El Pasabordo no es un log de errores; es la Caja Negra del vuelo del dato, registrando la física, la topología y la economía de su viaje."

**1. Concepto: El Viaje del Dato**

En la arquitectura tradicional, los datos se mueven "a ciegas". En APU Filter, implementamos el patrón de Propagación de Contexto Ciberfísico. Desde el momento de la ingesta, se instancia un **Pasabordo** (TelemetryContext).
Este documento viaja adjunto al payload de datos a través del Consejo de Sabios, acumulando "visados" (validaciones) y métricas físicas en cada aduana. Al final del ciclo, el Pasabordo se sella criptográficamente y se almacena como evidencia de auditoría.

--------------------------------------------------------------------------------
**2. Anatomía Inmutable del Pasaporte (telemetry_schemas)**

El pasaporte utiliza Estructuras de Datos Inmutables (*frozen dataclasses*). Una vez que un agente estampa su métrica, esta se convierte en un valor algebraico puro y a prueba de manipulaciones forenses.

*   **PhysicsMetrics (La Realidad):** Registra la Saturación, el Voltaje Flyback y el Exceso Hamiltoniano. (Si se viola la conservación de energía, el dato es corrupto).
*   **TopologicalMetrics (La Forma):** Registra los Números de Betti ($\beta_0, \beta_1$) y la Estabilidad Piramidal ($\Psi$).
*   **ThermodynamicMetrics (El Valor):** Registra la Entropía Financiera y la Temperatura del Sistema (Fiebre Inflacionaria).
*   **ControlMetrics (La Estabilidad):** Registra la ubicación de los Polos de Laplace.

**2.1 El Juez Final: TelemetryNarrator y la Clausura Transitiva**

El viaje del Pasaporte termina en el `TelemetryNarrator`. Este módulo lee todos los sellos y aplica la regla de **Clausura Transitiva**: *Un fallo en la base invalida toda la cima*.
Si el reporte detecta que la física de los datos falló (ej. archivo corrupto o turbulencia masiva en la ingesta), el Narrador emite el código `REJECTED_PHYSICS` y aborta inmediatamente cualquier análisis financiero estratégico. El sistema se niega a crear una "estrategia" sobre datos que físicamente no tienen integridad.

--------------------------------------------------------------------------------
**3. Protocolo de Control Fronterizo (Gobernanza)**

El sistema aplica Gobernanza Computacional en tiempo real. El pasabordo es verificado en cada "Punto de Control".

| Punto de Control | Agente Inspector | Criterio de Rechazo (Revocación de Pasabordo) | Acción |
| :--- | :--- | :--- | :--- |
| Ingesta | FluxCondenser | Entropía Térmica: Datos ilegibles. | Rechazo Inmediato. |
| Estabilidad | LaplaceOracle | Divergencia: El sistema es incontrolable. | Veto Técnico. |
| Estructura | Arquitecto | "Bucle de la Formaleta" (Ciclos Lógicos). | Alerta Bloqueante (Requiere intervención humana). |
| Finanzas | Oráculo | Inconsistencia: Rentabilidad alta con Estabilidad baja. | Degradación de Score (Marca de "Falso Positivo"). |

**La Ley de Clausura Transitiva (Fast-Fail):**
Nuestros esquemas son inmutables. Si un archivo intenta pedir un análisis financiero (Estrato 1) saltándose la validación estructural (Estrato 2), el sistema bloquea la operación. **No hay estrategia sin estructura.**

--------------------------------------------------------------------------------
**4. Ejemplo de Pasabordo Sellado (Acta Digital)**

Este es el documento final que recibe el auditor o el sistema cliente.

{
  "passport_id": "req_8f15c0a2-bcee-4b12",
  "status": "APPROVED_WITH_WARNINGS",
  "physics_stamp": {
    "agent": "FluxCondenser",
    "metrics": {
      "system_temperature": 68.5   // ALERTA: Fiebre Inflacionaria
    },
    "status": "STABLE"
  },
  "topology_stamp": {
    "agent": "BusinessTopologicalAnalyzer",
    "metrics": {
      "pyramid_stability": 0.69    // CRÍTICO: Monopolio Invisible detectado
    },
    "anomalies": ["Alto estrés en nodo: CEMENTO_GRIS"]
  },
  "final_narrative": {
    "verdict": "APROBACIÓN CONDICIONADA",
    "summary": "El proyecto es viable físicamente, pero presenta una estructura de Pirámide Invertida (Psi=0.69). Se requiere ampliar la base de proveedores de Cemento para mitigar el riesgo de colapso logístico."
  }
}

--------------------------------------------------------------------------------
Este documento define el estándar de interoperabilidad y trazabilidad del ecosistema APU Filter.

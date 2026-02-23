# 🛂 PASSPORT.md: El Pasabordo Ciberfísico (Identidad del Dato)

"En APU Filter, ningún dato viaja anónimo. Cada byte que ingresa al sistema recibe una identidad y un historial clínico inmutable. El Pasabordo no es un log de errores; es la Caja Negra del vuelo del dato, registrando la física, la topología y la economía de su viaje."

**1. Concepto: El Viaje del Dato**

En la arquitectura tradicional, los datos se mueven "a ciegas". En APU Filter, implementamos el patrón de Propagación de Contexto Ciberfísico. Desde el momento de la ingesta, se instancia un **Pasabordo** (TelemetryContext).
Este documento viaja adjunto al payload de datos a través del Consejo de Sabios, acumulando "visados" (validaciones) y métricas físicas en cada aduana. Al final del ciclo, el Pasabordo se sella criptográficamente y se almacena como evidencia de auditoría.

--------------------------------------------------------------------------------
**2. Anatomía del Pasabordo (La Pirámide Conectada a la Rentabilidad)**

El pasabordo está estructurado jerárquicamente, no como un JSON plano, sino como una pirámide DIKW que ancla la tecnología a la rentabilidad del negocio:

**Estrato 3: Cimentación (FÍSICA - Datos)**
*   **Emisor:** El Guardián.
*   **Validación:** ¿Tienen los datos la integridad termodinámica para existir?
*   **Métrica Clave:** Entropía (Orden) y Temperatura (Volatilidad). Si los datos son ruido ("basura"), se rechazan aquí para no gastar energía procesando errores.

**Estrato 2: Estructura (TÁCTICA - Información)**
*   **Emisor:** El Arquitecto.
*   **Validación:** ¿Es la topología del proyecto construible?
*   **Métrica Clave:** Números de Betti (Ciclos y Huecos). Aquí se detecta el "Bucle de la Formaleta". Si la estructura lógica falla, no se pasa al análisis financiero.

**Estrato 1: Finanzas (ESTRATEGIA - Conocimiento)**
*   **Emisor:** El Oráculo.
*   **Validación:** ¿Es el proyecto rentable bajo estrés?
*   **Métrica Clave:** VPN en Riesgo y Temperatura del Sistema (Inflación). Se simulan escenarios de crisis para probar la robustez del margen.

**Estrato 0: Retorno de Inversión (SABIDURÍA - Valor)**
*   **Emisor:** El Intérprete.
*   **Validación:** Veredicto Ejecutivo.
*   **Salida:** Un "Acta de Deliberación" en lenguaje natural que le dice al gerente exactamente qué decisión tomar para proteger su dinero.

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

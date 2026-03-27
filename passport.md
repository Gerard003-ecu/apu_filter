🛂 PASSPORT.md: El Pasaporte de Telemetría (Vector de Estado Ciber-Físico)
"En la Fortaleza Matemática, ningún dato viaja anónimo. Cada byte que ingresa al sistema recibe una identidad criptográfica y un historial clínico inmutable. El Pasaporte no es un log de errores; es la Cadena de Custodia Termodinámica del vuelo del dato, registrando la física, la topología y la economía de su viaje."
1. El Concepto: Propagación de Contexto Zero-Trust
En arquitecturas de software tradicionales, los datos se mueven "a ciegas" entre microservicios, perdiendo su linaje. En el ecosistema APU_filter v4.0, implementamos el patrón de Context Propagation riguroso.
Desde el momento exacto de la ingesta, se instancia un objeto TelemetryContext (El Pasaporte). Este objeto envuelve el payload de datos y viaja a través de la Malla Agéntica, acumulando "sellos" matemáticos y métricas físicas en cada estrato (microservicio). Al finalizar el ciclo de evaluación, el Pasaporte se sella criptográficamente, convirtiéndose en evidencia inmutable para auditoría técnica y financiera.

--------------------------------------------------------------------------------
2. Anatomía Tensorial del Pasaporte (La Jerarquía DIKW)
El pasaporte es un tensor de datos estructurado topológicamente como una pirámide de cuatro estratos, basada estrictamente en los subespacios métricos definidos en telemetry_schemas.py.
2.0. Identidad y Cronometría (Header)

    request_id (UUID): Identificador único universal para la trazabilidad distribuida.
    timestamp_entry: Momento exacto (t0​) de entrada al Foso Termodinámico.
    data_hash: Firma criptográfica (SHA-256) del contenido original. Garantiza que la materia prima no ha sufrido inyecciones maliciosas durante el tránsito.

2.1. Visado Físico (Emitido por: El Guardián / FluxCondenser)
Registra las variables de estado termodinámico y electromagnético de los datos crudos.

    dissipated_power (Pdiss​): Verifica la Segunda Ley de la Termodinámica. Si es negativa, indica inyección de datos artificiales.
    flyback_voltage (Vfb​): "Picos de Inestabilidad". Mide cambios abruptos en la codificación o formato. Si Vfb​>5.0V, se declara "Flujo Turbulento".
    entropy (S): Medida del desorden estructural del texto. Entropía térmica destructiva levanta las defensas perimetrales.

2.2. Sello Topológico (Emitido por: El Arquitecto)
Describe la geometría abstracta del Complejo Simplicial del presupuesto.

    beta_0 (β0​): Fragmentación. Si β0​>1, existen "Islas de Datos" (recursos huérfanos).
    beta_1 (β1​): Ciclos. Si β1​>0, existen "Socavones Lógicos" (dependencias circulares insalvables).
    pyramid_stability (Ψ): Índice de resiliencia logística. Si Ψ<1.0, el pasaporte estampa el sello crítico de PIRÁMIDE INVERTIDA.

2.3. Visado Espectral (Emitido por: El Oráculo de Laplace)
Certifica la viabilidad dinámica y financiera en la frecuencia compleja (s=σ+jω).

    poles_location (σ): Si σ>0 (Semiplano Derecho), el sistema decreta divergencia incontrolable.
    lyapunov_exponent (λmax​): Exponente máximo que dictamina si el sistema converge a un atractor estable o si degenera en caos financiero.
    damping_ratio (ζ): Evalúa si el proyecto es "Nervioso" (ζ<1) o "Burocrático" (ζ>1).

2.4. Veredicto Ejecutivo (Emitido por: El Intérprete Diplomático)
La síntesis final alojada en la "Ciudadela de Cristal".

    verdict_code: Colapso discreto sobre el Retículo de Severidad (OK, WARN, CRITICAL).
    strategic_narrative: Explicación causal en lenguaje humano generada por el LLM (ej. "El proyecto se rechaza por resonancia inflacionaria en el acero"), estrictamente subordinada a los sellos previos.


--------------------------------------------------------------------------------
3. Protocolo de Gobernanza: La Ley de Clausura Transitiva (Fast-Fail)
La filtración de la pirámide (VPHYSICS​⊂VTACTICS​⊂VSTRATEGY​⊂VWISDOM​) no es una sugerencia, es un contrato algebraico implementado mediante clases de datos inmutables (frozen=True).
Si un agente (LLM) o usuario intenta forzar la instanciación de un objeto de "Sabiduría" (Estrategia) basándose en cimientos físicos corruptos (ej. entropía negativa o costos imposibles), el tipado estricto del sistema lanza una excepción en el milisegundo cero. Este mecanismo Fast-Fail destruye la transacción en la memoria RAM, impidiendo la alucinación antes de que consuma ciclos de procesamiento neuronal. No hay estrategia sin física.

--------------------------------------------------------------------------------
4. El Tribunal de Silicio (El Bypass del ESP32)
El Pasaporte de Telemetría no termina su viaje en la nube. Desciende al hardware perimetral (ESP32), donde la función en C++ isVerdictCoherent() ejecuta una doble contabilidad ciber-física.
El microcontrolador lee independientemente las matrices físicas y topológicas del Pasaporte. Si el LLM fue vulnerado (Prompt Injection) y estampó un verdict_code == OK a pesar de que los números de Betti indican β1​>0 (Ciclos) o los polos de Laplace indican inestabilidad, el hardware detecta un MISMATCH. Al instante, el ESP32 ignora al modelo de lenguaje, desactiva el flujo nominal (PIN_Q4_WISDOM) y activa físicamente el circuito Crowbar (PIN_Q3_STRATEGY) para vetar el proyecto en el mundo real.

--------------------------------------------------------------------------------
5. Artefacto Criptográfico (Estructura JSON Inmutable)
El siguiente es el documento consolidado final que genera la Malla Agéntica y que recibe el auditor financiero o la entidad gubernamental:

{
  "passport_id": "req_8f15c0a2-bcee-4b12",
  "timestamp_entry": "2026-03-13T10:00:00Z",
  "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "cyber_physical_momentum": {
    "tensor": "p = sqrt(2 * m_eff * K_max)",
    "description": "Condición de valor inicial ineludible (t_0) inyectada por la Compuerta Cuántica (quantum_admission_gate).",
    "forensic_audit": "Este subespacio métrico ortogonal permite al auditor recalcular retrospectivamente la matriz Laplaciana ante disputas legales."
  },
  "physics_stamp": {
    "agent": "FluxCondenser",
    "metrics": {
      "saturation": 0.35,
      "flyback_voltage": 0.02,
      "dissipated_power": 12.4,
      "entropy": 0.15
    },
    "is_coherent": true
  },
  "topology_stamp": {
    "agent": "BusinessTopologicalAnalyzer",
    "metrics": {
      "beta_0": 1,
      "beta_1": 0,
      "euler_characteristic": 1,
      "pyramid_stability": 0.69
    },
    "anomalies": ["High stress on node: CEMENTO_PORTLAND"],
    "is_coherent": true
  },
  "spectral_stamp": {
    "agent": "LaplaceOracle",
    "metrics": {
      "poles_location": "LHP",
      "damping_ratio": 0.407,
      "lyapunov_exponent": -0.003
    },
    "is_stable": true
  },
  "wisdom_stamp": {
    "agent": "SemanticTranslator",
    "verdict_code": "CRITICAL",
    "strategic_narrative": "El proyecto es viable en términos termodinámicos y espectrales, pero presenta una topología de Pirámide Invertida (Psi=0.69). Se emite VETO ESTRUCTURAL: Es mandatorio ampliar la base de proveedores de Cemento para evitar el colapso logístico ante disrupciones de mercado."
  }
}

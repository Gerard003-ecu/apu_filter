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
    quantized_friction (Tor(H_0, Z)): Captura obligatoriamente la fricción de cuantización logística que el sistema disipa como remanente isoperimétrico.

2.2. Sello Topológico (Emitido por: El Arquitecto)
Describe la geometría abstracta del Complejo Simplicial del presupuesto y su lente gravitacional.

    beta_0 (β0​): Fragmentación. Si β0​>1, existen "Islas de Datos" (recursos huérfanos).
    beta_1 ($\beta_1$): Ciclos. Si $\beta_1^- > 0$ (sin DPP certificado), existen "Socavones Lógicos" (dependencias circulares insalvables). Genera **veto duro**.
    pyramid_stability ($\Psi$): Índice de resiliencia logística. **Definición formal** (Inversa del Índice de Simpson de Concentración):
    $$\Psi := \frac{\left(\sum_{j=1}^{n} \deg(p_j)\right)^2}{n \cdot \sum_{j=1}^{n} \deg(p_j)^2} \in (0, 1]$$
    donde $\deg(p_j)$ es el número de APUs que dependen del proveedor $j$. Si $\Psi < \Psi_{\min}$: **veto duro** (Fast-Fail, RECHAZO). Si $\Psi_{\min} \leq \Psi < 1.0$: veto suave (CRITICAL/WARN con plan correctivo). $\Psi_{\min}$ recomendado: 0.7 para infraestructura pública.
    beta_2 ($\beta_2$): Cavidades ternarias. Si $\beta_2 > 0$, existen interdependencias trilaterales irresolubles. Genera veto suave y penalización de tarifa.
    renormalized_inertial_drag ($m^{**}$): Documenta el arrastre de masa inercial renormalizada que deforma el espacio de fase, evidenciando un anclaje logístico severo en el presupuesto.

2.3. Visado Espectral (Emitido por: El Oráculo de Laplace)
Certifica la viabilidad dinámica y financiera en la frecuencia compleja (s=σ+jω).

    poles_location (σ): Si σ>0 (Semiplano Derecho), el sistema decreta divergencia incontrolable.
    lyapunov_exponent (λmax​): Exponente máximo que dictamina si el sistema converge a un atractor estable o si degenera en caos financiero.
    damping_ratio (ζ): Evalúa si el proyecto es "Nervioso" (ζ<1) o "Burocrático" (ζ>1).

2.4. Sello del Haz Γ (Emitido por: El Haz Tangente Generativo)
Certifica la rigurosidad del código y estrategias generadas por la IA.

    sheaf_cohomology (H1): Validación de la Cohomología de Haces. Si H1 > 0, detecta una paradoja lógica insalvable.
    dirichlet_energy (E(x)): Medida de la suavidad y estabilidad de la solución generada.
    symplectic_invariance: Certifica que el código preserva la estructura de fase y no inyecta entropía incontrolada.

2.5. Veredicto Ejecutivo (Emitido por: El Intérprete Diplomático)
La síntesis final alojada en la "Ciudadela de Cristal".

    verdict_code: Colapso discreto sobre el Retículo de Severidad (OK, WARN, CRITICAL, RECHAZO).
    strategic_narrative: Explicación causal en lenguaje humano generada por el LLM (ej. "El proyecto se rechaza por resonancia inflacionaria en el acero"), estrictamente subordinada a los sellos previos.


--------------------------------------------------------------------------------
3. Protocolo de Gobernanza: Ley de Clausura Transitiva y Niveles de Veto
La filtración de la pirámide (tabla canónica: $V_{\aleph_0} \subsetneq V_{\mathbb{P}} \subsetneq V_{\mathbb{T}} \subsetneq V_{\mathbb{S}} \subsetneq V_{\mathbb{W}}$) no es una sugerencia, es un contrato algebraico implementado mediante clases de datos inmutables (`frozen=True`).

**Distinción crítica: Veto Duro vs Veto Suave**

El sistema opera con dos categorías estrictas de respuesta, no intercambiables:

| Tipo | Código | Condición Disparadora | Acción |
|---|---|---|---|
| **Veto Duro (Fast-Fail)** | `RECHAZO` | $P_{\text{diss}} < 0$, $\beta_1^- > 0$, $\Psi < \Psi_{\min}$, $\sigma > 0$, o $|\mu_k| > 1$ | Transacción abortada en RAM. Imposible continuar sin sanear el invariante. ESP32 activa Crowbar. |
| **Veto Suave** | `CRITICAL` | $\Psi_{\min} \leq \Psi < 1.0$, $\beta_2 > 0$, PM $< 45°$ | Proyecto puede continuar bajo condiciones. Acta de Deliberación con plan correctivo obligatorio. |
| **Advertencia** | `WARN` | Límites próximos pero no superados | Registro de precaución. Sin bloqueo. |

La **Ley de Monotonicidad del Pasaporte** garantiza transiciones únicamente ascendentes: `PASS → WARN → CRITICAL → RECHAZO`. Un sello de RECHAZO en estrato $k$ imposibilita emitir sello de aprobación en cualquier estrato $k' > k$ dentro del mismo ciclo.

Si un agente (LLM) intenta forzar la instanciación de Sabiduría sobre cimientos físicos corruptos, el tipado estricto lanza una excepción en el milisegundo cero (Fast-Fail). No hay estrategia sin física.

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
      "beta_2": 0,
      "euler_characteristic": 1,
      "pyramid_stability": 0.69,
      "pyramid_stability_threshold": 0.70,
      "veto_type": "SOFT_WARN"
    },
    "anomalies": ["High stress on node: CEMENTO_PORTLAND", "Psi=0.69 < Psi_min=0.70: WARN — diversificar proveedores de cemento"],
    "is_coherent": true,
    "note": "IMPORTANT: is_coherent=true porque Psi=0.69 supera el umbral de veto DURO (Psi_min=0.70 en este proyecto). Un veto duro requeriria Psi < Psi_min, lo cual generaria is_coherent=false y Fast-Fail inmediato. El verdict_code=CRITICAL a continuacion es un veto SUAVE: el proyecto puede continuar bajo plan correctivo obligatorio de diversificacion de proveedores."
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

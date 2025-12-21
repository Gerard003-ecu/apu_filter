# 📂 Nivel 4: Los Archivos del Consejo

(La Evidencia Forense y la Caja de Cristal)
"La confianza no se exige; se demuestra. Aquí presentamos los entregables tangibles donde la matemática abstracta se convierte en decisiones de negocio."
En este nivel, aterrizamos la teoría. El sistema APU Filter no le entrega un simple "log" de errores; le entrega tres artefactos de inteligencia diseñados para la auditoría forense de su proyecto. Estos archivos son la materialización del trabajo del Consejo.

## 1. El Microscopio Estructural (Visualización Forense)

Fuente: Generado por topology_viz.txt y business_topology.txt.
El Arquitecto Estratega no solo calcula; dibuja. Transformamos su presupuesto en un grafo interactivo (usando Cytoscape.js) que permite una inspección visual inmediata de las patologías. No es un dibujo estático; es un mapa de calor de riesgos.
La Leyenda del Mapa (Decodificación Visual)
El sistema colorea y etiqueta los nodos basándose en la gravedad del hallazgo matemático:

| Elemento Visual | Significado en Obra | Causa Técnica (Código) |
| :--- | :--- | :--- | :--- |
| 🔴 Nodos ROJOS | Punto de Falla Crítico. Elemento en riesgo de colapso, bloqueo o aislamiento. | NodeColor.RED: Detectado en anomaly_data (ciclos, estrés, islas) |
| 🟠 Borde PUNTEADO | Conexión Fantasma. Una relación que debería existir pero falta (ej. un APU sin insumos). | style="dotted", color="orange": Detectado por get_missing_connections. |
| ⚡ Clase "STRESS" | Pirámide Invertida. Un insumo básico que soporta demasiada carga del proyecto. | NodeClass.STRESS: Insumos con in_degree desproporcionado (>30% del grafo). |
| 🔄 Clase "CIRCULAR" | Socavón Lógico. El nodo es parte de un bucle infinito de precios (A → B → A). | NodeClass.CIRCULAR: Pertenencia al conjunto nodes_in_cycles. |

La Experiencia de Usuario: Usted no busca errores fila por fila. Usted abre el visor y "sigue el rojo".

## 2. El Veredicto Ejecutivo (Narrativa Estratégica)

Fuente: Generado por semantic_translator.txt y business_agent.txt.
El Intérprete Diplomático toma los datos crudos y redacta un informe en lenguaje natural. No recibirá un JSON crudo; recibirá un Dictamen de Ingeniería.
Estructura del Informe (ConstructionRiskReport)
El sistema genera un objeto ConstructionRiskReport
 que se traduce en:
1. Puntuación de Integridad (Score 0-100):
    ◦ Cálculo: Se penaliza por ciclos (β1​>0), islas (β0​>1) y pirámides invertidas (Ψ<1).
    ◦ Ejemplo: "Integridad: 65/100 (Riesgo Alto)".
2. Narrativa de Estabilidad:
    ◦ Input: Ψ = 0.8
    ◦ Output: "📉 COLAPSO POR BASE ESTRECHA (Pirámide Invertida). La cimentación logística es demasiado angosta para soportar el peso táctico de los APUs."
3. Veredicto Financiero:
    ◦ Input: ROI < 0 o Volatilidad > 20%.
    ◦ Output: "🛑 Veredicto: RIESGO CRÍTICO. Estructura no financiable sin reingeniería."

## 3. El Electrocardiograma del Sistema (Telemetría OODA)

Fuente: Generado por apu_agent.txt y flux_condenser.txt.
El Guardián del Umbral y el Agente Autónomo monitorean la salud del proceso de análisis en tiempo real. Usted puede ver el "pulso" del sistema para saber si sus datos son estables o caóticos.
El Ciclo OODA en Acción
El Agente ejecuta un bucle continuo de Observar-Orientar-Decidir-Actuar
, generando métricas físicas que usted puede auditar:
• ⚡ Voltaje de Flyback (flyback_voltage):
    ◦ Significado: ¿Qué tan inestables son los datos entrantes?
    ◦ Alerta: Si V>0.8, el sistema reporta: "Alta inestabilidad detectada. Los datos contienen cambios bruscos de formato o calidad."
• 🔥 Potencia Disipada (dissipated_power):
    ◦ Significado: ¿Cuánta energía estamos desperdiciando procesando "basura" (filas vacías, errores)?
    ◦ Acción: Si P>50W, el sistema activa el "Freno de Emergencia" térmico para evitar colapsos.
• 🌊 Saturación (saturation):
    ◦ Significado: Presión de datos en el pipeline.
    ◦ Meta: Mantener un "Flujo Laminar" al 30% de capacidad para máxima precisión.

## 4. La Bitácora Inmutable (Audit Trail)

Fuente: data_validator.txt y telemetry.txt.
Finalmente, para el auditor financiero, el sistema genera una traza completa. Nada se pierde.
• Identidad de la Sesión: Cada análisis tiene un session_id y un request_id único

• Evidencia de Validación:
    ◦ Cantidad Recalculada: "El sistema detectó Cantidad=0 pero Valor Total > 0. Se recalculó la cantidad implícita."
    ◦ Descripción Inferida: "Descripción faltante. Se infirió 'Cemento Gris' usando búsqueda difusa (Fuzzy Matching) con 95% de confianza."
Conclusión del Nivel 4: Al llegar aquí, usted tiene:
1. Un Mapa Visual de sus riesgos (Topología).
2. Un Dictamen Escrito de viabilidad (Semántica).
3. Un Reporte de Signos Vitales de la calidad de sus datos (Física).
Esto es la Caja de Cristal.
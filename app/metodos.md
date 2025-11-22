## Ingeniería Bajo el Capó: La Garantía de Estabilidad

Aunque nuestra prioridad es su negocio, la solidez técnica es nuestra garantía. APU Filter está construido sobre una arquitectura modular que separa claramente las responsabilidades, garantizando robustez y escalabilidad. Sus tres pilares fundamentales son:

### 1. Condensador de Flujo de Datos (Data Flux Condenser)
- **Componente Clave:** `app/flux_condenser.py` (**El Estabilizador**)
- **Función:** Actúa como un amortiguador industrial a la entrada del sistema.

#### La Analogía del Amortiguador Industrial
Imagine que los datos de entrada son un vehículo transitando por un terreno agreste (archivos con formatos rotos, caracteres extraños, errores humanos). Sin suspensión, el motor (el procesador) se rompería con el primer bache.

Nuestro **Condensador de Flujo** funciona como una suspensión activa avanzada. Usa física real para absorber los impactos del "camino" (datos sucios), entregando un viaje suave y constante al "pasajero" (su lógica de negocio). Si el camino es muy malo, el sistema reduce la velocidad automáticamente para no volcar, pero **nunca se detiene**.

#### Ingeniería de Confiabilidad (SRE) aplicada a Datos
Esta no es una metáfora decorativa. Utilizamos ecuaciones de sistemas dinámicos para gestionar la "fricción" de los datos corruptos.

*   **Detectar Fricción:** Identificar cuándo la "suciedad" de los datos está generando resistencia.
*   **Disipar Calor:** Liberar la "presión" reduciendo la velocidad de ingesta antes de un fallo crítico.
*   **Mantener el Flujo:** Garantizar que el sistema procese lo recuperable sin detenerse.

#### ⚙️ Nivel 1: Motor de Física RLC (El Sensor)
El sistema evoluciona hacia un **Modelo Energético Escalar**.

1.  **Energía Potencial ($E_c$) - Presión de Datos:** La "carga" acumulada por el volumen de registros.
2.  **Energía Cinética ($E_l$) - Inercia de Calidad:** Un flujo de alta calidad tiene una inercia fuerte que resiste perturbaciones.
3.  **Potencia Disipada ($P$) - Calor/Fricción:**
    *   **Termodinámica del Software:** Calcula el "calor" generado por la resistencia de los datos sucios.
    *   Si el sistema gasta demasiada energía procesando basura, se activa el **Disyuntor Térmico** (Freno de Emergencia) para evitar el sobrecalentamiento lógico.

#### 🧠 Nivel 2: Controlador PI Discreto (El Cerebro)
Un **Lazo de Control Cerrado** que ajusta el tamaño del lote de procesamiento (*Batch Size*) en tiempo real para mantener un flujo laminar, protegiendo al sistema de la saturación.

### 2. Pipeline Narrativo de Datos
- **Componente Clave:** `app/procesador_csv.py` (**El Orquestador**)

El flujo de datos no es una simple tubería, es una historia de transformación contada en cuatro actos:

1.  **El Ingreso (Load):** Los datos crudos llegan a la recepción. Aquí, **El Guardia** (`ReportParserCrudo`) detiene a los sospechosos (líneas corruptas) en la puerta.
2.  **El Diagnóstico (Merge):** **El Cirujano** (`APUProcessor`) examina los pacientes admitidos. Cruza la información del APU con el catálogo maestro de insumos para completar los vacíos (precios faltantes).
3.  **La Operación (Calculate):** Se realiza la suma de alta precisión. Se agregan costos de materiales, mano de obra y equipos para obtener el costo real por unidad.
4.  **El Alta (Final Merge):** El APU curado y valorado se une al presupuesto general, listo para ser presentado en la oferta final.

### 3. Estimador Inteligente: Filosofía de "Caja Blanca"
- **Componente Clave:** `app/estimator.py` (**El Estratega**)

En ingeniería de costos, una "Caja Negra" (un sistema que da respuestas sin explicaciones) es inaceptable. Un gerente necesita saber **por qué** se sugiere un precio.

El Estratega opera con **Transparencia Radical**:

#### Evidencia, no Magia
Cuando el sistema sugiere un APU histórico para un nuevo concepto, no solo entrega el precio, entrega la **Evidencia Matemática** de su decisión.

*   **Coincidencia Semántica (El "Parecido Conceptual"):**
    *   El sistema entiende que *"Muro de ladrillo tolete"* es conceptualmente idéntico a *"Mampostería en arcilla cocida"*, aunque no compartan palabras.
    *   **Log:** `✅ Coincidencia semántica encontrada: 0.92` (El sistema tiene un 92% de certeza de que son lo mismo).

*   **Coincidencia por Palabras Clave (El "Parecido Exacto"):**
    *   Si no hay similitud conceptual, busca términos específicos.
    *   **Log:** `✅ Match FLEXIBLE encontrado (80%)` (Coincidieron 4 de 5 palabras clave).

Esto permite al ingeniero humano auditar al "robot", validando si un 92% de similitud es suficiente para aceptar el precio o si requiere revisión.

## Tecnologías Utilizadas

La plataforma está construida sobre una pila de tecnologías modernas de alto rendimiento:

- **Backend:** **Flask** y **Redis** para una API robusta y con estado.
- **Inteligencia Artificial:**
    - **Sentence-Transformers & FAISS:** El cerebro detrás de la búsqueda semántica y la memoria institucional.
- **Física de Datos:**
    - **Modelado RLC:** Algoritmos propios de control de flujo.
- **Calidad:**
    - **Pytest & Ruff:** Estándares de código de nivel industrial.

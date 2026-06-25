# conda_v5 + suturas_rigurosas fix plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Corregir los clusters de fallos descritos en `app/conda_v5.md` y alinear la base de código con las suturas operativas de `app/suturas_rigurosas.md`, minimizando cambios en tests cuando el contrato de producción sea el que está roto.

**Architecture:** La estrategia es por capas: primero restaurar compatibilidad de API/contratos (objetos, métodos, tipos y kwargs), luego endurecer la estabilidad numérica en módulos espectrales/geodésicos, y finalmente ajustar fixtures/tests para que reflejen invariantes reales y tolerancias numéricas razonables. Los arreglos se hacen de adentro hacia afuera: producción primero, tests después, salvo cuando el fallo viene de un supuesto obsoleto en la suite.

**Tech Stack:** Python 3.10, pytest, numpy, dataclasses, typing, optional dependency gating (`psutil`).

---

## Contexto de falla que guía este plan

`app/conda_v5.md` agrupa fallos en 5 familias:
1. Ruptura del fibrador estocástico/Choi-Stinespring por `complex128` y falta de `__call__`.
2. Inestabilidad métrica/geodésica en el fibrador de atención.
3. Singularidades logarítmicas y tolerancias inadecuadas en el minimizador MAC / auditor Bures-Uhlmann.
4. Desalineación dimensional en pruebas de anclaje escalar (longitud fija 5).
5. Ruptura funtorial / abstracciones permeables (`_HAS_PSUTIL`, `CategoricalState.metadata`, `validate_faithful_state`, kwargs no soportados).

`app/suturas_rigurosas.md` refuerza la doctrina: producción debe quedar contractualmente correcta; las pruebas pueden adaptarse si estaban codificando una suposición vieja.

---

## Task 1: Restaurar compatibilidad de la capa funtorial base y de los estados categóricos

**Objetivo:** Eliminar los errores de contrato que impiden instanciar o usar los objetos base (`CategoricalState`, funtores, validaciones auxiliares).

**Files:**
- Modify: `app/core/mic_algebra.py`
- Modify: `app/core/immune_system/topological_watcher.py`
- Modify: `app/agents/mic_agent.py`
- Modify: `app/adapters/mic_vectors.py`
- Test: `tests/unit/core/test_mic_algebra.py`
- Test: `tests/unit/core/immune_system/test_topological_watcher.py`
- Test: `tests/unit/agents/test_mic_agent.py`
- Test: `tests/integration/dynamic_stress/test_adversarial_homology_and_thermal_collapse_v2.py`

**Step 1: Reproducir el contrato roto con tests focalizados**

Run:
```bash
pytest -q tests/unit/core/test_mic_algebra.py::TestProperties::test_stratum_chain_total_order
pytest -q tests/unit/core/immune_system/test_topological_watcher.py -k 'metadata or success or validate_faithful_state'
pytest -q tests/unit/agents/test_mic_agent.py -k 'encapsulate_monad or force_override'
```
Expected: fallos explícitos por `metadata` ausente, kwargs no soportados o alias faltantes.

**Step 2: Extender `CategoricalState` sin romper compatibilidad**

En `app/core/mic_algebra.py`:
- Añadir soporte explícito para `metadata` si la suite lo sigue esperando, o exponer un alias derivado desde `context`/`forensic_evidence`.
- Mantener `success` y `error` coherentes con el contrato actual.
- Verificar que `add_trace()` preserve la inmutabilidad.

**Step 3: Publicar alias de API en `TomitaTakesakiTheory` o donde corresponda**

En `app/core/immune_system/topological_watcher.py`:
- Si `validate_faithful_state` existe sólo como privado, exponer alias público o ajustar los callers.
- Alinear `ValueError`/`TopologicalInvariantError` para que los tests no dependan de keyword args ilegales.

**Step 4: Reintroducir `_HAS_PSUTIL` con gating robusto**

En `app/adapters/mic_vectors.py`:
- Definir `_HAS_PSUTIL` y `_psutil_module` de forma segura.
- Garantizar que `_measure_memory_mb()` siga retornando `0.0` cuando psutil no esté presente.

**Step 5: Compatibilizar `MICAgent.encapsulate_monad()`**

En `app/agents/mic_agent.py`:
- Aceptar `force_override` si la integración dinámica aún lo usa, o refactorizar la llamada interna para no pasar ese kwarg.
- Mantener el comportamiento semántico esperado por los tests de estrés.

**Step 6: Verificación**

Run:
```bash
pytest -q tests/unit/core/test_mic_algebra.py
pytest -q tests/unit/core/immune_system/test_topological_watcher.py -q
pytest -q tests/unit/agents/test_mic_agent.py -q
pytest -q tests/integration/dynamic_stress/test_adversarial_homology_and_thermal_collapse_v2.py -q
```
Expected: sin `AttributeError`/`TypeError` por API rota.

---

## Task 2: Resolver el caso Stinespring/Choi-Jamiołkowski con complejos y callable contract

**Objetivo:** Hacer que el fibrador isométrico acepte estados complejos y que el objeto sea invocable como esperan los tests.

**Files:**
- Modify: `app/wisdom/stinespring_isometric_fibrator.py`
- Test: `tests/unit/wisdom/test_stinespring_isometric_fibrator.py`

**Step 1: Reproducir el fallo mínimo**

Run:
```bash
pytest -q tests/unit/wisdom/test_stinespring_isometric_fibrator.py -q
```
Expected: fallo por `dtype('complex128')` y/o `__call__` abstracto/ausente.

**Step 2: Ajustar el contrato de tipos numéricos**

En `app/wisdom/stinespring_isometric_fibrator.py`:
- Declarar explícitamente tensores/espectros/operadores como `np.complex128` donde exista traza cuántica/Choi.
- Evitar casts implícitos a `float64` sobre estados con parte imaginaria.
- Mantener el resto del pipeline en complex hasta la proyección final, si aplica.

**Step 3: Implementar `__call__` como interfaz pública estable**

- Definir `__call__(self, state: CategoricalState) -> CategoricalState`.
- Encapsular allí el flujo actual de fibración/validación para que el objeto cumpla el contrato funtorial esperado.

**Step 4: Ajustar tests sólo si el contrato cambia**

En `tests/unit/wisdom/test_stinespring_isometric_fibrator.py`:
- Si el test asumía `float64` en lugar de complejos, actualizar la aserción al dominio correcto.
- Añadir un test explícito para preservar la parte imaginaria cuando sea semánticamente necesaria.

**Step 5: Verificación**

Run:
```bash
pytest -q tests/unit/wisdom/test_stinespring_isometric_fibrator.py -q
```
Expected: el test suite pasa sin `Cannot cast ufunc 'multiply'...` y sin fallo por falta de `__call__`.

---

## Task 3: Endurecer el fibrador de atención geodésica frente a métricas degeneradas

**Objetivo:** Evitar que `g_eff` y el proyector `P` rompan covarianza/ortogonalidad por inestabilidad espectral.

**Files:**
- Modify: `app/boole/wisdom/geodesic_attention_fibrator.py`
- Test: `tests/unit/boole/wisdom/test_geodesic_attention_fibrator.py`

**Step 1: Reproducir los tests de estabilidad**

Run:
```bash
pytest -q tests/unit/boole/wisdom/test_geodesic_attention_fibrator.py -q
```
Expected: fallos en `effective_metric no es definida positiva`, `P no es covariante-ortogonal respecto a g_eff`, y transportes no preservados.

**Step 2: Introducir regularización de Tikhonov previa al proyector**

En `app/boole/wisdom/geodesic_attention_fibrator.py`:
- Aplicar desplazamiento espectral adaptativo antes de formar `P`.
- Garantizar que `g_eff_reg` sea SPD antes de evaluar ortogonalidad.
- Usar tolerancias coherentes para el chequeo `P^T g_eff P ≈ g_eff`.

**Step 3: Separar validación de forma vs validación numérica**

- Validar primero shape/simetría.
- Luego validar definitud positiva.
- Sólo después construir proyector y transporte.

**Step 4: Alinear tests con la geometría real**

En `tests/unit/boole/wisdom/test_geodesic_attention_fibrator.py`:
- Si el caso nominal usa métrica regularizada, afirmar contra `g_eff_reg` y no contra la métrica cruda.
- Mantener tests de casos degenerados como `xfail` sólo si el contrato de producción no puede garantizar la propiedad exacta.

**Step 5: Verificación**

Run:
```bash
pytest -q tests/unit/boole/wisdom/test_geodesic_attention_fibrator.py -q
```
Expected: `P` covariante-ortogonal bajo tolerancia y sin `NumericalInstabilityError` en los caminos nominales.

---

## Task 4: Corregir el MAC minimizer y la auditoría Bures-Uhlmann

**Objetivo:** Evitar `log(0)`, `invalid value encountered in log` y tolerancias demasiado estrictas en distancias cuánticas.

**Files:**
- Modify: `app/boole/tactics/mac_minimizer.py`
- Test: `tests/unit/boole/tactics/test_mac_minimizer.py`

**Step 1: Reproducir el bloque de fallos**

Run:
```bash
pytest -q tests/unit/boole/tactics/test_mac_minimizer.py -q
```
Expected: `FloatingPointError` en log y algunos `AssertionError` por tolerancias.

**Step 2: Proteger el cálculo de entropía**

En `app/boole/tactics/mac_minimizer.py`:
- Reemplazar `log(eigenvalues)` por `log(np.maximum(eigenvalues, eps))` en rutas de von Neumann / Shannon cuando corresponda.
- Preservar el límite `λ log λ -> 0` para estados puros.
- No introducir sesgo innecesario en estados ya estrictamente positivos.

**Step 3: Ajustar los tests de distancia/identidad cuántica**

En `tests/unit/boole/tactics/test_mac_minimizer.py`:
- Elevar tolerancias absolutas donde el valor cero es semántico pero la ruta numérica es ruidosa.
- Mantener chequeos estrictos en propiedades estructurales, no en el último decimal.

**Step 4: Verificación**

Run:
```bash
pytest -q tests/unit/boole/tactics/test_mac_minimizer.py -q
```
Expected: sin `divide by zero encountered in log` ni fallos por `invalid value encountered in log`.

---

## Task 5: Arreglar el caso dimensional fijo de longitud 5

**Objetivo:** Eliminar el desajuste entre fixtures y el tamaño geométrico esperado por la prueba de anclaje escalar.

**Files:**
- Modify: `tests/unit/boole/physics/test_ast_static_analyzer.py`
- Modify: fixtures relacionados en `tests/unit/boole/physics/`

**Step 1: Localizar el origen de la longitud incorrecta**

Run:
```bash
pytest -q tests/unit/boole/physics/test_ast_static_analyzer.py -q
```
Expected: error por longitud no coincidente o inferencia dimensional incorrecta.

**Step 2: Fijar el dominio a longitud 5 donde el contrato lo exige**

- Asegurar que el vector de entrada y `ϕ` tengan longitud exacta 5 en los fixtures relevantes.
- Mantener la lógica general del analizador sin forzar otros módulos a asumir dimensión 5 globalmente.

**Step 3: Verificación**

Run:
```bash
pytest -q tests/unit/boole/physics/test_ast_static_analyzer.py -q
```
Expected: el test de extracción/anclaje pasa con dimensión consistente.

---

## Task 6: Recuperar la integridad del flujo de negocio y la delegación MIC

**Objetivo:** Hacer que `evaluate_project()` siempre retorne un reporte válido y que la delegación al MIC preserve los campos requeridos.

**Files:**
- Modify: `app/strategy/business_agent.py`
- Modify: `tests/test_business_agent_integration.py`
- Modify: `tests/unit/strategy/test_business_agent.py`

**Step 1: Reproducir los fallos de retorno nulo**

Run:
```bash
pytest -q tests/test_business_agent_integration.py -q
```
Expected: `evaluate_project` devolviendo `None` y atributos faltantes en `strategic_narrative`/`details`.

**Step 2: Garantizar retorno no nulo y estructura estable**

En `app/strategy/business_agent.py`:
- Asegurar que `evaluate_project()` construya y retorne siempre un objeto de reporte.
- Rellenar `strategic_narrative`, `details` y métricas numéricas aun en casos vacíos o parciales.

**Step 3: Verificar la delegación MIC**

- Confirmar que el flujo invoca el camino esperado (`project_intent` u homólogo real) sin depender de mocks obsoletos.
- Mantener compatibilidad con los campos que la integración lee del payload.

**Step 4: Verificación**

Run:
```bash
pytest -q tests/test_business_agent_integration.py -q
pytest -q tests/unit/strategy/test_business_agent.py -q
```
Expected: reportes no nulos, narrativa estable y payload con campos requeridos.

---

## Task 7: Cierre de regresión y validación cruzada

**Objetivo:** Confirmar que los cambios no rompen las suturas y que las tolerancias quedaron coherentes.

**Files:**
- Any touched files above
- Optional: `app/suturas_rigurosas.md` only if the sutura necesita una nota de contrato nuevo

**Step 1: Ejecutar las suites focalizadas**

Run:
```bash
pytest -q tests/unit/wisdom/test_stinespring_isometric_fibrator.py -q
pytest -q tests/unit/boole/wisdom/test_geodesic_attention_fibrator.py -q
pytest -q tests/unit/boole/tactics/test_mac_minimizer.py -q
pytest -q tests/test_business_agent_integration.py -q
pytest -q tests/unit/core/immune_system/test_topological_watcher.py -q
pytest -q tests/integration/dynamic_stress/test_adversarial_homology_and_thermal_collapse_v2.py -q
```

**Step 2: Ejecutar un barrido de regresión del subconjunto afectado**

Run:
```bash
pytest -q tests/unit/wisdom tests/unit/boole/wisdom tests/unit/boole/tactics tests/unit/core tests/unit/agents -q
```
Expected: verde en el subconjunto afectado o fallos residuales claramente nuevos y aislados.

**Step 3: Documentar cualquier ajuste contractual nuevo**

- Si algún test tenía una expectativa obsoleta, registrar el cambio de contrato en `app/suturas_rigurosas.md` con una nota breve y precisa.
- Mantener el texto consistente con la doctrina de sutura: producción primero, tests como consumidores del contrato.

---

## Riesgos y tradeoffs

- Algunas fallas en `conda_v5.md` son efectos en cascada; conviene arreglar primero los contratos base (`CategoricalState`, funtores, optional deps) antes de tocar numerics complejos.
- No sobre-regularizar: en módulos espectrales/geodésicos, usar la mínima tolerancia que haga estable el caso nominal.
- Si un test exige una propiedad matemática que el código no puede garantizar de forma robusta, revisar si el test está codificando una idealización demasiado fuerte.

---

## Orden recomendado de ejecución

1. Contratos base y compatibilidad de API.
2. Stinespring / complejos.
3. Geodesic attention / métrica efectiva.
4. MAC minimizer / logs y tolerancias.
5. Dimensión fija 5.
6. Business agent / evaluación no nula.
7. Regresión completa del subconjunto afectado.

---

## Resultado esperado

- Los errores listados en `app/conda_v5.md` se reducen a cero en el subconjunto afectado.
- La base queda alineada con `app/suturas_rigurosas.md`.
- La próxima iteración puede centrarse en limpiar regresiones residuales de otras familias de tests sin reabrir estos contratos.

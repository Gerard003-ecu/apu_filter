# Gobernanza Computacional Federada y Malla Agéntica

El patrón jerárquico Manager-Worker para la prevención de resonancia destructiva, gobernado por la **Ley de Clausura Transitiva** (tabla canónica): $V_{\aleph_0} \subsetneq V_{\mathbb{P}} \subsetneq V_{\mathbb{T}} \subsetneq V_{\mathbb{S}} \subsetneq V_{\mathbb{W}}$.

```mermaid
graph TD
    classDef orchestrator fill:#0f3460,stroke:#e94560,stroke-width:3px,color:#fff;
    classDef delegates fill:#1a1a2e,stroke:#fff,stroke-width:2px,color:#fff;
    classDef workers fill:#16213e,stroke:#4a4e69,stroke-width:1px,color:#fff;
    classDef error_node fill:#ef4444,stroke:#000,stroke-width:3px,color:#fff;

    %% Vértice: Orquestador (Manager)
    O[Business Agent Orquestador<br>Política de Alto Nivel]:::orchestrator

    %% Estrato Medio: Sabios (Delegates)
    D1[Topological Watcher<br>Matriz Ortogonal Topológica]:::delegates
    D2[Laplace Oracle<br>Matriz Ortogonal Financiera]:::delegates

    %% Base: Trabajadores (Workers)
    W1[Worker 1<br>Policy-as-Code (APUs)]:::workers
    W2[Worker 2<br>Policy-as-Code (Insumos)]:::error_node
    W3[Worker 3<br>Policy-as-Code (Cantidades)]:::workers

    %% Propagación Descendente
    O --> D1
    O --> D2
    D1 --> W1
    D1 --> W2
    D2 --> W3

    %% Propagación de Cumplimiento (Mónadas de Error)
    W2 -. "Ciclo Mutante (β1>0)" .-> D1
    style D1 stroke:#ef4444,stroke-width:4px
    D1 -. "Veto Estructural Topológico" .-> O
    style O fill:#ef4444,stroke:#fff,stroke-width:4px
```

---

## 1. El Protocolo de Escalada: Mónadas de Error

En la arquitectura de Gobernanza Federada, los fallos no generan excepciones en cascada desordenadas. Se propagan hacia arriba de forma **algebraicamente controlada** mediante **Mónadas de Error** — contenedores inmutables que envuelven el fallo junto con su contexto matemático completo.

### 1.1 Condiciones de Escalada

El sistema reconoce dos rutas de escalada:

| Ruta | Condición de Disparo | Tipo de Mónada | Acción del Delegate |
|---|---|---|---|
| **W2 → D1** | Worker 2 detecta $\beta_1^- > 0$ en Insumos | `TopologicalFaultMonad` | D1 calcula el impacto en la homología global |
| **D1 → O** | D1 confirma que la fusión $K_A \cup K_B$ genera $\Delta\beta_1 > 0$ | `StructuralVetoMonad` | O emite Acta de Deliberación de RECHAZO |
| **W3 → D2** | Worker 3 detecta $\sigma > 0$ en Cantidades | `SpectralFaultMonad` | D2 recalcula los polos de Laplace |
| **D2 → O** | D2 confirma inestabilidad espectral $|\mu_k| > 1$ | `SpectralVetoMonad` | O emite Acta de Deliberación de RECHAZO |

### 1.2 Estructura de una Mónada de Error

```json
{
  "monad_type": "TopologicalFaultMonad",
  "source_worker": "Worker2_Insumos",
  "timestamp": "2026-03-28T20:00:00Z",
  "invariant_violated": "beta_1_minus",
  "value_detected": 2,
  "value_threshold": 0,
  "cycle_nodes": ["CEMENTO_A", "TRANSPORTE_B", "OBRA_C"],
  "propagates_to": "TopologicalWatcher_D1",
  "fast_fail": true,
  "audit_hash": "sha256:..."
}
```

---

## 2. Política de Cuarentena de Nodos

Cuando un Delegate (D1 o D2) emite un Veto Estructural hacia el Orquestador (O), el nodo fuente del fallo entra en **Cuarentena Topológica** siguiendo el protocolo:

### 2.1 Estados de Cuarentena

| Estado | Condición | Acciones Permitidas |
|---|---|---|
| `ACTIVE` | Sin anomalías | Todas |
| `WARN` | $\Psi_{\min} \leq \Psi < 1.0$ o $\beta_2 > 0$ | Lectura, recomendaciones correctivas |
| `QUARANTINE` | $\beta_1^- > 0$ o $\Psi < \Psi_{\min}$ o $\sigma > 0$ | Solo auditoría forense. Sin escritura. |
| `ANNIHILATED` | $P_{\text{diss}} < 0$ o hash inválido | Nodo desconectado del grafo. Fast-Fail total. |

### 2.2 Condición de Liberación de Cuarentena

Un nodo en estado `QUARANTINE` puede retornar a `ACTIVE` **únicamente** si:
1. El invariante violado ha sido saneado (verificado por el Delegate correspondiente).
2. Un nuevo ciclo completo de validación Multi-Estrato ha sido ejecutado desde el Nivel 0 (ℵ₀).
3. El Orquestador emite un `RevalidationToken` firmado criptográficamente.

**No existe liberación directa de software** — la restauración requiere atravesar todos los estratos de la Ley de Clausura Transitiva (Ley 5 de Irreversibilidad del Veto Físico).

### 2.3 Política cuando D1 emite Veto y O está en estado CRITICAL

Cuando el Orquestador O ya se encuentra en estado `CRITICAL` (veto suave activo) y simultáneamente recibe un `StructuralVetoMonad` de D1 (veto duro):

> **Ley de Supremo de Estados:** El sistema aplica la operación $\text{CRITICAL} \sqcup \text{RECHAZO} = \text{RECHAZO}$.
> El estado del Orquestador escala irrevocablemente a `RECHAZO`. Ningún operador puede revertir este estado sin el protocolo de Liberación de Cuarentena completo del § 2.2.

---

## 3. Garantía Algebraica de No-Resonancia

El patrón Manager-Worker previene la resonancia destructiva mediante la **Ley de Conservación del Rango-Nulidad**:

$$\text{rank}(\text{MIC}) = n = \text{número de agentes activos}$$

Si un Worker entra en `ANNIHILATED`, el rango efectivo de la MIC disminuye. El Orquestador detecta automáticamente $\text{rank}(\text{MIC}') < n$ y:
1. Suspende todas las deliberaciones activas.
2. Emite un `MICDegradationAlert`.
3. Exige reposición del Worker o aprobación explícita humana para operar con MIC degenerada.

Esto garantiza que **ninguna decisión de alta jerarquía se tome con información incompleta o colineal**, previniendo la resonancia sistémica entre agentes degradados.

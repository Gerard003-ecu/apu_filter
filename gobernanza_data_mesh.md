# Gobernanza Computacional Federada y Malla Agéntica

El patrón jerárquico Manager-Worker para la prevención de resonancia destructiva:


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

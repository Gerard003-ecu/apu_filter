---
name: r02-topological-acyclicity
description: "Regla R-02: Conservacion Topologica y Aciclicidad Estricta beta_1 = 0. Prohibe la generacion o aprobacion de grafos presupuestales con ciclos homologicos o fragmentacion algebraica."
trigger: model_decision
---

# Regla R-02: Conservación Topológica y Aciclicidad Estricta ($\beta_1 \equiv 0$)

Esta regla gobierna la topología simplicial del grafo de contratos, insumos y subcontratistas en APU Filter.

---

## 1. Declaración del Invariante Constitucional

Queda formalmente prohibida la validación de presupuestos, subcontratos o redes de transporte que contengan ciclos homológicos no nulos en su 1-esqueleto simplicial.

### Restricciones Topológicas Inviolables:
1. **Aciclicidad Homológica Estricta:**
   $$\beta_1 = |E| - |V| + \beta_0 \equiv 0$$
   Todo 1-ciclo cerrado $\gamma \in Z_1(K)$ debe ser frontera nula ($\partial_2 \equiv 0$). Un valor $\beta_1 > 0$ es evidencia matemática irrefutable de **triangulación dolosa de fletes o sobrefacturación circular**.
2. **Conectividad Algebraica de Fiedler:** El segundo autovalor del Laplaciano discreto $\mathbf{\Delta}_0 = \mathbf{d}_0^* \mathbf{d}_0$ debe ser estrictamente positivo:
   $$\lambda_2(\mathbf{\Delta}_0) > 10^{-6}$$
   Queda vetada la creación de componentes disjuntas o sub-grafos aislados que operen como "cajas negras" fuera de la supervisión de la interventoría.
3. **Ausencia de Torsión Discreta:** La forma normal de Smith $B_1 = U \cdot D \cdot V$ debe verificar que todos los divisores elementales satisfagan $d_i = 1$, garantizando $\operatorname{Tor}(H_k(K; \mathbb{Z})) \equiv \mathbf{0}$.

---

## 2. Directiva de Ejecución para el Agente

* **Auditoría en Cada Edición de Grafos:** Siempre que el agente agregue dependencias, cambie proveedores o ajuste cuadrillas en un APU, debe re-computar la característica de Euler-Poincaré y los números de Betti.
* **Bloqueo Inmediato:** Si el grafo resultante genera $\beta_1 \ge 1$, el agente no puede considerar la tarea finalizada; debe romper el ciclo eliminando la dependencia circular de costos.

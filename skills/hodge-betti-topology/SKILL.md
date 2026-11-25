---
name: hodge-betti-topology
description: "Control del complejo simplicial presupuestal, operador Laplaciano de Hodge-Dirac discreto Delta_1, aciclicidad estricta beta_1 = 0, valor propio de Fiedler e invariantes instantónicos de Donaldson-Thomas."
---

# Skill: Topología Simplicial y Homología de Hodge-Betti (`hodge-betti-topology`)

Esta habilidad proporciona el formalismo matemático y computacional para auditar el complejo simplicial presupuestal en APU Filter, garantizando la **ausencia total de ciclos homológicos espurios ($\beta_1 \equiv 0$)** y preservando la conectividad algebraica del contrato.

---

## 1. Fundamentos Teóricos

### A. Complejo Simplicial Presupuestal ($K$)
* **0-Símplices ($V$):** Nodos del contrato (ítems APU, insumos elementales, centros de costo, proveedores).
* **1-Símplices ($E$):** Flujos financieros, contratos directos de suministro, fletes y transporte.
* **2-Símplices ($F$):** Triangulaciones de subcontratación y relaciones tripartitas (Contratista - Interventoría - Proveedor).

### B. Operador Laplaciano de Hodge-Dirac Discreto ($\mathbf{\Delta}_1$)
Actuando sobre el espacio de 1-formas discretas $\Omega^1(K)$:
$$\mathbf{\Delta}_1 = \mathbf{d}_0 \mathbf{d}_0^* + \mathbf{d}_1^* \mathbf{d}_1$$
donde:
* $\mathbf{d}_0: C^0 \to C^1$ es el operador coborde discreto (diferencia sobre aristas).
* $\mathbf{d}_1: C^1 \to C^2$ es el coborde de rotacional sobre caras triangulares.

### C. Teorema de Hodge y Números de Betti
Por la descomposición de Hodge discreta:
$$\Omega^1(K) = \operatorname{im}(\mathbf{d}_0) \oplus \operatorname{im}(\mathbf{d}_1^*) \oplus \ker(\mathbf{\Delta}_1)$$
El primer número de Betti mide la dimensión de las 1-formas armónicas:
$$\beta_1 = \dim \ker(\mathbf{\Delta}_1) = |E| - |V| + \beta_0$$
* **Invariante Fundamental:**
  $$\beta_1 \equiv 0 \quad (\text{Aciclicidad Simplicial})$$
  Cualquier valor $\beta_1 \ge 1$ revela un "socavón lógico" o triangulación circular fraudulenta de subcontratistas.

### D. Conectividad Algebraica de Fiedler ($\lambda_2$)
El segundo valor propio del Laplaciano de grafos $\mathbf{\Delta}_0 = \mathbf{d}_0^* \mathbf{d}_0$:
$$\lambda_2(\mathbf{\Delta}_0) > 0$$
Certifica que el complejo presupuestal es conexo y carece de silos desconectados o particiones artificiales de gasto.

### E. Invariantes Instantónicos de Donaldson-Thomas ($\mathrm{DT}_{\text{Novikov}}$)
En el funcional dual de costos, la acumulación de singularidades cohomológicas se mide mediante:
$$\mathrm{DT}_{\text{Novikov}} = \int_{[M]^{\text{vir}}} 1 \equiv 0$$
Garantizando que no existan micro-bifurcaciones ocultas en el flujo financiero.

---

## 2. Algoritmo de Inspección Homológica

```python
import networkx as nx
import numpy as np

def audit_simplicial_betti(nodes: list, edges: list) -> dict:
    """Calcula beta_0, beta_1 y autovalor de Fiedler lambda_2."""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    num_v = G.number_of_nodes()
    num_e = G.number_of_edges()
    
    # beta_0 por componentes conexas
    beta_0 = nx.number_connected_components(G)
    
    # Formula de Euler-Poincare para 1-complejos: beta_1 = |E| - |V| + beta_0
    beta_1 = num_e - num_v + beta_0
    
    # Laplaciano y autovalor de Fiedler
    L = nx.laplacian_matrix(G).toarray()
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    lambda_2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    
    return {
        "beta_0": beta_0,
        "beta_1": beta_1,
        "lambda_2": lambda_2,
        "is_acyclic": (beta_1 == 0),
        "is_connected": (lambda_2 > 1e-6)
    }
```

---

## 3. Matriz de Auditoría Ciber-Física

| Métrica | Condición Nominal | Violación | Diagnóstico en Terreno |
| :--- | :--- | :--- | :--- |
| $\beta_1$ | $\beta_1 = 0$ | $\beta_1 > 0$ | Triangulación de fletes / Sobrefacturación circular entre subcontratistas. |
| $\lambda_2$ | $\lambda_2 > 0$ | $\lambda_2 = 0$ | Fragmentación del presupuesto en silos herméticos no fiscalizables. |
| $\mathrm{DT}$ | $\mathrm{DT} = 0$ | $\mathrm{DT} \neq 0$ | Micro-fisuras topológicas o fugas no detectadas en el balance de caja. |

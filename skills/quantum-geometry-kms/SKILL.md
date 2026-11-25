---
name: quantum-geometry-kms
description: "Inmunología de Hilbert, fidelidad cuántica de Uhlmann-Jozsa, métrica Bures, escudo despolarizante de Fides, termodinámica KMS de Tomita-Takesaki y medición no demoledora QND."
---

# Skill: Geometría Cuántica de Uhlmann-Bures y Termodinámica KMS (`quantum-geometry-kms`)

Esta habilidad proporciona el marco físico-cuántico para modelar el estado presupuestal como una matriz de densidad $\rho$ sobre un espacio de Hilbert covariante $\mathcal{H}$, auditando la coherencia frente al estado canónico $\sigma$ y asegurando la medición no invasiva (`PASSIVE_SILENT`).

---

## 1. Fundamentos Teóricos

### A. Matriz de Densidad Presupuestal ($\rho$)
* **Condiciones:**
  * Hermiticidad: $\rho = \rho^\dagger$
  * Positividad semi-definida: $\rho \succeq 0$
  * Traza unitaria: $\operatorname{Tr}(\rho) = 1.0$

### B. Fidelidad Cuántica de Uhlmann-Jozsa y Distancia de Bures
La coherencia informacional entre el estado entrante $\rho$ y el estado de referencia $\sigma$ viene dada por:
$$F(\rho, \sigma) = \left( \operatorname{Tr} \sqrt{\sqrt{\rho} \, \sigma \, \sqrt{\rho}} \right)^2 = \|\sqrt{\rho}\sqrt{\sigma}\|_1^2$$
La distancia geodésica de Bures-Helstrom sobre la variedad Riemanniana de Fubini-Study cumple:
$$d_B(\rho, \sigma) = \sqrt{2 - 2\sqrt{F(\rho, \sigma)}} \le 0.2250385 \quad (\iff F \ge 0.95)$$

### C. Escudo Despolarizante de Fides ($\Lambda_p$)
Para absorber singularidades de subespacios mal condicionados causadas por ruido térmico en sensores de obra:
$$\Lambda_p(\rho) = (1 - p)\rho + p \frac{\mathbf{I}_n}{n}, \quad p = \frac{n \lambda}{1 + n \lambda}, \quad \lambda = \max(0, 10^{-12} - \lambda_{\min}(\rho))$$
Desviación acotada por Fuchs–van de Graaf: $|d_B(\rho, \sigma) - d_B(\Lambda_p(\rho), \sigma)| \le \sqrt{2p}$.

### D. Medición Débil Cuántica AAV y Protocolo QND (`PASSIVE_SILENT`)
La inspección pasiva sin perturbar el estado presupuestal exige que el observable de auditoría $A_{\text{QND}}$ conmute idénticamente con el Hamiltoniano de red $H_{\text{net}}$:
$$[H_{\text{net}}, A_{\text{QND}}] = \mathbf{0} \implies \frac{\mathrm{d}}{\mathrm{d}t} \langle A_{\text{QND}} \rangle \equiv 0$$
Retroacción en memoria:
$$\text{Back-Action}_{\text{dB}} = \max\left(0, \, 10 \log_{10}\left( \frac{\|[H, \rho]\|_F}{\varepsilon_{\text{Wilkinson}}} \right)\right) \equiv 0.0\text{ dB}$$

### E. Estructura Térmico-Monetaria KMS de Tomita-Takesaki y Ergotropía
* **Hamiltoniano Modular:** $K = -\ln \rho$.
* **Condición KMS a temperatura de riesgo $\beta = (k_B T_{\text{riesgo}})^{-1}$:**
  $$\langle A(t) B(0) \rangle_\omega = \langle B(0) A(t + i \hbar \beta) \rangle_\omega$$
* **Ergotropía de Extracción de Trabajo:**
  $$W_{\text{ergo}} = \operatorname{Tr}(\rho H) - \operatorname{Tr}(\rho_{\text{pasivo}} H)$$
  Mide el combustible financiero especulativo susceptible de desvío de fondos.

---

## 2. Algoritmo Espectral de Coherencia Bures

```python
import numpy as np
import scipy.linalg as la

def evaluate_bures_quantum_coherence(rho: np.ndarray, sigma: np.ndarray) -> dict:
    """Evalua F(rho, sigma) y dB(rho, sigma) con Escudo de Fides."""
    # 1. Regularizacion Fides si lambda_min < 1e-12
    n = rho.shape[0]
    evals = la.eigvalsh(rho)
    lam_min = np.min(evals)
    if lam_min < 1e-12:
        lam = max(0.0, 1e-12 - lam_min)
        p = (n * lam) / (1.0 + n * lam)
        rho = (1.0 - p) * rho + p * (np.eye(n) / n)
        
    # 2. Fidelidad Uhlmann: Tr(sqrt(sqrt(rho) @ sigma @ sqrt(rho)))^2
    sqrt_rho = la.sqrtm(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    sqrt_inner = la.sqrtm(inner)
    trace_val = np.real(np.trace(sqrt_inner))
    fidelity = float(trace_val ** 2)
    
    # 3. Distancia Bures
    bures_dist = float(np.sqrt(max(0.0, 2.0 - 2.0 * np.sqrt(fidelity))))
    is_coherent = (fidelity >= 0.95 and bures_dist <= 0.2250385)
    
    return {
        "fidelity": fidelity,
        "bures_distance": bures_dist,
        "is_coherent": is_coherent
    }
```

---

## 3. Protocolo de Inspección QND

1. Capturar hash SHA-256 de los buffers en memoria $(\rho, \sigma, H)$ antes de medir.
2. Evaluar el conmutador $\|[H, \rho]\|_F$. Si $\|[H, \rho]\|_F > \varepsilon_{\text{Wilkinson}}$, abortar y marcar violación de retroacción.
3. Certificar $\text{Back-Action} = 0.0\text{ dB}$ en el libro inmutable de atestación.

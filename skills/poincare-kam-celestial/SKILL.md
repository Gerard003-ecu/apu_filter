---
name: poincare-kam-celestial
description: "Mecánica celeste de Poincaré, teoría KAM, integración simpléctica Ruth-Forest-Yoshida, triángulo de Lie-Deprit, escisión asintótica de Lazutkin-Treschev y solapamiento de Chirikov."
---

# Skill: Mecánica Celeste de Poincaré-KAM y Dinámica No Lineal (`poincare-kam-celestial`)

Esta habilidad proporciona el motor dinámico para proyectar flujos presupuestales sobre el espacio de fases canónico de Acción-Ángulo de Delaunay-Poincaré $(J, \theta) \in T^*\mathbb{T}^n$, asegurando la estabilidad KAM frente a resonancias y perturbaciones macroeconómicas.

---

## 1. Fundamentos Teóricos

### A. Coordenadas Canónicas y Ecuación de Kepler
* Coordenadas de Delaunay: $(L, G, H, l, g, h) \mapsto (J_1, J_2, J_3, \theta_1, \theta_2, \theta_3)$.
* **Resolución super-cúbica de Halley:**
  $$E_{k+1} = E_k - \frac{f(E_k)}{f'(E_k) - \frac{f''(E_k)f(E_k)}{2f'(E_k)}}$$
  para $f(E) = E - e \sin E - M$, garantizando convergencia incondicional de orden 3.

### B. Integración Simpléctica de Ruth-Forest-Yoshida (Orden 4)
Para preservar estrictamente la 2-forma simpléctica $\omega = \sum \mathrm{d}p_i \wedge \mathrm{d}q_i$:
$$q_{k+1} = q_k + c_i \frac{\partial H}{\partial p} \Delta t, \quad p_{k+1} = p_k - d_i \frac{\partial H}{\partial q} \Delta t$$
con coeficientes universales:
$$w_1 = \frac{1}{2 - 2^{1/3}}, \quad w_0 = -\frac{2^{1/3}}{2 - 2^{1/3}}$$
$$c_1 = c_4 = \frac{w_1}{2}, \quad c_2 = c_3 = \frac{w_0 + w_1}{2}$$
$$d_1 = d_3 = w_1, \quad d_2 = w_0, \quad d_4 = 0$$

### C. Triángulo Homológico de Lie-Deprit (Orden 2)
Elimina los términos seculares de orden perturbativo en la función generatriz:
$$\{H_0, W_1\} + \tilde{H}_1 = 0$$
$$\{H_0, W_2\} + \frac{1}{2}\{W_1, H_1 + \tilde{H}_1\} + 2\tilde{H}_2 = 0$$

### D. Solapamiento de Resonancias de Chirikov ($K_{\text{Chirikov}}$)
$$\Delta J_{\text{separatriz}} = 4\sqrt{\frac{V_0}{m\omega^2}}, \quad K_{\text{Chirikov}} = \left( \frac{\Delta J_1 + \Delta J_2}{2 \, \delta J_{12}} \right)^2$$
* **Invariante:** $K_{\text{Chirikov}} < 1.0 \implies$ Confinamiento en toros KAM (Ausencia de caos difusivo de Arnol'd).

### E. Escisión Asintótica de Lazutkin-Treschev ($\theta_{\text{Laz}}$)
Para perturbaciones de alta frecuencia $\omega$:
$$\theta_{\text{Laz}} \approx \frac{C_{\text{Laz}}}{\varepsilon} \exp\left( -\frac{\pi \omega}{2\sqrt{\varepsilon}} \right), \quad C_{\text{Laz}} \approx 1118.82770595$$
Evalúa la permeabilidad del lóbulo turnstile y la tasa de fuga financiera.

---

## 2. Algoritmo de Diagnóstico de Estabilidad KAM

```python
import numpy as np

def audit_kam_stability(eps: float, omega: float, k_chirikov: float) -> dict:
    """Verifica criterios KAM y escision de Lazutkin."""
    c_laz = 1118.82770595
    # Angulo de escision de Lazutkin-Treschev
    arg_exp = -(np.pi * omega) / (2.0 * np.sqrt(max(1e-9, eps)))
    theta_laz = (c_laz / eps) * np.exp(arg_exp) if arg_exp > -700 else 0.0
    
    # Criterio de confinamiento de Chirikov
    is_confined = (k_chirikov < 1.0)
    # Cota de fuga por lobulo turnstile
    is_leak_free = (theta_laz <= 0.25)
    
    return {
        "k_chirikov": k_chirikov,
        "theta_laz": theta_laz,
        "kam_stable": is_confined and is_leak_free
    }
```

---

## 3. Matriz de Dinámica Celeste en Obra

| Indicador | Condición Estable | Violación | Fenómeno en Megaproyecto |
| :--- | :--- | :--- | :--- |
| $K_{\text{Chirikov}}$ | $< 1.0$ | $\ge 1.0$ | Resonancia destructiva: volatilidad de insumos devora el fondo de imprevistos. |
| $\theta_{\text{Laz}}$ | $\le 0.25$ | $> 0.25$ | Fuga transitoria de liquidez a través de la separatriz presupuestal. |
| Integración Simpléctica | Error energía $< 10^{-10}$ | Deriva monótona | Algoritmo numérico no conservativo disipando caja erróneamente. |

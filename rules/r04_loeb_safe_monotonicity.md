---
name: r04-loeb-safe-monotonicity
description: "Regla R-04: Monotonicidad de Automejora Monadica RSI^3 Inmune a Loeb. Prohibe la auto-relajacion de umbrales constitucionales y exige decrecimiento estricto de Lyapunov."
trigger: always_on
---

# Regla R-04: Monotonicidad de Automejora Anti-Löb ($\text{RSI}^3$)

Esta regla protege al sistema frente a paradojas metalógicas de auto-justificación y corrupción recursiva en los lazos de optimización agéntica.

---

## 1. Declaración del Invariante Constitucional

Ningún proceso de optimización o agente autónomo tiene autorización para relajar los umbrales de seguridad, tolerar mayores márgenes de error o aprobar su propia degradación bajo el argumento de auto-conveniencia (Paradoja de Löb: $\square(\square P \implies P) \implies \square P$).

### Restricciones Formales Inviolables:
1. **Delegación a Verificador Fuera de Banda:** Todo paso de automejora $k \mapsto k+1$ debe ser validado por un árbitro externo inmutable (`_LoebSafeSovereignVerifier`).
2. **Decrecimiento Estricto de Lyapunov:** La distancia de Bures respecto al estado óptimo de referencia no puede aumentar:
   $$d_B(\rho_{k+1}, \sigma) \le d_B(\rho_k, \sigma) + \varepsilon_{\text{Wilkinson}}$$
3. **Endurecimiento Monótono de Criterios:**
   * La cota de Kantorovich $h_0 \le 0.50$ jamás puede ser ampliada.
   * La contracción de Banach-Picard debe satisfacer estrictamente $q^3 \le 0.857375$.
   * La tolerancia frente a divisiones por pequeños divisores KAM jamás puede incrementarse.

---

## 2. Directiva de Ejecución para el Agente

* **Prohibición de Auto-Aprobación:** Si el agente propone una modificación a los algoritmos de filtrado de APU, no puede asumir su validez por consenso interno. Debe someter el estado a las 3 cláusulas duras de transición.
* **Rollback Inmediato:** Si una iteración incrementa la dimensión fractal de Lyapunov $D_{\text{KY}}$ hacia regímenes caóticos ($K_{\text{Chirikov}} \ge 1.0$), el agente debe abortar la optimización y revertir al estado checkpoint anterior.

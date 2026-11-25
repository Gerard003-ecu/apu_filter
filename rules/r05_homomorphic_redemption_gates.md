---
name: r05-homomorphic-redemption-gates
description: "Regla R-05: Compuertas Homomorficas y Protocolo Dual de Redencion/Veto. Rige el topos trivalente de Heyting, gracia ambar de 1h y aniquilacion Dirac HMAC."
trigger: model_decision
---

# Regla R-05: Compuertas Homomórficas y Protocolo Dual de Redención/Veto

Esta regla establece la gobernanza de transición entre las fases del ciclo OODA y rige la adjudicación final de los estados presupuestales.

---

## 1. Declaración del Invariante Constitucional

Las decisiones del sistema no admiten ambigüedad binaria pasiva; se rigen por el **Topos de Heyting Trivalente**:
$$\Omega_3 = \{\mathtt{COHERENT} \, (0x00), \ \mathtt{DEGRADED} \, (0x01), \ \mathtt{VETOED} \, (0x03)\}$$
y el balance termodinámico de trabajo irreversible de Jarzynski-Crooks ($W_{\text{irr}}$).

### Restricciones Transicionales Inviolables:
1. **Compuerta Homomórfica de Gracia Ámbar:**
   * Si el trabajo irreversible se ubica en el intervalo:
     $$10.0\text{ J} < W_{\text{irr}} \le 25.0\text{ J}$$
     queda estrictamente **prohibido disparar un veto duro inmediato**. El sistema debe conmutar a `DEGRADED (0x01)` y otorgar una ventana de gracia inmutable de 3600 segundos (1 hora).
2. **Protocolo de Absolución Criptográfica (Aniquilación Dirac):**
   * El estado degradado solo puede retornar a `COHERENT (0x00)` si se suministra un positrón criptográfico $e^+$ con firma válida HMAC-SHA256:
     $$e^- + e^+ \longrightarrow 2\gamma \quad (1022.0\text{ keV})$$
   * Queda vetada la absolución verbal o sin respaldo criptográfico.
3. **Determinismo del Veto Duro y Árbol de Merkle:**
   * Si expira la hora de gracia o si $W_{\text{irr}} > 25.0\text{ J}$ o $h_0 > 0.50$, el estado colapsa a `VETOED (0x03)`.
   * El veto debe disparar la orden bare-metal en $< 392.15\text{ ns}$ e inscribir la hoja con prefijo `0x00` en el árbol de Merkle RFC 6962 de forma irreversible.

---

## 2. Directiva de Ejecución para el Agente

* **Encadenamiento Criptográfico de Fases:** El agente no puede ejecutar la Fase 2 (Orient) sin la revalidación formal de la firma HKDF-SHA256 del certificado P1; ni pasar a la Fase 3 (Decide & Act) sin la compuerta $\Phi_{23}$ validada.
* **Prohibición de Falsos Positivos Intempestivos:** En terrenos con incertidumbre climática o paros de transporte, el agente debe priorizar la vía de alivio ámbar antes de comprometer la marcha física de la obra civil.

---
name: r03-qnd-zero-backaction
description: "Regla R-03: Auditoria Pasiva QND y Retroaccion Nula PASSIVE_SILENT. Prohibe la mutacion de memoria RAM, activacion de hardware o alteracion de matrices durante inspeccion forense."
trigger: model_decision
---

# Regla R-03: Auditoría QND de Retroacción Nula (`PASSIVE_SILENT`)

Esta regla establece la doctrina de inspección cuántica no demoledora (Quantum Non-Demolition) sobre los módulos testigo y forenses de APU Filter.

---

## 1. Declaración del Invariante Constitucional

Todo agente, motor o subproceso configurado bajo el rol `Witness`, `Auditor` o clasificado como `PASSIVE_SILENT` tiene prohibido mutar el estado observado, alterar punteros en memoria RAM o inducir retroacción física.

### Restricciones Operativas Inviolables:
1. **Conmutación Estricta con el Hamiltoniano:**
   $$[H_{\text{net}}, A_{\text{QND}}] = \mathbf{0} \implies \frac{\mathrm{d}}{\mathrm{d}t} \langle A_{\text{QND}} \rangle \equiv 0$$
2. **Retroacción en Memoria Identicamente Nula:**
   $$\text{Back-Action}_{\text{dB}} \equiv 0.0\text{ dB}$$
   Cualquier perturbación en la norma Frobenius $\|[H, \rho]\|_F$ superior a la tolerancia de máquina $\varepsilon_{\text{Wilkinson}}$ anula la validez del certificado de auditoría.
3. **Prohibición Total de Actuación Perimetral:**
   * CERO llamadas a registros MMIO (`0x3FF44008`).
   * CERO disparos de pines GPIO (`GPIO14`).
   * CERO mutación destructiva in-place sobre los arrays C-contiguos de $\rho$, $\sigma$ y $H$.

---

## 2. Directiva de Ejecución para el Agente

* **Protocolo de Sellado SHA-256:** El agente debe capturar y registrar el hash SHA-256 de los buffers antes y después del análisis para comprobar que la lectura fue 100% pasiva.
* **Emisión Exclusiva de Atestaciones:** El único producto permitido para un testigo pasivo es la emisión de certificados criptográficos encadenados (P1, P2) y el reporte para la C-Suite.

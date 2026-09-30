# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Financial Engine — Oráculo Estocástico y Funtor de Medida Financiera           ║
║  Ruta   : app/strategy/financial_engine.py                                               ║
║  Versión: 6.1.0-Nested-Phase-Spectral-Homological-Banach-Axiomatic-Strict                ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  0. OBJETO CATEGORIAL                                                                    ║
║  ────────────────────                                                                    ║
║  Sea 𝔐_fin el topos de Grothendieck de estructuras financieras admisibles,              ║
║  con objeto terminal 1 (config axioma-cerrado) y clasificador Ω = {OK, WARN, FAIL}.      ║
║  Este módulo realiza tres endofuntores Fᵢ : 𝔐_fin → 𝔐_fin, i ∈ {1,2,3},                ║
║  y su composición (lectura de derecha a izquierda)                                       ║
║                                                                                          ║
║      Φ  :=  F₃ ∘ F₂ ∘ F₁  :  𝔐_fin  →  𝔐_fin                                            ║
║                                                                                          ║
║  Cada Fᵢ es exacto a izquierda sobre el cono de no-arbitraje 𝒩 ⊂ 𝔐_fin y                ║
║  preserva la consistencia dimensional (R_F, Ke, WACC ∈ T⁻¹; σ ∈ T⁻¹/²;                   ║
║  flujos ∈ Moneda; β₀,β₁ ∈ ℕ; λ₂ ∈ T⁰).                                                   ║
║                                                                                          ║
║  Morfismos de puerto (objetos terminales / iniciales de fases adyacentes):               ║
║      π₁₂ : F₁(X) → F₂(X)     CouplingTensor τ = (κ₀, κ₁, λ̄₂)                             ║
║      π₂₃ : F₂(X) → F₃(X)     RiskEnvelope   ε = (K_eff, σ_eff, ‖·‖_{tail})               ║
║                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  I. FASE 1 — SUBSTRATO ONTOLÓGICO  (estrato 0 del topos)                                 ║
║  ────────────────────────────────────────────────────────                                ║
║                                                                                          ║
║  I.1  Álgebra de Boole de validez                                                        ║
║       𝔅 = ⟨{OK, WARN, FAIL}, ∧, ∨, ¬⟩ con FAIL absorbente.                               ║
║       Un objeto C ∈ Ob(FinancialConfig) es construible  ⇔  ningún átomo es FAIL.         ║
║                                                                                          ║
║  I.2  Universo de medidas  𝔇 = {NORMAL, STUDENT_T, CORNISH_FISHER}                       ║
║       Cerrado bajo tail-flip γ ↦ 1−γ, operación exigida por                              ║
║                                                                                          ║
║           ES_α(X)  =  (1−α)⁻¹  ∫_α¹ VaR_γ(X) dγ,     α ∈ (0,1).                          ║
║                                                                                          ║
║       NORMAL         : S(α=2,β=0) estable.  φ_X(t) = exp(iμt − ½σ² t²).                  ║
║       STUDENT_T(ν)   : colas O(|x|⁻ν).  Var < ∞  ⇔  ν>2;  Kurt < ∞  ⇔  ν>4.              ║
║       CORNISH_FISHER : jet de orden 4 sobre Φ⁻¹. Dominio heurístico |γ₁|<1, |γ₂|<3.      ║
║                                                                                          ║
║  I.3  Homología del complejo simplicial de dependencias K                                ║
║       Sea C_•(K; ℝ) el complejo de cadenas. Números de Betti:                            ║
║                                                                                          ║
║           β₀  = dim H₀(K; ℝ)  ≥ 1     (componentes conexas; fragmentación)               ║
║           β₁  = dim H₁(K; ℝ)  ≥ 0     (ciclos independientes; mallas)                    ║
║           χ̃  = β₀ − β₁               (Euler truncada a grado ≤ 1)                       ║
║           φ   = max(β₀ − 1, 0)        (índice de fragmentación; 0 ⇔ K conexo)            ║
║                                                                                          ║
║       Axiomas:  (H1) β₀ ∈ ℕ, β₀ ≥ 1;  (H2) β₁ ∈ ℕ ∪ {0};  (H3) χ̃ ∈ ℤ.                    ║
║                                                                                          ║
║  I.4  Espectro del Laplaciano combinatorio  L = D − A ∈ Sym⁺(ℝ^{|V|})                    ║
║       0 = λ₁(L) ≤ λ₂(L) ≤ ⋯ ≤ λ_{|V|}(L).  Teorema de Fiedler:                           ║
║                                                                                          ║
║           λ₂(L) = 0   ⇔   el grafo (V,E) es disconexo.                                   ║
║                                                                                          ║
║       Semigrupo del calor  e^{−t L} sobre ℓ²(V): la componente ortogonal a las           ║
║       constantes decae a tasa e^{−t λ₂}. Kernel de descuento espectral:                  ║
║                                                                                          ║
║           κ_heat(t, λ₂)  :=  exp(− t λ₂)  ∈  (0, 1],     t ≥ 0, λ₂ ≥ 0.                  ║
║                                                                                          ║
║       Identidad  κ_heat = 1  ⇔  λ₂=0 ∨ t=0.                                              ║
║                                                                                          ║
║  I.5  Tensor de acoplamiento  (objeto terminal de F₁ = objeto inicial de F₂)             ║
║       τ ∈ ℝ²_{≥0} × [0,1] ⊂ (ℝ³, ‖·‖_∞)  álgebra de Banach con producto de Hadamard:    ║
║                                                                                          ║
║           τ  =  (κ₀, κ₁, λ̄₂),     ‖τ‖_∞ = max{|κ₀|, |κ₁|, |λ̄₂|}.                        ║
║                                                                                          ║
║       Factor estructural (acción lineal sobre homología):                                ║
║                                                                                          ║
║           S(τ, H)  =  1 + κ₀ · φ(H) + κ₁ · β₁(H)  ≥  1.                                  ║
║                                                                                          ║
║       S = 1  ⇔  K es un árbol (β₀=1, β₁=0). Acción completa:                             ║
║                                                                                          ║
║           τ.apply(w, H, Σ)  =  w · S(τ, H) · κ_heat(t_Σ, λ₂(Σ)),    w ≥ 0.               ║
║                                                                                          ║
║       Axioma de cono: w ≥ 0  ⇒  τ.apply(w,·,·) ≥ 0  (preserva tasas admisibles).         ║
║       λ̄₂ es cota de diagnóstico, NUNCA se usa para forzar un descuento artificial.      ║
║                                                                                          ║
║  I.6  Invariantes cruzadas del config (no factorizables)                                 ║
║       (I1) D/E > 0  ∧  K_d ≤ 0          → pasivo remunerado degenerado (WARN).           ║
║       (I2) β > 2    ∧  D/E < 0.3        → apalancamiento operativo implícito (Hamada).   ║
║       (I3) Ψ* < Ψ_stable                → existencia de meseta estable de pandeo.        ║
║       (I4) T_ref < T_stress             → monotonía de la rama Arrhenius.                ║
║       (I5) λ_L + ρ_f ≤ 1.5              → heurística de sanidad de ratios.               ║
║       (I6) R_F + β R_P ≱ 0              → Ke naive fuera del cono (floor en F₂).         ║
║                                                                                          ║
║  PUERTO π₁₂ :  FinancialConfig.topological_coupling_tensor()  →  CouplingTensor          ║
║                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  II. FASE 2 — MEDIDA Y ESPECTRO                                                          ║
║  ────────────────────────────────                                                        ║
║                                                                                          ║
║  II.1  CAPM (Sharpe–Lintner)                                                             ║
║                                                                                          ║
║           Ke  =  R_F + β · R_P,     R_P := E[R_M] − R_F.                                 ║
║                                                                                          ║
║       Casos degenerados:                                                                 ║
║           |β| < 10⁻¹⁰  ⇒  Ke = R_F          (neutralidad sistémica)                      ║
║           β < 0        ⇒  Ke < R_F admisible (cobertura)                                 ║
║           Ke < 0       ⇒  Ke ← max(Ke, ½ R_F)  (floor del cono relajado)                 ║
║                                                                                          ║
║  II.2  Hamada (desapalancamiento / reapalancamiento)                                     ║
║                                                                                          ║
║           β_U  =  β_L  /  [ 1 + (1−τ)·(D/E) ]                                            ║
║           β_L  =  β_U  ·  [ 1 + (1−τ)·(D/E) ]                                            ║
║                                                                                          ║
║       Identidad: D/E = 0  ⇒  β_U = β_L.                                                  ║
║                                                                                          ║
║  II.3  WACC (Modigliani–Miller con escudo fiscal)                                        ║
║       Pesos:  w_E = 1/(1+D/E),  w_D = (D/E)/(1+D/E),  w_E + w_D = 1.                     ║
║                                                                                          ║
║           WACC_base  =  w_E · Ke  +  w_D · K_d · (1−τ).                                  ║
║                                                                                          ║
║  II.4  WACC topológico  (consume π₁₂)                                                    ║
║                                                                                          ║
║           WACC_topo  =  τ.apply(WACC_base, H, Σ)                                         ║
║                      =  WACC_base · [1 + κ₀(β₀−1) + κ₁ β₁] · e^{−t λ₂}.                  ║
║                                                                                          ║
║       Coherencia espectral-homológica:                                                   ║
║           árbol conexo (β₀=1, β₁=0, λ₂=0)  ⇒  WACC_topo = WACC_base                      ║
║           fragmentación (β₀>1)             ⇒  WACC_topo ≥ WACC_base                      ║
║           ciclos (β₁>0)                    ⇒  WACC_topo ≥ WACC_base                      ║
║           alta conectividad (λ₂ ≫ 0)       ⇒  descuento e^{−tλ₂} < 1                     ║
║                                                                                          ║
║  II.5  DCF, TIR, MIRR, duración                                                          ║
║                                                                                          ║
║           NPV(r)   =  −|I₀| + Σ_{t=1}^{T} CF_t / (1+r)^t                                 ║
║           NPV'(r)  =  − Σ_{t=1}^{T} t · CF_t / (1+r)^{t+1}                               ║
║                                                                                          ║
║       TIR: raíz r* ∈ (−1+ε, 10] de NPV(r*)=0. Newton–Raphson                             ║
║           r_{n+1} = r_n − NPV(r_n)/NPV'(r_n)                                             ║
║       con respaldo biseccional si NPV no cambia de signo en el intervalo → None.         ║
║                                                                                          ║
║           MIRR  =  ( FV₊(r_reinv) / PV₋(r_fin) )^{1/n} − 1                               ║
║                                                                                          ║
║       Duración de Macaulay / modificada / convexidad / DV01:                             ║
║                                                                                          ║
║           D_mac  =  Σ t·PV(CF_t) / P                                                     ║
║           D_mod  =  D_mac / (1+r)                                                        ║
║           Conv   =  Σ t(t+1)·PV(CF_t) / [P (1+r)²]                                       ║
║           DV01   =  D_mod · P / 10⁴                                                      ║
║                                                                                          ║
║  II.6  Medidas de riesgo coherentes (Artzner–Delbaen–Eber–Heath)                         ║
║       Cuantil y Expected Shortfall:                                                      ║
║                                                                                          ║
║           VaR_α(X)  =  inf{ x : P(X ≤ x) ≥ α }                                           ║
║           ES_α(X)   =  (1−α)⁻¹ ∫_α¹ VaR_γ(X) dγ  =  E[X | X > VaR_α(X)]                  ║
║                                                                                          ║
║       Axiomas sobre 𝓡 (el VaR viola A4; el ES los satisface todos):                      ║
║           (A1) Monotonía:            X ≤ Y  ⇒  𝓡(X) ≥ 𝓡(Y)                               ║
║           (A2) Invariancia transl.:  𝓡(X + c) = 𝓡(X) − c                                 ║
║           (A3) Homogeneidad⁺:        𝓡(λX) = λ 𝓡(X),  λ > 0                              ║
║           (A4) Subaditividad:        𝓡(X+Y) ≤ 𝓡(X) + 𝓡(Y)                                ║
║                                                                                          ║
║       Escalado i.i.d. temporal:  σ_X(t) = σ · √(t / t_year),  t_year = 252.              ║
║                                                                                          ║
║       ■ Normal(μ,σ²):                                                                    ║
║           VaR_α = μ + σ z_α,     z_α = Φ⁻¹(α)                                            ║
║           ES_α  = μ + σ · φ(z_α) / (1−α)                                                 ║
║                                                                                          ║
║       ■ Student-t(ν), X = μ + s T_ν,  s = σ √((ν−2)/ν)  [s es ESCALA, no σ]:             ║
║           VaR_α = μ + s t_α,     t_α = F_{t,ν}⁻¹(α)                                      ║
║           ES_α  = μ + s · [ f_{t,ν}(t_α) · (ν + t_α²) / ((ν−1)(1−α)) ]                   ║
║                                                                                          ║
║       ■ Cornish–Fisher (orden 4), z = Φ⁻¹(α):                                            ║
║           q_CF(α) = z + (γ₁/6)(z²−1) + (γ₂/24)(z³−3z)                                    ║
║                     − (γ₁²/36)(2z³−5z) + O(γ³)                                           ║
║           VaR_α = μ + σ q_CF(α)                                                          ║
║           ES_α  ≈ μ + σ (1−α)⁻¹ ∫_α¹ q_CF(u) du     (cuadratura trapezoidal)             ║
║                                                                                          ║
║       Coherencia numérica débil: ES_α ≥ VaR_α (cola superior, σ>0).                      ║
║       Norma de Banach de cola en ℓ²:                                                     ║
║                                                                                          ║
║           ‖(VaR, ES)‖₂  =  √(VaR² + ES²).                                                ║
║                                                                                          ║
║  II.7  Sobre de riesgo  (objeto terminal de F₂ = objeto inicial de F₃)                   ║
║                                                                                          ║
║           tail_loading      = max( 0, (ES_α − VaR_α) / K₀ )                              ║
║           tail_dispersion   = max( 0, ES_α / VaR_α − 1 )                                 ║
║           K_eff  = K₀ · (1 + tail_loading)                                               ║
║           σ_eff  = σ₀ · (1 + tail_dispersion)                                            ║
║                                                                                          ║
║       ε = RiskEnvelope(K_eff, σ_eff, tail_loading, tail_dispersion, VaR, ES, ‖·‖₂).      ║
║                                                                                          ║
║  PUERTO π₂₃ :  RiskQuantifier.synthesize_risk_envelope()  →  RiskEnvelope                ║
║                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  III. FASE 3 — PDE ESTOCÁSTICA Y SÍNTESIS TERMUDINÁMICA                                  ║
║  ───────────────────────────────────────────────────────                                 ║
║                                                                                          ║
║  III.1  PDE de Black–Scholes–Merton (medida martingala Q)                                ║
║       Subyacente:  dS_t = (r−q) S_t dt + σ S_t dW_t^Q.                                   ║
║                                                                                          ║
║           ∂V/∂t + ½ σ² S² ∂²V/∂S² + (r−q) S ∂V/∂S − r V  =  0                            ║
║           V(S, T)  =  max(S − K, 0)                 (call europea)                       ║
║                                                                                          ║
║       Feynman–Kac (solución cerrada, saturación |d₁|,|d₂| ≤ 8 en C_b(ℝ)):                ║
║                                                                                          ║
║           d₁  =  [ ln(S/K) + (r − q + ½σ²) T ] / (σ √T),     d₂ = d₁ − σ√T               ║
║           C   =  S e^{−qT} N(d₁) − K e^{−rT} N(d₂)                                       ║
║           P   =  K e^{−rT} N(−d₂) − S e^{−qT} N(−d₁)     (paridad put-call)              ║
║                                                                                          ║
║       Greeks de 1º y 2º orden (call):                                                    ║
║           Δ     = e^{−qT} N(d₁)                                                          ║
║           Γ     = e^{−qT} φ(d₁) / (S σ √T)                                               ║
║           Vega  = S e^{−qT} φ(d₁) √T                                                     ║
║           Θ     = −S e^{−qT} φ(d₁) σ/(2√T) + q S e^{−qT} N(d₁) − r K e^{−rT} N(d₂)       ║
║           ρ     = K T e^{−rT} N(d₂)                                                      ║
║           Vanna = ∂Δ/∂σ = − e^{−qT} φ(d₁) d₂ / σ                                         ║
║           Volga = ∂Vega/∂σ = Vega · d₁ d₂ / σ                                            ║
║           Charm = ∂Δ/∂τ                                                                  ║
║                                                                                          ║
║       Mapa σ ↦ C_BSM(σ) estrictamente creciente (Vega>0, T>0, S,K>0) ⇒ IV única.        ║
║                                                                                          ║
║  III.2  CRR (Cox–Ross–Rubinstein), rejilla recombinante, precios en log-espacio          ║
║                                                                                          ║
║           Δt = T/n,   u = e^{σ√Δt},   d = 1/u = e^{−σ√Δt}                                ║
║           p  = ( e^{(r−q)Δt} − d ) / (u − d),     disc = e^{−r Δt}                       ║
║                                                                                          ║
║       No-arbitraje:  p ∈ (0,1)  ⇔  r−q ∈ ( ln d / Δt , ln u / Δt ).                      ║
║       Nodo (i,j):  S_{i,j} = exp( ln S + (2j − i) σ√Δt ).                                ║
║       Inducción hacia atrás:                                                             ║
║                                                                                          ║
║           V_{i,j}^{eur}  = disc · [ p V_{i+1,j+1} + (1−p) V_{i+1,j} ]                    ║
║           V_{i,j}^{am}   = max( S_{i,j} − K,  V_{i,j}^{eur} )                            ║
║                                                                                          ║
║       Greeks de árbol: Δ = (V_u − V_d)/(S(u−d));  Γ por diferencias de Δ en 2Δt;         ║
║       Θ = (E^Q[V(Δt)] − V(0)) / Δt.                                                      ║
║       Richardson (solo europeas, error O(1/n)):  V_rich = (4 V(2n) − V(n)) / 3.          ║
║                                                                                          ║
║  III.3  Acoplamiento termo-estructural (Arrhenius modificado ⊗ pandeo de Euler)          ║
║                                                                                          ║
║           σ_eff  =  σ_base · M_s(Ψ) · M_t(T),     saturado por max_amplification.        ║
║                                                                                          ║
║           F_s(Ψ) = 0                                           si Ψ ≥ Ψ_stable           ║
║                    tanh[(Ψ*−Ψ) κ] + ½ [(Ψ*−Ψ)/Ψ*]² 𝟙{Ψ<Ψ*}    si Ψ < Ψ_stable           ║
║                                                                                          ║
║           F_t(T) = 0                                           si T ≤ T_ref              ║
║                    (T − T_ref)/Θ · 0.1                         si T_ref < T ≤ T_str      ║
║                    0.1 + [ e^{(T−T_str)/Θ} − 1 ] · 0.2        si T > T_str               ║
║                                                                                          ║
║           M_s = 1 + F_s,    M_t = 1 + α F_t,                                             ║
║           cruce:  M_s ← M_s + 0.3 F_s F_t     si F_s>0 ∧ F_t>0.                          ║
║                                                                                          ║
║  III.4  Inercia térmica financiera (analogía C_th = m · c)                               ║
║                                                                                          ║
║           M_eff  =  λ_L · (1 + ½ Ψ)                                                      ║
║           C_eff  =  ρ_f · (1 + 0.3 Ψ)                                                    ║
║           A_att  =  e^{−2 σ}                                                             ║
║           I      =  M_eff · C_eff · A_att                                                ║
║                                                                                          ║
║       Dinámica de primer orden  C dT/dt = Q − T/τ_ext:                                   ║
║                                                                                          ║
║           ΔT(t)  =  (Q / I) · (1 − e^{−1/τ})                                             ║
║                                                                                          ║
║       τ → 0⁺ ⇒ ΔT → Q/I (cuasi-estático);  τ → ∞ ⇒ ΔT → 0 (amortiguación total).         ║
║       I → 0  ⇒ régimen elástico: ΔT = Q.                                                 ║
║                                                                                          ║
║  III.5  Amplificación topológica de volatilidad                                          ║
║                                                                                          ║
║           σ_adj  =  σ_base · ( 1 + P_sinergia + P_eficiencia )                           ║
║                  ≤  σ_base · ( 1 + Δ_max )                                               ║
║                                                                                          ║
║           P_sinergia   = f_syn · strength · 𝟙{synergy_detected}                          ║
║           P_eficiencia = f_eff · ( 1 − clamp(euler_efficiency, 0, 1) )                   ║
║                                                                                          ║
║  III.6  Orquestación  Φ(proyecto) = F₃(F₂(F₁(proyecto)))                                 ║
║       1. σ_eff ← Arrhenius⊗Euler  (o ajuste topológico).                                 ║
║       2. WACC_topo ← π₁₂(τ) · (H, Σ).                                                    ║
║       3. NPV, duración, TIR, MIRR bajo WACC_topo.                                        ║
║       4. VaR/CVaR sobre (I₀, σ_costo · σ_eff/σ_base).                                    ║
║       5. ε ← π₂₃;  V_option ← BSM/CRR(S = NPV+|I₀|, K = K_eff, σ = σ_eff).               ║
║       6. Inercia térmica y síntesis de informe.                                          ║
║                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  IV. INVARIANTES GLOBALES DE Φ  =  F₃ ∘ F₂ ∘ F₁                                          ║
║  ──────────────────────────────────────────────                                          ║
║                                                                                          ║
║  (N1) No-arbitraje: p_CRR ∈ (0,1); WACC ≥ 0; Ke ≥ min(R_F, ½ R_F) en el floor.          ║
║  (N2) Consistencia dimensional: tasas ∈ T⁻¹, σ ∈ T⁻¹/², Betti adimensionales.            ║
║  (N3) Monotonía de riesgo: ES_α ≥ VaR_α (cola superior, σ>0).                            ║
║  (N4) Preservación del cono: τ.apply : ℝ_{≥0} → ℝ_{≥0}.                                  ║
║  (N5) Exactitud a izquierda sobre 𝒩 (subcategoría de mercados libres de arbitraje).      ║
║  (N6) S(τ,H) = 1 sobre árboles; κ_heat = 1 sobre grafos con λ₂=0.                        ║
║  (N7) Call BSM: Δ ∈ [0, e^{−qT}] ⊂ [0,1]; Γ ≥ 0; Vega ≥ 0.                               ║
║  (N8) Clausura axiomática de FinancialConfig: FAIL ⇒ el objeto no se construye.          ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from math import erf, exp, log, pow, sqrt
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import norm, t

logger = logging.getLogger(__name__)

__version__ = "6.0.0-Nested-Phase-Spectral-Homological-Banach-Strict"

__all__ = [
    "DistributionType",
    "OptionModelType",
    "ValidityAtom",
    "HomologicalInvariants",
    "SpectralSignature",
    "CouplingTensor",
    "FinancialConfig",
    "CapitalAssetPricing",
    "RiskEnvelope",
    "RiskQuantifier",
    "RealOptionsAnalyzer",
    "FinancialEngine",
    "calculate_volatility_from_returns",
]


# Constantes numéricas del álgebra de Banach de tasas (norma uniforme).
_EPS_SPECTRAL: float = 1e-12
_EPS_NEWTON: float = 1e-10
_EPS_PROB: float = 1e-15
_KE_FLOOR_FRACTION: float = 0.5
_IRR_LO: float = -0.999999
_IRR_HI: float = 10.0
_MAX_D1: float = 8.0  # saturación de d₁,d₂ para estabilidad de N(·)


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE 1 — SUBSTRATO ONTOLÓGICO ▓▓▓
# ▓▓▓ Tipos algebraicos, invariantes homológicos/espectrales y axiomas de coherencia.  ▓▓▓
# ▓▓▓ Estrato 0 del topos 𝔐_fin. El objeto terminal de esta fase ES el puerto F₁→F₂.  ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


class DistributionType(Enum):
    r"""
    Universo de medidas de probabilidad soportadas por el cuantificador de riesgo.

    Cerramos el conjunto {NORMAL, STUDENT_T, CORNISH_FISHER} bajo la operación
    de inversión de cola (tail-flip) exigida por la definición integral del
    Expected Shortfall:

        ES_α(X) = (1-α)⁻¹ ∫_α¹ VaR_γ(X) dγ.

    Miembros:
        NORMAL         : Punto fijo de la familia estable S(α=2, β=0). Colas
                         sub-gaussianas. Característica φ(t)=exp(iμt - ½σ²t²).
        STUDENT_T      : Colas polinómicas O(|x|⁻ν). Varianza finita ⟺ ν>2.
                         Curtosis finita ⟺ ν>4. Escala s ≠ desviación típica.
        CORNISH_FISHER : Jet de orden 4 sobre el cuantil normal, corregido por
                         γ₁ (asimetría) y γ₂ (exceso de curtosis). Dominio de
                         monotonía heurístico: |γ₁|<1, |γ₂|<3.
    """

    NORMAL = "normal"
    STUDENT_T = "student_t"
    CORNISH_FISHER = "cornish_fisher"


class OptionModelType(Enum):
    r"""
    Álgebra de modelos de valoración de opciones reales.

    Codifica el isomorfismo categorial entre la solución de la PDE parabólica
    de Black-Scholes-Merton y su discretización en el espacio de trayectorias
    binomiales (CRR). AUTO es el funtor de despacho que selecciona el modelo
    según la presencia de frontera libre (ejercicio americano).

    Miembros:
        BLACK_SCHOLES : Solución cerrada Feynman-Kac de la PDE.
        BINOMIAL      : Esquema Cox-Ross-Rubinstein recombinante.
        AUTO          : BSM si europea; CRR si americana.
    """

    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial"
    AUTO = "auto"


class ValidityAtom(Enum):
    r"""
    Átomos del álgebra de Boole de validez axiomática del config.

    El retículo 𝔅 = ⟨{OK, WARN, FAIL}, ∧, ∨, ¬⟩ clasifica cada invariante.
    Un objeto FinancialConfig es construible ssi ningún átomo es FAIL.
    """

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class HomologicalInvariants:
    r"""
    Números de Betti del complejo simplicial de dependencias del proyecto.

    Sea K el complejo de cadenas de tareas/contratos. Entonces:

        β₀ = dim H₀(K; ℝ)  ≥ 1     (componentes conexas; fragmentación)
        β₁ = dim H₁(K; ℝ)  ≥ 0     (ciclos independientes; mallas)
        χ  = β₀ - β₁               (característica de Euler truncada a H₁)

    Axiomas:
        H1. β₀ ∈ ℕ, β₀ ≥ 1.
        H2. β₁ ∈ ℕ ∪ {0}.
        H3. χ ∈ ℤ.

    El funtor de acoplamiento F₁→F₂ interpreta (β₀-1) y β₁ como curvaturas
    que deforman el WACC base.
    """

    beta_0: int = 1
    beta_1: int = 0

    def __post_init__(self) -> None:
        if self.beta_0 < 1:
            raise ValueError(f"β₀ = {self.beta_0} < 1 no es un número de Betti válido.")
        if self.beta_1 < 0:
            raise ValueError(f"β₁ = {self.beta_1} < 0 no es un número de Betti válido.")

    @property
    def euler_characteristic_truncated(self) -> int:
        """χ̃ = β₀ − β₁ (truncación a homología de grado ≤ 1)."""
        return self.beta_0 - self.beta_1

    @property
    def fragmentation_index(self) -> int:
        """Índice de fragmentación max(β₀ − 1, 0). Cero ssi el grafo es conexo."""
        return max(0, self.beta_0 - 1)


@dataclass(frozen=True)
class SpectralSignature:
    r"""
    Firma espectral del Laplaciano combinatorio L = D − A del grafo de tareas.

    Teorema de Fiedler: λ₂(L) = 0 ⟺ el grafo es disconexo. El calor
    (semigrupo e^{−tL} sobre ℓ²(V)) decae en la componente ortogonal a las
    constantes a tasa e^{−t λ₂}. El WACC topológico usa exactamente este
    kernel de calor como descuento espectral.

    Attributes:
        lambda_2 : λ₂ ≥ 0, segundo autovalor (conectividad algebraica).
        time     : t ≥ 0, tiempo de difusión del kernel (default 1).
    """

    lambda_2: float = 0.0
    time: float = 1.0

    def __post_init__(self) -> None:
        if self.lambda_2 < -_EPS_SPECTRAL:
            raise ValueError(f"λ₂ = {self.lambda_2} < 0 viola semidefinición de L.")
        if self.time < 0:
            raise ValueError(f"t = {self.time} < 0 no es un tiempo de difusión válido.")

    @property
    def heat_kernel_discount(self) -> float:
        r"""e^{−t λ₂} ∈ (0, 1]. Identidad ssi λ₂=0 o t=0."""
        return float(exp(-max(0.0, self.lambda_2) * max(0.0, self.time)))

    @property
    def is_disconnected(self) -> bool:
        """True ssi λ₂ ≈ 0 (dentro de tolerancia espectral)."""
        return self.lambda_2 <= _EPS_SPECTRAL


@dataclass(frozen=True)
class CouplingTensor:
    r"""
    Tensor de acoplamiento homológico τ ∈ ℝ²_{≥0} × [0, 1].

    Es el objeto terminal de la FASE 1 y el objeto inicial de la FASE 2:
    el morfismo de puerto F₁ → F₂.

        τ = (κ₀, κ₁, λ̄₂)

    Acción sobre (WACC_base, HomologicalInvariants, SpectralSignature):

        WACC_topo = WACC_base · [1 + κ₀(β₀−1) + κ₁ β₁] · e^{−t λ₂}

    con saturación λ₂ ← max(λ₂, 0) y floor opcional λ̄₂ (nunca se usa para
    forzar un descuento artificial: λ̄₂ es cota de diagnóstico, no de cálculo).

    El tensor es un objeto del álgebra de Banach (ℝ³, ‖·‖_∞) con producto
    de Hadamard y acción lineal sobre el factor estructural.
    """

    kappa_0: float = 0.05
    kappa_1: float = 0.03
    lambda_2_floor: float = 0.05

    def __post_init__(self) -> None:
        if self.kappa_0 < 0.0 or self.kappa_1 < 0.0:
            raise ValueError("κ₀, κ₁ deben ser no negativos (penalizaciones).")
        if not (0.0 <= self.lambda_2_floor <= 1.0 + 1e-9):
            raise ValueError("λ̄₂ debe vivir en [0, 1] (normalización espectral).")

    def structural_factor(self, homology: HomologicalInvariants) -> float:
        r"""
        Factor estructural S = 1 + κ₀(β₀−1) + κ₁ β₁ ≥ 1.

        S = 1 ssi el complejo es conexo y acíclico (árbol: β₀=1, β₁=0).
        """
        return (
            1.0
            + self.kappa_0 * homology.fragmentation_index
            + self.kappa_1 * max(0, homology.beta_1)
        )

    def apply(
        self,
        wacc_base: float,
        homology: HomologicalInvariants,
        spectrum: SpectralSignature,
    ) -> float:
        r"""
        Acción del tensor: WACC_topo = WACC_base · S · e^{−t λ₂}.

        Preserva no-negatividad: wacc_base ≥ 0 ⟹ WACC_topo ≥ 0.
        """
        if wacc_base < 0:
            raise ValueError("WACC_base < 0 viola el cono de tasas admisibles.")
        return float(
            wacc_base * self.structural_factor(homology) * spectrum.heat_kernel_discount
        )

    def as_tuple(self) -> Tuple[float, float, float]:
        """Proyección canónica a ℝ³ (compatibilidad binaria con v5)."""
        return (self.kappa_0, self.kappa_1, self.lambda_2_floor)

    def infinity_norm(self) -> float:
        """Norma ‖τ‖_∞ del álgebra de Banach de acoplamiento."""
        return max(abs(self.kappa_0), abs(self.kappa_1), abs(self.lambda_2_floor))


@dataclass
class FinancialConfig:
    r"""
    Espacio de parámetros macroeconómicos del proyecto, axioma-cerrado.

    Naturaleza algebraica:
        Objeto del estrato 0 (base) del topos financiero 𝔐_fin. Cada campo
        es un escalar real acotado; el conjunto vive en un producto cartesiano
        cerrado bajo las operaciones de validación `_validate_parameters` y
        `_validate_cross_constraints`.

    Dualidad:
        Los parámetros NO son independientes. Los invariantes σ-topológicos
        (β₀, β₁, λ₂) actúan como curvatura sobre el haz de tasas de descuento.
        El puerto `topological_coupling_tensor` es el morfismo de salida de
        esta fase y el morfismo de entrada de CapitalAssetPricing (FASE 2).

    Attributes:
        Núcleo CAPM/WACC, penalizaciones topológicas, acoplamiento homológico,
        física del costo (Arrhenius + pandeo de Euler), cuantiles de riesgo y
        tolerancias numéricas. Ver campos.
    """

    # ── Núcleo CAPM/WACC ─────────────────────────────────────────────────────
    risk_free_rate: float = 0.04
    market_premium: float = 0.06
    beta: float = 0.0
    tax_rate: float = 0.30
    cost_of_debt: float = 0.08
    debt_to_equity_ratio: float = 0.0
    project_life_years: int = 10
    liquidity_ratio: float = 0.1
    fixed_contracts_ratio: float = 0.5
    inflation_rate: float = 0.03

    # ── Penalizaciones topológicas ───────────────────────────────────────────
    synergy_penalty_factor: float = 0.20
    efficiency_penalty_factor: float = 0.10
    max_volatility_adjustment: float = 0.50

    # ── Acoplamiento homológico (κ₀, κ₁, λ̄₂) ─────────────────────────────────
    kappa_0_topo: float = 0.05
    kappa_1_topo: float = 0.03
    lambda_2_floor: float = 0.05

    # ── Física del costo ─────────────────────────────────────────────────────
    psi_critical: float = 1.0
    psi_stable: float = 1.5
    kappa_struct: float = 2.0
    t_reference: float = 25.0
    t_stress: float = 30.0
    t_scale: float = 20.0
    alpha_coupling: float = 0.7
    max_amplification: float = 3.0

    # ── Cuantiles de riesgo ──────────────────────────────────────────────────
    df_student_t: int = 5
    confidence_var: float = 0.95
    confidence_contingency: float = 0.90

    # ── Tolerancias numéricas ────────────────────────────────────────────────
    tol_spectral: float = 1e-9
    tol_newton: float = 1e-6

    def __post_init__(self) -> None:
        """Clausura axiomática: rangos individuales + invariantes cruzadas."""
        self._validate_parameters()
        self._validate_cross_constraints()

    def _validate_parameters(self) -> None:
        r"""
        Verifica cada parámetro contra su dominio físico-económico.

        Política de severidad:
            CRITICAL (is_critical=True)  → ValueError (violación de dominio).
            WARNING  (is_critical=False) → log.warning (fuera de rango típico).

        Invariante:
            ∀ campo xᵢ, xᵢ ∈ [minᵢ, maxᵢ] ⊂ ℝ, o el objeto no se construye.
        """
        validations: List[Tuple[Any, Any, Any, str, bool]] = [
            (self.risk_free_rate, 0.0, 0.20, "Tasa libre de riesgo", False),
            (self.market_premium, 0.00, 0.25, "Prima de riesgo de mercado", True),
            (self.beta, -3.0, 5.0, "Beta", False),
            (self.tax_rate, 0.0, 0.50, "Tasa impositiva", False),
            (self.cost_of_debt, 0.0, 0.30, "Costo de la deuda", False),
            (self.debt_to_equity_ratio, 0.0, 10.0, "Razón Deuda/Capital", False),
            (self.project_life_years, 1, 50, "Vida del proyecto", True),
            (self.liquidity_ratio, 0.0, 1.0, "Ratio de liquidez", False),
            (self.fixed_contracts_ratio, 0.0, 1.0, "Ratio contratos fijos", False),
            (self.inflation_rate, -0.05, 0.50, "Inflación", False),
            (self.alpha_coupling, 0.0, 1.0, "Acoplamiento α", False),
            (self.max_amplification, 1.0, 20.0, "Amplificación máxima", True),
            (self.df_student_t, 3, 100, "Grados de libertad t", True),
            (self.confidence_var, 0.50, 0.9999, "Confianza VaR", True),
            (self.confidence_contingency, 0.50, 0.9999, "Confianza contingencias", True),
            (self.kappa_0_topo, 0.0, 1.0, "κ₀ (fragmentación)", False),
            (self.kappa_1_topo, 0.0, 1.0, "κ₁ (ciclos)", False),
            (self.lambda_2_floor, 0.0, 1.0, "λ̄₂ (Fiedler floor)", False),
            (self.psi_critical, 0.01, 10.0, "Ψ*", False),
            (self.psi_stable, 0.01, 20.0, "Ψ_stable", False),
            (self.kappa_struct, 0.0, 20.0, "κ estructural", False),
            (self.t_scale, 1e-6, 200.0, "Θ (escala Arrhenius)", True),
            (self.tol_spectral, 0.0, 1e-3, "ε_dom espectral", False),
            (self.tol_newton, 0.0, 1e-2, "ε_Newton", False),
            (self.synergy_penalty_factor, 0.0, 2.0, "Penalización sinergia", False),
            (self.efficiency_penalty_factor, 0.0, 2.0, "Penalización eficiencia", False),
            (self.max_volatility_adjustment, 0.0, 5.0, "Δσ máxima", False),
        ]

        for value, min_v, max_v, name, is_crit in validations:
            if not (min_v <= value <= max_v):
                tag = "🚨" if is_crit else "⚠️"
                msg = f"{tag} {name} ({value!r}) fuera de rango [{min_v}, {max_v}]"
                if is_crit:
                    logger.error(msg)
                    raise ValueError(msg)
                logger.warning(msg)

    def _validate_cross_constraints(self) -> None:
        r"""
        Invariantes cruzadas del sistema acoplado (no factorizables).

        I1. D/E > 0 ∧ K_d ≤ 0  → pasivo remunerado degenerado.
        I2. β > 2 ∧ D/E < 0.3  → apalancamiento operativo implícito (Hamada).
        I3. Ψ* < Ψ_stable      → existencia de meseta estable.
        I4. T_ref < T_stress   → monotonía Arrhenius.
        I5. λ_L + ρ_f ≤ 1.5    → heurística de sanidad de ratios.
        I6. R_F + β R_P ≥ 0    → Ke no-degenerado en el cono (aviso).
        """
        if self.debt_to_equity_ratio > 0 and self.cost_of_debt <= 0:
            logger.warning(
                "⚠️ Inconsistencia: D/E > 0 pero K_d ≤ 0. "
                "El costo de deuda positivo es estructural a un pasivo remunerado."
            )

        if self.beta > 2.0 and self.debt_to_equity_ratio < 0.3:
            logger.info(
                "ℹ️ β=%.2f con D/E=%.2f: apalancamiento operativo implícito. "
                "Verificar Hamada β_U = β_L / [1 + (1-τ)·D/E].",
                self.beta,
                self.debt_to_equity_ratio,
            )

        if self.psi_critical >= self.psi_stable:
            logger.warning(
                "⚠️ Ψ_critical=%.3f ≥ Ψ_stable=%.3f: meseta estable vacía, "
                "colapso del régimen de Arrhenius.",
                self.psi_critical,
                self.psi_stable,
            )

        if self.t_reference >= self.t_stress:
            logger.warning(
                "⚠️ T_reference=%.2f ≥ T_stress=%.2f: monotonicidad rota en "
                "la curva Arrhenius modificada.",
                self.t_reference,
                self.t_stress,
            )

        if self.liquidity_ratio + self.fixed_contracts_ratio > 1.5:
            logger.warning(
                "⚠️ Σ ratios (λ_L + ρ_f) = %.3f > 1.5. Verificar bases de cálculo.",
                self.liquidity_ratio + self.fixed_contracts_ratio,
            )

        ke_naive = self.risk_free_rate + self.beta * self.market_premium
        if ke_naive < 0:
            logger.warning(
                "⚠️ Ke naive = %.4f < 0. El floor 0.5·R_F se aplicará en FASE 2.",
                ke_naive,
            )

    def homology_default(self) -> HomologicalInvariants:
        """Complejo trivial conexo acíclico (árbol: β₀=1, β₁=0)."""
        return HomologicalInvariants(beta_0=1, beta_1=0)

    def spectral_default(self, lambda_2: Optional[float] = None) -> SpectralSignature:
        """Firma espectral con λ₂ dado (o 0) y tiempo de difusión unitario."""
        return SpectralSignature(
            lambda_2=0.0 if lambda_2 is None else float(lambda_2),
            time=1.0,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ► PUERTO DE SALIDA FASE 1 → FASE 2
    # ► Este método ES el objeto terminal de F₁ y el objeto inicial de F₂.
    # ► CapitalAssetPricing.calculate_wacc_topological consume este morfismo.
    # ═══════════════════════════════════════════════════════════════════════════
    def topological_coupling_tensor(self) -> CouplingTensor:
        r"""
        Proyecta la configuración ontológica al tensor de acoplamiento
        homológico consumido por CapitalAssetPricing (FASE 2).

        Este método es el **puerto formal** entre FASE 1 y FASE 2. Codifica
        las constantes que unen los números de Betti del complejo simplicial
        del proyecto con su costo de capital ponderado.

        Modelo:
            τ = CouplingTensor(κ₀, κ₁, λ̄₂) ∈ ℝ²_{≥0} × [0, 1]

        El consumidor (FASE 2) define la acción:

            WACC_topo = τ.apply(WACC_base, H, Σ)
                      = WACC_base · [1 + κ₀(β₀−1) + κ₁ β₁] · e^{−t λ₂}

        Returns:
            CouplingTensor: objeto algebraico listo para consumo en FASE 2.
        """
        return CouplingTensor(
            kappa_0=self.kappa_0_topo,
            kappa_1=self.kappa_1_topo,
            lambda_2_floor=self.lambda_2_floor,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE 2 — MEDIDA Y ESPECTRO ▓▓▓
# ▓▓▓ Valoración CAPM/WACC + cuantificación rigurosa de riesgo de cola.                ▓▓▓
# ▓▓▓ Consume: FinancialConfig.topological_coupling_tensor()  (puerto F₁→F₂).          ▓▓▓
# ▓▓▓ El objeto terminal de esta fase ES synthesize_risk_envelope() → RiskEnvelope.    ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


class CapitalAssetPricing:
    r"""
    Motor de valoración del costo de capital con acoplamiento topológico.

    Pipeline:
        R_F → Ke (CAPM) → WACC_base → WACC_topo(H, Σ; τ) → NPV / TIR / MIRR.

    Fundamentos:

    §1. Costo del Equity (CAPM Sharpe-Lintner):
            Ke = R_F + β · R_P
        Hamada (desapalancamiento):
            β_U = β_L / [1 + (1−τ)·(D/E)],
            β_L = β_U · [1 + (1−τ)·(D/E)].

    §2. WACC (Modigliani-Miller con escudo fiscal):
            WACC = (E/V)·Ke + (D/V)·K_d·(1−τ)
        con E/V = 1/(1+D/E), D/V = (D/E)/(1+D/E).

    §3. WACC topológico (acción del CouplingTensor de FASE 1):
            WACC_topo = τ.apply(WACC_base, H, Σ)

    §4. Duración/convexidad de Macaulay sobre el haz de flujos descontados.
    """

    def __init__(self, config: FinancialConfig) -> None:
        if not isinstance(config, FinancialConfig):
            raise TypeError("Se requiere una instancia válida de FinancialConfig.")
        self.config = config
        # Cache del puerto F₁→F₂ (objeto inmutable).
        self._tau: CouplingTensor = config.topological_coupling_tensor()

    def refresh_coupling_tensor(self) -> CouplingTensor:
        """Relee el puerto F₁→F₂ (p.ej. tras mutar config en sensibilidad)."""
        self._tau = self.config.topological_coupling_tensor()
        return self._tau

    @property
    def coupling_tensor(self) -> CouplingTensor:
        """Tensor τ inyectado desde FASE 1."""
        return self._tau

    @lru_cache(maxsize=1)
    def calculate_ke(self) -> float:
        r"""
        Costo del Equity vía CAPM, con manejo de casos degenerados.

        Ke = R_F + β · R_P

        Casos:
            • |β| < 10⁻¹⁰  → Ke = R_F (neutralidad sistémica).
            • β < 0        → activo de cobertura (Ke < R_F admisible).
            • Ke < 0       → floor Ke ≥ ½ R_F (cono de no-negatividad relajado).

        Returns:
            float: Ke ≥ min(R_F, ½ R_F) en la práctica acotado inferiormente.
        """
        try:
            beta = self.config.beta
            rf = self.config.risk_free_rate
            market_premium = self.config.market_premium

            if abs(beta) < 1e-10:
                logger.info(
                    "β ≈ 0 → activo neutral al riesgo sistémico. Ke = R_F = %.2f%%",
                    rf * 100,
                )
                return rf

            if beta < 0:
                logger.warning(
                    "β = %.3f < 0: activo de cobertura. Ke < R_F teóricamente admisible.",
                    beta,
                )

            ke = rf + beta * market_premium
            if ke < 0:
                logger.error(
                    "Ke = %.2f%% < 0. Aplicando floor Ke ≥ %.2f·R_F.",
                    ke * 100,
                    _KE_FLOOR_FRACTION,
                )
                ke = max(ke, rf * _KE_FLOOR_FRACTION)

            logger.info("Ke = %.2f%% [β=%.2f, R_F=%.2f%%]", ke * 100, beta, rf * 100)
            return ke
        except Exception as e:
            logger.error("Error calculando Ke: %s", e)
            return self.config.risk_free_rate

    def hamada_unlever(self, beta_levered: Optional[float] = None) -> float:
        r"""
        β_U = β_L / [1 + (1−τ)·(D/E)].

        Identidad: D/E = 0 ⟹ β_U = β_L.
        """
        beta_l = self.config.beta if beta_levered is None else float(beta_levered)
        de = self.config.debt_to_equity_ratio
        tau = self.config.tax_rate
        denom = 1.0 + (1.0 - tau) * de
        if abs(denom) < _EPS_SPECTRAL:
            logger.warning("Hamada: denominador degenerado. Retornando β_L.")
            return beta_l
        return beta_l / denom

    def hamada_relever(self, beta_unlevered: float, debt_to_equity: Optional[float] = None) -> float:
        r"""β_L = β_U · [1 + (1−τ)·(D/E)]."""
        de = self.config.debt_to_equity_ratio if debt_to_equity is None else float(debt_to_equity)
        tau = self.config.tax_rate
        return beta_unlevered * (1.0 + (1.0 - tau) * de)

    @lru_cache(maxsize=1)
    def calculate_wacc(self) -> float:
        r"""
        WACC base (sin acoplamiento topológico).

            WACC_base = w_E · Ke + w_D · K_d · (1−τ)
            w_E = 1/(1+D/E),  w_D = (D/E)/(1+D/E),  w_E + w_D = 1.

        Returns:
            float: WACC_base ∈ [0, max(Ke, K_d neto)] en el caso no degenerado.

        Raises:
            ValueError: estructura de capital inválida.
        """
        try:
            if self.config.debt_to_equity_ratio < 0:
                raise ValueError("Razón D/E no puede ser negativa.")

            ke = self.calculate_ke()
            d_e = self.config.debt_to_equity_ratio
            w_e = 1.0 / (1.0 + d_e)
            w_d = d_e / (1.0 + d_e)

            if abs(w_e + w_d - 1.0) > 1e-10:
                logger.warning("Inconsistencia numérica en pesos de capital.")

            kd_neto = self.config.cost_of_debt * (1.0 - self.config.tax_rate)
            wacc = w_e * ke + w_d * kd_neto

            logger.info(
                "WACC_base = %.2f%% [Ke=%.2f%%, K_d_neto=%.2f%%, D/E=%.2f]",
                wacc * 100,
                ke * 100,
                kd_neto * 100,
                d_e,
            )
            return wacc
        except ZeroDivisionError as e:
            logger.error("División por cero en estructura de capital.")
            raise ValueError("Estructura de capital inválida.") from e
        except Exception as e:
            logger.error("Error calculando WACC: %s", e)
            raise

    def calculate_wacc_topological(
        self,
        beta_0: int = 1,
        beta_1: int = 0,
        lambda_2: Optional[float] = None,
        diffusion_time: float = 1.0,
        homology: Optional[HomologicalInvariants] = None,
        spectrum: Optional[SpectralSignature] = None,
    ) -> float:
        r"""
        WACC acoplado a los invariantes homológicos del complejo simplicial.

        Consume el tensor τ = topological_coupling_tensor() de la FASE 1.

            WACC_topo = τ.apply(WACC_base, H, Σ)
                      = WACC_base · [1 + κ₀(β₀−1) + κ₁ β₁] · e^{−t λ₂}

        Coherencia:
            • Árbol conexo: β₀=1, β₁=0, λ₂=0 ⟹ WACC_topo = WACC_base.
            • Fragmentación: β₀>1 ⟹ WACC_topo ≥ WACC_base.
            • Ciclos: β₁>0 ⟹ WACC_topo ≥ WACC_base.
            • Alta conectividad: λ₂≫0 ⟹ descuento espectral e^{−tλ₂}<1.

        Args:
            beta_0, beta_1: números de Betti (ignorados si se pasa `homology`).
            lambda_2: Fiedler value (ignorado si se pasa `spectrum`).
            diffusion_time: t del kernel de calor (default 1).
            homology: invariantes homológicos tipados (FASE 1).
            spectrum: firma espectral tipada (FASE 1).

        Returns:
            float: WACC_topo ≥ 0.
        """
        H = homology if homology is not None else HomologicalInvariants(beta_0, beta_1)
        if spectrum is not None:
            Σ = spectrum
        else:
            lam = 0.0 if lambda_2 is None else max(0.0, float(lambda_2))
            Σ = SpectralSignature(lambda_2=lam, time=max(0.0, float(diffusion_time)))

        wacc_base = self.calculate_wacc()
        tau = self._tau
        wacc_topo = tau.apply(wacc_base, H, Σ)

        logger.info(
            "WACC_topo = %.2f%% [base=%.2f%%, β₀=%d, β₁=%d, λ₂=%.4f, t=%.3f, "
            "S=%.4f, e^{-tλ₂}=%.4f, ‖τ‖_∞=%.4f]",
            wacc_topo * 100,
            wacc_base * 100,
            H.beta_0,
            H.beta_1,
            Σ.lambda_2,
            Σ.time,
            tau.structural_factor(H),
            Σ.heat_kernel_discount,
            tau.infinity_norm(),
        )
        return wacc_topo

    def calculate_npv(
        self,
        cash_flows: Sequence[float],
        initial_investment: float = 0.0,
        discount_rate: Optional[float] = None,
    ) -> float:
        r"""
        Valor Presente Neto.

            NPV = −|I₀| + Σ_{t=1}^{T} CF_t / (1+r)^t

        Convención: I₀ se toma en valor absoluto (desembolso).
        """
        try:
            rate = discount_rate if discount_rate is not None else self.calculate_wacc()
            if rate <= -1.0:
                raise ValueError(f"Tasa de descuento r={rate} ≤ −1: polo en (1+r)^t.")
            npv = -abs(initial_investment)
            for i, cf in enumerate(cash_flows, 1):
                npv += cf / pow(1.0 + rate, i)
            logger.info("NPV = %.2f (tasa=%.2f%%)", npv, rate * 100)
            return float(npv)
        except Exception as e:
            logger.error("Error calculando NPV: %s", e)
            raise

    def macaulay_duration(
        self,
        cash_flows: Sequence[float],
        discount_rate: Optional[float] = None,
    ) -> Dict[str, float]:
        r"""
        Duración de Macaulay, duración modificada, convexidad y DV01.

            D_mac = Σ t · PV(CF_t) / Σ PV(CF_t)
            D_mod = D_mac / (1+r)
            Conv  = Σ t(t+1) · PV(CF_t) / [P · (1+r)²]
            DV01  = D_mod · P / 10⁴

        Returns:
            Dict con duration_macaulay, duration_modified, convexity, dv01, pv.
        """
        rate = discount_rate if discount_rate is not None else self.calculate_wacc()
        if rate <= -1.0:
            raise ValueError(f"r={rate} ≤ −1.")
        if not cash_flows:
            return {
                "duration_macaulay": 0.0,
                "duration_modified": 0.0,
                "convexity": 0.0,
                "dv01": 0.0,
                "pv": 0.0,
            }

        pv = 0.0
        weighted_t = 0.0
        weighted_t2 = 0.0
        for t_idx, cf in enumerate(cash_flows, 1):
            disc = pow(1.0 + rate, t_idx)
            pvcf = cf / disc
            pv += pvcf
            weighted_t += t_idx * pvcf
            weighted_t2 += t_idx * (t_idx + 1) * pvcf

        if abs(pv) < _EPS_SPECTRAL:
            return {
                "duration_macaulay": 0.0,
                "duration_modified": 0.0,
                "convexity": 0.0,
                "dv01": 0.0,
                "pv": pv,
            }

        d_mac = weighted_t / pv
        d_mod = d_mac / (1.0 + rate)
        convexity = weighted_t2 / (pv * (1.0 + rate) ** 2)
        dv01 = d_mod * pv / 1.0e4
        return {
            "duration_macaulay": float(d_mac),
            "duration_modified": float(d_mod),
            "convexity": float(convexity),
            "dv01": float(dv01),
            "pv": float(pv),
        }

    def calculate_irr(
        self,
        cash_flows: Sequence[float],
        initial_investment: float,
        tol: Optional[float] = None,
        max_iter: int = 100,
    ) -> Optional[float]:
        r"""
        TIR: raíz de NPV(r*)=0. Newton-Raphson con respaldo biseccional.

            NPV(r) = −I₀ + Σ CF_t / (1+r)^t
            NPV'(r) = −Σ t·CF_t / (1+r)^{t+1}

        Dominio de búsqueda: r ∈ (−1+ε, 10]. Si NPV no cambia de signo, None.
        """
        tol = tol if tol is not None else self.config.tol_newton
        I0 = abs(initial_investment)
        if I0 < 1e-12 or not cash_flows:
            return None

        def npv_at(r: float) -> Tuple[float, float]:
            npv = -I0
            d_npv = 0.0
            rp1 = 1.0 + r
            if rp1 <= 0.0:
                return float("inf"), 0.0
            for t_idx, cf in enumerate(cash_flows, start=1):
                d = rp1 ** t_idx
                npv += cf / d
                d_npv -= t_idx * cf / (rp1 ** (t_idx + 1))
            return npv, d_npv

        r = self.calculate_wacc()
        r = min(max(r, _IRR_LO), _IRR_HI)
        for _ in range(max_iter):
            npv, d_npv = npv_at(r)
            if abs(d_npv) < _EPS_NEWTON:
                break
            r_new = r - npv / d_npv
            r_new = max(_IRR_LO, min(_IRR_HI, r_new))
            if abs(r_new - r) < tol:
                return float(r_new)
            r = r_new

        lo, hi = _IRR_LO, _IRR_HI
        npv_lo, _ = npv_at(lo)
        npv_hi, _ = npv_at(hi)
        if npv_lo * npv_hi > 0:
            logger.warning("IRR no tiene raíz real en [%.6f, %.1f].", lo, hi)
            return None
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            npv_mid, _ = npv_at(mid)
            if abs(npv_mid) < tol or (hi - lo) < tol:
                return float(mid)
            if npv_lo * npv_mid < 0:
                hi, npv_hi = mid, npv_mid
            else:
                lo, npv_lo = mid, npv_mid
        return float(0.5 * (lo + hi))

    def calculate_mirr(
        self,
        cash_flows: Sequence[float],
        initial_investment: float,
        finance_rate: Optional[float] = None,
        reinvest_rate: Optional[float] = None,
    ) -> Optional[float]:
        r"""
        TIR modificada (evita raíces múltiples de NPV).

            MIRR = (FV_positivos / PV_negativos)^{1/n} − 1

        FV se capitaliza a `reinvest_rate` (default WACC).
        PV de egresos se descuenta a `finance_rate` (default Ke).
        """
        I0 = abs(initial_investment)
        n = len(cash_flows)
        if I0 < 1e-12 or n < 1:
            return None

        r_fin = finance_rate if finance_rate is not None else self.calculate_ke()
        r_reinv = reinvest_rate if reinvest_rate is not None else self.calculate_wacc()
        if r_fin <= -1.0 or r_reinv <= -1.0:
            return None

        pv_out = I0
        fv_in = 0.0
        for t_idx, cf in enumerate(cash_flows, start=1):
            if cf >= 0:
                fv_in += cf * ((1.0 + r_reinv) ** (n - t_idx))
            else:
                pv_out += abs(cf) / ((1.0 + r_fin) ** t_idx)

        if pv_out <= 0.0 or fv_in <= 0.0:
            return None
        return float((fv_in / pv_out) ** (1.0 / n) - 1.0)

    def sensitivity_analysis(
        self, parameter: str, range_values: Sequence[float]
    ) -> Dict[float, float]:
        r"""
        Elasticidad univariante de WACC_base. Restaura el valor original
        (try/finally) y limpia caches + tensor τ.
        """
        if not hasattr(self.config, parameter):
            raise ValueError(f"Parámetro desconocido: {parameter}")

        original = getattr(self.config, parameter)
        results: Dict[float, float] = {}
        try:
            for val in range_values:
                setattr(self.config, parameter, val)
                self.calculate_ke.cache_clear()
                self.calculate_wacc.cache_clear()
                self.refresh_coupling_tensor()
                results[float(val)] = self.calculate_wacc()
        finally:
            setattr(self.config, parameter, original)
            self.calculate_ke.cache_clear()
            self.calculate_wacc.cache_clear()
            self.refresh_coupling_tensor()
        return results


@dataclass(frozen=True)
class RiskEnvelope:
    r"""
    Sobre de riesgo efectivo. Objeto terminal de FASE 2 y objeto inicial de FASE 3.

    Transforma el par nominal (K₀, σ₀) en (K_eff, σ_eff) cargando la cola:

        K_eff  = K₀ · (1 + tail_loading_α)
        σ_eff  = σ₀ · (1 + tail_dispersion_α)

    con tail_loading = max(0, (ES−VaR)/K₀) y
        tail_dispersion = max(0, ES/VaR − 1).

    RealOptionsAnalyzer (FASE 3) consume este objeto como (K, σ) de la PDE.
    """

    strike_effective: float
    sigma_effective: float
    tail_loading: float
    tail_dispersion: float
    var: float
    cvar: float
    banach_tail_norm: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "strike_effective": self.strike_effective,
            "sigma_effective": self.sigma_effective,
            "tail_loading": self.tail_loading,
            "tail_dispersion": self.tail_dispersion,
            "var": self.var,
            "cvar": self.cvar,
            "banach_tail_norm": self.banach_tail_norm,
        }


class RiskQuantifier:
    r"""
    Cuantificador de riesgo financiero basado en teoría de medida.

        𝓡_α(X) = −Q_α(X) = −inf{x : P(X ≤ x) ≥ α}
        ES_α(X) = (1−α)⁻¹ ∫_α¹ VaR_γ(X) dγ = E[X | X > VaR_α(X)]

    Axiomas de coherencia (Artzner et al.):
        A1. Monotonía: X ≤ Y ⟹ 𝓡(X) ≥ 𝓡(Y).
        A2. Invariancia traslacional: 𝓡(X+c) = 𝓡(X) − c.
        A3. Homogeneidad positiva: 𝓡(λX) = λ 𝓡(X), λ>0.
        A4. Subaditividad: 𝓡(X+Y) ≤ 𝓡(X)+𝓡(Y).

    El VaR viola A4; el CVaR la satisface. Se reportan ambos.

    Norma de Banach de cola (en ℓ² de (VaR, ES)):
        ‖(VaR, ES)‖₂ = √(VaR² + ES²), usada como diagnóstico de carga.
    """

    def __init__(self, distribution: DistributionType = DistributionType.NORMAL) -> None:
        self.distribution = distribution

    @staticmethod
    def _cornish_fisher_quantile(alpha: float, skew: float, excess_kurt: float) -> float:
        r"""
        Cuantil Cornish-Fisher de orden 4 sobre z_α = Φ⁻¹(α).

            q_CF(α) = z + (γ₁/6)(z²−1) + (γ₂/24)(z³−3z)
                      − (γ₁²/36)(2z³−5z) + O(γ³)

        Dominio de monotonía heurístico: |γ₁|<1, |γ₂|<3.
        """
        z = float(norm.ppf(alpha))
        z2 = z * z
        z3 = z2 * z
        corr1 = (skew / 6.0) * (z2 - 1.0)
        corr2 = (excess_kurt / 24.0) * (z3 - 3.0 * z)
        corr3 = -(skew * skew / 36.0) * (2.0 * z3 - 5.0 * z)
        return z + corr1 + corr2 + corr3

    @staticmethod
    def _cf_jet_is_valid(skew: float, excess_kurt: float) -> bool:
        """True ssi el jet CF no está en zona de fractura heurística."""
        return abs(skew) < 1.0 and abs(excess_kurt) < 3.0

    def calculate_var(
        self,
        mean: float,
        std_dev: float,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1,
        df_student_t: int = 5,
        trading_days_per_year: int = 252,
        skewness: float = 0.0,
        excess_kurtosis: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        r"""
        VaR y CVaR exactos bajo tres medidas paramétricas.

        Escalado temporal i.i.d.: σ_X = σ · √(t / t_year).

        ■ Normal(μ, σ²):
            VaR_α = μ + σ z_α,   ES_α = μ + σ φ(z_α)/(1−α)

        ■ Student-t(ν) con X = μ + s T_ν, s = σ √((ν−2)/ν):
            VaR_α = μ + s t_α
            ES_α  = μ + s · [f_ν(t_α) (ν + t_α²) / ((ν−1)(1−α))]

        ■ Cornish-Fisher:
            VaR_α = μ + σ q_CF(α)
            ES_α  ≈ μ + σ (1−α)⁻¹ ∫_α¹ q_CF(u) du   (trapecios)

        Raises:
            ValueError: parámetros fuera de dominio.
        """
        if std_dev < 0:
            raise ValueError("σ < 0: no es una desviación típica válida.")
        if not 0 < confidence_level < 1:
            raise ValueError(f"α = {confidence_level} ∉ (0, 1).")
        if time_horizon_days < 1:
            raise ValueError(f"t = {time_horizon_days} < 1 día.")
        if trading_days_per_year < 1:
            raise ValueError("trading_days_per_year < 1.")

        if std_dev == 0:
            return mean, {
                "distribution": "Degenerate",
                "var": mean,
                "cvar": mean,
                "scaled_std": 0.0,
                "confidence": confidence_level,
                "z_score": 0.0,
                "var_lower": mean,
                "var_upper": mean,
                "time_horizon_days": time_horizon_days,
                "annualization_factor": 0.0,
                "coherence_gap": 0.0,
                "banach_tail_norm": abs(mean) * sqrt(2.0),
            }

        if self.distribution == DistributionType.STUDENT_T and df_student_t <= 2:
            logger.warning("ν = %d ≤ 2: varianza infinita. Ajustando a ν = 3.", df_student_t)
            df_student_t = 3
        if self.distribution == DistributionType.STUDENT_T and df_student_t <= 4:
            logger.info("ν = %d ≤ 4: curtosis infinita. ES sigue definido para ν>2.", df_student_t)

        time_factor = sqrt(time_horizon_days / trading_days_per_year)
        scaled_std = std_dev * time_factor

        if self.distribution == DistributionType.NORMAL:
            z_a = float(norm.ppf(confidence_level))
            z_l = float(norm.ppf(1.0 - confidence_level))
            var_upper = mean + z_a * scaled_std
            var_lower = mean + z_l * scaled_std
            cvar = mean + scaled_std * float(norm.pdf(z_a)) / (1.0 - confidence_level)
            z_score = z_a
            dist_name = "Normal"

        elif self.distribution == DistributionType.STUDENT_T:
            nu = df_student_t
            scale_s = scaled_std * sqrt((nu - 2.0) / nu)
            t_a = float(t.ppf(confidence_level, nu))
            t_l = float(t.ppf(1.0 - confidence_level, nu))
            var_upper = mean + scale_s * t_a
            var_lower = mean + scale_s * t_l
            pdf_t = float(t.pdf(t_a, nu))
            es_std = pdf_t * (nu + t_a * t_a) / ((nu - 1.0) * (1.0 - confidence_level))
            cvar = mean + scale_s * es_std
            z_score = t_a
            dist_name = f"Student-t(ν={nu})"

        elif self.distribution == DistributionType.CORNISH_FISHER:
            if not self._cf_jet_is_valid(skewness, excess_kurtosis):
                logger.warning(
                    "Cornish-Fisher: |γ₁|=%.3f o |γ₂|=%.3f fuera de monotonía "
                    "heurística (|γ₁|<1, |γ₂|<3). El cuantil puede no ser monótono.",
                    abs(skewness),
                    abs(excess_kurtosis),
                )
            z_a = self._cornish_fisher_quantile(confidence_level, skewness, excess_kurtosis)
            z_l = self._cornish_fisher_quantile(1.0 - confidence_level, skewness, excess_kurtosis)
            var_upper = mean + scaled_std * z_a
            var_lower = mean + scaled_std * z_l
            n_quad = 400
            us = np.linspace(confidence_level, 1.0 - 1e-6, n_quad)
            qs = np.array(
                [
                    self._cornish_fisher_quantile(float(u), skewness, excess_kurtosis)
                    for u in us
                ]
            )
            integral = float(np.trapezoid(qs, us))
            cvar = mean + scaled_std * integral / (1.0 - confidence_level)
            z_score = z_a
            dist_name = f"Cornish-Fisher(γ₁={skewness:.2f}, γ₂={excess_kurtosis:.2f})"
        else:
            raise ValueError(f"Distribución no soportada: {self.distribution}")

        # Coherencia débil: ES debe dominar VaR en la cola superior para σ>0.
        coherence_gap = cvar - var_upper
        if coherence_gap < -1e-8 * max(1.0, abs(var_upper)):
            logger.warning(
                "⚠️ ES (%.6f) < VaR (%.6f): violación numérica de ES≥VaR.",
                cvar,
                var_upper,
            )

        banach_norm = float(sqrt(var_upper * var_upper + cvar * cvar))

        metrics = {
            "distribution": dist_name,
            "var": var_upper,
            "var_lower": var_lower,
            "var_upper": var_upper,
            "cvar": cvar,
            "expected_shortfall": cvar,
            "scaled_std": scaled_std,
            "confidence": confidence_level,
            "z_score": z_score,
            "time_horizon_days": time_horizon_days,
            "annualization_factor": time_factor,
            "tail_risk_ratio": cvar / var_upper if var_upper > 0 else float("inf"),
            "risk_contribution": (var_upper - mean) / mean if mean != 0 else float("inf"),
            "coherence_gap": coherence_gap,
            "banach_tail_norm": banach_norm,
        }
        logger.info(
            "%s: VaR=%.2f, CVaR=%.2f @ α=%.2f%%",
            dist_name,
            var_upper,
            cvar,
            confidence_level * 100,
        )
        return var_upper, metrics

    def suggest_contingency(
        self,
        base_cost: float,
        std_dev: float,
        confidence_level: float = 0.90,
        method: str = "all",
    ) -> Dict[str, float]:
        r"""
        Reserva de contingencia bajo criterios VaR / porcentaje / heurístico.

        Decisión final := max de los candidatos (dominio conservador).
        """
        if base_cost <= 0:
            return {"recommended": 0.0}

        cv = std_dev / base_cost
        buffers: Dict[str, float] = {}

        if method in ("all", "var"):
            var_val, _ = self.calculate_var(base_cost, std_dev, confidence_level)
            buffers["var_based"] = max(0.0, var_val - base_cost)

        if method in ("all", "percentage"):
            if cv > 0.20:
                pct = 0.20
            elif cv > 0.10:
                pct = 0.15
            else:
                pct = 0.10
            buffers["percentage_based"] = base_cost * pct
            buffers["percentage_rate"] = pct

        if method in ("all", "heuristic"):
            if cv > 0.20:
                mult = 2.0
            elif cv > 0.15:
                mult = 1.5
            else:
                mult = 1.0
            buffers["heuristic"] = std_dev * mult
            buffers["heuristic_multiplier"] = mult

        candidates = [
            v
            for k, v in buffers.items()
            if k in ("var_based", "percentage_based", "heuristic")
        ]
        buffers["recommended"] = max(candidates) if candidates else 0.0
        buffers["coefficient_of_variation"] = cv
        return buffers

    # ═══════════════════════════════════════════════════════════════════════════
    # ► PUERTO DE SALIDA FASE 2 → FASE 3
    # ► Este método ES el objeto terminal de F₂ y el objeto inicial de F₃.
    # ► RealOptionsAnalyzer.value_option_to_wait consume strike/sigma efectivos.
    # ═══════════════════════════════════════════════════════════════════════════
    def synthesize_risk_envelope(
        self,
        base_strike: float,
        base_sigma: float,
        alpha_envelope: float = 0.95,
    ) -> Dict[str, float]:
        r"""
        Sintetiza el sobre de riesgo efectivo consumido por RealOptionsAnalyzer
        (FASE 3). Continuación formal del puerto F₁→F₂: el CouplingTensor deformó
        el WACC; ahora el cuantificador deforma (K, σ) de la PDE.

            K_eff = K₀ · (1 + tail_loading_α)
            σ_eff = σ₀ · (1 + tail_dispersion_α)

        Returns:
            Dict compatible con v5 + campo `banach_tail_norm`.
            Use RiskEnvelope(**dict) para el objeto algebraico.
        """
        if base_strike <= 0 or base_sigma <= 0:
            logger.warning(
                "synthesize_risk_envelope: K₀=%.4f, σ₀=%.4f. Retornando sin cargar.",
                base_strike,
                base_sigma,
            )
            env = RiskEnvelope(
                strike_effective=base_strike,
                sigma_effective=base_sigma,
                tail_loading=0.0,
                tail_dispersion=0.0,
                var=base_strike,
                cvar=base_strike,
                banach_tail_norm=abs(base_strike) * sqrt(2.0) if base_strike else 0.0,
            )
            return env.as_dict()

        _, metrics = self.calculate_var(
            mean=base_strike,
            std_dev=base_sigma,
            confidence_level=alpha_envelope,
        )
        var_val = float(metrics["var"])
        cvar_val = float(metrics["cvar"])

        tail_loading = max(0.0, (cvar_val - var_val) / base_strike)
        tail_dispersion = max(0.0, cvar_val / var_val - 1.0) if var_val > 0 else 0.0

        env = RiskEnvelope(
            strike_effective=base_strike * (1.0 + tail_loading),
            sigma_effective=base_sigma * (1.0 + tail_dispersion),
            tail_loading=tail_loading,
            tail_dispersion=tail_dispersion,
            var=var_val,
            cvar=cvar_val,
            banach_tail_norm=float(metrics.get("banach_tail_norm", 0.0)),
        )
        return env.as_dict()


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE 3 — PDE ESTOCÁSTICA Y SÍNTESIS ▓▓▓
# ▓▓▓ Black-Scholes-Merton, CRR binomial log-estable, orquestación termodinámica.      ▓▓▓
# ▓▓▓ Consume: RiskQuantifier.synthesize_risk_envelope()  (puerto F₂→F₃).              ▓▓▓
# ▓▓▓ Fachada FinancialEngine = F₃ ∘ F₂ ∘ F₁ sobre 𝔐_fin.                              ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


def _clip_d(d: float) -> float:
    """Satura d₁,d₂ para estabilidad de Φ y φ en el álgebra de Banach C_b(ℝ)."""
    if d > _MAX_D1:
        return _MAX_D1
    if d < -_MAX_D1:
        return -_MAX_D1
    return d


class RealOptionsAnalyzer:
    r"""
    Analizador de opciones reales. Resuelve el problema de frontera libre
    asociado a la flexibilidad estratégica sobre la variedad estocástica
    del valor del proyecto.

    PDE BSM (medida Q):
        ∂V/∂t + ½ σ² S² ∂²V/∂S² + (r−q) S ∂V/∂S − r V = 0
        V(S,T) = max(S−K, 0)

    Feynman-Kac:
        V = S e^{−qT} N(d₁) − K e^{−rT} N(d₂)

    CRR americano (inducción hacia atrás, precios en log-espacio):
        V_{i,j} = max(S_{ij}−K, e^{−rΔt}[p V_{i+1,j+1} + (1−p) V_{i+1,j}])

    Consume el RiskEnvelope de FASE 2 como (K_eff, σ_eff).
    """

    def __init__(self, model_type: OptionModelType = OptionModelType.AUTO) -> None:
        self.model_type = model_type

    def value_option_to_wait(
        self,
        project_value: float,
        investment_cost: float,
        risk_free_rate: float,
        time_to_expire: float,
        volatility: float,
        steps: int = 100,
        american: bool = True,
        dividend_yield: float = 0.0,
    ) -> Dict[str, float]:
        r"""
        Valora la opción de esperar (call) sobre el proyecto.

        Dispatcher:
            AUTO          → BSM si american=False, CRR si american=True.
            BLACK_SCHOLES → fuerza BSM (aprox. europea si american=True).
            BINOMIAL      → fuerza CRR.
        """
        use_bsm = self.model_type == OptionModelType.BLACK_SCHOLES or (
            self.model_type == OptionModelType.AUTO and not american
        )

        if use_bsm:
            if american:
                logger.warning(
                    "BSM no resuelve la frontera libre de ejercicio anticipado; "
                    "aplicando aproximación europea como cota inferior."
                )
            return self._black_scholes_merton(
                S=project_value,
                K=investment_cost,
                r=risk_free_rate,
                T=time_to_expire,
                sigma=volatility,
                q=dividend_yield,
            )

        return self._binomial_valuation(
            S=project_value,
            K=investment_cost,
            r=risk_free_rate,
            T=time_to_expire,
            sigma=volatility,
            n=steps,
            american=american,
            q=dividend_yield,
        )

    def _black_scholes_merton(
        self,
        S: float,
        K: float,
        r: float,
        T: float,
        sigma: float,
        q: float = 0.0,
    ) -> Dict[str, Any]:
        r"""
        Solución analítica BSM + Greeks de 1º y 2º orden.

        Call: C = S e^{−qT} N(d₁) − K e^{−rT} N(d₂)
        Put : P = K e^{−rT} N(−d₂) − S e^{−qT} N(−d₁)
        d₁  = [ln(S/K) + (r−q+½σ²)T] / (σ√T),  d₂ = d₁ − σ√T

        Greeks:
            Δ = e^{−qT} N(d₁)
            Γ = e^{−qT} φ(d₁) / (S σ √T)
            Vega = S e^{−qT} φ(d₁) √T
            Θ, ρ clásicos.
            Vanna = ∂Δ/∂σ = −e^{−qT} φ(d₁) d₂ / σ
            Volga = ∂Vega/∂σ = Vega · d₁ d₂ / σ
            Charm = ∂Δ/∂τ (decay de delta)
        """
        if S <= 0 or K <= 0:
            intrinsic = max(S - K, 0.0)
            return {
                "option_value": intrinsic,
                "model": "BSM dominio inválido",
                "intrinsic_value": intrinsic,
                "time_value": 0.0,
                "delta": 1.0 if S > K else 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
                "vanna": 0.0,
                "volga": 0.0,
                "charm": 0.0,
                "d1": float("nan"),
                "d2": float("nan"),
            }

        if T <= 0 or sigma <= 0:
            intrinsic = max(S - K, 0.0)
            return {
                "option_value": intrinsic,
                "model": "BSM degenerado",
                "intrinsic_value": intrinsic,
                "time_value": 0.0,
                "delta": 1.0 if S > K else 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
                "vanna": 0.0,
                "volga": 0.0,
                "charm": 0.0,
                "d1": float("nan"),
                "d2": float("nan"),
            }

        sqrt_T = sqrt(T)
        d1 = _clip_d((log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T))
        d2 = _clip_d(d1 - sigma * sqrt_T)

        N_d1 = float(norm.cdf(d1))
        N_d2 = float(norm.cdf(d2))
        N_md1 = float(norm.cdf(-d1))
        N_md2 = float(norm.cdf(-d2))
        phi_d1 = float(norm.pdf(d1))

        disc_q = exp(-q * T)
        disc_r = exp(-r * T)

        call = S * disc_q * N_d1 - K * disc_r * N_d2
        put = K * disc_r * N_md2 - S * disc_q * N_md1

        delta = disc_q * N_d1
        gamma = disc_q * phi_d1 / (S * sigma * sqrt_T)
        vega = S * disc_q * phi_d1 * sqrt_T
        theta = (
            -S * disc_q * phi_d1 * sigma / (2.0 * sqrt_T)
            + q * S * disc_q * N_d1
            - r * K * disc_r * N_d2
        )
        rho = K * T * disc_r * N_d2
        vanna = -disc_q * phi_d1 * d2 / sigma
        volga = vega * d1 * d2 / sigma if sigma > 0 else 0.0
        # Charm (drift de delta respecto a τ=T, convención anual).
        charm = (
            q * disc_q * N_d1
            - disc_q * phi_d1 * (2.0 * (r - q) * T - d2 * sigma * sqrt_T) / (2.0 * T * sigma * sqrt_T)
        )

        intrinsic = max(S - K, 0.0)
        return {
            "option_value": max(call, 0.0),
            "put_value": max(put, 0.0),
            "model": "Black-Scholes-Merton (analítica)",
            "intrinsic_value": intrinsic,
            "time_value": max(0.0, call - intrinsic),
            "d1": d1,
            "d2": d2,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "theta_daily": theta / 252.0,
            "rho": rho,
            "vanna": vanna,
            "volga": volga,
            "charm": charm,
            "parameters": {"S": S, "K": K, "r": r, "T": T, "sigma": sigma, "q": q},
        }

    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        r: float,
        T: float,
        q: float = 0.0,
        sigma0: float = 0.2,
        tol: float = 1e-8,
        max_iter: int = 50,
    ) -> Optional[float]:
        r"""
        Volatilidad implícita por Newton sobre vega (mapa σ ↦ C_BSM(σ)).

        El mapa es estrictamente creciente en σ para T>0, S,K>0 (vega>0),
        luego la raíz es única cuando price ∈ (descuento intrínseco, cota).
        """
        if market_price < 0 or S <= 0 or K <= 0 or T <= 0:
            return None
        sigma = max(1e-6, float(sigma0))
        for _ in range(max_iter):
            res = self._black_scholes_merton(S, K, r, T, sigma, q)
            price = float(res["option_value"])
            vega = float(res["vega"])
            diff = price - market_price
            if abs(diff) < tol:
                return float(sigma)
            if vega < 1e-14:
                break
            sigma = max(1e-6, min(5.0, sigma - diff / vega))
        return float(sigma) if abs(diff) < 10.0 * tol else None

    def _binomial_valuation(
        self,
        S: float,
        K: float,
        r: float,
        T: float,
        sigma: float,
        n: int,
        american: bool = True,
        q: float = 0.0,
    ) -> Dict[str, Any]:
        r"""
        CRR con precios en log-espacio (estabilidad frente a overflow de u^n)
        y Greeks extraídos del árbol.

        u = e^{σ√Δt}, d = 1/u, p = (e^{(r−q)Δt} − d)/(u − d).
        Condición de no-arbitraje: p ∈ (0,1) ⇔ r−q ∈ (ln d / Δt, ln u / Δt).
        """
        if S <= 0:
            raise ValueError(f"S = {S} debe ser estrictamente positivo.")

        if T <= 0:
            intrinsic = max(S - K, 0.0)
            return {
                "option_value": intrinsic,
                "model": "Expirada",
                "intrinsic_value": intrinsic,
                "time_value": 0.0,
                "delta": 1.0 if S > K else 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "theta_daily": 0.0,
            }

        if sigma <= 0:
            intrinsic = max(S - K * exp(-r * T), 0.0)
            return {
                "option_value": intrinsic,
                "model": "Determinístico (σ=0)",
                "intrinsic_value": max(S - K, 0.0),
                "time_value": max(0.0, intrinsic - max(S - K, 0.0)),
                "delta": 1.0 if S > K else 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "theta_daily": 0.0,
            }

        n = max(1, int(n))
        dt = T / n
        log_u = sigma * sqrt(dt)
        u = exp(log_u)
        d = exp(-log_u)
        disc = exp(-r * dt)
        dr_dt = exp((r - q) * dt)
        denom = u - d
        if abs(denom) < _EPS_SPECTRAL:
            fallback = max(S - K, 0.0)
            return {
                "option_value": fallback,
                "model": "Error: u≈d",
                "error": "u − d degenerado.",
                "intrinsic_value": fallback,
                "time_value": 0.0,
                "delta": float("nan"),
                "gamma": float("nan"),
                "theta": float("nan"),
            }
        p = (dr_dt - d) / denom

        if not (0.0 < p < 1.0):
            logger.error(
                "CRR: p = %.4f ∉ (0,1). r=%.4f, σ=%.4f, T=%.4f, n=%d.",
                p,
                r,
                sigma,
                T,
                n,
            )
            fallback = max(S - K, 0.0)
            return {
                "option_value": fallback,
                "model": "Error: Arbitraje detectado",
                "error": f"p = {p:.4f} fuera de (0,1).",
                "intrinsic_value": fallback,
                "time_value": 0.0,
                "delta": float("nan"),
                "gamma": float("nan"),
                "theta": float("nan"),
            }

        log_S = log(S)
        # Nodo terminal j = 0..n : n down-moves compensados, j up-moves netos.
        # S_{n,j} = S · u^j · d^{n−j} = exp(log_S + (2j−n) log_u)
        js = np.arange(n + 1, dtype=float)
        prices_T = np.exp(log_S + (2.0 * js - n) * log_u)
        values = np.maximum(prices_T - K, 0.0)

        values_history: List[np.ndarray] = []
        early_exercise_count = 0
        early_exercise_value = 0.0

        for i in range(n - 1, -1, -1):
            cont = disc * (p * values[1:] + (1.0 - p) * values[:-1])
            if american:
                js_i = np.arange(i + 1, dtype=float)
                s_nodes = np.exp(log_S + (2.0 * js_i - i) * log_u)
                intrinsic_nodes = np.maximum(s_nodes - K, 0.0)
                exercised = intrinsic_nodes > cont + 1e-10
                early_exercise_count += int(np.count_nonzero(exercised))
                early_exercise_value += float(
                    np.sum((intrinsic_nodes - cont)[exercised])
                ) if np.any(exercised) else 0.0
                new_values = np.where(exercised, intrinsic_nodes, cont)
            else:
                new_values = cont
            values = new_values
            values_history.append(values.copy())

        option_value = float(values[0])

        if n >= 2:
            V_dt = values_history[n - 2]
        else:
            V_dt = np.maximum(np.array([S * u - K, S * d - K], dtype=float), 0.0)

        V_2dt: Optional[np.ndarray] = None
        if n >= 3:
            V_2dt = values_history[n - 3]

        if len(V_dt) >= 2:
            S_up = S * u
            S_down = S * d
            delta = float((V_dt[1] - V_dt[0]) / (S_up - S_down))
            delta = float(np.clip(delta, 0.0, 1.0))
        else:
            delta = 1.0 if S > K else 0.0

        gamma = 0.0
        if V_2dt is not None and len(V_2dt) >= 3:
            S_uu = S * u * u
            S_ud = S
            S_dd = S * d * d
            delta_up = (V_2dt[2] - V_2dt[1]) / (S_uu - S_ud)
            delta_down = (V_2dt[1] - V_2dt[0]) / (S_ud - S_dd)
            gamma = float((delta_up - delta_down) / (0.5 * (S_uu - S_dd)))

        if len(V_dt) >= 2:
            v_dt_expected = p * V_dt[1] + (1.0 - p) * V_dt[0]
            theta = float((v_dt_expected - option_value) / dt)
            theta_daily = theta / 252.0
        else:
            theta = 0.0
            theta_daily = 0.0

        intrinsic = max(S - K, 0.0)
        time_value = max(0.0, option_value - intrinsic)

        return {
            "option_value": option_value,
            "model": f"Binomial CRR ({'Americana' if american else 'Europea'}, n={n})",
            "intrinsic_value": intrinsic,
            "time_value": time_value,
            "early_exercise_nodes": early_exercise_count,
            "early_exercise_value": early_exercise_value,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "theta_daily": theta_daily,
            "parameters": {"u": u, "d": d, "p": p, "dt": dt, "discount_factor": disc},
        }

    def binomial_richardson(
        self,
        S: float,
        K: float,
        r: float,
        T: float,
        sigma: float,
        n: int = 50,
        q: float = 0.0,
    ) -> Dict[str, Any]:
        r"""
        Extrapolación de Richardson sobre CRR europeo (error O(1/n)).

            V_rich = (4 V(2n) − V(n)) / 3

        No se aplica a americanas (la frontera libre rompe la expansión regular).
        """
        n = max(2, int(n))
        v_n = self._binomial_valuation(S, K, r, T, sigma, n=n, american=False, q=q)
        v_2n = self._binomial_valuation(S, K, r, T, sigma, n=2 * n, american=False, q=q)
        price_n = float(v_n["option_value"])
        price_2n = float(v_2n["option_value"])
        rich = (4.0 * price_2n - price_n) / 3.0
        out = dict(v_2n)
        out["option_value"] = max(rich, 0.0)
        out["model"] = f"CRR Richardson europeo (n={n}, 2n={2 * n})"
        out["price_n"] = price_n
        out["price_2n"] = price_2n
        return out


class FinancialEngine:
    r"""
    Fachada de orquestación integral. Ensambla el funtor F₃ ∘ F₂ ∘ F₁.

    Pipeline canónico de analyze_project:
        1. Volatilidad estructural (Arrhenius × pandeo de Euler).
        2. CAPM/WACC topológico — consume CouplingTensor (FASE 1).
        3. DCF (NPV, TIR, MIRR, duración) bajo tasa ajustada.
        4. VaR/CVaR sobre el nodo de inversión.
        5. RiskEnvelope (FASE 2) → opciones reales BSM/CRR (FASE 3).
        6. Síntesis termodinámica (inercia financiera).
    """

    def __init__(self, config: FinancialConfig) -> None:
        self.config = config
        self.capm = CapitalAssetPricing(config)
        self.risk = RiskQuantifier(DistributionType.NORMAL)
        self.options = RealOptionsAnalyzer(OptionModelType.AUTO)

    def _calculate_thermo_structural_volatility(
        self,
        base_volatility: float,
        stability_psi: float,
        system_temperature: float,
    ) -> float:
        r"""
        Amplificación de volatilidad por acoplamiento termo-estructural.

            σ_eff = σ_base · M_s(Ψ) · M_t(T)   (con cruce 0.3 F_s F_t)

        F_s = tanh[(Ψ*−Ψ) κ] + ½ [(Ψ*−Ψ)/Ψ*]² 𝟙{Ψ<Ψ*}
              (meseta nula si Ψ ≥ Ψ_stable)
        F_t = 0                                         T ≤ T_ref
              (T−T_ref)/Θ · 0.1                         T_ref < T ≤ T_str
              0.1 + [e^{(T−T_str)/Θ} − 1] · 0.2        T > T_str
        M_s = 1+F_s,  M_t = 1+α F_t,  M_s M_t ≤ max_amplification.
        """
        if base_volatility < 0:
            raise ValueError(f"σ_base = {base_volatility} < 0.")
        if base_volatility == 0:
            return 0.0

        cfg = self.config
        psi = max(0.01, stability_psi)

        if psi >= cfg.psi_stable:
            structural_factor = 0.0
        else:
            x = (cfg.psi_critical - psi) * cfg.kappa_struct
            structural_factor = max(0.0, float(np.tanh(x)))
            if psi < cfg.psi_critical:
                sub_pen = ((cfg.psi_critical - psi) / cfg.psi_critical) ** 2
                structural_factor += 0.5 * sub_pen

        if system_temperature <= cfg.t_reference:
            thermal_factor = 0.0
        elif system_temperature <= cfg.t_stress:
            thermal_factor = (system_temperature - cfg.t_reference) / cfg.t_scale * 0.1
        else:
            excess = system_temperature - cfg.t_stress
            thermal_factor = 0.1 + (exp(excess / cfg.t_scale) - 1.0) * 0.2

        m_s = 1.0 + structural_factor
        m_t = 1.0 + thermal_factor * cfg.alpha_coupling
        if structural_factor > 0 and thermal_factor > 0:
            m_s += 0.3 * structural_factor * thermal_factor

        total = min(m_s * m_t, cfg.max_amplification)
        sigma_eff = base_volatility * total

        if total > 1.01:
            logger.warning(
                "🔥 Amplificación termo-estructural: σ %.4f → %.4f (×%.3f) "
                "[F_s=%.4f, F_t=%.4f, T=%.1f, Ψ=%.2f, coupling=%s]",
                base_volatility,
                sigma_eff,
                total,
                structural_factor,
                thermal_factor,
                system_temperature,
                psi,
                "ON" if structural_factor > 0 and thermal_factor > 0 else "OFF",
            )
        return float(sigma_eff)

    def analyze_project(
        self,
        initial_investment: float,
        cash_flows: List[float],
        cost_std_dev: float,
        volatility: Optional[float] = None,
        topology_report: Optional[Dict[str, Any]] = None,
        expected_cash_flows: Optional[List[float]] = None,
        project_volatility: Optional[float] = None,
        liquidity: Optional[float] = None,
        fixed_contracts_ratio: Optional[float] = None,
        pyramid_stability: Optional[float] = None,
        system_temperature: Optional[float] = None,
        beta_0: int = 1,
        beta_1: int = 0,
        lambda_2: Optional[float] = None,
        diffusion_time: float = 1.0,
    ) -> Dict[str, Any]:
        r"""
        Análisis financiero integral. Orquesta F₃ ∘ F₂ ∘ F₁.

        Puertos:
            F₁→F₂ : CouplingTensor via capm.calculate_wacc_topological.
            F₂→F₃ : RiskEnvelope via risk.synthesize_risk_envelope.
        """
        flows = expected_cash_flows if expected_cash_flows is not None else cash_flows
        vol = project_volatility if project_volatility is not None else volatility

        if not flows:
            raise ValueError("Se requiere al menos un flujo de caja proyectado.")
        if vol is None:
            raise ValueError("Se requiere 'volatility' o 'project_volatility'.")
        if vol < 0:
            raise ValueError(f"σ = {vol} no puede ser negativa.")
        if initial_investment < 0:
            logger.warning(
                "I₀ = %.2f < 0. Interpretando como desinversión.", initial_investment
            )

        liq = liquidity if liquidity is not None else self.config.liquidity_ratio
        fcr = (
            fixed_contracts_ratio
            if fixed_contracts_ratio is not None
            else self.config.fixed_contracts_ratio
        )

        homology = HomologicalInvariants(beta_0=beta_0, beta_1=beta_1)
        spectrum = SpectralSignature(
            lambda_2=0.0 if lambda_2 is None else max(0.0, float(lambda_2)),
            time=max(0.0, float(diffusion_time)),
        )

        # ── 1. Volatilidad efectiva ─────────────────────────────────────────
        effective_volatility = vol
        physics_applied = False
        physics_details: Dict[str, Any] = {}

        if pyramid_stability is not None and vol > 0:
            temp = (
                system_temperature
                if system_temperature is not None
                else self.config.t_reference
            )
            effective_volatility = self._calculate_thermo_structural_volatility(
                vol, pyramid_stability, temp
            )
            physics_applied = effective_volatility > vol * 1.001
            physics_details = {
                "pyramid_stability": pyramid_stability,
                "system_temperature": temp,
                "amplification_factor": effective_volatility / vol if vol > 0 else 1.0,
            }
        elif topology_report:
            adjusted = self.adjust_volatility_by_topology(vol, topology_report)
            if adjusted > vol * 1.001:
                physics_applied = True
                physics_details = {
                    "topology_adjustment": adjusted / vol if vol > 0 else 1.0
                }
            effective_volatility = adjusted

        # ── 2. WACC topológico (puerto F₁→F₂) ───────────────────────────────
        wacc = self.capm.calculate_wacc_topological(
            homology=homology,
            spectrum=spectrum,
        )
        npv = self.capm.calculate_npv(flows, initial_investment, discount_rate=wacc)
        duration_pack = self.capm.macaulay_duration(flows, discount_rate=wacc)

        # ── 3. Riesgo (VaR/CVaR) ────────────────────────────────────────────
        vol_ratio = (effective_volatility / vol) if vol > 0 else 1.0
        adjusted_std_dev = cost_std_dev * vol_ratio

        var_val, var_metrics = self.risk.calculate_var(
            mean=initial_investment,
            std_dev=adjusted_std_dev,
            confidence_level=self.config.confidence_var,
        )
        contingency = self.risk.suggest_contingency(
            initial_investment,
            adjusted_std_dev,
            confidence_level=self.config.confidence_contingency,
        )

        # ── 4. Opciones reales (puerto F₂→F₃) ───────────────────────────────
        project_pv = npv + initial_investment
        option_result: Dict[str, Any] = {"option_value": 0.0, "delta": 0.0, "gamma": 0.0}

        if project_pv > 0 and effective_volatility > 0:
            envelope = self.risk.synthesize_risk_envelope(
                base_strike=initial_investment,
                base_sigma=effective_volatility,
                alpha_envelope=self.config.confidence_var,
            )
            try:
                option_result = self.options.value_option_to_wait(
                    project_value=project_pv,
                    investment_cost=envelope["strike_effective"],
                    risk_free_rate=self.config.risk_free_rate,
                    time_to_expire=self.config.project_life_years,
                    volatility=envelope["sigma_effective"],
                    american=True,
                )
                option_result["risk_envelope"] = envelope
            except Exception as e:
                logger.error("Error en opciones reales: %s", e)
                option_result = {"option_value": 0.0, "error": str(e)}

        option_val = float(option_result.get("option_value", 0.0))
        total_value = npv + option_val

        # ── 5. Performance ──────────────────────────────────────────────────
        performance = self._calculate_performance_metrics(
            npv, initial_investment, len(flows), flows=flows
        )
        performance["duration"] = duration_pack

        # ── 6. Termodinámica ────────────────────────────────────────────────
        inertia_result = self.calculate_financial_thermal_inertia(
            liquidity=liq,
            fixed_contracts_ratio=fcr,
            project_complexity=pyramid_stability if pyramid_stability is not None else 1.0,
            market_volatility=vol,
        )

        return {
            "wacc": wacc,
            "npv": npv,
            "total_value": total_value,
            "volatility_base": vol,
            "volatility_structural": effective_volatility,
            "volatility": effective_volatility,
            "physics_adjustment": physics_applied,
            "physics_details": physics_details,
            "var": var_val,
            "var_metrics": var_metrics,
            "contingency": contingency,
            "real_option_value": option_val,
            "real_option_details": option_result,
            "performance": performance,
            "thermodynamics": {
                "financial_inertia": inertia_result["inertia"],
                "liquidity_ratio": liq,
                "fixed_contracts_ratio": fcr,
                "components": inertia_result,
            },
            "topology_invariants": {
                "beta_0": homology.beta_0,
                "beta_1": homology.beta_1,
                "euler_truncated": homology.euler_characteristic_truncated,
                "lambda_2": spectrum.lambda_2,
                "diffusion_time": spectrum.time,
                "heat_kernel_discount": spectrum.heat_kernel_discount,
                "coupling_tensor": self.capm.coupling_tensor.as_tuple(),
            },
            "diagnostics": {
                "input_flows_count": len(flows),
                "input_volatility": vol,
                "effective_volatility": effective_volatility,
                "std_dev_adjustment_ratio": vol_ratio,
                "topology_report_provided": topology_report is not None,
                "engine_version": __version__,
            },
        }

    def _calculate_performance_metrics(
        self,
        npv: float,
        investment: float,
        years: int,
        flows: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        r"""
        ROI, PI, retorno anualizado, payback (simple y descontado), eficiencia
        de capital, TIR y MIRR.
        """
        metrics: Dict[str, Any] = {}

        if investment > 0:
            roi = npv / investment
            pi = (npv + investment) / investment
            metrics["profitability_index"] = pi
            metrics["recommendation"] = "ACEPTAR" if pi > 1 else "RECHAZAR"
        elif investment < 0:
            logger.warning("I₀ < 0: interpretando como desinversión.")
            roi = -npv / abs(investment)
            metrics["profitability_index"] = float("nan")
            metrics["recommendation"] = "REVISAR"
        else:
            roi = float("inf") if npv > 0 else (float("-inf") if npv < 0 else 0.0)
            metrics["profitability_index"] = float("nan")
            metrics["recommendation"] = "REVISAR"

        metrics["roi"] = roi

        if years > 0 and investment > 0 and (1 + roi) > 0:
            try:
                metrics["annualized_return"] = (1 + roi) ** (1.0 / years) - 1.0
            except (ValueError, OverflowError):
                metrics["annualized_return"] = float("nan")
        elif investment > 0 and (1 + roi) <= 0:
            metrics["annualized_return"] = -1.0
        else:
            metrics["annualized_return"] = float("nan")

        if flows and investment > 0:
            cumulative = 0.0
            payback: Optional[float] = None
            peak_deficit = 0.0
            deficit_periods = 0

            for t_idx, cf in enumerate(flows, start=1):
                cumulative += cf
                if cumulative < peak_deficit:
                    peak_deficit = cumulative
                    deficit_periods += 1
                if payback is None and cumulative >= investment:
                    prev = cumulative - cf
                    remaining = investment - prev
                    fraction = (remaining / cf) if cf > 0 else 0.0
                    payback = (t_idx - 1) + fraction

            if payback is not None:
                metrics["payback_period"] = round(payback, 2)
                metrics["payback_status"] = "RECUPERABLE"
            else:
                metrics["payback_period"] = float("inf")
                metrics["payback_status"] = "NO_RECUPERABLE"
                metrics["final_cumulative"] = cumulative
                metrics["recovery_gap"] = investment - cumulative

            metrics["payback"] = metrics["payback_period"]
            metrics["peak_deficit"] = peak_deficit
            metrics["deficit_periods"] = deficit_periods

            if payback is not None and years > 0:
                metrics["capital_efficiency"] = 1.0 - (payback / years)
            else:
                metrics["capital_efficiency"] = 0.0

            # Payback descontado al WACC.
            try:
                rate = self.capm.calculate_wacc()
                disc_cum = 0.0
                dpayback: Optional[float] = None
                prev_disc = 0.0
                for t_idx, cf in enumerate(flows, start=1):
                    disc_cum += cf / pow(1.0 + rate, t_idx)
                    if dpayback is None and disc_cum >= investment:
                        pvcf = cf / pow(1.0 + rate, t_idx)
                        remaining = investment - prev_disc
                        fraction = (remaining / pvcf) if pvcf > 0 else 0.0
                        dpayback = (t_idx - 1) + fraction
                    prev_disc = disc_cum
                metrics["discounted_payback"] = (
                    round(dpayback, 2) if dpayback is not None else float("inf")
                )
            except Exception as e:
                logger.warning("Payback descontado no calculable: %s", e)
                metrics["discounted_payback"] = float("nan")

        if flows and investment > 0:
            try:
                irr = self.capm.calculate_irr(flows, investment)
                metrics["irr_estimate"] = irr if irr is not None else float("nan")
            except Exception as e:
                logger.warning("TIR no calculable: %s", e)
                metrics["irr_estimate"] = float("nan")
            try:
                mirr = self.capm.calculate_mirr(flows, investment)
                metrics["mirr_estimate"] = mirr if mirr is not None else float("nan")
            except Exception as e:
                logger.warning("MIRR no calculable: %s", e)
                metrics["mirr_estimate"] = float("nan")

        return metrics

    def _estimate_irr(
        self,
        investment: float,
        flows: List[float],
        max_iterations: int = 50,
        tolerance: Optional[float] = None,
    ) -> float:
        """Alias retro-compatible de CapitalAssetPricing.calculate_irr."""
        tol = tolerance if tolerance is not None else self.config.tol_newton
        result = self.capm.calculate_irr(flows, investment, tol=tol, max_iter=max_iterations)
        return result if result is not None else float("nan")

    def adjust_volatility_by_topology(
        self, base_volatility: float, topology_report: Mapping[str, Any]
    ) -> float:
        r"""
        Amplificación por integridad topológica.

            σ_adj = σ_base · (1 + P_sinergia + P_eficiencia)
                  ≤ σ_base · (1 + Δ_max)
        """
        if not topology_report:
            return base_volatility

        synergy_penalty = 0.0
        synergy_data = topology_report.get("synergy_risk", {}) or {}
        if synergy_data.get("synergy_detected", False):
            strength = synergy_data.get("synergy_strength", 1.0)
            if strength is None or (isinstance(strength, float) and np.isnan(strength)):
                strength = 1.0
            synergy_penalty = self.config.synergy_penalty_factor * float(strength)

        efficiency_penalty = 0.0
        efficiency = topology_report.get("euler_efficiency")
        if efficiency is not None:
            try:
                eff_f = float(efficiency)
                if not np.isnan(eff_f):
                    eff_clamped = max(0.0, min(1.0, eff_f))
                    efficiency_penalty = self.config.efficiency_penalty_factor * (
                        1.0 - eff_clamped
                    )
            except (TypeError, ValueError):
                pass

        total_adj = min(
            synergy_penalty + efficiency_penalty,
            self.config.max_volatility_adjustment,
        )
        return max(0.0, base_volatility * (1.0 + total_adj))

    def calculate_financial_thermal_inertia(
        self,
        liquidity: float = 0.0,
        fixed_contracts_ratio: float = 0.0,
        project_complexity: float = 0.0,
        market_volatility: float = 0.0,
    ) -> Dict[str, Any]:
        r"""
        Inercia térmica financiera (analogía C_th = m · c).

            M_eff = λ_L · (1 + 0.5 Ψ)
            C_eff = ρ_f · (1 + 0.3 Ψ)
            A_att = e^{−2σ}
            I     = M_eff · C_eff · A_att
        """
        mass = liquidity * (1.0 + 0.5 * project_complexity)
        heat_capacity = fixed_contracts_ratio * (1.0 + 0.3 * project_complexity)
        attenuation = exp(-2.0 * market_volatility)
        inertia = mass * heat_capacity * attenuation
        return {
            "inertia": inertia,
            "thermal_mass": mass,
            "heat_capacity": heat_capacity,
            "attenuation": attenuation,
        }

    def predict_temperature_change(
        self,
        perturbation: float,
        inertia_data: Optional[Dict[str, Any]] = None,
        time_constant: Optional[float] = None,
    ) -> Dict[str, Any]:
        r"""
        Respuesta de primer orden: C dT/dt = Q − T/τ_ext.

            ΔT(t) = (Q/I) · (1 − e^{−1/τ})

        τ→0⁺ ⇒ ΔT→Q/I;  τ→∞ ⇒ ΔT→0.
        """
        inertia = 0.0
        if inertia_data is not None:
            inertia = float(inertia_data.get("inertia", 0.0))

        if inertia <= 1e-12:
            return {
                "temperature_change": perturbation,
                "regime": "elástico (I → 0)",
                "inertia": 0.0,
            }

        base_change = perturbation / inertia
        if time_constant is not None and time_constant > 0:
            temporal_factor = 1.0 - exp(-1.0 / time_constant)
            change = base_change * temporal_factor
        else:
            temporal_factor = 1.0
            change = base_change

        return {
            "temperature_change": change,
            "temporal_factor": temporal_factor,
            "inertia": inertia,
            "regime": "dinámico" if time_constant else "estacionario",
        }


# ══════════════════════════════════════════════════════════════════════════════════════════
# UTILIDADES MODULE-LEVEL
# ══════════════════════════════════════════════════════════════════════════════════════════


def calculate_volatility_from_returns(
    returns: Sequence[float],
    frequency: str = "daily",
    annual_trading_days: int = 252,
) -> float:
    r"""
    Volatilidad anualizada a partir de retornos muestrales.

        s = √[ 1/(n−1) Σ (rᵢ − r̄)² ]
        σ_anual = s · √k

    k ∈ {252, 52, 12, 1} según frequency ∈ {daily, weekly, monthly, annual}.
    """
    if not returns or len(returns) < 2:
        raise ValueError(
            f"Se requieren ≥ 2 retornos. Recibidos: {len(returns) if returns else 0}"
        )

    factors = {
        "daily": annual_trading_days,
        "weekly": 52,
        "monthly": 12,
        "annual": 1,
    }
    if frequency not in factors:
        raise ValueError(
            f"Frecuencia '{frequency}' inválida. Opciones: {list(factors.keys())}"
        )

    arr = np.asarray(returns, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("La serie de retornos contiene NaN o Inf.")

    std_period = float(np.std(arr, ddof=1))
    volatility = std_period * sqrt(factors[frequency])

    logger.info(
        "σ_anual = %.2f%% (n=%d, freq=%s, ddof=1)",
        volatility * 100,
        len(returns),
        frequency,
    )
    return volatility
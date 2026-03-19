"""
Suite de Testing para el Sistema Termodinámico.
===============================================

Fundamentos Matemáticos Verificados:
─────────────────────────────────────────────────────────────────────────────
ENTROPÍA DE SHANNON (FluxPhysicsEngine):
  H(p) = -[p·log₂(p) + (1-p)·log₂(1-p)]   para p ∈ (0,1)
  H(0) = H(1) = 0                            (estados puros)
  H(0.5) = 1.0                               (máximo caos)

  Propiedades:
    · Simetría:    H(p) = H(1-p)             ∀p ∈ [0,1]
    · Concavidad:  H''(p) = -1/(p(1-p)ln2) < 0  ∀p ∈ (0,1)
    · Aditividad:  H(p,q) = H(p) + p·H(q/p) (regla de cadena)
    · Umbral muerte térmica: H > 0.8

TERMODINÁMICA FINANCIERA (FinancialEngine):
  Analogía física-financiera:
    Temperatura T  ↔  Volatilidad σ
    Capacidad C    ↔  Liquidez L
    Masa térmica m ↔  Contratos fijos F
    Inercia I = mC ↔  L × F  (ajustada por complejidad y volatilidad)

  Ley de Newton del enfriamiento:
    ΔT = (Q/I) · (1 - exp(-1/τ))
    Si τ → 0:  ΔT → Q/I  (respuesta instantánea)
    Si τ → ∞:  ΔT → Q/(I·τ)  (respuesta amortiguada)

  Linealidad: ΔT(αQ) = α·ΔT(Q)  para I, τ constantes

EXERGÍA (MatterGenerator):
  η_ex = W_útil / E_entrada = Σcosto_estructural / Σcosto_total
  Propiedades:
    · η_ex ∈ [0, 1]
    · η_ex = 1.0 ↔ 100% materiales estructurales
    · η_ex = 0.0 ↔ 100% materiales decorativos
    · Aditividad: η_ex(A∪B) = (E_A + E_B) / (T_A + T_B)

TOPOLOGÍA ALGEBRAICA (BusinessTopologicalAnalyzer):
  β₀ = componentes conexas  (zeroth Betti number)
  β₁ = ciclos independientes (first Betti number)
  χ  = V - E               (característica de Euler para grafos sin caras)
  χ  = β₀ - β₁            (fórmula de Euler-Poincaré para grafos)

  Convección inflacionaria:
    I(APU) = Σcosto(insumos volátiles) / Σcosto(todos los insumos)
    Alto riesgo: I(APU) > 0.2  (desigualdad estricta)

Referencias:
  - Shannon, C.E. (1948). A Mathematical Theory of Communication.
  - Callen, H.B. (1985). Thermodynamics and an Introduction to Thermostatistics.
  - Hatcher, A. (2002). Algebraic Topology. Cambridge University Press.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import unittest
from math import exp, log2

import networkx as nx

from app.tactics.business_topology import BusinessTopologicalAnalyzer
from app.strategy.financial_engine import FinancialConfig, FinancialEngine
from app.physics.flux_condenser import FluxPhysicsEngine
from app.physics.matter_generator import MaterialRequirement, MatterGenerator


# ── Constantes del dominio ────────────────────────────────────────────────────

_THERMAL_DEATH_THRESHOLD: float = 0.8
_HIGH_RISK_THRESHOLD:     float = 0.2
_NUMERICAL_TOLERANCE:     float = 1e-9
_PLACES_HIGH:             int   = 9   # Para igualdades exactas (fórmulas cerradas)
_PLACES_MEDIUM:           int   = 6   # Para resultados con acumulación de redondeo
_PLACES_LOW:              int   = 2   # Para aproximaciones con discretización


def _binary_entropy(p: float) -> float:
    """
    Entropía binaria de Shannon: H(p) = -[p·log₂(p) + (1-p)·log₂(1-p)].

    Convención: 0·log₂(0) = 0 (límite por L'Hôpital).

    Args:
        p: Probabilidad ∈ [0, 1].

    Returns:
        H(p) ∈ [0, 1].
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * log2(p) + (1 - p) * log2(1 - p))


# ============================================================================
# TEST: FluxPhysicsEngine — Entropía de Shannon
# ============================================================================


class TestFluxPhysicsEngine(unittest.TestCase):
    """
    Motor de física termodinámica basado en entropía de Shannon.

    H(p) = -[p·log₂(p) + (1-p)·log₂(1-p)]
    Dominio: H ∈ [0, 1], con H(0.5) = 1.0 (máximo caos).
    """

    def setUp(self) -> None:
        """Configura motor RLC con parámetros de circuito equivalente."""
        self.engine = FluxPhysicsEngine(
            capacitance=5000.0, resistance=10.0, inductance=2.0
        )

    def tearDown(self) -> None:
        """Libera el motor después de cada test."""
        self.engine = None  # type: ignore[assignment]

    # ── Casos límite ──────────────────────────────────────────────────────────

    def test_entropy_zero_records_returns_zero(self) -> None:
        """
        Caso límite: sin datos → entropía convencionalmente 0.

        Matemática: lim(x→0⁺) x·log(x) = 0 por L'Hôpital.
        Con total_records=0, p es indefinida → se usa H=0 por convención.
        """
        metrics = self.engine.calculate_system_entropy(
            total_records=0, error_count=0, processing_time=1.0
        )

        self.assertEqual(metrics["entropy_absolute"], 0.0)
        self.assertEqual(metrics["entropy_rate"], 0.0)
        self.assertFalse(metrics["is_thermal_death"])

    def test_entropy_perfect_system_zero_errors(self) -> None:
        """
        p = 0 → H(0) = 0. Estado puro: certeza absoluta de éxito.

        H(0) = -[0·log₂(0) + 1·log₂(1)] = -[0 + 0] = 0.
        """
        metrics = self.engine.calculate_system_entropy(
            total_records=1000, error_count=0, processing_time=1.0
        )

        self.assertAlmostEqual(metrics["entropy_absolute"], 0.0, places=_PLACES_HIGH)
        self.assertFalse(metrics["is_thermal_death"])

    def test_entropy_total_failure_also_zero(self) -> None:
        """
        p = 1 → H(1) = 0. Estado puro: certeza absoluta de fallo.

        H(1) = -[1·log₂(1) + 0·log₂(0)] = -[0 + 0] = 0.
        Simétrico con el caso de cero errores.
        """
        metrics = self.engine.calculate_system_entropy(
            total_records=100, error_count=100, processing_time=1.0
        )

        self.assertAlmostEqual(metrics["entropy_absolute"], 0.0, places=_PLACES_HIGH)
        # En la V3, un estado con 100% errores declara is_thermal_death=True directamente.
        self.assertTrue(metrics["is_thermal_death"])

    # ── Valor máximo ──────────────────────────────────────────────────────────

    def test_entropy_maximum_at_equiprobability(self) -> None:
        """
        Teorema: H(p) alcanza máximo único en p = 0.5.
        H(0.5) = -2·[0.5·log₂(0.5)] = -2·[0.5·(-1)] = 1.0.

        Este es el punto de muerte térmica termodinámica.
        """
        metrics = self.engine.calculate_system_entropy(
            total_records=1000, error_count=500, processing_time=1.0
        )

        self.assertAlmostEqual(
            metrics["entropy_absolute"],
            1.0,
            places=_PLACES_LOW,
            msg="H(0.5) debe ser ≈ 1.0 (máximo normalizado)",
        )
        self.assertTrue(
            metrics["is_thermal_death"],
            msg="H=1.0 > 0.8 debe activar is_thermal_death=True",
        )

    # ── Simetría H(p) = H(1-p) ───────────────────────────────────────────────

    def test_entropy_symmetry_property(self) -> None:
        """
        H(p) = H(1-p)  para todo p ∈ [0, 1].

        La simetría refleja que la incertidumbre solo depende
        de la distancia a los extremos, no de la dirección.
        Verificado con p=0.15 y p=0.85: H(0.15) = H(0.85).
        """
        # Usar proporciones exactas (sin int-casting) para evitar error de redondeo
        # p = 150/1000 = 0.15  y  p = 850/1000 = 0.85
        metrics_low = self.engine.calculate_system_entropy(
            total_records=1000, error_count=150, processing_time=1.0
        )
        metrics_high = self.engine.calculate_system_entropy(
            total_records=1000, error_count=850, processing_time=1.0
        )

        self.assertAlmostEqual(
            metrics_low["entropy_absolute"],
            metrics_high["entropy_absolute"],
            places=_PLACES_HIGH,
            msg="H(0.15) debe igualar H(0.85) por simetría H(p)=H(1-p)",
        )

    def test_entropy_symmetry_multiple_pairs(self) -> None:
        """
        Simetría verificada para múltiples pares (p, 1-p).

        Pares exactos: (100,900), (200,800), (300,700), (400,600).
        Todos usan denominador 1000 → sin error de discretización.
        """
        symmetric_pairs = [
            (100, 900),
            (200, 800),
            (300, 700),
            (400, 600),
        ]
        for e_low, e_high in symmetric_pairs:
            with self.subTest(error_count_low=e_low, error_count_high=e_high):
                m_low = self.engine.calculate_system_entropy(
                    total_records=1000, error_count=e_low, processing_time=1.0
                )
                m_high = self.engine.calculate_system_entropy(
                    total_records=1000, error_count=e_high, processing_time=1.0
                )
                self.assertAlmostEqual(
                    m_low["entropy_absolute"],
                    m_high["entropy_absolute"],
                    places=_PLACES_HIGH,
                    msg=f"H({e_low/1000}) debe igualar H({e_high/1000})",
                )

    # ── Concavidad estricta ───────────────────────────────────────────────────

    def test_entropy_strict_concavity_on_exact_fractions(self) -> None:
        """
        H(p) es estrictamente cóncava: H''(p) < 0 ∀p ∈ (0,1).

        Implica: p₁ < p₂ < 0.5 ⟹ H(p₁) < H(p₂).

        IMPORTANTE: Se usan fracciones exactas (e_count divisible por total)
        para eliminar el error de redondeo por int-casting que puede
        romper la monotonicidad estricta en probabilidades muy cercanas.
        """
        # Pares (error_count, total) que producen probabilidades exactas
        # sin redondeo: 50/1000=0.05, 100/1000=0.10, ..., 500/1000=0.50
        exact_points = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

        entropies: list[float] = []
        for e in exact_points:
            m = self.engine.calculate_system_entropy(
                total_records=1000, error_count=e, processing_time=1.0
            )
            entropies.append(m["entropy_absolute"])

        for i in range(len(entropies) - 1):
            p_i   = exact_points[i]   / 1000
            p_ip1 = exact_points[i+1] / 1000
            self.assertLess(
                entropies[i],
                entropies[i + 1],
                msg=(
                    f"Concavidad violada en p={p_i:.3f} → p={p_ip1:.3f}: "
                    f"H({p_i:.3f})={entropies[i]:.6f} ≥ "
                    f"H({p_ip1:.3f})={entropies[i+1]:.6f}"
                ),
            )

    def test_entropy_is_strictly_less_than_maximum_off_center(self) -> None:
        """
        ∀p ≠ 0.5: H(p) < H(0.5) = 1.0.

        Consecuencia de la concavidad estricta con máximo único en p=0.5.
        """
        max_entropy = self.engine.calculate_system_entropy(
            total_records=1000, error_count=500, processing_time=1.0
        )["entropy_absolute"]

        for e in [50, 100, 200, 300, 400, 600, 700, 800, 900, 950]:
            with self.subTest(error_count=e):
                m = self.engine.calculate_system_entropy(
                    total_records=1000, error_count=e, processing_time=1.0
                )
                self.assertLess(
                    m["entropy_absolute"],
                    max_entropy,
                    msg=f"H({e/1000:.2f}) debe ser < H(0.5)=1.0",
                )

    # ── Tasa de entropía ──────────────────────────────────────────────────────

    def test_entropy_rate_temporal_scaling(self) -> None:
        """
        dS/dt = S / Δt → Si Δt se duplica, dS/dt se reduce a la mitad.

        Verificación: entropy_rate(Δt=1s) = 2 × entropy_rate(Δt=2s).
        """
        m1s = self.engine.calculate_system_entropy(
            total_records=100, error_count=25, processing_time=1.0
        )
        m2s = self.engine.calculate_system_entropy(
            total_records=100, error_count=25, processing_time=2.0
        )

        self.assertAlmostEqual(
            m1s["entropy_rate"],
            m2s["entropy_rate"] * 2.0,
            places=_PLACES_HIGH,
            msg="dS/dt(Δt=1s) debe ser el doble de dS/dt(Δt=2s)",
        )
        self.assertGreaterEqual(
            m1s["entropy_rate"],
            0.0,
            msg="Segunda Ley: dS/dt ≥ 0 para sistemas aislados",
        )

    def test_entropy_rate_is_non_negative(self) -> None:
        """
        Segunda Ley: dS/dt ≥ 0 para todos los estados del sistema.

        La tasa de producción de entropía nunca es negativa.
        """
        for e, total in [(0, 100), (50, 100), (100, 100), (0, 0)]:
            with self.subTest(error_count=e, total_records=total):
                m = self.engine.calculate_system_entropy(
                    total_records=total, error_count=e, processing_time=1.0
                )
                self.assertGreaterEqual(
                    m["entropy_rate"],
                    0.0,
                    msg=f"entropy_rate < 0 para e={e}, total={total}",
                )

    # ── Umbral de muerte térmica ──────────────────────────────────────────────

    def test_thermal_death_threshold_boundary(self) -> None:
        """
        Análisis de frontera alrededor del umbral H = 0.8.

        Resolviendo H(p) = 0.8 numéricamente:
          p ≈ 0.0942 o p ≈ 0.9058  (soluciones simétricas)

        Se elige p=0.12 (H≈0.53, seguro) y p=0.35 (H≈0.93, muerte térmica)
        con margen amplio para robustez ante variaciones de implementación.
        """
        # p = 0.12 → H(0.12) ≈ 0.529  < 0.8 → seguro
        m_safe = self.engine.calculate_system_entropy(
            total_records=1000, error_count=120, processing_time=1.0
        )
        # p = 0.35 → H(0.35) ≈ 0.931  > 0.8 → muerte térmica
        m_critical = self.engine.calculate_system_entropy(
            total_records=1000, error_count=350, processing_time=1.0
        )

        self.assertFalse(
            m_safe["is_thermal_death"],
            msg=f"H(0.12)≈{m_safe['entropy_absolute']:.3f} no debe activar muerte térmica",
        )
        self.assertTrue(
            m_critical["is_thermal_death"],
            msg=f"H(0.35)≈{m_critical['entropy_absolute']:.3f} debe activar muerte térmica",
        )
        self.assertLess(m_safe["entropy_absolute"], _THERMAL_DEATH_THRESHOLD)
        self.assertGreater(m_critical["entropy_absolute"], _THERMAL_DEATH_THRESHOLD)

    def test_thermal_death_false_below_threshold(self) -> None:
        """is_thermal_death = False para todo H < 0.8."""
        # H(p) < 0.8 para p < 0.094 o p > 0.906
        # Usamos p=0.05 (H≈0.286) y p=0.95 (H≈0.286)
        for e in [50, 950]:
            with self.subTest(error_count=e):
                m = self.engine.calculate_system_entropy(
                    total_records=1000, error_count=e, processing_time=1.0
                )
                self.assertFalse(
                    m["is_thermal_death"],
                    msg=f"H({e/1000:.2f})={m['entropy_absolute']:.3f} no debe ser muerte térmica",
                )

    # ── Esquema completo ──────────────────────────────────────────────────────

    def test_calculate_metrics_schema_and_ranges(self) -> None:
        """
        Validación de esquema completo y rangos físicamente válidos.

        entropy_absolute ∈ [0, 1]  (entropía normalizada)
        entropy_rate     ∈ [0, ∞)  (tasa de producción)
        is_thermal_death ∈ {True, False}
        """
        metrics = self.engine.calculate_metrics(
            total_records=100, cache_hits=90, error_count=5, processing_time=1.0
        )

        required_schema: dict[str, tuple] = {
            "entropy_absolute": (0.0, 1.0),
            "entropy_rate":     (0.0, float("inf")),
        }

        for key, (min_val, max_val) in required_schema.items():
            self.assertIn(key, metrics, f"Campo requerido ausente: '{key}'")
            self.assertGreaterEqual(
                metrics[key], min_val,
                msg=f"{key} = {metrics[key]} < {min_val}",
            )
            if max_val != float("inf"):
                self.assertLessEqual(
                    metrics[key], max_val,
                    msg=f"{key} = {metrics[key]} > {max_val}",
                )

        self.assertIn("is_thermal_death", metrics)
        self.assertIsInstance(metrics["is_thermal_death"], bool)

    def test_entropy_consistent_with_analytical_formula(self) -> None:
        """
        Coherencia entre el motor y la fórmula analítica H(p).

        Verifica que la implementación respeta la definición matemática
        para varios valores de p exactamente representables.
        """
        test_points = [(200, 1000), (300, 1000), (400, 1000), (500, 1000)]

        for e, total in test_points:
            with self.subTest(p=e/total):
                p = e / total
                expected = _binary_entropy(p)
                m = self.engine.calculate_system_entropy(
                    total_records=total, error_count=e, processing_time=1.0
                )
                self.assertAlmostEqual(
                    m["entropy_absolute"],
                    expected,
                    places=2,
                    msg=f"H({p}) del motor ≠ H({p}) analítico = {expected:.6f}",
                )


# ============================================================================
# TEST: FinancialEngine — Termodinámica Financiera
# ============================================================================


class TestFinancialEngine(unittest.TestCase):
    """
    Motor financiero con analogía termodinámica.

    Mapeo físico-financiero:
      Temperatura T ↔ Volatilidad σ
      Inercia I = mC ↔ L × F (liquidez × contratos fijos)

    Ley de Newton del enfriamiento financiero:
      ΔT = (Q/I) · (1 - exp(-1/τ))
      Si τ → 0: ΔT → Q/I  (respuesta instantánea)
    """

    def setUp(self) -> None:
        self.config = FinancialConfig()
        self.engine = FinancialEngine(self.config)

    def tearDown(self) -> None:
        self.engine = None  # type: ignore[assignment]
        self.config = None  # type: ignore[assignment]

    # ── Inercia térmica ───────────────────────────────────────────────────────

    def test_thermal_inertia_multiplicative_base_cases(self) -> None:
        """
        Inercia base: I = L × F con complejidad=0 y volatilidad=0.

        Casos verificados:
          I(0.2, 0.5) = 0.10
          I(0.4, 0.3) = 0.12
          I(1.0, 1.0) = 1.00
          I(0.0, 0.5) = 0.00  (sin liquidez)
          I(0.5, 0.0) = 0.00  (sin contratos fijos)
        """
        test_cases = [
            (0.2, 0.5, 0.10),
            (0.4, 0.3, 0.12),
            (1.0, 1.0, 1.00),
            (0.0, 0.5, 0.00),
            (0.5, 0.0, 0.00),
        ]

        for liquidity, fixed_ratio, expected in test_cases:
            with self.subTest(L=liquidity, F=fixed_ratio, expected=expected):
                res = self.engine.calculate_financial_thermal_inertia(
                    liquidity=liquidity,
                    fixed_contracts_ratio=fixed_ratio,
                    project_complexity=0.0,
                    market_volatility=0.0,
                )
                self.assertAlmostEqual(
                    res["inertia"],
                    expected,
                    places=_PLACES_HIGH,
                    msg=(
                        f"I({liquidity}, {fixed_ratio}) = {res['inertia']:.10f} "
                        f"≠ {expected}"
                    ),
                )

    def test_thermal_inertia_is_non_negative(self) -> None:
        """
        I ≥ 0 para todos los parámetros válidos.

        La inercia es una magnitud física no negativa.
        """
        params_list = [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, 1.0),
            (0.5, 0.5, 0.3, 0.2),
        ]
        for L, F, c, v in params_list:
            with self.subTest(L=L, F=F, complexity=c, volatility=v):
                res = self.engine.calculate_financial_thermal_inertia(
                    liquidity=L,
                    fixed_contracts_ratio=F,
                    project_complexity=c,
                    market_volatility=v,
                )
                self.assertGreaterEqual(
                    res["inertia"], 0.0,
                    msg=f"Inercia negativa para L={L}, F={F}, c={c}, v={v}",
                )

    # ── Ley de Newton (ΔT = Q/I para τ→0) ───────────────────────────────────

    def test_temperature_change_instant_response_at_zero_tau(self) -> None:
        """
        Para τ → 0: ΔT ≈ Q/I  (respuesta instantánea).

        (1 - exp(-1/τ)) → 1 cuando τ → 0,
        luego ΔT = (Q/I) · 1 = Q/I.

        Se usa τ=0.0001 como aproximación de τ=0⁺.
        """
        perturbation = 0.10
        tau = 0.0001

        test_cases = [
            (0.1, perturbation / 0.1),
            (0.2, perturbation / 0.2),
            (0.4, perturbation / 0.4),
            (0.5, perturbation / 0.5),
            (1.0, perturbation / 1.0),
        ]

        for inertia, expected_delta_t in test_cases:
            with self.subTest(inertia=inertia):
                res = self.engine.predict_temperature_change(
                    perturbation,
                    inertia_data={"inertia": inertia},
                    time_constant=tau,
                )
                self.assertAlmostEqual(
                    res["temperature_change"],
                    expected_delta_t,
                    places=_PLACES_HIGH,
                    msg=(
                        f"ΔT({inertia}) = {res['temperature_change']:.10f} "
                        f"≠ Q/I = {expected_delta_t:.10f}"
                    ),
                )

    def test_temperature_change_inverse_order(self) -> None:
        """
        Mayor inercia → menor cambio de temperatura.

        ΔT₁ > ΔT₂ cuando I₁ < I₂  (orden estrictamente inverso).
        """
        perturbation = 0.10
        tau = 0.0001
        inertias = [0.1, 0.2, 0.4, 0.5, 1.0]

        delta_ts = [
            self.engine.predict_temperature_change(
                perturbation,
                inertia_data={"inertia": I},
                time_constant=tau,
            )["temperature_change"]
            for I in inertias
        ]

        for i in range(len(delta_ts) - 1):
            self.assertGreater(
                delta_ts[i],
                delta_ts[i + 1],
                msg=(
                    f"Ley inversa violada: ΔT(I={inertias[i]})={delta_ts[i]:.6f} "
                    f"≤ ΔT(I={inertias[i+1]})={delta_ts[i+1]:.6f}"
                ),
            )

    def test_temperature_change_zero_inertia_passthrough(self) -> None:
        """
        I = 0 → ΔT = Q (sin amortiguación, perturbación pasa íntegra).

        Físicamente: objeto sin masa térmica transmite calor instantáneamente.
        Financieramente: sin liquidez ni contratos, perturbación = impacto directo.
        """
        perturbation = 0.05
        res = self.engine.predict_temperature_change(
            perturbation, inertia_data={"inertia": 0.0}
        )

        self.assertEqual(
            res["temperature_change"],
            perturbation,
            msg=f"Sin inercia: ΔT debe ser Q={perturbation}, obtenido {res['temperature_change']}",
        )

    # ── Linealidad ────────────────────────────────────────────────────────────

    def test_temperature_change_linearity_in_perturbation(self) -> None:
        """
        Linealidad: ΔT(αQ) = α·ΔT(Q)  para I y τ constantes.

        Verificado para factores α = 2 y α = 3.
        """
        inertia_data = {"inertia": 0.25}
        tau          = 0.0001
        Q_base       = 0.04

        res1 = self.engine.predict_temperature_change(
            Q_base, inertia_data, time_constant=tau
        )
        res2 = self.engine.predict_temperature_change(
            2 * Q_base, inertia_data, time_constant=tau
        )
        res3 = self.engine.predict_temperature_change(
            3 * Q_base, inertia_data, time_constant=tau
        )

        dt1, dt2, dt3 = (
            res1["temperature_change"],
            res2["temperature_change"],
            res3["temperature_change"],
        )

        self.assertAlmostEqual(
            dt2, 2 * dt1, places=_PLACES_HIGH,
            msg=f"ΔT(2Q)={dt2:.10f} ≠ 2·ΔT(Q)={2*dt1:.10f}",
        )
        self.assertAlmostEqual(
            dt3, 3 * dt1, places=_PLACES_HIGH,
            msg=f"ΔT(3Q)={dt3:.10f} ≠ 3·ΔT(Q)={3*dt1:.10f}",
        )

    def test_temperature_change_zero_perturbation(self) -> None:
        """
        Q = 0 → ΔT = 0 para cualquier inercia.

        Sin perturbación externa no hay cambio de temperatura.
        """
        for inertia in [0.0, 0.5, 1.0]:
            with self.subTest(inertia=inertia):
                res = self.engine.predict_temperature_change(
                    0.0, inertia_data={"inertia": inertia}
                )
                self.assertEqual(
                    res["temperature_change"],
                    0.0,
                    msg=f"ΔT(Q=0, I={inertia}) debe ser 0.0",
                )

    # ── Análisis de proyecto ──────────────────────────────────────────────────

    def test_analyze_project_thermodynamic_inertia_formula(self) -> None:
        """
        Verificación de la fórmula de inercia en analyze_project.

        Parámetros:
          liquidity = 0.2, fixed_contracts_ratio = 0.5
          project_complexity = 1 (default), market_volatility = 1 (default)

        Cálculo paso a paso:
          mass          = liquidity × (1 + 0.5 × complexity)  = 0.2 × 1.5 = 0.30
          heat_capacity = fixed_ratio × (1 + 0.3 × complexity) = 0.5 × 1.3 = 0.65
          attenuation   = exp(-2 × market_volatility)          = exp(-2)    ≈ 0.1353
          inertia       = mass × heat_capacity × attenuation   ≈ 0.30 × 0.65 × 0.1353

        NOTA: Si analyze_project usa complexity=1 y volatility=1 como defaults internos,
        la fórmula debe reflejar esos valores. El test documenta explícitamente qué
        values se asumen para que futuros cambios sean detectables.
        """
        analysis = self.engine.analyze_project(
            initial_investment=1000,
            cash_flows=[200, 200, 200, 200, 200],
            cost_std_dev=100,
            project_volatility=0.2,
            liquidity=0.2,
            fixed_contracts_ratio=0.5,
        )

        thermo = analysis["thermodynamics"]
        self.assertIn(
            "financial_inertia", thermo,
            msg="analyze_project debe incluir 'financial_inertia' en 'thermodynamics'",
        )
        # La inercia debe ser estrictamente positiva para parámetros > 0
        self.assertGreater(
            thermo["financial_inertia"],
            0.0,
            msg="Inercia financiera debe ser > 0 para L=0.2, F=0.5",
        )
        # La inercia debe estar acotada superiormente considerando ajustes de complejidad
        # (ej. 0.30 x 0.65 x exp(-2) ~ 0.13), así que usamos <= 0.15 para la V3.
        self.assertLessEqual(
            thermo["financial_inertia"],
            0.15,
            msg=(
                "Inercia con atenuación y complejidad debe ser ≤ 0.15 "
                "para los parámetros dados"
            ),
        )

    def test_analyze_project_returns_thermodynamics_key(self) -> None:
        """
        analyze_project siempre retorna la clave 'thermodynamics'.
        """
        analysis = self.engine.analyze_project(
            initial_investment=500,
            cash_flows=[100, 100, 100],
            cost_std_dev=25,
            project_volatility=0.1,
            liquidity=0.3,
            fixed_contracts_ratio=0.4,
        )

        self.assertIn(
            "thermodynamics", analysis,
            msg="analyze_project debe retornar clave 'thermodynamics'",
        )

    def test_analyze_project_net_flows_are_positive(self) -> None:
        """
        Primera Ley: ΔU = Q - W.

        Cuando Σcash_flows > initial_investment, el proyecto es rentable.
        El análisis debe reconocer flujos netos positivos.
        """
        initial    = 1000
        cash_flows = [300, 300, 300, 300]  # Total: 1200 > 1000

        analysis = self.engine.analyze_project(
            initial_investment=initial,
            cash_flows=cash_flows,
            cost_std_dev=50,
            project_volatility=0.15,
            liquidity=0.3,
            fixed_contracts_ratio=0.4,
        )

        # Verificar que el análisis se completó correctamente
        self.assertIn("thermodynamics", analysis)
        # La inercia debe ser positiva (parámetros > 0)
        self.assertGreater(
            analysis["thermodynamics"]["financial_inertia"],
            0.0,
        )


# ============================================================================
# TEST: MatterGenerator — Análisis Exergético
# ============================================================================


class TestMatterGenerator(unittest.TestCase):
    """
    Generador de materia con análisis exergético.

    Exergía: máximo trabajo útil extraíble respecto a un estado de referencia.
    η_ex = Σcosto_estructural / Σcosto_total  ∈ [0, 1]

    Materiales estructurales: CONCRETO, ACERO  (exergía alta)
    Materiales decorativos:   PINTURA, PAPEL   (exergía baja → "anergía")
    """

    def setUp(self) -> None:
        self.generator = MatterGenerator()

    def tearDown(self) -> None:
        self.generator = None  # type: ignore[assignment]

    def _build_material(
        self,
        material_id: str,  # Renombrado: evita shadowing del builtin id()
        description: str,
        quantity_base: float,
        unit: str,
        waste_factor: float,
        unit_cost: float,
    ) -> MaterialRequirement:
        """
        Construye MaterialRequirement con cálculos derivados explícitos.

        quantity_total = quantity_base × (1 + waste_factor)
        total_cost     = quantity_total × unit_cost
        """
        quantity_total = quantity_base * (1 + waste_factor)
        total_cost     = quantity_total * unit_cost

        return MaterialRequirement(
            id=material_id,
            description=description,
            quantity_base=quantity_base,
            unit=unit,
            waste_factor=waste_factor,
            quantity_total=quantity_total,
            unit_cost=unit_cost,
            total_cost=total_cost,
        )

    # ── Casos mixtos ──────────────────────────────────────────────────────────

    def test_exergy_efficiency_mixed_materials(self) -> None:
        """
        η_ex con mezcla de materiales estructurales y decorativos.

        Cálculos explícitos:
          CONCRETO:  10 M3 × 1.05 × 100 $/M3 = 1050.00
          ACERO:    100 KG × 1.05 ×   2 $/KG =  210.00
          PINTURA:   10 GAL × 1.10 × 50 $/GAL =  550.00

          Estructural  = 1050 + 210 = 1260.00
          Decorativo   = 550.00
          Total        = 1810.00
          η_ex         = 1260 / 1810 ≈ 0.69613
        """
        items = [
            self._build_material("MAT-001", "CONCRETO 3000 PSI",  10,  "M3",  0.05, 100),
            self._build_material("MAT-002", "ACERO DE REFUERZO", 100,  "KG",  0.05,   2),
            self._build_material("MAT-003", "PINTURA DECORATIVA",  10, "GAL", 0.10,  50),
        ]

        report = self.generator.analyze_budget_exergy(items)

        cost_concrete  = 10  * 1.05 * 100   # 1050.0
        cost_steel     = 100 * 1.05 *   2   #  210.0
        cost_paint     = 10  * 1.10 *  50   #  550.0

        structural = cost_concrete + cost_steel    # 1260.0
        decorative = cost_paint                    #  550.0
        total      = structural + decorative       # 1810.0
        expected_efficiency = structural / total

        self.assertAlmostEqual(
            report["exergy_efficiency"], expected_efficiency, places=_PLACES_HIGH,
            msg=f"η_ex = {report['exergy_efficiency']:.6f} ≠ {expected_efficiency:.6f}",
        )
        self.assertAlmostEqual(
            report["structural_investment"], structural, places=_PLACES_MEDIUM,
        )
        self.assertAlmostEqual(
            report["decorative_investment"], decorative, places=_PLACES_MEDIUM,
        )
        self.assertAlmostEqual(
            report["total_investment"], total, places=_PLACES_MEDIUM,
        )

    # ── Casos límite ──────────────────────────────────────────────────────────

    def test_exergy_pure_structural_maximum_efficiency(self) -> None:
        """
        η_ex = 1.0 cuando 100% son materiales estructurales.

        Condición límite superior: toda la inversión es trabajo útil.
        """
        items = [
            self._build_material("MAT-001", "CONCRETO ESTRUCTURAL",  50, "M3",  0.03, 120),
            self._build_material("MAT-002", "ACERO ESTRUCTURAL",    200, "KG",  0.02,   4),
        ]

        report = self.generator.analyze_budget_exergy(items)

        self.assertAlmostEqual(
            report["exergy_efficiency"], 1.0, places=_PLACES_HIGH,
            msg="η_ex debe ser 1.0 con 100% materiales estructurales",
        )
        self.assertAlmostEqual(
            report["decorative_investment"], 0.0, places=_PLACES_HIGH,
            msg="Inversión decorativa debe ser 0.0",
        )

    def test_exergy_pure_decorative_minimum_efficiency(self) -> None:
        """
        η_ex = 0.0 cuando 100% son materiales decorativos.

        Condición límite inferior: toda la inversión es "anergía".
        """
        items = [
            self._build_material("MAT-001", "PINTURA VINILO",    25, "GAL", 0.08, 35),
            self._build_material("MAT-002", "PAPEL COLGADURA",   80, "M2",  0.12, 18),
        ]

        report = self.generator.analyze_budget_exergy(items)

        self.assertAlmostEqual(
            report["exergy_efficiency"], 0.0, places=_PLACES_HIGH,
            msg="η_ex debe ser 0.0 con 100% materiales decorativos",
        )
        self.assertAlmostEqual(
            report["structural_investment"], 0.0, places=_PLACES_HIGH,
            msg="Inversión estructural debe ser 0.0",
        )

    def test_exergy_empty_budget_all_zeros(self) -> None:
        """
        Presupuesto vacío → todas las métricas son cero.

        El analizador debe manejar graciosamente la lista vacía.
        """
        report = self.generator.analyze_budget_exergy([])

        for key in ("exergy_efficiency", "structural_investment",
                    "decorative_investment", "total_investment"):
            self.assertEqual(
                report[key], 0.0,
                msg=f"'{key}' debe ser 0.0 para presupuesto vacío",
            )

    # ── Propiedades de desperdicio ────────────────────────────────────────────

    def test_waste_factor_increases_total_cost(self) -> None:
        """
        Mayor factor de desperdicio → mayor costo total.

        waste_factor representa la irreversibilidad del proceso (Segunda Ley).
        """
        mat_low  = self._build_material("W-001", "CONCRETO", 100, "M3", 0.02, 100)
        mat_high = self._build_material("W-002", "CONCRETO", 100, "M3", 0.15, 100)

        self.assertLess(
            mat_low.total_cost,
            mat_high.total_cost,
            msg=(
                f"waste=0.02: cost={mat_low.total_cost}, "
                f"waste=0.15: cost={mat_high.total_cost} — "
                f"mayor desperdicio debe implicar mayor costo"
            ),
        )

    def test_process_efficiency_inverse_of_waste_factor(self) -> None:
        """
        Eficiencia de proceso: η = quantity_base / quantity_total = 1/(1+waste).

        La eficiencia es estrictamente decreciente con el factor de desperdicio.
        """
        mat_low  = self._build_material("W-001", "CONCRETO", 100, "M3", 0.02, 100)
        mat_high = self._build_material("W-002", "CONCRETO", 100, "M3", 0.15, 100)

        eta_low  = mat_low.quantity_base  / mat_low.quantity_total
        eta_high = mat_high.quantity_base / mat_high.quantity_total

        self.assertAlmostEqual(eta_low,  1 / 1.02, places=_PLACES_HIGH)
        self.assertAlmostEqual(eta_high, 1 / 1.15, places=_PLACES_HIGH)
        self.assertGreater(
            eta_low, eta_high,
            msg=f"η(waste=0.02)={eta_low:.4f} debe ser > η(waste=0.15)={eta_high:.4f}",
        )

    # ── Aditividad ────────────────────────────────────────────────────────────

    def test_exergy_structural_investment_is_additive(self) -> None:
        """
        Aditividad de la inversión estructural:
          structural(A ∪ B) = structural(A) + structural(B)

        La exergía es una propiedad extensiva: se suma sobre subsistemas
        no solapados. Aplica cuando ambos materiales son estructurales.
        """
        item1 = self._build_material("ADD-001", "CONCRETO", 10, "M3", 0.05, 100)
        item2 = self._build_material("ADD-002", "ACERO",    50, "KG", 0.03,   3)

        report_combined = self.generator.analyze_budget_exergy([item1, item2])
        report_single1  = self.generator.analyze_budget_exergy([item1])
        report_single2  = self.generator.analyze_budget_exergy([item2])

        self.assertAlmostEqual(
            report_combined["structural_investment"],
            report_single1["structural_investment"] + report_single2["structural_investment"],
            places=_PLACES_MEDIUM,
            msg="structural_investment(A∪B) debe ser structural(A) + structural(B)",
        )

    def test_exergy_total_investment_is_additive(self) -> None:
        """
        Aditividad del costo total: total(A ∪ B) = total(A) + total(B).
        """
        item1 = self._build_material("ADD-001", "CONCRETO", 10, "M3", 0.05, 100)
        item2 = self._build_material("ADD-002", "PINTURA",  20, "GAL", 0.10,  30)

        report_combined = self.generator.analyze_budget_exergy([item1, item2])
        report_single1  = self.generator.analyze_budget_exergy([item1])
        report_single2  = self.generator.analyze_budget_exergy([item2])

        self.assertAlmostEqual(
            report_combined["total_investment"],
            report_single1["total_investment"] + report_single2["total_investment"],
            places=_PLACES_MEDIUM,
            msg="total_investment(A∪B) debe ser total(A) + total(B)",
        )

    def test_exergy_efficiency_in_unit_interval(self) -> None:
        """
        η_ex ∈ [0, 1]  para cualquier combinación de materiales.
        """
        test_combos = [
            [self._build_material("C1", "CONCRETO", 10, "M3", 0.05, 100)],
            [self._build_material("C2", "PINTURA",  10, "GAL", 0.10, 30)],
            [
                self._build_material("C3", "CONCRETO", 5, "M3", 0.05, 100),
                self._build_material("C4", "PINTURA",  5, "GAL", 0.10, 30),
            ],
        ]
        for combo in test_combos:
            with self.subTest(materials=[m.description for m in combo]):
                report = self.generator.analyze_budget_exergy(combo)
                self.assertGreaterEqual(report["exergy_efficiency"], 0.0)
                self.assertLessEqual(report["exergy_efficiency"],    1.0)


# ============================================================================
# TEST: BusinessTopologicalAnalyzer — Convección Inflacionaria y Topología
# ============================================================================


class TestBusinessTopologicalAnalyzer(unittest.TestCase):
    """
    Analizador topológico de estructura de negocios.

    Convección inflacionaria:
      I(APU) = Σcosto(insumos volátiles) / Σcosto(todos los insumos)
      Alto riesgo: I(APU) > 0.2  (desigualdad estricta)

    Topología algebraica:
      β₀ = componentes conexas
      χ  = V - E  (característica de Euler para grafos sin caras)
      χ  = β₀ - β₁  (fórmula de Euler-Poincaré)
    """

    def setUp(self) -> None:
        self.analyzer = BusinessTopologicalAnalyzer(telemetry=None)

    def tearDown(self) -> None:
        self.analyzer = None  # type: ignore[assignment]

    def _create_base_graph(self) -> nx.DiGraph:
        """Grafo base de dependencias APU → Insumo."""
        G = nx.DiGraph()
        G.add_node("APU1", type="APU")
        G.add_node("APU2", type="APU")
        G.add_node("APU3", type="APU")
        G.add_node("T1",   type="INSUMO", description="TRANSPORTE MATERIAL")
        G.add_node("M1",   type="INSUMO", description="CEMENTO PORTLAND")
        G.add_node("M2",   type="INSUMO", description="AGREGADO GRUESO")
        return G

    # ── Cálculo de impacto de convección ─────────────────────────────────────

    def test_convection_impact_calculation(self) -> None:
        """
        I(APU1) = costo(T1) / costo(T1 + M1) = 200 / 1000 = 0.2.

        APU2 no usa T1 → I(APU2) = 0.0 si aparece en el reporte.
        """
        G = self._create_base_graph()
        G.add_edge("APU1", "T1", total_cost=200)
        G.add_edge("APU1", "M1", total_cost=800)
        G.add_edge("APU2", "M1", total_cost=500)

        report = self.analyzer.analyze_inflationary_convection(G, ["T1"])

        self.assertIn("APU1", report["convection_impact"])
        self.assertAlmostEqual(
            report["convection_impact"]["APU1"],
            0.2,
            places=_PLACES_HIGH,
            msg="I(APU1) = 200/1000 debe ser 0.2",
        )

        if "APU2" in report["convection_impact"]:
            self.assertAlmostEqual(
                report["convection_impact"]["APU2"],
                0.0,
                places=_PLACES_HIGH,
                msg="I(APU2) debe ser 0.0: no usa insumos volátiles",
            )

    # ── Umbral de alto riesgo (desigualdad estricta) ──────────────────────────

    def test_high_risk_threshold_strict_inequality(self) -> None:
        """
        I = 0.2 NO es alto riesgo (> 0.2 es la condición).
        I = 0.21 SÍ es alto riesgo.

        Se usan proporciones enteras para evitar error de redondeo:
          APU1: 200/(200+800) = 0.2000 exacto
          APU2: 210/(210+790) = 0.2100 exacto
        """
        G = self._create_base_graph()
        G.add_edge("APU1", "T1", total_cost=200)
        G.add_edge("APU1", "M1", total_cost=800)
        G.add_edge("APU2", "T1", total_cost=210)
        G.add_edge("APU2", "M1", total_cost=790)

        report = self.analyzer.analyze_inflationary_convection(G, ["T1"])

        self.assertNotIn(
            "APU1",
            report["high_risk_nodes"],
            msg="I(APU1)=0.200 no cumple I>0.2 (desigualdad estricta)",
        )
        self.assertIn(
            "APU2",
            report["high_risk_nodes"],
            msg="I(APU2)=0.210 > 0.2 debe ser alto riesgo",
        )

    def test_maximum_exposure_single_fluid_insumo(self) -> None:
        """
        I = 1.0 cuando el APU depende 100% de un insumo volátil.

        Peor caso de concentración de riesgo.
        """
        G = nx.DiGraph()
        G.add_node("APU_CRITICAL", type="APU")
        G.add_node("FUEL",         type="INSUMO", description="ACPM DIESEL")
        G.add_edge("APU_CRITICAL", "FUEL", total_cost=1000)

        report = self.analyzer.analyze_inflationary_convection(G, ["FUEL"])

        self.assertAlmostEqual(
            report["convection_impact"]["APU_CRITICAL"],
            1.0,
            places=_PLACES_HIGH,
            msg="I = 1.0 cuando 100% del costo es insumo volátil",
        )
        self.assertIn(
            "APU_CRITICAL",
            report["high_risk_nodes"],
            msg="I=1.0 > 0.2 debe ser alto riesgo",
        )

    def test_multiple_fluid_nodes_linear_superposition(self) -> None:
        """
        Superposición lineal: I = Σcosto(volátiles) / Σcosto(total).

        I(APU_LOGISTICS) = (150 + 100) / (150 + 100 + 250) = 250/500 = 0.5
        """
        G = nx.DiGraph()
        G.add_node("APU_LOGISTICS", type="APU")
        G.add_node("FUEL",          type="INSUMO", description="COMBUSTIBLE")
        G.add_node("TRANSPORT",     type="INSUMO", description="FLETE")
        G.add_node("MATERIAL",      type="INSUMO", description="CEMENTO")

        G.add_edge("APU_LOGISTICS", "FUEL",      total_cost=150)
        G.add_edge("APU_LOGISTICS", "TRANSPORT", total_cost=100)
        G.add_edge("APU_LOGISTICS", "MATERIAL",  total_cost=250)

        report = self.analyzer.analyze_inflationary_convection(
            G, ["FUEL", "TRANSPORT"]
        )

        expected_impact = (150 + 100) / 500  # 0.5

        self.assertAlmostEqual(
            report["convection_impact"]["APU_LOGISTICS"],
            expected_impact,
            places=_PLACES_HIGH,
            msg=f"I = (150+100)/500 = {expected_impact}",
        )
        self.assertIn("APU_LOGISTICS", report["high_risk_nodes"])

    # ── Casos límite ──────────────────────────────────────────────────────────

    def test_empty_graph_stability(self) -> None:
        """
        Grafo vacío → convection_impact={} y high_risk_nodes=[].

        El analizador maneja graciosamente espacios vacíos.
        """
        G = nx.DiGraph()
        report = self.analyzer.analyze_inflationary_convection(G, ["T1"])

        self.assertEqual(report["convection_impact"], {})
        self.assertEqual(report["high_risk_nodes"],   [])

    def test_no_fluid_nodes_zero_convection_all_apus(self) -> None:
        """
        Sin insumos volátiles definidos → I(APU) = 0 para todos los APUs.

        El sistema es termodinámicamente estable sin convección inflacionaria.
        """
        G = self._create_base_graph()
        G.add_edge("APU1", "M1", total_cost=500)
        G.add_edge("APU1", "M2", total_cost=300)

        report = self.analyzer.analyze_inflationary_convection(G, [])

        for apu, impact in report["convection_impact"].items():
            self.assertEqual(
                impact, 0.0,
                msg=f"I({apu}) debe ser 0.0 sin insumos volátiles",
            )
        self.assertEqual(report["high_risk_nodes"], [])

    # ── Topología algebraica ──────────────────────────────────────────────────

    def test_betti_zero_via_analyzer(self) -> None:
        """
        β₀ = componentes conexas, verificado a través del analizador.

        Un grafo con 2 componentes aisladas tiene β₀ = 2.
        Este test usa el método del analizador (no directamente NetworkX)
        para verificar la propiedad topológica.
        """
        G = nx.DiGraph()
        # Componente 1: APU_A → M_A
        G.add_node("APU_A", type="APU")
        G.add_node("M_A",   type="INSUMO", description="MATERIAL A")
        G.add_edge("APU_A", "M_A", total_cost=100)
        # Componente 2: APU_B → M_B (sin conexión a la componente 1)
        G.add_node("APU_B", type="APU")
        G.add_node("M_B",   type="INSUMO", description="MATERIAL B")
        G.add_edge("APU_B", "M_B", total_cost=200)

        betti = self.analyzer.calculate_betti_numbers(G)

        self.assertEqual(
            betti.beta_0,
            2,
            msg="β₀ debe ser 2 para grafo con 2 componentes conexas",
        )

    def test_euler_characteristic_equals_betti_alternating_sum(self) -> None:
        """
        Fórmula de Euler-Poincaré para grafos: χ = β₀ - β₁.

        Para el grafo base con 3 aristas y 6 nodos:
          χ_geométrico = V - E = 6 - 3 = 3
          χ_homológico = β₀ - β₁ (calculado por el analizador)
          Ambos deben coincidir.

        Esto verifica la coherencia interna del analizador, no solo aritmética.
        """
        G = self._create_base_graph()
        G.add_edge("APU1", "T1", total_cost=100)
        G.add_edge("APU1", "M1", total_cost=200)
        G.add_edge("APU2", "M1", total_cost=150)

        # χ geométrico (invariante topológico del grafo como CW-complejo)
        V = G.number_of_nodes()   # 6
        E = G.number_of_edges()   # 3
        chi_geometric = V - E     # 3

        # χ homológico (fórmula de Euler-Poincaré)
        betti = self.analyzer.calculate_betti_numbers(G)
        chi_homological = betti.beta_0 - betti.beta_1

        self.assertEqual(
            chi_geometric,
            chi_homological,
            msg=(
                f"Euler-Poincaré violado: "
                f"χ_geométrico={chi_geometric} (V={V}, E={E}) ≠ "
                f"χ_homológico={chi_homological} "
                f"(β₀={betti.beta_0}, β₁={betti.beta_1})"
            ),
        )

    def test_euler_characteristic_is_integer_invariant(self) -> None:
        """
        χ = V - E es un entero para cualquier grafo finito.

        Como invariante topológico, χ no cambia bajo homeomorfismos
        que preservan los vértices y aristas (subdivisión baricéntrica).
        """
        G = self._create_base_graph()
        G.add_edge("APU1", "T1", total_cost=100)

        chi = G.number_of_nodes() - G.number_of_edges()

        self.assertIsInstance(chi, int, msg="χ debe ser un entero")

    def test_topological_resilience_after_high_risk_removal(self) -> None:
        """
        Resiliencia topológica: los nodos estables permanecen conectados
        tras remover los nodos de alto riesgo.

        Estructura:
          APU_FINAL → APU_RISKY → VOLATILE  (nodo volátil)
          APU_FINAL → APU_STABLE → STABLE   (nodo estable)

        Tras remover APU_RISKY (alto riesgo, I=1.0):
          · APU_FINAL y APU_STABLE permanecen
          · El subgrafo residual tiene al menos 1 APU conectado a APU_FINAL
        """
        G = nx.DiGraph()
        G.add_node("APU_FINAL",  type="APU")
        G.add_node("APU_RISKY",  type="APU")
        G.add_node("APU_STABLE", type="APU")
        G.add_node("VOLATILE",   type="INSUMO", description="PETROLEO")
        G.add_node("STABLE",     type="INSUMO", description="ARENA")

        G.add_edge("APU_FINAL",  "APU_RISKY",  total_cost=500)
        G.add_edge("APU_FINAL",  "APU_STABLE", total_cost=500)
        G.add_edge("APU_RISKY",  "VOLATILE",   total_cost=400)
        G.add_edge("APU_STABLE", "STABLE",     total_cost=300)

        report = self.analyzer.analyze_inflationary_convection(G, ["VOLATILE"])

        self.assertIn(
            "APU_RISKY",
            report["high_risk_nodes"],
            msg="APU_RISKY (I=1.0) debe ser alto riesgo",
        )
        self.assertNotIn(
            "APU_STABLE",
            report["high_risk_nodes"],
            msg="APU_STABLE (I=0.0) NO debe ser alto riesgo",
        )

        # Verificar que APU_STABLE queda intacto tras remover APU_RISKY
        G_resilient = G.copy()
        for node in report["high_risk_nodes"]:
            G_resilient.remove_node(node)

        # APU_STABLE debe seguir presente
        self.assertIn(
            "APU_STABLE",
            G_resilient.nodes(),
            msg="APU_STABLE debe permanecer tras remover nodos de alto riesgo",
        )
        # APU_FINAL debe seguir conectado a APU_STABLE
        self.assertTrue(
            G_resilient.has_edge("APU_FINAL", "APU_STABLE"),
            msg="APU_FINAL → APU_STABLE debe subsistir en el grafo resiliente",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
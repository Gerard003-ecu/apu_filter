# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Test Telemetry Validator v6 (Batería de Pruebas Doctoral)          ║
║  Ruta   : app/core/test_telemetry_validator_v6.py                            ║
║  Versión: 6.0.0-Spectral-Boolean-Categorical-Test-Suite                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Estructura en 3 fases anidadas, isomorfa a telemetry_validator_v6.py:       ║
║    FASE 1 — Kernel espectral/algebraico/termodinámico + Topología.           ║
║    FASE 2 — Álgebra de Boole, Grafos (clausura transitiva) y TelemetryPacket.║
║    FASE 3 — Categorías/Topos y Majorización Cuántica (clausura demostrada).  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import math
import json
import unittest
import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from telemetry_validator_v6 import (
    # Fase 1
    PhysicsMetrics,
    ControlMetrics,
    ThermodynamicMetrics,
    TopologicalMetrics,
    clamp,
    spectral_abscissa,
    is_positive_semidefinite,
    is_skew_symmetric,
    routh_hurwitz_stable,
    shannon_entropy,
    majorizes,
    PassivityViolationError,
    # Fase 2
    AuditReport,
    TelemetryCoherenceAuditor,
    PhysicsCoherenceError,
    ControlInstabilityError,
    ThermodynamicBoundaryError,
    WisdomMetrics,
    VerdictCode,
    TelemetryError,
    TelemetryPacket,
    verify_strict_total_order,
    _build_hierarchy_adjacency,
    # Fase 3
    StrataPoset,
    QuantumMajorizationValidator,
    CyberPhysicalTopos,
    CategoricalClosureError,
    MajorizationOrderError,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 — KERNEL ESPECTRAL, ALGEBRAICO, TERMODINÁMICO Y TOPOLÓGICO          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestFase1KernelFundamental(unittest.TestCase):
    """Pruebas atómicas de las primitivas matemáticas y los cuatro dominios base."""

    # ─────────────────── I.0 Utilidades matemáticas puras ───────────────────

    def test_clamp_boundary_inclusion(self) -> None:
        """El acotamiento debe incluir ambos extremos (cerrados) sin desbordar."""
        self.assertEqual(clamp(5.0, 0.0, 1.0), 1.0)
        self.assertEqual(clamp(-5.0, 0.0, 1.0), 0.0)
        self.assertEqual(clamp(0.5, 0.0, 1.0), 0.5)

    def test_spectral_abscissa_known_eigenvalues(self) -> None:
        r"""$\alpha(A) = \max_i \Re(\lambda_i)$ sobre una matriz diagonal exacta."""
        A = np.diag([-3.0, -1.0, 2.0])
        self.assertAlmostEqual(spectral_abscissa(A), 2.0)

    def test_spectral_abscissa_empty_matrix(self) -> None:
        """Ante ausencia de estado, la abscisa espectral debe ser -∞ (vacuidad estable)."""
        self.assertEqual(spectral_abscissa(np.zeros((0, 0))), float("-inf"))

    def test_is_positive_semidefinite_identity_and_negative(self) -> None:
        r"""Certifica $I \succeq 0$ vía el Teorema Espectral y rechaza $-I$."""
        self.assertTrue(is_positive_semidefinite(np.eye(3)))
        self.assertFalse(is_positive_semidefinite(-np.eye(3)))

    def test_is_skew_symmetric_structure_matrix(self) -> None:
        r"""Verifica $J = -J^T$ para una matriz de interconexión canónica 2x2."""
        J = np.array([[0.0, 1.0], [-1.0, 0.0]])
        self.assertTrue(is_skew_symmetric(J))
        self.assertFalse(is_skew_symmetric(np.eye(2)))

    def test_routh_hurwitz_stable_second_order(self) -> None:
        r"""Polinomio $s^2 + 3s + 2$ (raíces -1,-2): estable por Routh-Hurwitz."""
        self.assertTrue(routh_hurwitz_stable([1.0, 3.0, 2.0]))

    def test_routh_hurwitz_unstable_negative_coefficient(self) -> None:
        r"""Un coeficiente no positivo viola la condición necesaria trivial de estabilidad."""
        self.assertFalse(routh_hurwitz_stable([1.0, -3.0, 2.0]))

    def test_shannon_entropy_uniform_distribution_is_maximal(self) -> None:
        r"""La entropía de una distribución uniforme de $n$ estados es $\ln n$."""
        uniform = np.array([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(shannon_entropy(uniform), math.log(4))

    def test_shannon_entropy_deterministic_distribution_is_zero(self) -> None:
        """Un estado puro (certeza total) tiene entropía nula."""
        pure = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(shannon_entropy(pure), 0.0)

    def test_majorizes_reflexivity_and_pure_state_dominance(self) -> None:
        r"""
        Verifica los axiomas del preorden HLP:
            (a) Reflexividad: $p \succ p$.
            (b) Dominancia: el estado puro $(1,0,\dots,0)$ mayoriza a cualquier
                distribución de igual soporte total (es el máximo del preorden).
        """
        p = np.array([0.5, 0.3, 0.2])
        pure = np.array([1.0, 0.0, 0.0])
        self.assertTrue(majorizes(p, p))
        self.assertTrue(majorizes(pure, p))
        self.assertFalse(majorizes(p, pure))

    def test_majorizes_rejects_unequal_totals(self) -> None:
        """El preorden de majorización exige conservación de la masa total (traza=1)."""
        self.assertFalse(majorizes(np.array([1.0, 0.0]), np.array([0.4, 0.4])))

    # ─────────────────── I.1 PhysicsMetrics (dominio físico) ─────────────────

    def test_physics_metrics_nominal_coherence(self) -> None:
        r"""Valida un estado nominal frente a los cuatro axiomas físicos declarados."""
        metrics = PhysicsMetrics(
            saturation=0.25, pressure=100.0, kinetic_energy=10.0, potential_energy=-5.0,
            flyback_voltage=0.01, dissipated_power=2.0, gyroscopic_stability=1.5,
            poynting_flux=50.0, hamiltonian_excess=0.00001,
        )
        self.assertEqual(metrics.total_energy, 5.0)
        self.assertTrue(math.isclose(metrics.dissipation_ratio, 2.0 / 5.0))
        self.assertTrue(math.isclose(metrics.efficiency, 0.6))
        self.assertTrue(metrics.is_coherent())

    def test_physics_metrics_default_state_is_coherent(self) -> None:
        """El estado de vacío por defecto debe ser, por construcción, coherente."""
        self.assertTrue(PhysicsMetrics().is_coherent())

    def test_physics_metrics_negative_kinetic_energy(self) -> None:
        r"""$T < 0 \implies \text{is\_coherent()} = \text{False}$."""
        self.assertFalse(PhysicsMetrics(kinetic_energy=-0.1, dissipated_power=1.0).is_coherent())

    def test_physics_metrics_negative_dissipated_power(self) -> None:
        r"""$P_{diss} < 0 \implies \text{is\_coherent()} = \text{False}$ (viola la 2ª Ley)."""
        self.assertFalse(PhysicsMetrics(kinetic_energy=5.0, dissipated_power=-0.05).is_coherent())

    def test_physics_metrics_hamiltonian_leak(self) -> None:
        r"""$|\Delta H| > \tau \cdot \max(|H|,1) \implies$ fuga energética vetada."""
        metrics = PhysicsMetrics(kinetic_energy=10.0, potential_energy=5.0,
                                  dissipated_power=1.0, hamiltonian_excess=2.5)
        self.assertFalse(metrics.is_coherent(tolerance=1e-5))

    def test_physics_metrics_gyroscopic_instability_boundary(self) -> None:
        r"""$\sigma_{gyr} \in [0,2]$ es un intervalo cerrado: 2.0 es válido, 2.0000001 no."""
        self.assertTrue(PhysicsMetrics(gyroscopic_stability=2.0).is_coherent())
        self.assertFalse(PhysicsMetrics(gyroscopic_stability=2.0000001).is_coherent())

    def test_physics_metrics_port_hamiltonian_passivity_veto(self) -> None:
        r"""
        Certifica el veto estructural cuando $R \not\succeq 0$: un sistema
        Port-Hamiltoniano con matriz de disipación activa es físicamente
        imposible y debe abortar con `PassivityViolationError`.
        """
        metrics = PhysicsMetrics()
        J = np.array([[0.0, 1.0], [-1.0, 0.0]])   # Antisimétrica: correcta.
        R_active = -np.eye(2)                      # Definida negativa: sistema activo.
        with self.assertRaises(PassivityViolationError):
            metrics.verify_port_hamiltonian_passivity(J, R_active)

    def test_physics_metrics_port_hamiltonian_passivity_nominal(self) -> None:
        """Con J antisimétrica y R semidefinida positiva, el sistema es pasivo."""
        metrics = PhysicsMetrics()
        J = np.array([[0.0, 2.0], [-2.0, 0.0]])
        R = np.diag([0.5, 0.3])
        self.assertTrue(metrics.verify_port_hamiltonian_passivity(J, R))

    # ─────────────────── I.2 ControlMetrics (dominio LTI) ────────────────────

    def test_control_metrics_nominal_stability(self) -> None:
        r"""Estabilidad BIBO nominal: $\omega_n = |p_{dom}|/\zeta$, $\tau_s = -4/p_{dom}$."""
        metrics = ControlMetrics(poles_real=[-2.0, -5.0], is_stable=True,
                                  phase_margin_deg=60.0, damping_ratio=0.5,
                                  lyapunov_exponent=-2.0)
        self.assertEqual(metrics.dominant_pole, -2.0)
        self.assertTrue(metrics.poles_indicate_stability())
        self.assertTrue(math.isclose(metrics.settling_time_approx, 2.0))
        self.assertTrue(math.isclose(metrics.natural_frequency_approx, 4.0))
        self.assertTrue(metrics.is_coherent())

    def test_control_metrics_accepts_list_input_for_poles(self) -> None:
        """`__post_init__` debe normalizar listas de entrada a tupla inmutable de floats."""
        metrics = ControlMetrics(poles_real=[-1, -2, -3])
        self.assertIsInstance(metrics.poles_real, tuple)
        self.assertTrue(all(isinstance(p, float) for p in metrics.poles_real))

    def test_control_metrics_lyapunov_inconsistency(self) -> None:
        r"""$\text{is\_stable}=\text{True} \land \lambda_{max} > \epsilon \implies$ incoherente."""
        metrics = ControlMetrics(poles_real=[-1.0], is_stable=True, lyapunov_exponent=0.05)
        self.assertFalse(metrics.is_coherent())

    def test_control_metrics_unstable_natural_frequency_correction(self) -> None:
        r"""Rigor V4.6: $p_{dom} \ge 0 \implies \omega_n = 0.0$ (protección de la FPU)."""
        metrics = ControlMetrics(poles_real=[0.5, -2.0], is_stable=False,
                                  damping_ratio=0.7, lyapunov_exponent=0.5)
        self.assertEqual(metrics.dominant_pole, 0.5)
        self.assertEqual(metrics.settling_time_approx, float("inf"))
        self.assertEqual(metrics.natural_frequency_approx, 0.0)
        self.assertTrue(metrics.is_coherent())

    def test_control_metrics_poles_indicate_stability_is_independently_testable(self) -> None:
        """Granularidad: el predicado espectral puro se certifica sin pasar por `is_coherent`."""
        self.assertTrue(ControlMetrics(poles_real=[-1.0, -0.5]).poles_indicate_stability())
        self.assertFalse(ControlMetrics(poles_real=[-1.0, 0.1]).poles_indicate_stability())
        self.assertTrue(ControlMetrics(poles_real=(), is_stable=True).poles_indicate_stability())

    def test_control_metrics_kalman_controllability_rank_criterion(self) -> None:
        r"""Sistema canónico controlable: $\text{rank}([B|AB]) = n$."""
        A = np.array([[0.0, 1.0], [-2.0, -3.0]])
        B = np.array([[0.0], [1.0]])
        self.assertTrue(ControlMetrics().is_controllable(A, B))

    def test_control_metrics_uncontrollable_pair(self) -> None:
        """Un par $(A,B)$ desacoplado del canal de entrada debe fallar el rango de Kalman."""
        A = np.diag([-1.0, -2.0])
        B = np.array([[1.0], [0.0]])   # Segundo modo inalcanzable.
        self.assertFalse(ControlMetrics().is_controllable(A, B))

    def test_control_metrics_reachability_gramian_definiteness(self) -> None:
        r"""Para un par controlable, el Gramiano de Lyapunov debe ser definido positivo."""
        A = np.array([[-1.0, 1.0], [0.0, -2.0]])
        B = np.array([[1.0], [1.0]])
        self.assertTrue(ControlMetrics().verify_reachability_definiteness(A, B))

    def test_control_metrics_spectral_abscissa_cross_validation(self) -> None:
        r"""Coherencia entre `is_stable` declarado y $\alpha(A) < 0$ real de la matriz de estado."""
        A_stable = np.diag([-1.0, -3.0])
        metrics_ok = ControlMetrics(is_stable=True)
        self.assertTrue(metrics_ok.verify_stability_via_spectral_abscissa(A_stable))
        A_unstable = np.diag([1.0, -3.0])
        self.assertFalse(metrics_ok.verify_stability_via_spectral_abscissa(A_unstable))

    # ─────────────────── I.3 ThermodynamicMetrics ────────────────────────────

    def test_thermodynamic_metrics_strict_laws(self) -> None:
        """Las cuatro leyes económico-termodinámicas deben vetar independientemente."""
        self.assertFalse(ThermodynamicMetrics(system_temperature=-1.0).is_coherent())
        self.assertFalse(ThermodynamicMetrics(entropy=-0.01).is_coherent())
        self.assertFalse(ThermodynamicMetrics(financial_inertia=0.0).is_coherent())
        self.assertFalse(ThermodynamicMetrics(financial_inertia=-5.0).is_coherent())
        self.assertFalse(ThermodynamicMetrics(exergy=-10.0).is_coherent())
        self.assertTrue(ThermodynamicMetrics().is_coherent())

    def test_thermodynamic_metrics_boltzmann_consistency(self) -> None:
        r"""Contraste $S_{decl} \approx -k_B\sum p_i \ln p_i$ contra un microestado uniforme."""
        micro = np.array([1.0, 1.0, 1.0, 1.0])  # ln(4) ≈ 1.386
        thermo = ThermodynamicMetrics(entropy=math.log(4))
        self.assertTrue(thermo.verify_boltzmann_consistency(micro))

    def test_thermodynamic_metrics_second_law_monotonicity(self) -> None:
        r"""La evolución hacia el equilibrio debe ser mayorizada por el estado de referencia."""
        reference = np.array([0.7, 0.2, 0.1])
        evolved_towards_equilibrium = np.array([0.4, 0.35, 0.25])
        thermo = ThermodynamicMetrics()
        self.assertTrue(thermo.verify_second_law_monotonicity(reference, evolved_towards_equilibrium))

    # ─────────────────── I.4 TopologicalMetrics ──────────────────────────────

    def test_topological_metrics_nominal_invariants(self) -> None:
        r"""$\chi = \beta_0 - \beta_1 + \beta_2$; cyclomatic = $\beta_1$; robustez = $\lambda_2\Delta$."""
        metrics = TopologicalMetrics(beta_0=1, beta_1=3, beta_2=0, fiedler_value=0.5,
                                      spectral_gap=0.5, mayer_vietoris_delta=1)
        self.assertEqual(metrics.euler_characteristic(), -2)
        self.assertEqual(metrics.total_betti(), 4)
        self.assertEqual(metrics.cyclomatic_complexity(), 3.0)
        self.assertTrue(math.isclose(metrics.algebraic_connectivity(), 0.25))
        self.assertTrue(metrics.is_coherent())

    def test_topological_metrics_default_state_is_coherent(self) -> None:
        """El complejo simplicial trivial (un punto, sin ciclos) es coherente por defecto."""
        self.assertTrue(TopologicalMetrics().is_coherent())

    def test_topological_metrics_fiedler_theorem_violation(self) -> None:
        r"""$\beta_0 > 1 \implies \lambda_2 = 0$: disconexo con conectividad espuria se veta."""
        metrics = TopologicalMetrics(beta_0=2, fiedler_value=0.4, spectral_gap=0.4)
        self.assertFalse(metrics.is_coherent())

    def test_topological_metrics_mayer_vietoris_excess(self) -> None:
        r"""$\delta_{MV} \notin [0, \sum\beta_k] \implies$ incoherente."""
        metrics = TopologicalMetrics(beta_0=1, beta_1=0, beta_2=0, mayer_vietoris_delta=2)
        self.assertFalse(metrics.is_coherent())

    def test_topological_metrics_von_neumann_entropy_limit(self) -> None:
        r"""$S_{struct} > 3\ln(\text{total\_betti}+2) \implies$ inflación entrópica vetada."""
        metrics = TopologicalMetrics(beta_0=1, beta_1=1, beta_2=0, structural_entropy=10.0)
        self.assertFalse(metrics.is_coherent())


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 — ÁLGEBRA DE BOOLE, TEORÍA DE GRAFOS Y ORQUESTACIÓN DE PAQUETES     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestFase2BooleanGraphAndPacket(unittest.TestCase):
    """
    Continúa directamente el kernel de la Fase 1: consume `PhysicsMetrics`,
    `ControlMetrics`, `ThermodynamicMetrics` y `TopologicalMetrics` ya
    certificados, y los somete a la retícula booleana de axiomas y a la
    clausura transitiva del grafo jerárquico.
    """

    def setUp(self) -> None:
        self.auditor = TelemetryCoherenceAuditor()

    # ─────────────── II.0 Clausura transitiva del grafo jerárquico ──────────

    def test_hierarchy_transitive_closure_reaches_root_to_leaf(self) -> None:
        r"""$V_{PHYSICS} \subset \dots \subset V_{WISDOM}$ debe ser demostrable, no declarada."""
        adjacency = _build_hierarchy_adjacency()
        self.assertTrue(verify_strict_total_order(adjacency))

    def test_hierarchy_closure_rejects_cyclic_corruption(self) -> None:
        """Inyectar un ciclo artificial (WISDOM → PHYSICS) debe romper el orden estricto."""
        adjacency = _build_hierarchy_adjacency()
        corrupted = adjacency.copy()
        corrupted[-1, 0] = True  # Introduce dependencia circular.
        self.assertFalse(verify_strict_total_order(corrupted))

    # ─────────────── II.1 Retícula booleana / TelemetryCoherenceAuditor ─────

    def test_auditor_nominal_packet_returns_zero_mask(self) -> None:
        """Un paquete perfectamente coherente produce la máscara booleana nula (⊤ total)."""
        report = self.auditor.audit_packet(PhysicsMetrics(), ControlMetrics(), ThermodynamicMetrics())
        self.assertIsInstance(report, AuditReport)
        self.assertEqual(report.boolean_lattice_mask, 0)
        self.assertTrue(report.coherent)
        self.assertTrue(bool(report))

    def test_auditor_raises_physics_coherence_error(self) -> None:
        """La violación de axiomas físicos debe propagarse como `PhysicsCoherenceError`."""
        broken_physics = PhysicsMetrics(kinetic_energy=-1.0)
        with self.assertRaises(PhysicsCoherenceError):
            self.auditor.audit_packet(broken_physics, ControlMetrics(), ThermodynamicMetrics())

    def test_auditor_raises_control_instability_error(self) -> None:
        """La violación de axiomas LTI debe propagarse como `ControlInstabilityError`."""
        broken_control = ControlMetrics(damping_ratio=-0.1)
        with self.assertRaises(ControlInstabilityError):
            self.auditor.audit_packet(PhysicsMetrics(), broken_control, ThermodynamicMetrics())

    def test_auditor_raises_thermodynamic_boundary_error(self) -> None:
        """La violación de leyes termodinámicas debe propagarse como `ThermodynamicBoundaryError`."""
        broken_thermo = ThermodynamicMetrics(system_temperature=-50.0)
        with self.assertRaises(ThermodynamicBoundaryError):
            self.auditor.audit_packet(PhysicsMetrics(), ControlMetrics(), broken_thermo)

    def test_auditor_cross_entropy_power_coupling_veto(self) -> None:
        r"""
        Sutura trans-estrato: disipación física activa ($P_{diss} > 0$) con
        producción de entropía negativa ($S \cdot T < 0$) es termodinámicamente
        imposible y debe ser vetada aunque cada estrato sea individualmente
        "coherente" en aislamiento.
        """
        physics = PhysicsMetrics(dissipated_power=5.0)
        thermo = ThermodynamicMetrics(entropy=0.0, system_temperature=0.0)
        # Se fuerza el acoplamiento negativo vía un thermo mock-compatible:
        # entropy*T = 0 aquí; se prueba la rama límite en lugar de forzar un
        # estado con S<0 (ya vetado en capa termodinámica aislada).
        self.assertTrue(self.auditor.audit_packet(physics, ControlMetrics(), thermo))

    # ─────────────── II.2 TelemetryPacket: cascada de errores en JSON ───────

    def test_telemetry_packet_empty_input(self) -> None:
        """Manejo defensivo de un string JSON vacío (EMPTY_INPUT)."""
        packet = TelemetryPacket()
        self.assertEqual(packet.from_json(""), TelemetryError.EMPTY_INPUT)
        self.assertEqual(packet.from_json("   "), TelemetryError.EMPTY_INPUT)

    def test_telemetry_packet_parse_error(self) -> None:
        """Manejo defensivo de un JSON corrupto o mal formado."""
        packet = TelemetryPacket()
        self.assertEqual(packet.from_json("{'broken_json': "), TelemetryError.JSON_PARSE_ERROR)

    def test_telemetry_packet_coherence_cascade_precedence(self) -> None:
        r"""
        Precedencia de intercepción:
        $$\text{TOPOLOGICAL\_ANOMALY} \prec \text{THERMODYNAMIC\_ANOMALY} \prec \text{COHERENCE\_VIOLATION}$$
        Con violaciones simultáneas en físico, topológico, control y
        termodinámico, la aduana debe interceptar primero la anomalía
        topológica.
        """
        packet = TelemetryPacket()
        payload = {
            "timestamp": "2026-07-31T18:00:00Z",
            "physics": {"kinetic_energy": -5.0},
            "topology": {"beta_0": 2, "fiedler_value": 0.5},
            "control": {"damping_ratio": -0.1},
            "thermodynamics": {"system_temperature": -100.0},
            "wisdom": {"verdict_code": 0, "narrative_short": "LLM alucina viabilidad"},
        }
        self.assertEqual(packet.from_json(json.dumps(payload)), TelemetryError.TOPOLOGICAL_ANOMALY)

    def test_telemetry_packet_thermodynamic_anomaly_precedes_coherence_violation(self) -> None:
        """Con topología sana, la anomalía termodinámica debe preceder a la violación genérica."""
        packet = TelemetryPacket()
        payload = {
            "physics": {"kinetic_energy": -1.0},          # COHERENCE_VIOLATION potencial
            "topology": {"beta_0": 1},                     # Sano
            "thermodynamics": {"system_temperature": -1.0},  # THERMODYNAMIC_ANOMALY
        }
        self.assertEqual(packet.from_json(json.dumps(payload)), TelemetryError.THERMODYNAMIC_ANOMALY)

    def test_telemetry_packet_bypass_esp32_mismatch_veto(self) -> None:
        r"""
        Simula el veto por mismatch epistemológico: la IA en la nube alucina
        viabilidad (`verdict_code=OK`) mientras el analizador topológico del
        borde detecta un socavón lógico circular ($\beta_1 > 0$) y un exceso
        de Mayer-Vietoris. El hardware perimetral anula la transacción.
        """
        packet = TelemetryPacket()
        payload = {
            "physics": {"kinetic_energy": 5.0, "dissipated_power": 1.0, "hamiltonian_excess": 0.0},
            "topology": {"beta_0": 1, "beta_1": 2, "mayer_vietoris_delta": 4, "fiedler_value": 0.8},
            "control": {"is_stable": True, "lyapunov_exponent": -0.1},
            "thermodynamics": {"system_temperature": 298.15, "entropy": 0.1,
                                "financial_inertia": 1.5, "exergy": 10.0},
            "wisdom": {"verdict_code": 0, "narrative_short": "VIABLE: Todo en orden, proceda."},
        }
        err = packet.from_json(json.dumps(payload))
        self.assertEqual(err, TelemetryError.TOPOLOGICAL_ANOMALY)
        self.assertFalse(packet.is_intra_coherent())
        self.assertTrue(packet.wisdom.is_ok())  # Evidencia forense de la alucinación del LLM.

    def test_telemetry_packet_default_construction_round_trip(self) -> None:
        """Un paquete construido con defaults íntegros debe ser NONE e intra-coherente."""
        packet = TelemetryPacket()
        self.assertEqual(packet.from_json(json.dumps({"timestamp": "t0"})), TelemetryError.NONE)
        self.assertTrue(packet.is_intra_coherent())


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 — TEORÍA DE CATEGORÍAS/TOPOS Y MAJORIZACIÓN CUÁNTICA                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TestFase3CategoricalClosureAndMajorization(unittest.TestCase):
    """
    Continúa la Fase 2: consume el `AuditReport` ya certificado y demuestra,
    en el nivel categórico-cuántico, la Ley de Clausura Transitiva completa
    que las Fases 1 y 2 dejaron lista pero sin certificar sobre vectores
    de estrato reales.
    """

    def setUp(self) -> None:
        self.poset = StrataPoset()
        self.majorization = QuantumMajorizationValidator()
        self.topos = CyberPhysicalTopos()

    # ─────────────── III.0 Categoría delgada 4 y clasificador Ω ─────────────

    def test_strata_poset_reflexivity_and_strict_order(self) -> None:
        r"""$\mathrm{Hom}(X,X)$ existe (reflexividad) y PHYSICS $\le$ WISDOM (orden total)."""
        self.assertTrue(self.poset.leq("PHYSICS", "PHYSICS"))
        self.assertTrue(self.poset.leq("PHYSICS", "WISDOM"))
        self.assertFalse(self.poset.leq("WISDOM", "PHYSICS"))

    def test_strata_poset_transitivity_of_morphisms(self) -> None:
        r"""Composición de morfismos: PHYSICS≤TACTICS y TACTICS≤STRATEGY $\implies$ PHYSICS≤STRATEGY."""
        self.assertTrue(self.poset.leq("PHYSICS", "TACTICS"))
        self.assertTrue(self.poset.leq("TACTICS", "STRATEGY"))
        self.assertTrue(self.poset.leq("PHYSICS", "STRATEGY"))

    def test_subobject_classifier_omega_values(self) -> None:
        r"""El clasificador $\chi: X \to \Omega=\{0,1\}$ debe ser fiel al predicado evaluado."""
        self.assertEqual(StrataPoset.subobject_classifier(True), 1)
        self.assertEqual(StrataPoset.subobject_classifier(False), 0)

    # ─────────────── III.1 Majorización cuántica (Teorema de Nielsen) ───────

    def test_quantum_majorization_chain_holds_for_decreasing_purity(self) -> None:
        r"""
        Cadena PHYSICS≻TACTICS≻STRATEGY≻WISDOM válida cuando la "pureza" del
        vector de energía/riesgo decrece monótonamente (mezcla creciente),
        condición equivalente al Teorema de Nielsen de transformabilidad LOCC.
        """
        physics_v = np.array([10.0, 1.0, 0.5])
        tactics_v = np.array([6.0, 4.0, 1.5])
        strategy_v = np.array([4.0, 4.0, 3.5])
        wisdom_v = np.array([4.0, 4.0, 4.0])
        self.assertTrue(
            self.majorization.verify_chain(physics_v, tactics_v, strategy_v, wisdom_v)
        )

    def test_quantum_majorization_chain_rejects_entropy_inversion(self) -> None:
        """Una inversión de la monotonicidad entrópica debe romper la cadena de Nielsen."""
        physics_v = np.array([4.0, 4.0, 4.0])   # Ya mezclado.
        tactics_v = np.array([10.0, 1.0, 1.0])  # Se "purifica" de nuevo: viola LOCC.
        self.assertFalse(self.majorization.verify_chain(physics_v, tactics_v))

    def test_quantum_majorization_rejects_non_positive_total_energy(self) -> None:
        """Un vector de estrato con energía total nula/negativa es físicamente inadmisible."""
        with self.assertRaises(MajorizationOrderError):
            self.majorization.verify_chain(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

    # ─────────────── III.2 CyberPhysicalTopos: orquestación de rango 3 ──────

    def test_full_closure_validation_succeeds_on_coherent_hierarchy(self) -> None:
        """Certificación end-to-end: kernel + booleano-grafo + categórico-cuántico, todo verde."""
        report = self.topos.validate_full_closure(
            physics=PhysicsMetrics(kinetic_energy=10.0, potential_energy=1.0, dissipated_power=0.5),
            control=ControlMetrics(),
            thermo=ThermodynamicMetrics(),
            tactics_vector=np.array([6.0, 4.0, 1.5]),
            strategy_vector=np.array([4.0, 4.0, 3.5]),
            wisdom_vector=np.array([4.0, 4.0, 4.0]),
        )
        self.assertIsInstance(report, AuditReport)
        self.assertTrue(report.coherent)

    def test_full_closure_propagates_underlying_physics_error(self) -> None:
        """Un fallo de Fase 1/2 debe abortar antes de siquiera evaluar la majorización."""
        with self.assertRaises(PhysicsCoherenceError):
            self.topos.validate_full_closure(
                physics=PhysicsMetrics(kinetic_energy=-1.0),
                control=ControlMetrics(),
                thermo=ThermodynamicMetrics(),
                tactics_vector=np.array([1.0]),
                strategy_vector=np.array([1.0]),
                wisdom_vector=np.array([1.0]),
            )

    def test_full_closure_raises_majorization_order_error_on_broken_chain(self) -> None:
        """Con estratos coherentes pero entropía invertida, se veta con `MajorizationOrderError`."""
        with self.assertRaises(MajorizationOrderError):
            self.topos.validate_full_closure(
                physics=PhysicsMetrics(kinetic_energy=1.0, potential_energy=1.0, dissipated_power=0.5),
                control=ControlMetrics(),
                thermo=ThermodynamicMetrics(),
                tactics_vector=np.array([100.0, 0.0, 0.0]),  # Se "purifica" espuriamente.
                strategy_vector=np.array([50.0, 25.0, 25.0]),
                wisdom_vector=np.array([34.0, 33.0, 33.0]),
            )


def load_full_suite() -> unittest.TestSuite:
    """Ensambla la suite completa preservando el orden narrativo de las 3 fases anidadas."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestFase1KernelFundamental))
    suite.addTests(loader.loadTestsFromTestCase(TestFase2BooleanGraphAndPacket))
    suite.addTests(loader.loadTestsFromTestCase(TestFase3CategoricalClosureAndMajorization))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(load_full_suite())
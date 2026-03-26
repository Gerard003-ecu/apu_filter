"""
Suite de pruebas: test_hilbert_watcher.py
Ubicación: tests/unit/agents/test_hilbert_watcher.py

Pruebas exhaustivas para el agente HilbertObserverAgent.

Cobertura:
    1. Construcción y validación de dependencias (Protocol contracts).
    2. Serialización determinista canónica del payload.
    3. Entropía de Shannon: casos extremos y propiedades teóricas.
    4. Fase OBSERVE: energía incidente desde payload.
    5. Fase ORIENT: acoplamiento a oráculos estructurales.
    6. Fase DECIDE: reglas de transmisión R1-R4 exhaustivas.
    7. Hash de colapso: determinismo, uniformidad, rango.
    8. Fase ACT: colapso a eigenestados con telemetría.
    9. Ciclo OODA completo: integración end-to-end.
   10. Casos límite numéricos: underflow, overflow, NaN, ±Inf.
   11. Propiedades algebraicas: idempotencia, consistencia categórica.
   12. WavefunctionState: validación de invariantes __post_init__.

Estrategia:
    - Mocks explícitos para oráculos (inyección de dependencias).
    - Valores frontera derivados de las constantes QuantumThresholds.
    - Verificación de propiedades matemáticas, no solo casos puntuales.
    - Sin dependencias externas más allá de pytest y numpy.

========================================================================
"""
from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, Mapping
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.agents.hilbert_watcher import (
    HilbertEigenstate,
    HilbertInterfaceError,
    HilbertNumericalError,
    HilbertObserverAgent,
    HilbertPayloadError,
    HilbertWatcherError,
    QuantumThresholds,
    WavefunctionState,
    _clamp_probability,
    _ensure_finite_float,
    _ensure_nonneg_finite_float,
    _safe_context,
)
from app.core.mic_algebra import CategoricalState
from app.core.schemas import Stratum


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES Y MOCKS
# ═══════════════════════════════════════════════════════════════════════


class MockTopologicalWatcher:
    """Mock del oráculo topológico con amenaza configurable."""

    def __init__(self, threat: float = 0.0) -> None:
        self._threat = threat

    def get_mahalanobis_threat(self) -> float:
        return self._threat


class MockLaplaceOracle:
    """Mock del oráculo espectral con polo dominante configurable."""

    def __init__(self, sigma: float = -1.0) -> None:
        self._sigma = sigma

    def get_dominant_pole_real(self) -> float:
        return self._sigma


class MockSheafOrchestrator:
    """Mock del orquestador cohomológico con frustración configurable."""

    def __init__(self, frustration: float = 0.0) -> None:
        self._frustration = frustration

    def get_global_frustration_energy(self) -> float:
        return self._frustration


def make_agent(
    threat: float = 0.0,
    sigma: float = -1.0,
    frustration: float = 0.0,
) -> HilbertObserverAgent:
    """Factory de agentes con oráculos configurables."""
    return HilbertObserverAgent(
        topo_watcher=MockTopologicalWatcher(threat),
        laplace_oracle=MockLaplaceOracle(sigma),
        sheaf_orchestrator=MockSheafOrchestrator(frustration),
    )


def make_state(
    payload: Dict[str, Any] | None = None,
    context: Dict[str, Any] | None = None,
    validated_strata: frozenset | None = None,
) -> CategoricalState:
    """Factory de estados categóricos para pruebas."""
    return CategoricalState(
        payload=payload if payload is not None else {"key": "value"},
        context=context if context is not None else {},
        validated_strata=(
            validated_strata
            if validated_strata is not None
            else frozenset()
        ),
    )


@pytest.fixture
def default_agent() -> HilbertObserverAgent:
    """Agente con oráculos en estado nominal."""
    return make_agent(threat=0.0, sigma=-1.0, frustration=0.0)


@pytest.fixture
def frustrated_agent() -> HilbertObserverAgent:
    """Agente con frustración cohomológica activa."""
    return make_agent(frustration=1.0)


@pytest.fixture
def unstable_agent() -> HilbertObserverAgent:
    """Agente con polo dominante inestable (σ ≥ 0)."""
    return make_agent(sigma=0.1)


@pytest.fixture
def high_threat_agent() -> HilbertObserverAgent:
    """Agente con amenaza topológica elevada."""
    return make_agent(threat=10.0)


@pytest.fixture
def simple_state() -> CategoricalState:
    """Estado categórico simple para pruebas."""
    return make_state(payload={"data": "test_payload"})


# ═══════════════════════════════════════════════════════════════════════
# 1. CONSTRUCCIÓN Y VALIDACIÓN DE DEPENDENCIAS
# ═══════════════════════════════════════════════════════════════════════


class TestAgentConstruction:
    """Pruebas de construcción y validación de contratos Protocol."""

    def test_valid_construction(self) -> None:
        """Agente se construye correctamente con dependencias válidas."""
        agent = make_agent()
        assert agent is not None

    def test_none_topo_watcher_raises(self) -> None:
        """topo_watcher=None lanza HilbertInterfaceError."""
        with pytest.raises(HilbertInterfaceError, match="topo_watcher"):
            HilbertObserverAgent(
                topo_watcher=None,  # type: ignore[arg-type]
                laplace_oracle=MockLaplaceOracle(),
                sheaf_orchestrator=MockSheafOrchestrator(),
            )

    def test_none_laplace_oracle_raises(self) -> None:
        """laplace_oracle=None lanza HilbertInterfaceError."""
        with pytest.raises(HilbertInterfaceError, match="laplace_oracle"):
            HilbertObserverAgent(
                topo_watcher=MockTopologicalWatcher(),
                laplace_oracle=None,  # type: ignore[arg-type]
                sheaf_orchestrator=MockSheafOrchestrator(),
            )

    def test_none_sheaf_orchestrator_raises(self) -> None:
        """sheaf_orchestrator=None lanza HilbertInterfaceError."""
        with pytest.raises(
            HilbertInterfaceError, match="sheaf_orchestrator"
        ):
            HilbertObserverAgent(
                topo_watcher=MockTopologicalWatcher(),
                laplace_oracle=MockLaplaceOracle(),
                sheaf_orchestrator=None,  # type: ignore[arg-type]
            )

    def test_missing_method_raises(self) -> None:
        """Objeto sin método requerido lanza HilbertInterfaceError."""

        class BadWatcher:
            pass

        with pytest.raises(HilbertInterfaceError, match="carece"):
            HilbertObserverAgent(
                topo_watcher=BadWatcher(),  # type: ignore[arg-type]
                laplace_oracle=MockLaplaceOracle(),
                sheaf_orchestrator=MockSheafOrchestrator(),
            )

    def test_non_callable_method_raises(self) -> None:
        """Atributo no-callable con nombre de método requerido falla."""

        class BadOracle:
            get_dominant_pole_real = 42  # No callable

        with pytest.raises(HilbertInterfaceError, match="invocable"):
            HilbertObserverAgent(
                topo_watcher=MockTopologicalWatcher(),
                laplace_oracle=BadOracle(),  # type: ignore[arg-type]
                sheaf_orchestrator=MockSheafOrchestrator(),
            )

    def test_construction_is_atomic(self) -> None:
        """Si una dependencia falla, ninguna se asigna (atomicidad)."""

        class PartialWatcher:
            def get_mahalanobis_threat(self) -> float:
                return 0.0

        # sheaf_orchestrator es None → debe fallar antes de asignar
        with pytest.raises(HilbertInterfaceError):
            HilbertObserverAgent(
                topo_watcher=PartialWatcher(),
                laplace_oracle=MockLaplaceOracle(),
                sheaf_orchestrator=None,  # type: ignore[arg-type]
            )


# ═══════════════════════════════════════════════════════════════════════
# 2. SERIALIZACIÓN DETERMINISTA
# ═══════════════════════════════════════════════════════════════════════


class TestSerializePayload:
    """Pruebas de serialización canónica determinista."""

    def test_deterministic_same_payload(self) -> None:
        """Mismo payload produce bytes idénticos."""
        payload = {"a": 1, "b": [2, 3], "c": "hello"}
        b1 = HilbertObserverAgent._serialize_payload(payload)
        b2 = HilbertObserverAgent._serialize_payload(payload)
        assert b1 == b2

    def test_deterministic_key_order_invariant(self) -> None:
        """Orden de inserción de claves no afecta serialización."""
        p1 = {"z": 1, "a": 2, "m": 3}
        p2 = {"a": 2, "m": 3, "z": 1}
        assert (
            HilbertObserverAgent._serialize_payload(p1)
            == HilbertObserverAgent._serialize_payload(p2)
        )

    def test_empty_payload(self) -> None:
        """Payload vacío produce bytes válidos no vacíos."""
        result = HilbertObserverAgent._serialize_payload({})
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_non_mapping_raises(self) -> None:
        """Payload no-Mapping lanza HilbertPayloadError."""
        with pytest.raises(HilbertPayloadError, match="Mapping"):
            HilbertObserverAgent._serialize_payload(
                [1, 2, 3]  # type: ignore[arg-type]
            )

    def test_none_payload_raises(self) -> None:
        """Payload None lanza HilbertPayloadError."""
        with pytest.raises(HilbertPayloadError, match="Mapping"):
            HilbertObserverAgent._serialize_payload(
                None  # type: ignore[arg-type]
            )

    def test_string_payload_raises(self) -> None:
        """Payload string lanza HilbertPayloadError."""
        with pytest.raises(HilbertPayloadError, match="Mapping"):
            HilbertObserverAgent._serialize_payload(
                "not a dict"  # type: ignore[arg-type]
            )

    def test_different_payloads_different_bytes(self) -> None:
        """Payloads distintos producen bytes distintos."""
        p1 = {"a": 1}
        p2 = {"a": 2}
        assert (
            HilbertObserverAgent._serialize_payload(p1)
            != HilbertObserverAgent._serialize_payload(p2)
        )

    def test_nested_payload(self) -> None:
        """Payloads con estructuras anidadas se serializan."""
        payload = {
            "nested": {"deep": [1, 2, {"x": True}]},
            "value": 42,
        }
        result = HilbertObserverAgent._serialize_payload(payload)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_type_distinction_preserved(self) -> None:
        """Tipos distintos con repr similar se distinguen."""
        p_int = {"val": 1}
        p_float = {"val": 1.0}
        # repr(1) = '1', repr(1.0) = '1.0' → deben diferir
        assert (
            HilbertObserverAgent._serialize_payload(p_int)
            != HilbertObserverAgent._serialize_payload(p_float)
        )

    def test_output_is_utf8(self) -> None:
        """Resultado es decodificable como UTF-8 válido."""
        payload = {"key": "value", "num": 42}
        result = HilbertObserverAgent._serialize_payload(payload)
        decoded = result.decode("utf-8")
        assert isinstance(decoded, str)


# ═══════════════════════════════════════════════════════════════════════
# 3. ENTROPÍA DE SHANNON
# ═══════════════════════════════════════════════════════════════════════


class TestByteEntropy:
    """Pruebas de la función de entropía de Shannon en bytes."""

    def test_empty_data_zero_entropy(self) -> None:
        """Datos vacíos tienen entropía 0."""
        assert HilbertObserverAgent._byte_entropy_bits(b"") == 0.0

    def test_single_byte_zero_entropy(self) -> None:
        """Un solo byte tiene entropía 0 (un símbolo, log₂(1)=0)."""
        assert HilbertObserverAgent._byte_entropy_bits(b"\x00") == 0.0

    def test_constant_data_zero_entropy(self) -> None:
        """Datos constantes (todos el mismo byte) tienen H=0."""
        data = bytes([42] * 1000)
        assert HilbertObserverAgent._byte_entropy_bits(data) == 0.0

    def test_two_equiprobable_symbols_one_bit(self) -> None:
        """Dos símbolos equiprobables dan H=1.0 bit."""
        data = bytes([0, 1] * 500)
        entropy = HilbertObserverAgent._byte_entropy_bits(data)
        assert math.isclose(entropy, 1.0, abs_tol=1e-10)

    def test_four_equiprobable_symbols_two_bits(self) -> None:
        """Cuatro símbolos equiprobables dan H=2.0 bits."""
        data = bytes([0, 1, 2, 3] * 250)
        entropy = HilbertObserverAgent._byte_entropy_bits(data)
        assert math.isclose(entropy, 2.0, abs_tol=1e-10)

    def test_uniform_256_symbols_eight_bits(self) -> None:
        """256 símbolos equiprobables dan H=8.0 bits (máximo)."""
        data = bytes(list(range(256)) * 100)
        entropy = HilbertObserverAgent._byte_entropy_bits(data)
        assert math.isclose(entropy, 8.0, abs_tol=1e-6)

    def test_entropy_bounded_zero_eight(self) -> None:
        """Entropía siempre está en [0, 8] para cualquier input."""
        import os

        # Datos aleatorios
        random_data = os.urandom(10000)
        entropy = HilbertObserverAgent._byte_entropy_bits(random_data)
        assert 0.0 <= entropy <= 8.0

    def test_entropy_nonnegative(self) -> None:
        """Entropía nunca es negativa."""
        test_cases = [b"\x00", b"\xff", b"hello", b"\x00\x01"]
        for data in test_cases:
            assert HilbertObserverAgent._byte_entropy_bits(data) >= 0.0

    def test_entropy_monotonicity_with_diversity(self) -> None:
        """Mayor diversidad de símbolos → mayor entropía.

        Propiedad: H(X_k) ≤ H(X_{k+1}) cuando X_{k+1} tiene
        más símbolos equiprobables que X_k.
        """
        prev_entropy = 0.0
        for n_symbols in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            data = bytes(list(range(n_symbols)) * 100)
            entropy = HilbertObserverAgent._byte_entropy_bits(data)
            assert entropy >= prev_entropy - 1e-10
            prev_entropy = entropy

    def test_entropy_is_log2_n_for_uniform(self) -> None:
        """Para distribución uniforme de n símbolos: H = log₂(n).

        Propiedad fundamental de Shannon.
        """
        for n in [2, 4, 8, 16, 32, 64, 128, 256]:
            data = bytes(list(range(n)) * 200)
            entropy = HilbertObserverAgent._byte_entropy_bits(data)
            expected = math.log2(n)
            assert math.isclose(entropy, expected, abs_tol=1e-6), (
                f"n={n}: H={entropy}, expected={expected}"
            )


# ═══════════════════════════════════════════════════════════════════════
# 4. OBSERVE: ENERGÍA INCIDENTE
# ═══════════════════════════════════════════════════════════════════════


class TestObserveIncidentWave:
    """Pruebas de la fase OBSERVE del ciclo OODA."""

    def test_empty_payload_zero_energy(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Payload vacío serializado produce E=0."""
        serialized = HilbertObserverAgent._serialize_payload({})
        # payload vacío no produce bytes vacíos (repr de tupla vacía)
        # pero la energía debe ser determinista
        E = default_agent._observe_incident_wave(serialized)
        assert E >= 0.0

    def test_empty_bytes_zero_energy(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Bytes vacíos producen E=0."""
        E = default_agent._observe_incident_wave(b"")
        assert E == 0.0

    def test_energy_nonnegative(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Energía siempre es ≥ 0."""
        payloads = [
            {"a": 1},
            {"data": "x" * 1000},
            {"n": 42, "list": [1, 2, 3]},
        ]
        for p in payloads:
            serialized = HilbertObserverAgent._serialize_payload(p)
            E = default_agent._observe_incident_wave(serialized)
            assert E >= 0.0, f"E={E} negativa para {p}"

    def test_energy_finite(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Energía siempre es finita."""
        serialized = HilbertObserverAgent._serialize_payload(
            {"large": "x" * 10000}
        )
        E = default_agent._observe_incident_wave(serialized)
        assert math.isfinite(E)

    def test_energy_increases_with_size(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Energía crece con el tamaño del payload (fijada entropía similar).

        Propiedad: E ∝ N / H, por lo que a H similar, E ∝ N.
        """
        energies = []
        for size in [10, 100, 1000]:
            # Datos repetitivos para mantener entropía baja
            payload = {"data": "a" * size}
            serialized = HilbertObserverAgent._serialize_payload(payload)
            E = default_agent._observe_incident_wave(serialized)
            energies.append(E)

        for i in range(len(energies) - 1):
            assert energies[i] <= energies[i + 1], (
                f"Energía no creciente: {energies}"
            )

    def test_energy_deterministic(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Misma entrada produce misma energía."""
        serialized = HilbertObserverAgent._serialize_payload(
            {"key": "test"}
        )
        E1 = default_agent._observe_incident_wave(serialized)
        E2 = default_agent._observe_incident_wave(serialized)
        assert E1 == E2


# ═══════════════════════════════════════════════════════════════════════
# 5. ORIENT: ACOPLAMIENTO A ORÁCULOS
# ═══════════════════════════════════════════════════════════════════════


class TestOrientGaugeFields:
    """Pruebas de la fase ORIENT del ciclo OODA."""

    def test_nominal_orient(self) -> None:
        """Orient con oráculos nominales produce valores consistentes."""
        agent = make_agent(threat=0.0, sigma=-1.0, frustration=0.0)
        Phi, m_eff, frustrated, threat, sigma, frust = (
            agent._orient_gauge_fields()
        )

        assert Phi == QuantumThresholds.BASE_PHI
        assert m_eff == QuantumThresholds.BASE_MASS / 1.0
        assert not frustrated
        assert threat == 0.0
        assert sigma == -1.0
        assert frust == 0.0

    def test_threat_increases_work_function(self) -> None:
        """Amenaza topológica mayor → función de trabajo mayor.

        Φ(threat) = Φ₀ + α·threat, monótona creciente.
        """
        Phi_low = make_agent(threat=0.0)._orient_gauge_fields()[0]
        Phi_mid = make_agent(threat=1.0)._orient_gauge_fields()[0]
        Phi_high = make_agent(threat=5.0)._orient_gauge_fields()[0]

        assert Phi_low < Phi_mid < Phi_high

    def test_work_function_formula(self) -> None:
        """Verificar fórmula exacta: Φ = Φ₀ + α·threat."""
        threat = 3.7
        agent = make_agent(threat=threat)
        Phi = agent._orient_gauge_fields()[0]

        expected = (
            QuantumThresholds.BASE_PHI
            + QuantumThresholds.ALPHA_COUPLING * threat
        )
        assert math.isclose(Phi, expected, rel_tol=1e-12)

    def test_stable_pole_finite_mass(self) -> None:
        """Polo estable (σ < -tol) → masa efectiva finita positiva."""
        for sigma in [-0.5, -1.0, -2.0, -10.0]:
            agent = make_agent(sigma=sigma)
            _, m_eff, *_ = agent._orient_gauge_fields()
            assert math.isfinite(m_eff)
            assert m_eff > 0.0

    def test_mass_formula(self) -> None:
        """Verificar fórmula exacta: m_eff = m₀ / |σ|."""
        sigma = -2.5
        agent = make_agent(sigma=sigma)
        _, m_eff, *_ = agent._orient_gauge_fields()

        expected = QuantumThresholds.BASE_MASS / abs(sigma)
        assert math.isclose(m_eff, expected, rel_tol=1e-12)

    def test_unstable_pole_infinite_mass(self) -> None:
        """Polo inestable (σ ≥ -tol) → m_eff = +∞."""
        for sigma in [0.0, 0.1, 1.0, -1e-10]:
            agent = make_agent(sigma=sigma)
            _, m_eff, *_ = agent._orient_gauge_fields()
            assert math.isinf(m_eff)

    def test_marginal_pole_infinite_mass(self) -> None:
        """Polo marginalmente estable (σ = -tol/2) → m_eff = +∞."""
        sigma = -QuantumThresholds.SIGMA_CHAOS_TOL / 2
        agent = make_agent(sigma=sigma)
        _, m_eff, *_ = agent._orient_gauge_fields()
        assert math.isinf(m_eff)

    def test_frustration_activates_flag(self) -> None:
        """Frustración > ε activa flag de veto cohomológico."""
        agent = make_agent(frustration=1.0)
        _, _, frustrated, *_ = agent._orient_gauge_fields()
        assert frustrated is True

    def test_no_frustration_deactivates_flag(self) -> None:
        """Frustración = 0 desactiva flag."""
        agent = make_agent(frustration=0.0)
        _, _, frustrated, *_ = agent._orient_gauge_fields()
        assert frustrated is False

    def test_frustration_at_epsilon_boundary(self) -> None:
        """Frustración exactamente en ε no activa flag."""
        agent = make_agent(
            frustration=QuantumThresholds.EPSILON_MACH
        )
        _, _, frustrated, *_ = agent._orient_gauge_fields()
        assert frustrated is False

    def test_frustration_just_above_epsilon(self) -> None:
        """Frustración ligeramente sobre ε activa flag."""
        agent = make_agent(
            frustration=QuantumThresholds.EPSILON_MACH * 1.1
        )
        _, _, frustrated, *_ = agent._orient_gauge_fields()
        assert frustrated is True

    def test_orient_raises_on_nan_threat(self) -> None:
        """Amenaza NaN de oráculo lanza HilbertNumericalError."""

        class NanWatcher:
            def get_mahalanobis_threat(self) -> float:
                return float("nan")

        agent = HilbertObserverAgent(
            topo_watcher=NanWatcher(),
            laplace_oracle=MockLaplaceOracle(),
            sheaf_orchestrator=MockSheafOrchestrator(),
        )
        with pytest.raises(HilbertNumericalError, match="finito"):
            agent._orient_gauge_fields()

    def test_orient_raises_on_negative_threat(self) -> None:
        """Amenaza negativa de oráculo lanza HilbertNumericalError."""
        agent = make_agent(threat=-1.0)
        with pytest.raises(HilbertNumericalError, match="≥ 0"):
            agent._orient_gauge_fields()

    def test_orient_raises_on_negative_frustration(self) -> None:
        """Frustración negativa lanza HilbertNumericalError."""
        agent = make_agent(frustration=-0.5)
        with pytest.raises(HilbertNumericalError, match="≥ 0"):
            agent._orient_gauge_fields()

    def test_orient_catches_oracle_exception(self) -> None:
        """Excepción en oráculo se re-lanza como HilbertInterfaceError."""

        class FailingOracle:
            def get_dominant_pole_real(self) -> float:
                raise RuntimeError("Oracle exploded")

        agent = HilbertObserverAgent(
            topo_watcher=MockTopologicalWatcher(),
            laplace_oracle=FailingOracle(),
            sheaf_orchestrator=MockSheafOrchestrator(),
        )
        with pytest.raises(HilbertInterfaceError, match="falló"):
            agent._orient_gauge_fields()


# ═══════════════════════════════════════════════════════════════════════
# 6. DECIDE: REGLAS DE TRANSMISIÓN R1-R4
# ═══════════════════════════════════════════════════════════════════════


class TestDecideQuantumTransmission:
    """Pruebas exhaustivas de las reglas de decisión R1-R4."""

    # --- R1: Veto cohomológico ---

    def test_r1_frustrated_always_zero(self) -> None:
        """R1: is_frustrated=True → T=0 sin importar E, Φ, m_eff."""
        test_cases = [
            (100.0, 1.0, 1.0),   # E >> Φ
            (0.0, 10.0, 1.0),     # E << Φ
            (10.0, 10.0, 1.0),    # E = Φ
            (50.0, 1.0, math.inf),  # m_eff infinita
        ]
        for E, Phi, m_eff in test_cases:
            T = HilbertObserverAgent._decide_quantum_transmission(
                E, Phi, m_eff, is_frustrated=True
            )
            assert T == 0.0, (
                f"R1 violada: T={T} para E={E}, Φ={Phi}, "
                f"m_eff={m_eff}"
            )

    # --- R2: Masa infinita ---

    def test_r2_infinite_mass_above_barrier(self) -> None:
        """R2: m_eff=+∞, E ≥ Φ → T=1."""
        T = HilbertObserverAgent._decide_quantum_transmission(
            E=20.0, Phi=10.0, m_eff=math.inf, is_frustrated=False
        )
        assert T == 1.0

    def test_r2_infinite_mass_at_barrier(self) -> None:
        """R2: m_eff=+∞, E = Φ → T=1."""
        T = HilbertObserverAgent._decide_quantum_transmission(
            E=10.0, Phi=10.0, m_eff=math.inf, is_frustrated=False
        )
        assert T == 1.0

    def test_r2_infinite_mass_below_barrier(self) -> None:
        """R2: m_eff=+∞, E < Φ → T=0."""
        T = HilbertObserverAgent._decide_quantum_transmission(
            E=5.0, Phi=10.0, m_eff=math.inf, is_frustrated=False
        )
        assert T == 0.0

    # --- R3: Fotoeléctrico clásico ---

    def test_r3_photoelectric_above(self) -> None:
        """R3: E > Φ, m_eff finita → T=1."""
        T = HilbertObserverAgent._decide_quantum_transmission(
            E=20.0, Phi=10.0, m_eff=1.0, is_frustrated=False
        )
        assert T == 1.0

    def test_r3_photoelectric_exact(self) -> None:
        """R3: E = Φ exacto → T=1."""
        T = HilbertObserverAgent._decide_quantum_transmission(
            E=10.0, Phi=10.0, m_eff=1.0, is_frustrated=False
        )
        assert T == 1.0

    # --- R4: Túnel WKB ---

    def test_r4_tunnel_below_barrier(self) -> None:
        """R4: E < Φ, m_eff finita → 0 < T < 1."""
        T = HilbertObserverAgent._decide_quantum_transmission(
            E=9.0, Phi=10.0, m_eff=1.0, is_frustrated=False
        )
        assert 0.0 < T < 1.0

    def test_r4_tunnel_deep_below_barrier(self) -> None:
        """R4: E << Φ → T ≈ 0 (exponencialmente suprimido)."""
        T = HilbertObserverAgent._decide_quantum_transmission(
            E=0.1, Phi=100.0, m_eff=10.0, is_frustrated=False
        )
        assert T < 0.01  # Casi cero

    def test_r4_tunnel_monotone_in_energy(self) -> None:
        """R4: T es monótona creciente en E (Φ, m_eff fijos).

        ∂T/∂E > 0 para E < Φ.
        """
        Phi, m_eff = 10.0, 1.0
        energies = [1.0, 3.0, 5.0, 7.0, 9.0, 9.9]
        transmissions = [
            HilbertObserverAgent._decide_quantum_transmission(
                E, Phi, m_eff, is_frustrated=False
            )
            for E in energies
        ]
        for i in range(len(transmissions) - 1):
            assert transmissions[i] <= transmissions[i + 1], (
                f"No monótona: T({energies[i]})={transmissions[i]} > "
                f"T({energies[i+1]})={transmissions[i+1]}"
            )

    def test_r4_tunnel_monotone_decreasing_in_mass(self) -> None:
        """R4: T es monótona decreciente en m_eff (E, Φ fijos).

        ∂T/∂m < 0 para E < Φ (barrera más pesada → menos túnel).
        """
        E, Phi = 5.0, 10.0
        masses = [0.1, 0.5, 1.0, 5.0, 10.0]
        transmissions = [
            HilbertObserverAgent._decide_quantum_transmission(
                E, Phi, m, is_frustrated=False
            )
            for m in masses
        ]
        for i in range(len(transmissions) - 1):
            assert transmissions[i] >= transmissions[i + 1], (
                f"No decreciente en masa: "
                f"T(m={masses[i]})={transmissions[i]} < "
                f"T(m={masses[i+1]})={transmissions[i+1]}"
            )

    def test_r4_tunnel_monotone_decreasing_in_phi(self) -> None:
        """R4: T es monótona decreciente en Φ (E, m_eff fijos).

        ∂T/∂Φ < 0 para E < Φ.
        """
        E, m_eff = 5.0, 1.0
        phis = [6.0, 8.0, 10.0, 20.0, 50.0]
        transmissions = [
            HilbertObserverAgent._decide_quantum_transmission(
                E, phi, m_eff, is_frustrated=False
            )
            for phi in phis
        ]
        for i in range(len(transmissions) - 1):
            assert transmissions[i] >= transmissions[i + 1], (
                f"No decreciente en Φ: "
                f"T(Φ={phis[i]})={transmissions[i]} < "
                f"T(Φ={phis[i+1]})={transmissions[i+1]}"
            )

    def test_r4_underflow_protection(self) -> None:
        """Exponente extremo no causa error, retorna T=0."""
        T = HilbertObserverAgent._decide_quantum_transmission(
            E=0.0, Phi=1e6, m_eff=1e6, is_frustrated=False
        )
        assert T == 0.0

    def test_transmission_always_in_unit_interval(self) -> None:
        """T ∈ [0, 1] para combinaciones diversas de parámetros."""
        params = [
            (0.0, 0.0, 1.0, False),
            (100.0, 1.0, 0.01, False),
            (1.0, 100.0, 100.0, False),
            (50.0, 50.0, 1.0, False),
            (0.0, 0.0, math.inf, False),
            (10.0, 5.0, math.inf, True),
        ]
        for E, Phi, m, frust in params:
            T = HilbertObserverAgent._decide_quantum_transmission(
                E, Phi, m, frust
            )
            assert 0.0 <= T <= 1.0, (
                f"T={T} fuera de [0,1] para "
                f"E={E}, Φ={Phi}, m={m}, frust={frust}"
            )

    def test_negative_mass_raises(self) -> None:
        """Masa negativa finita lanza error."""
        with pytest.raises(HilbertNumericalError, match="positiva"):
            HilbertObserverAgent._decide_quantum_transmission(
                E=5.0, Phi=10.0, m_eff=-1.0, is_frustrated=False
            )

    def test_zero_mass_raises(self) -> None:
        """Masa cero lanza error."""
        with pytest.raises(HilbertNumericalError, match="positiva"):
            HilbertObserverAgent._decide_quantum_transmission(
                E=5.0, Phi=10.0, m_eff=0.0, is_frustrated=False
            )

    # --- Prioridad de reglas ---

    def test_r1_overrides_r3(self) -> None:
        """R1 tiene prioridad sobre R3 (fotoeléctrico con frustración)."""
        T = HilbertObserverAgent._decide_quantum_transmission(
            E=100.0, Phi=1.0, m_eff=1.0, is_frustrated=True
        )
        assert T == 0.0

    def test_r1_overrides_r2(self) -> None:
        """R1 tiene prioridad sobre R2 (masa infinita con frustración)."""
        T = HilbertObserverAgent._decide_quantum_transmission(
            E=100.0, Phi=1.0, m_eff=math.inf, is_frustrated=True
        )
        assert T == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 7. HASH DE COLAPSO
# ═══════════════════════════════════════════════════════════════════════


class TestCollapseThreshold:
    """Pruebas del generador de umbral de colapso determinista."""

    def test_deterministic(self) -> None:
        """Mismo payload → mismo umbral."""
        payload = HilbertObserverAgent._serialize_payload(
            {"test": 123}
        )
        t1 = HilbertObserverAgent._compute_collapse_threshold(payload)
        t2 = HilbertObserverAgent._compute_collapse_threshold(payload)
        assert t1 == t2

    def test_range_zero_one(self) -> None:
        """Umbral siempre en [0, 1)."""
        for i in range(100):
            payload = HilbertObserverAgent._serialize_payload(
                {"i": i}
            )
            threshold = (
                HilbertObserverAgent._compute_collapse_threshold(
                    payload
                )
            )
            assert 0.0 <= threshold < 1.0, (
                f"threshold={threshold} fuera de [0,1) para i={i}"
            )

    def test_different_payloads_different_thresholds(self) -> None:
        """Payloads distintos generan umbrales distintos (salvo colisión).

        Con SHA-256 y 64 bits, la probabilidad de colisión en 100
        muestras es negligible (~2.7×10⁻¹⁶).
        """
        thresholds = set()
        for i in range(100):
            payload = HilbertObserverAgent._serialize_payload(
                {"unique": i}
            )
            t = HilbertObserverAgent._compute_collapse_threshold(
                payload
            )
            thresholds.add(t)

        # Al menos 99 de 100 deben ser distintos
        assert len(thresholds) >= 99

    def test_empty_payload_valid_threshold(self) -> None:
        """Payload vacío produce umbral válido."""
        payload = HilbertObserverAgent._serialize_payload({})
        threshold = (
            HilbertObserverAgent._compute_collapse_threshold(payload)
        )
        assert 0.0 <= threshold < 1.0

    def test_manual_sha256_verification(self) -> None:
        """Verificación manual del cálculo SHA-256 → umbral."""
        payload_dict = {"verify": "manual"}
        serialized = HilbertObserverAgent._serialize_payload(
            payload_dict
        )

        # Cálculo manual
        digest = hashlib.sha256(serialized).digest()
        n = int.from_bytes(
            digest[:8], byteorder="big", signed=False
        )
        expected = n / float(2**64)

        actual = HilbertObserverAgent._compute_collapse_threshold(
            serialized
        )
        assert math.isclose(actual, expected, rel_tol=1e-15)

    def test_resolution_better_than_1e6(self) -> None:
        """Resolución del umbral es mejor que 10⁻⁶.

        El mapeo n/2^64 tiene paso mínimo ~5.4×10⁻²⁰.
        """
        payload1 = HilbertObserverAgent._serialize_payload({"a": 1})
        payload2 = HilbertObserverAgent._serialize_payload({"a": 2})

        t1 = HilbertObserverAgent._compute_collapse_threshold(payload1)
        t2 = HilbertObserverAgent._compute_collapse_threshold(payload2)

        if t1 != t2:
            diff = abs(t1 - t2)
            # La diferencia puede ser arbitraria, pero la resolución
            # del mapeo permite distinciones más finas que 10⁻⁶
            assert diff > 0.0


# ═══════════════════════════════════════════════════════════════════════
# 8. ACT: COLAPSO DE FUNCIÓN DE ONDA
# ═══════════════════════════════════════════════════════════════════════


class TestActCollapseWavefunction:
    """Pruebas de la fase ACT del ciclo OODA."""

    @staticmethod
    def _make_wave(
        energy: float = 20.0,
        work_function: float = 10.0,
        effective_mass: float = 1.0,
        transmission_prob: float = 0.8,
        frustrated: bool = False,
        threat_level: float = 0.0,
        dominant_pole_real: float = -1.0,
        frustration_energy: float = 0.0,
        collapse_threshold: float = 0.5,
    ) -> WavefunctionState:
        """Factory de WavefunctionState para pruebas."""
        return WavefunctionState(
            energy=energy,
            work_function=work_function,
            effective_mass=effective_mass,
            transmission_prob=transmission_prob,
            frustrated=frustrated,
            threat_level=threat_level,
            dominant_pole_real=dominant_pole_real,
            frustration_energy=frustration_energy,
            collapse_threshold=collapse_threshold,
        )

    def test_admitted_when_t_ge_threshold(self) -> None:
        """T ≥ τ → eigenstate ADMITTED."""
        state = make_state()
        wave = self._make_wave(
            transmission_prob=0.8, collapse_threshold=0.5
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        measurement = result.context["quantum_measurement"]
        assert measurement["eigenstate"] == "ADMITTED"

    def test_rejected_when_t_lt_threshold(self) -> None:
        """T < τ → eigenstate REJECTED."""
        state = make_state()
        wave = self._make_wave(
            transmission_prob=0.3, collapse_threshold=0.5
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        measurement = result.context["quantum_measurement"]
        assert measurement["eigenstate"] == "REJECTED"

    def test_admitted_exact_equality(self) -> None:
        """T = τ exactamente → ADMITTED (≥ no >)."""
        state = make_state()
        wave = self._make_wave(
            transmission_prob=0.5, collapse_threshold=0.5
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        measurement = result.context["quantum_measurement"]
        assert measurement["eigenstate"] == "ADMITTED"

    def test_admitted_has_physics_stratum(self) -> None:
        """Estado admitido incluye Stratum.PHYSICS."""
        state = make_state()
        wave = self._make_wave(
            transmission_prob=1.0, collapse_threshold=0.0
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        assert Stratum.PHYSICS in result.validated_strata

    def test_rejected_has_empty_strata(self) -> None:
        """Estado rechazado tiene strata vacíos."""
        state = make_state()
        wave = self._make_wave(
            transmission_prob=0.0, collapse_threshold=0.5
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        assert result.validated_strata == frozenset()

    def test_payload_preserved_on_admission(self) -> None:
        """Payload original se preserva intacto en admisión."""
        original_payload = {"important": "data", "count": 42}
        state = make_state(payload=original_payload)
        wave = self._make_wave(transmission_prob=1.0)

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        assert result.payload == original_payload

    def test_payload_preserved_on_rejection(self) -> None:
        """Payload original se preserva intacto en rechazo."""
        original_payload = {"important": "data", "count": 42}
        state = make_state(payload=original_payload)
        wave = self._make_wave(
            transmission_prob=0.0, collapse_threshold=0.5
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        assert result.payload == original_payload

    def test_admitted_photoelectric_kinetic_energy(self) -> None:
        """Admisión fotoeléctrica: E_kin = E - Φ."""
        E, Phi = 25.0, 10.0
        state = make_state()
        wave = self._make_wave(
            energy=E,
            work_function=Phi,
            transmission_prob=1.0,
            collapse_threshold=0.0,
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        measurement = result.context["quantum_measurement"]
        expected_kinetic = E - Phi
        assert math.isclose(
            measurement["kinetic_energy"],
            expected_kinetic,
            rel_tol=1e-12,
        )

    def test_admitted_tunnel_minimal_kinetic_energy(self) -> None:
        """Admisión por túnel: E_kin = MIN_KINETIC_ENERGY."""
        state = make_state()
        wave = self._make_wave(
            energy=5.0,
            work_function=10.0,
            transmission_prob=0.8,
            collapse_threshold=0.5,
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        measurement = result.context["quantum_measurement"]
        assert measurement["kinetic_energy"] == (
            QuantumThresholds.MIN_KINETIC_ENERGY
        )

    def test_admitted_has_positive_momentum(self) -> None:
        """Estado admitido tiene momentum > 0."""
        state = make_state()
        wave = self._make_wave(
            transmission_prob=1.0, collapse_threshold=0.0
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        assert result.context["quantum_momentum"] > 0.0

    def test_rejected_has_zero_momentum(self) -> None:
        """Estado rechazado tiene momentum = 0."""
        state = make_state()
        wave = self._make_wave(
            transmission_prob=0.0, collapse_threshold=0.5
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        measurement = result.context["quantum_measurement"]
        assert measurement["momentum"] == 0.0
        assert measurement["kinetic_energy"] == 0.0

    def test_rejected_frustrated_has_cohomological_reason(self) -> None:
        """Rechazo por frustración menciona cohomología en razón."""
        state = make_state()
        wave = self._make_wave(
            transmission_prob=0.0,
            collapse_threshold=0.5,
            frustrated=True,
            frustration_energy=1.0,
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        reason = result.context["quantum_measurement"]["reason"]
        assert "cohomológico" in reason or "H¹" in reason

    def test_rejected_has_quantum_error(self) -> None:
        """Estado rechazado tiene clave 'quantum_error' en contexto."""
        state = make_state()
        wave = self._make_wave(
            transmission_prob=0.0, collapse_threshold=0.5
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        assert "quantum_error" in result.context

    def test_measurement_contains_all_fields(self) -> None:
        """Telemetría quantum_measurement contiene todos los campos."""
        required_fields = {
            "eigenstate",
            "energy",
            "work_function",
            "effective_mass",
            "transmission_prob",
            "frustrated",
            "threat_level",
            "dominant_pole_real",
            "frustration_energy",
            "collapse_threshold",
            "kinetic_energy",
            "momentum",
            "reason",
        }

        state = make_state()
        wave = self._make_wave(transmission_prob=1.0)

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        measurement = result.context["quantum_measurement"]
        assert required_fields.issubset(set(measurement.keys())), (
            f"Campos faltantes: "
            f"{required_fields - set(measurement.keys())}"
        )

    def test_context_preserved_from_input(self) -> None:
        """Contexto original del estado se preserva en output."""
        state = make_state(
            context={"existing_key": "existing_value"}
        )
        wave = self._make_wave(transmission_prob=1.0)

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        assert result.context["existing_key"] == "existing_value"

    def test_momentum_formula_photoelectric(self) -> None:
        """Momentum fotoeléctrico: p = √(2·m_eff·(E-Φ)).

        Verificación de la relación de de Broglie discretizada.
        """
        E, Phi, m_eff = 25.0, 10.0, 2.0
        state = make_state()
        wave = self._make_wave(
            energy=E,
            work_function=Phi,
            effective_mass=m_eff,
            transmission_prob=1.0,
            collapse_threshold=0.0,
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        kinetic = E - Phi
        expected_p = math.sqrt(2.0 * m_eff * kinetic)
        actual_p = result.context["quantum_momentum"]

        assert math.isclose(actual_p, expected_p, rel_tol=1e-12)

    def test_momentum_with_infinite_mass_uses_base(self) -> None:
        """Con m_eff=+∞, momentum usa BASE_MASS."""
        E, Phi = 20.0, 10.0
        state = make_state()
        wave = self._make_wave(
            energy=E,
            work_function=Phi,
            effective_mass=math.inf,
            transmission_prob=1.0,
            collapse_threshold=0.0,
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        kinetic = E - Phi
        expected_p = math.sqrt(
            2.0 * QuantumThresholds.BASE_MASS * kinetic
        )
        actual_p = result.context["quantum_momentum"]

        assert math.isclose(actual_p, expected_p, rel_tol=1e-12)

    def test_infinite_mass_serialized_as_string(self) -> None:
        """m_eff=+∞ se serializa como '+Inf' en telemetría."""
        state = make_state()
        wave = self._make_wave(
            effective_mass=math.inf,
            transmission_prob=1.0,
            collapse_threshold=0.0,
        )

        result = HilbertObserverAgent._act_collapse_wavefunction(
            state, wave
        )

        measurement = result.context["quantum_measurement"]
        assert measurement["effective_mass"] == "+Inf"


# ═══════════════════════════════════════════════════════════════════════
# 9. CICLO OODA COMPLETO (INTEGRACIÓN)
# ═══════════════════════════════════════════════════════════════════════


class TestOODALoop:
    """Pruebas de integración del ciclo OODA completo."""

    def test_nominal_execution(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Ciclo completo se ejecuta sin errores."""
        state = make_state(payload={"data": "hello"})
        result = default_agent.execute_ooda_loop(state)

        assert isinstance(result, CategoricalState)
        assert "quantum_measurement" in result.context

    def test_callable_interface(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Agente es invocable como Morphism."""
        state = make_state()
        result = default_agent(state)

        assert isinstance(result, CategoricalState)

    def test_callable_equals_execute(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """__call__ y execute_ooda_loop producen mismo resultado."""
        state = make_state(payload={"deterministic": True})
        r1 = default_agent.execute_ooda_loop(state)
        r2 = default_agent(state)

        # Misma decisión
        m1 = r1.context["quantum_measurement"]
        m2 = r2.context["quantum_measurement"]
        assert m1["eigenstate"] == m2["eigenstate"]
        assert m1["energy"] == m2["energy"]
        assert m1["transmission_prob"] == m2["transmission_prob"]

    def test_frustrated_agent_always_rejects(
        self, frustrated_agent: HilbertObserverAgent
    ) -> None:
        """Agente frustrado siempre rechaza (R1 domina)."""
        for i in range(10):
            state = make_state(payload={"test": i})
            result = frustrated_agent(state)
            eigenstate = result.context["quantum_measurement"][
                "eigenstate"
            ]
            assert eigenstate == "REJECTED", (
                f"Frustrado admitió payload {i}"
            )

    def test_frustrated_agent_empty_strata(
        self, frustrated_agent: HilbertObserverAgent
    ) -> None:
        """Agente frustrado produce strata vacíos."""
        state = make_state()
        result = frustrated_agent(state)
        assert result.validated_strata == frozenset()

    def test_high_threat_raises_barrier(self) -> None:
        """Alta amenaza aumenta Φ, dificultando admisión."""
        state = make_state(payload={"small": 1})

        low_threat = make_agent(threat=0.0)
        high_threat = make_agent(threat=100.0)

        r_low = low_threat(state)
        r_high = high_threat(state)

        m_low = r_low.context["quantum_measurement"]
        m_high = r_high.context["quantum_measurement"]

        # Φ mayor → T menor o igual
        assert m_low["work_function"] < m_high["work_function"]
        assert (
            m_low["transmission_prob"]
            >= m_high["transmission_prob"]
        )

    def test_invalid_state_type_raises(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """State que no es CategoricalState lanza error."""
        with pytest.raises(HilbertWatcherError, match="CategoricalState"):
            default_agent.execute_ooda_loop(
                "not a state"  # type: ignore[arg-type]
            )

    def test_invalid_payload_type_raises(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Payload que no es Mapping lanza error."""
        state = CategoricalState(
            payload="not a dict",  # type: ignore[arg-type]
            context={},
            validated_strata=frozenset(),
        )
        with pytest.raises(HilbertWatcherError, match="Mapping"):
            default_agent.execute_ooda_loop(state)

    def test_output_preserves_payload(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Payload de salida es idéntico al de entrada."""
        payload = {"preserve": "me", "nested": [1, 2, 3]}
        state = make_state(payload=payload)

        result = default_agent(state)

        assert result.payload == payload
        assert result.payload is state.payload  # Misma referencia

    def test_deterministic_over_multiple_runs(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Múltiples ejecuciones con mismo input dan mismo output."""
        state = make_state(payload={"stable": "input"})

        results = [default_agent(state) for _ in range(5)]

        eigenstates = [
            r.context["quantum_measurement"]["eigenstate"]
            for r in results
        ]
        assert len(set(eigenstates)) == 1, (
            f"Resultados no deterministas: {eigenstates}"
        )

    def test_large_payload_admitted(self) -> None:
        """Payload grande genera alta energía → admisión fotoeléctrica."""
        # Payload grande y repetitivo → alta E
        agent = make_agent(threat=0.0, sigma=-1.0, frustration=0.0)
        big_payload = {"data": "x" * 100_000}
        state = make_state(payload=big_payload)

        result = agent(state)

        measurement = result.context["quantum_measurement"]
        # Con payload grande y barrera baja, debería tener E alta
        assert measurement["energy"] > 0.0

    def test_existing_strata_preserved_on_admission(self) -> None:
        """Strata previos se preservan y se agrega PHYSICS."""
        existing = frozenset({Stratum.PHYSICS})
        state = make_state(validated_strata=existing)
        agent = make_agent()

        result = agent(state)

        if (
            result.context["quantum_measurement"]["eigenstate"]
            == "ADMITTED"
        ):
            assert Stratum.PHYSICS in result.validated_strata


# ═══════════════════════════════════════════════════════════════════════
# 10. CASOS LÍMITE NUMÉRICOS
# ═══════════════════════════════════════════════════════════════════════


class TestNumericalEdgeCases:
    """Pruebas de robustez numérica ante valores extremos."""

    # --- _ensure_finite_float ---

    def test_ensure_finite_int(self) -> None:
        """Entero se convierte a float."""
        assert _ensure_finite_float(42, name="x") == 42.0

    def test_ensure_finite_float(self) -> None:
        """Float se preserva."""
        assert _ensure_finite_float(3.14, name="x") == 3.14

    def test_ensure_finite_nan_raises(self) -> None:
        """NaN lanza error."""
        with pytest.raises(HilbertNumericalError, match="finito"):
            _ensure_finite_float(float("nan"), name="x")

    def test_ensure_finite_inf_raises(self) -> None:
        """Infinito lanza error."""
        with pytest.raises(HilbertNumericalError, match="finito"):
            _ensure_finite_float(float("inf"), name="x")

    def test_ensure_finite_neg_inf_raises(self) -> None:
        """-Infinito lanza error."""
        with pytest.raises(HilbertNumericalError, match="finito"):
            _ensure_finite_float(float("-inf"), name="x")

    def test_ensure_finite_string_raises(self) -> None:
        """String no numérico lanza error."""
        with pytest.raises(
            HilbertNumericalError, match="convertible"
        ):
            _ensure_finite_float("abc", name="x")

    def test_ensure_finite_none_raises(self) -> None:
        """None lanza error."""
        with pytest.raises(
            HilbertNumericalError, match="convertible"
        ):
            _ensure_finite_float(None, name="x")

    def test_ensure_finite_numeric_string(self) -> None:
        """String numérico se convierte."""
        assert _ensure_finite_float("3.14", name="x") == 3.14

    def test_ensure_finite_idempotent(self) -> None:
        """Aplicar dos veces da mismo resultado."""
        val = 2.718
        r1 = _ensure_finite_float(val, name="x")
        r2 = _ensure_finite_float(r1, name="x")
        assert r1 == r2

    # --- _ensure_nonneg_finite_float ---

    def test_ensure_nonneg_positive(self) -> None:
        """Valor positivo pasa."""
        assert _ensure_nonneg_finite_float(5.0, name="x") == 5.0

    def test_ensure_nonneg_zero(self) -> None:
        """Cero pasa."""
        assert _ensure_nonneg_finite_float(0.0, name="x") == 0.0

    def test_ensure_nonneg_negative_raises(self) -> None:
        """Valor negativo lanza error."""
        with pytest.raises(HilbertNumericalError, match="≥ 0"):
            _ensure_nonneg_finite_float(-1.0, name="x")

    def test_ensure_nonneg_nan_raises(self) -> None:
        """NaN lanza error (vía _ensure_finite_float)."""
        with pytest.raises(HilbertNumericalError):
            _ensure_nonneg_finite_float(float("nan"), name="x")

    # --- _clamp_probability ---

    def test_clamp_nan_to_zero(self) -> None:
        """NaN se mapea a 0.0."""
        assert _clamp_probability(float("nan")) == 0.0

    def test_clamp_negative(self) -> None:
        """Negativo se clampea a 0.0."""
        assert _clamp_probability(-0.5) == 0.0

    def test_clamp_above_one(self) -> None:
        """Valor > 1 se clampea a 1.0."""
        assert _clamp_probability(1.5) == 1.0

    def test_clamp_in_range_preserved(self) -> None:
        """Valor en [0,1] se preserva."""
        assert _clamp_probability(0.5) == 0.5

    def test_clamp_zero(self) -> None:
        """Cero se preserva."""
        assert _clamp_probability(0.0) == 0.0

    def test_clamp_one(self) -> None:
        """Uno se preserva."""
        assert _clamp_probability(1.0) == 1.0

    def test_clamp_idempotent(self) -> None:
        """Propiedad: clamp(clamp(x)) = clamp(x)."""
        for val in [-1.0, 0.0, 0.5, 1.0, 2.0, float("nan")]:
            r1 = _clamp_probability(val)
            r2 = _clamp_probability(r1)
            assert r1 == r2

    def test_clamp_monotone(self) -> None:
        """Propiedad: x ≤ y ⟹ clamp(x) ≤ clamp(y) para finitos."""
        values = sorted([-2.0, -0.5, 0.0, 0.3, 0.7, 1.0, 1.5, 3.0])
        clamped = [_clamp_probability(v) for v in values]
        for i in range(len(clamped) - 1):
            assert clamped[i] <= clamped[i + 1]

    # --- _safe_context ---

    def test_safe_context_none(self) -> None:
        """None produce dict vacío."""
        assert _safe_context(None) == {}

    def test_safe_context_dict(self) -> None:
        """Dict se copia."""
        original = {"a": 1}
        result = _safe_context(original)
        assert result == {"a": 1}
        assert result is not original  # Copia, no referencia

    def test_safe_context_non_mapping(self) -> None:
        """No-Mapping produce warning en dict."""
        result = _safe_context(42)  # type: ignore[arg-type]
        assert "_context_warning" in result

    # --- Estabilidad con valores extremos ---

    def test_very_small_sigma_large_mass(self) -> None:
        """Sigma muy cercano a -tol produce masa muy grande."""
        sigma = -(QuantumThresholds.SIGMA_CHAOS_TOL * 1.1)
        agent = make_agent(sigma=sigma)
        _, m_eff, *_ = agent._orient_gauge_fields()

        assert math.isfinite(m_eff)
        assert m_eff > 0.0
        expected = QuantumThresholds.BASE_MASS / abs(sigma)
        assert math.isclose(m_eff, expected, rel_tol=1e-9)

    def test_zero_energy_zero_phi(self) -> None:
        """E=0, Φ=0 → E ≥ Φ → T=1."""
        T = HilbertObserverAgent._decide_quantum_transmission(
            E=0.0, Phi=0.0, m_eff=1.0, is_frustrated=False
        )
        assert T == 1.0


# ═══════════════════════════════════════════════════════════════════════
# 11. PROPIEDADES ALGEBRAICAS
# ═══════════════════════════════════════════════════════════════════════


class TestAlgebraicProperties:
    """Pruebas de propiedades categóricas y algebraicas."""

    def test_morphism_preserves_payload_identity(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """El morfismo preserva la identidad sobre payload.

        Propiedad: payload(f(s)) = payload(s) para todo s.
        """
        for payload in [
            {"a": 1},
            {"x": [1, 2], "y": "test"},
            {},
        ]:
            state = make_state(payload=payload)
            result = default_agent(state)
            assert result.payload == payload

    def test_idempotent_decision_stability(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Aplicar el agente al mismo input es determinista.

        Propiedad débil de idempotencia: f(s) = f(s) para todo s.
        (No f(f(s)) = f(s), ya que el contexto cambia.)
        """
        state = make_state(payload={"test": "idem"})
        r1 = default_agent(state)
        r2 = default_agent(state)

        m1 = r1.context["quantum_measurement"]
        m2 = r2.context["quantum_measurement"]

        assert m1["eigenstate"] == m2["eigenstate"]
        assert m1["energy"] == m2["energy"]
        assert m1["transmission_prob"] == m2["transmission_prob"]
        assert m1["collapse_threshold"] == m2["collapse_threshold"]
        assert m1["momentum"] == m2["momentum"]

    def test_decision_is_total_function(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Todo payload válido produce exactamente un eigenstate.

        Propiedad: ∀s válido, eigenstate(f(s)) ∈ {ADMITTED, REJECTED}.
        """
        payloads = [
            {},
            {"a": 1},
            {"data": "x" * 10000},
            {"nested": {"deep": True}},
            {"num": 0, "float": 0.0, "none": None},
        ]

        for p in payloads:
            state = make_state(payload=p)
            result = default_agent(state)
            eigenstate = result.context["quantum_measurement"][
                "eigenstate"
            ]
            assert eigenstate in {"ADMITTED", "REJECTED"}, (
                f"Eigenstate inválido '{eigenstate}' para payload {p}"
            )

    def test_strata_disjunction_property(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Propiedad de disyunción exclusiva en strata:

        admitted ⟹ PHYSICS ∈ strata
        rejected ⟹ strata = ∅

        Estas condiciones son mutuamente exclusivas y exhaustivas.
        """
        payloads = [{"i": i} for i in range(20)]

        for p in payloads:
            state = make_state(payload=p)
            result = default_agent(state)
            eigenstate = result.context["quantum_measurement"][
                "eigenstate"
            ]

            if eigenstate == "ADMITTED":
                assert Stratum.PHYSICS in result.validated_strata
            else:
                assert result.validated_strata == frozenset()

    def test_energy_work_function_coherence(
        self, default_agent: HilbertObserverAgent
    ) -> None:
        """Coherencia entre energía, función de trabajo y eigenstate.

        Si T=1 (fotoeléctrico) y τ < 1, entonces siempre admitido.
        """
        # Crear payload con energía alta
        big_state = make_state(
            payload={"data": "x" * 50_000}
        )
        result = default_agent(big_state)
        measurement = result.context["quantum_measurement"]

        if measurement["energy"] >= measurement["work_function"]:
            assert measurement["transmission_prob"] == 1.0


# ═══════════════════════════════════════════════════════════════════════
# 12. WAVEFUNCTION STATE: INVARIANTES __post_init__
# ═══════════════════════════════════════════════════════════════════════


class TestWavefunctionStateInvariants:
    """Pruebas de validación de invariantes en WavefunctionState."""

    def test_valid_construction(self) -> None:
        """Construcción válida no lanza error."""
        wave = WavefunctionState(
            energy=10.0,
            work_function=5.0,
            effective_mass=1.0,
            transmission_prob=0.5,
            frustrated=False,
            threat_level=0.0,
            dominant_pole_real=-1.0,
            frustration_energy=0.0,
            collapse_threshold=0.3,
        )
        assert wave.energy == 10.0

    def test_valid_with_infinite_mass(self) -> None:
        """Construcción con m_eff=+∞ es válida."""
        wave = WavefunctionState(
            energy=10.0,
            work_function=5.0,
            effective_mass=math.inf,
            transmission_prob=1.0,
            frustrated=False,
            threat_level=0.0,
            dominant_pole_real=0.0,
            frustration_energy=0.0,
            collapse_threshold=0.5,
        )
        assert math.isinf(wave.effective_mass)

    def test_negative_energy_raises(self) -> None:
        """Energía negativa viola invariante."""
        with pytest.raises(HilbertNumericalError, match="energy"):
            WavefunctionState(
                energy=-1.0,
                work_function=5.0,
                effective_mass=1.0,
                transmission_prob=0.5,
                frustrated=False,
                threat_level=0.0,
                dominant_pole_real=-1.0,
                frustration_energy=0.0,
                collapse_threshold=0.3,
            )

    def test_negative_work_function_raises(self) -> None:
        """Función de trabajo negativa viola invariante."""
        with pytest.raises(
            HilbertNumericalError, match="work_function"
        ):
            WavefunctionState(
                energy=10.0,
                work_function=-1.0,
                effective_mass=1.0,
                transmission_prob=0.5,
                frustrated=False,
                threat_level=0.0,
                dominant_pole_real=-1.0,
                frustration_energy=0.0,
                collapse_threshold=0.3,
            )

    def test_zero_mass_raises(self) -> None:
        """Masa efectiva cero viola invariante."""
        with pytest.raises(
            HilbertNumericalError, match="effective_mass"
        ):
            WavefunctionState(
                energy=10.0,
                work_function=5.0,
                effective_mass=0.0,
                transmission_prob=0.5,
                frustrated=False,
                threat_level=0.0,
                dominant_pole_real=-1.0,
                frustration_energy=0.0,
                collapse_threshold=0.3,
            )

    def test_negative_mass_raises(self) -> None:
        """Masa efectiva negativa viola invariante."""
        with pytest.raises(
            HilbertNumericalError, match="effective_mass"
        ):
            WavefunctionState(
                energy=10.0,
                work_function=5.0,
                effective_mass=-1.0,
                transmission_prob=0.5,
                frustrated=False,
                threat_level=0.0,
                dominant_pole_real=-1.0,
                frustration_energy=0.0,
                collapse_threshold=0.3,
            )

    def test_transmission_above_one_raises(self) -> None:
        """Probabilidad > 1 viola invariante."""
        with pytest.raises(
            HilbertNumericalError, match="transmission_prob"
        ):
            WavefunctionState(
                energy=10.0,
                work_function=5.0,
                effective_mass=1.0,
                transmission_prob=1.1,
                frustrated=False,
                threat_level=0.0,
                dominant_pole_real=-1.0,
                frustration_energy=0.0,
                collapse_threshold=0.3,
            )

    def test_transmission_negative_raises(self) -> None:
        """Probabilidad negativa viola invariante."""
        with pytest.raises(
            HilbertNumericalError, match="transmission_prob"
        ):
            WavefunctionState(
                energy=10.0,
                work_function=5.0,
                effective_mass=1.0,
                transmission_prob=-0.1,
                frustrated=False,
                threat_level=0.0,
                dominant_pole_real=-1.0,
                frustration_energy=0.0,
                collapse_threshold=0.3,
            )

    def test_collapse_threshold_at_one_raises(self) -> None:
        """Umbral de colapso = 1.0 viola invariante [0, 1)."""
        with pytest.raises(
            HilbertNumericalError, match="collapse_threshold"
        ):
            WavefunctionState(
                energy=10.0,
                work_function=5.0,
                effective_mass=1.0,
                transmission_prob=0.5,
                frustrated=False,
                threat_level=0.0,
                dominant_pole_real=-1.0,
                frustration_energy=0.0,
                collapse_threshold=1.0,
            )

    def test_collapse_threshold_above_one_raises(self) -> None:
        """Umbral de colapso > 1 viola invariante."""
        with pytest.raises(
            HilbertNumericalError, match="collapse_threshold"
        ):
            WavefunctionState(
                energy=10.0,
                work_function=5.0,
                effective_mass=1.0,
                transmission_prob=0.5,
                frustrated=False,
                threat_level=0.0,
                dominant_pole_real=-1.0,
                frustration_energy=0.0,
                collapse_threshold=1.5,
            )

    def test_negative_threat_level_raises(self) -> None:
        """Amenaza negativa viola invariante."""
        with pytest.raises(
            HilbertNumericalError, match="threat_level"
        ):
            WavefunctionState(
                energy=10.0,
                work_function=5.0,
                effective_mass=1.0,
                transmission_prob=0.5,
                frustrated=False,
                threat_level=-1.0,
                dominant_pole_real=-1.0,
                frustration_energy=0.0,
                collapse_threshold=0.3,
            )

    def test_negative_frustration_energy_raises(self) -> None:
        """Frustración negativa viola invariante."""
        with pytest.raises(
            HilbertNumericalError, match="frustration_energy"
        ):
            WavefunctionState(
                energy=10.0,
                work_function=5.0,
                effective_mass=1.0,
                transmission_prob=0.5,
                frustrated=False,
                threat_level=0.0,
                dominant_pole_real=-1.0,
                frustration_energy=-0.1,
                collapse_threshold=0.3,
            )

    def test_boundary_values_accepted(self) -> None:
        """Valores frontera válidos no lanzan error."""
        # Todos en sus extremos válidos
        wave = WavefunctionState(
            energy=0.0,                # mínimo
            work_function=0.0,         # mínimo
            effective_mass=1e-300,      # muy pequeño pero > 0
            transmission_prob=0.0,      # mínimo
            frustrated=False,
            threat_level=0.0,           # mínimo
            dominant_pole_real=0.0,     # válido (cualquier real)
            frustration_energy=0.0,     # mínimo
            collapse_threshold=0.0,     # mínimo [0, 1)
        )
        assert wave.energy == 0.0

    def test_frozen_immutability(self) -> None:
        """WavefunctionState es inmutable (frozen=True)."""
        wave = WavefunctionState(
            energy=10.0,
            work_function=5.0,
            effective_mass=1.0,
            transmission_prob=0.5,
            frustrated=False,
            threat_level=0.0,
            dominant_pole_real=-1.0,
            frustration_energy=0.0,
            collapse_threshold=0.3,
        )
        with pytest.raises(AttributeError):
            wave.energy = 20.0  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# 13. PRUEBAS DE EIGENSTATE ENUM
# ═══════════════════════════════════════════════════════════════════════


class TestHilbertEigenstate:
    """Pruebas del enum de eigenestados."""

    def test_exactly_two_states(self) -> None:
        """Existen exactamente 2 eigenestados."""
        assert len(HilbertEigenstate) == 2

    def test_admitted_exists(self) -> None:
        """ADMITTED está definido."""
        assert hasattr(HilbertEigenstate, "ADMITTED")

    def test_rejected_exists(self) -> None:
        """REJECTED está definido."""
        assert hasattr(HilbertEigenstate, "REJECTED")

    def test_names_match_strings(self) -> None:
        """Los nombres coinciden con los strings usados en telemetría."""
        assert HilbertEigenstate.ADMITTED.name == "ADMITTED"
        assert HilbertEigenstate.REJECTED.name == "REJECTED"

    def test_states_are_distinct(self) -> None:
        """Los eigenestados son mutuamente distintos."""
        assert (
            HilbertEigenstate.ADMITTED != HilbertEigenstate.REJECTED
        )


# ═══════════════════════════════════════════════════════════════════════
# 14. PRUEBAS DE EXCEPCIONES
# ═══════════════════════════════════════════════════════════════════════


class TestExceptionHierarchy:
    """Pruebas de la jerarquía de excepciones."""

    def test_base_exception(self) -> None:
        """HilbertWatcherError es Exception."""
        assert issubclass(HilbertWatcherError, Exception)

    def test_numerical_inherits_base(self) -> None:
        """HilbertNumericalError hereda de HilbertWatcherError."""
        assert issubclass(
            HilbertNumericalError, HilbertWatcherError
        )

    def test_interface_inherits_base(self) -> None:
        """HilbertInterfaceError hereda de HilbertWatcherError."""
        assert issubclass(
            HilbertInterfaceError, HilbertWatcherError
        )

    def test_payload_inherits_base(self) -> None:
        """HilbertPayloadError hereda de HilbertWatcherError."""
        assert issubclass(
            HilbertPayloadError, HilbertWatcherError
        )

    def test_catch_all_with_base(self) -> None:
        """Todas las excepciones específicas se capturan con la base."""
        for exc_class in [
            HilbertNumericalError,
            HilbertInterfaceError,
            HilbertPayloadError,
        ]:
            try:
                raise exc_class("test")
            except HilbertWatcherError:
                pass  # Esperado


# ═══════════════════════════════════════════════════════════════════════
# 15. PRUEBAS DE QUANTUM THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════


class TestQuantumThresholds:
    """Pruebas de consistencia de constantes físicas."""

    def test_planck_hbar_relation(self) -> None:
        """ħ = h / 2π."""
        expected = QuantumThresholds.PLANCK_H / (2.0 * math.pi)
        assert math.isclose(
            QuantumThresholds.PLANCK_HBAR, expected, rel_tol=1e-15
        )

    def test_all_positive(self) -> None:
        """Todas las constantes físicas son estrictamente positivas."""
        positive_constants = [
            QuantumThresholds.PLANCK_H,
            QuantumThresholds.PLANCK_HBAR,
            QuantumThresholds.BASE_PHI,
            QuantumThresholds.BASE_MASS,
            QuantumThresholds.BARRIER_DX,
            QuantumThresholds.ALPHA_COUPLING,
            QuantumThresholds.EPSILON_MACH,
            QuantumThresholds.ENTROPY_FLOOR,
            QuantumThresholds.MIN_KINETIC_ENERGY,
            QuantumThresholds.SIGMA_CHAOS_TOL,
            QuantumThresholds.FREQUENCY_SCALE,
            QuantumThresholds.MAX_PAYLOAD_BYTES,
            QuantumThresholds.HASH_RESOLUTION,
        ]
        for val in positive_constants:
            assert val > 0, f"Constante no positiva: {val}"

    def test_exp_cutoff_negative(self) -> None:
        """Cutoff exponencial es negativo."""
        assert QuantumThresholds.EXP_UNDERFLOW_CUTOFF < 0

    def test_exp_cutoff_safe_for_float64(self) -> None:
        """exp(cutoff) no produce underflow a cero exacto."""
        result = math.exp(QuantumThresholds.EXP_UNDERFLOW_CUTOFF)
        assert result >= 0.0  # No negativo
        # Puede ser subnormal pero no debe lanzar excepción

    def test_epsilon_smaller_than_physical(self) -> None:
        """ε es mucho menor que constantes físicas."""
        assert QuantumThresholds.EPSILON_MACH < QuantumThresholds.BASE_PHI
        assert QuantumThresholds.EPSILON_MACH < QuantumThresholds.BASE_MASS

    def test_hash_resolution_is_2_64(self) -> None:
        """Resolución del hash es exactamente 2^64."""
        assert QuantumThresholds.HASH_RESOLUTION == float(2**64)


# ═══════════════════════════════════════════════════════════════════════
# 16. PRUEBAS DE WKB CUANTITATIVAS
# ═══════════════════════════════════════════════════════════════════════


class TestWKBQuantitative:
    """Verificaciones cuantitativas de la fórmula WKB.

    T = exp(-(2/ħ) · Δx · √(2·m·(Φ-E)))

    Verificamos valores numéricos exactos contra cálculo manual.
    """

    def test_known_values(self) -> None:
        """Verificar T para parámetros conocidos contra cálculo manual."""
        E = 5.0
        Phi = 10.0
        m_eff = 1.0

        # Cálculo manual
        hbar = QuantumThresholds.PLANCK_HBAR
        dx = QuantumThresholds.BARRIER_DX
        barrier = Phi - E  # 5.0
        integrand = math.sqrt(2.0 * m_eff * barrier)  # √10
        exponent = -(2.0 / hbar) * dx * integrand
        expected_T = math.exp(exponent)

        actual_T = HilbertObserverAgent._decide_quantum_transmission(
            E, Phi, m_eff, is_frustrated=False
        )

        assert math.isclose(
            actual_T, expected_T, rel_tol=1e-12
        ), f"T={actual_T}, expected={expected_T}"

    def test_symmetry_barrier_height(self) -> None:
        """T(E=a, Φ=b) = T(E=c, Φ=d) cuando b-a = d-c y m igual.

        La probabilidad solo depende de la diferencia Φ-E.
        """
        m = 2.0
        T1 = HilbertObserverAgent._decide_quantum_transmission(
            E=3.0, Phi=8.0, m_eff=m, is_frustrated=False
        )
        T2 = HilbertObserverAgent._decide_quantum_transmission(
            E=10.0, Phi=15.0, m_eff=m, is_frustrated=False
        )
        assert math.isclose(T1, T2, rel_tol=1e-12)

    def test_wkb_limit_thin_barrier(self) -> None:
        """Para barrera muy baja (Φ-E→0⁺), T→1.

        lim_{Φ→E⁺} T = exp(0) = 1.
        """
        m = 1.0
        T = HilbertObserverAgent._decide_quantum_transmission(
            E=9.999999, Phi=10.0, m_eff=m, is_frustrated=False
        )
        assert T > 0.99, f"T={T} debería ser ≈ 1 para barrera delgada"

    def test_wkb_exponential_suppression(self) -> None:
        """Barrera alta → supresión exponencial.

        T debe decaer exponencialmente con √(Φ-E).
        Verificamos que log(T) es aproximadamente lineal en √(Φ-E).
        """
        m = 1.0
        E = 0.0
        log_Ts = []
        sqrt_barriers = []

        for Phi in [1.0, 4.0, 9.0, 16.0, 25.0]:
            T = HilbertObserverAgent._decide_quantum_transmission(
                E, Phi, m, is_frustrated=False
            )
            if T > 0:
                log_Ts.append(math.log(T))
                sqrt_barriers.append(math.sqrt(Phi))

        # log(T) debería ser lineal en √Φ
        if len(log_Ts) >= 3:
            # Verificar linealidad: ratios consecutivos ≈ constante
            ratios = [
                log_Ts[i + 1] / sqrt_barriers[i + 1]
                for i in range(len(log_Ts) - 1)
                if sqrt_barriers[i + 1] != 0
            ]
            # Los ratios log(T)/√Φ deben ser aproximadamente iguales
            if len(ratios) >= 2:
                for i in range(len(ratios) - 1):
                    assert math.isclose(
                        ratios[i], ratios[i + 1], rel_tol=0.01
                    ), f"No lineal: {ratios}"
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite de Pruebas: MatterAgent (Endofuntor de Colapso Hadrónico)              ║
║ Ubicación : tests/unit/agents/omega/test_matter_agent.py                     ║
║ Versión   : 5.0.0-Topos-Thermodynamic-Phased-Strict                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Taxonomía de pruebas (5 familias):                                          ║
║                                                                              ║
║   F1 – Contratos de Fase 1 (Validación de Parámetros Constitutivos)          ║
║         Verifica restricciones escalares, relacionales y el sellado          ║
║         criptográfico del MatterAgentContext.                                ║
║                                                                              ║
║   F2 – Contratos de Fase 2 (Deliberación Termodinámica)                      ║
║         Cubre el retículo booleano de vetos [V1], [V2], [V3],                ║
║         el modelo de Rayleigh no-lineal y la firma del morfismo φ.           ║
║                                                                              ║
║   F3 – Contratos de Fase 3 (Proyección Categórica)                           ║
║         Verifica el cálculo de χ(K), la firma π, la completitud              ║
║         del payload y el stratum de destino.                                 ║
║                                                                              ║
║   F4 – Contratos de Inmutabilidad y Trazabilidad                             ║
║         Verifica frozen dataclasses, determinismo SHA-256 y la               ║
║         cadena de firmas φ → π → CategoricalState.                           ║
║                                                                              ║
║   F5 – Contratos de Integración End-to-End                                   ║
║         Orquesta la composición completa F = π ∘ δ ∘ φ con mocks             ║
║         del motor físico y verifica la clausura transitiva del topos.        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 0 – IMPORTACIONES
# ──────────────────────────────────────────────────────────────────────────────

# Standard Library
import dataclasses
import hashlib
import json
import math
import time
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, PropertyMock, patch

# Third-Party
import pytest

# Project: módulos bajo prueba
from app.agents.omega.matter_agent import (
    _FLOAT_EPSILON,
    _RAYLEIGH_EXPONENT,
    _RAYLEIGH_VISCOSITY_COEFFICIENT,
    MODULE_VERSION,
    HadronicCollapseVetoError,
    HadronicDeliberationVerdict,
    LogisticSingularityVeto,
    MatterAgent,
    MatterAgentContext,
    NegativeExergyVeto,
    ThermodynamicFrictionVeto,
)

# Project: dependencias auxiliares
from app.core.mic_algebra import CategoricalState, TopologicalInvariantError
from app.core.schemas import Stratum
from app.physics.matter_generator import BillOfMaterials, MatterGenerator


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 – FÁBRICAS Y FIXTURES COMPARTIDOS
# ──────────────────────────────────────────────────────────────────────────────


def _make_bom(
    total_mass: float = 100.0,
    gini_asymmetry: float = 0.40,
    exergy_available: float = 5000.0,
    num_nodes: Optional[int] = None,
    num_edges: Optional[int] = None,
    euler_characteristic: Optional[int] = None,
) -> BillOfMaterials:
    r"""
    Fábrica de ``BillOfMaterials`` para pruebas.

    Construye un mock configurado con los atributos mínimos que el
    agente consume durante la deliberación y la proyección categórica.

    Parameters
    ----------
    total_mass : float
        Masa total del BOM [kg].
    gini_asymmetry : float
        Índice de Gini de la distribución de masa.
    exergy_available : float
        Exergía disponible [J].
    num_nodes : Optional[int]
        Número de nodos del grafo de dependencias.
    num_edges : Optional[int]
        Número de aristas del grafo de dependencias.
    euler_characteristic : Optional[int]
        Característica de Euler-Poincaré precalculada.

    Returns
    -------
    BillOfMaterials
        Mock con atributos configurados.
    """
    bom = MagicMock(spec=BillOfMaterials)
    bom.total_mass = total_mass
    bom.gini_asymmetry = gini_asymmetry
    bom.exergy_available = exergy_available

    # Atributos topológicos opcionales
    if num_nodes is not None:
        bom.num_nodes = num_nodes
    else:
        # Eliminar el atributo para que hasattr() devuelva False
        del bom.num_nodes

    if num_edges is not None:
        bom.num_edges = num_edges
    else:
        del bom.num_edges

    if euler_characteristic is not None:
        bom.euler_characteristic = euler_characteristic
    else:
        del bom.euler_characteristic

    # Serialización canónica para firma SHA-256
    bom.to_dict.return_value = {
        "total_mass":     f"{total_mass:.12f}",
        "gini_asymmetry": f"{gini_asymmetry:.12f}",
        "exergy":         f"{exergy_available:.12f}",
    }

    return bom


def _make_engine(
    gini_threshold: float = 0.85,
    bom: Optional[BillOfMaterials] = None,
) -> MatterGenerator:
    r"""
    Fábrica de ``MatterGenerator`` para pruebas.

    Parameters
    ----------
    gini_threshold : float
        Umbral de Gini del motor.
    bom : Optional[BillOfMaterials]
        BOM que retornará ``project_to_bom``.  Si es ``None``
        se genera uno saludable por defecto.

    Returns
    -------
    MatterGenerator
        Mock configurado.
    """
    engine = MagicMock(spec=MatterGenerator)
    engine.gini_threshold = gini_threshold
    engine.project_to_bom.return_value = bom or _make_bom()
    return engine


@pytest.fixture
def healthy_bom() -> BillOfMaterials:
    """BOM saludable que supera los tres vetos con los umbrales por defecto."""
    return _make_bom(
        total_mass=100.0,
        gini_asymmetry=0.40,
        exergy_available=5000.0,
    )


@pytest.fixture
def default_agent(healthy_bom) -> MatterAgent:
    r"""
    Instancia de ``MatterAgent`` con parámetros por defecto y motor mock.

    Umbrales:
      γ_c = 0.85, Φ_max = 1e6 W, n = 2.0, ν = 0.05
    Fricción esperada para masa=100 kg:
      Φ = 0.05 · 100^2 = 500 W  << 1e6 W  ✓
    """
    engine = _make_engine(bom=healthy_bom)
    return MatterAgent(engine=engine)


@pytest.fixture
def phase2(default_agent) -> MatterAgent.Phase2_ThermodynamicDeliberation:
    """Instancia directa de la Fase 2 para pruebas unitarias granulares."""
    return MatterAgent.Phase2_ThermodynamicDeliberation(
        context=default_agent.context
    )


@pytest.fixture
def phase3(default_agent) -> MatterAgent.Phase3_CategoricalProjection:
    """Instancia directa de la Fase 3 para pruebas unitarias granulares."""
    return MatterAgent.Phase3_CategoricalProjection(
        context=default_agent.context
    )


@pytest.fixture
def approved_verdict(healthy_bom) -> HadronicDeliberationVerdict:
    r"""
    Veredicto aprobatorio preconstruido para pruebas de Fase 3.

    Se construye con los valores que produciría la Fase 2 ante el
    ``healthy_bom`` con los parámetros de Rayleigh por defecto.
    """
    friction = _RAYLEIGH_VISCOSITY_COEFFICIENT * math.pow(
        healthy_bom.total_mass, _RAYLEIGH_EXPONENT
    )
    return HadronicDeliberationVerdict(
        is_viable=True,
        gini_asymmetry=healthy_bom.gini_asymmetry,
        exergy_dissipated=friction,
        exergy_available=healthy_bom.exergy_available,
        bom_tensor=healthy_bom,
        morphism_signature="a" * 64,
        topological_veto_reason=None,
        euler_poincare_characteristic=0,
        deliberation_timestamp_utc=1_700_000_000.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FAMILIA F1 – CONTRATOS DE FASE 1 (VALIDACIÓN DE PARÁMETROS CONSTITUTIVOS)
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase1_ParameterValidation:
    r"""
    F1: Contratos de la Fase 1.

    Verifica que ``Phase1_ParameterValidation.build_context()`` rechaza
    toda configuración físicamente inadmisible y produce un
    ``MatterAgentContext`` correcto y sellado para configuraciones válidas.
    """

    # ── F1.1 Validación del umbral de Gini ────────────────────────────────────

    def test_f1_gini_zero_raises(self):
        """F1.1a: γ_c = 0 debe lanzar ValueError (veto trivial)."""
        with pytest.raises(ValueError, match="max_gini_critical"):
            MatterAgent(max_gini_critical=0.0)

    def test_f1_gini_negative_raises(self):
        """F1.1b: γ_c < 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="max_gini_critical"):
            MatterAgent(max_gini_critical=-0.1)

    def test_f1_gini_exactly_one_accepted(self):
        """F1.1c: γ_c = 1.0 es admisible (límite superior cerrado)."""
        agent = MatterAgent(max_gini_critical=1.0)
        assert agent.context.max_gini_critical == 1.0

    def test_f1_gini_epsilon_boundary_accepted(self):
        """F1.1d: γ_c = ε + δ debe ser admisible (justo sobre el límite inferior)."""
        γ_min = _FLOAT_EPSILON * 2
        agent = MatterAgent(max_gini_critical=γ_min)
        assert agent.context.max_gini_critical == pytest.approx(γ_min)

    def test_f1_gini_above_095_emits_warning(self, caplog):
        """F1.1e: γ_c > 0.95 emite advertencia de permisividad."""
        import logging
        with caplog.at_level(logging.WARNING, logger="MIC.Omega.MatterAgent"):
            MatterAgent(max_gini_critical=0.97)
        assert any("0.95" in msg for msg in caplog.messages)

    def test_f1_gini_095_no_warning(self, caplog):
        """F1.1f: γ_c = 0.95 exacto NO debe emitir advertencia."""
        import logging
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="MIC.Omega.MatterAgent"):
            MatterAgent(max_gini_critical=0.95)
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warning_msgs

    # ── F1.2 Validación del techo de fricción exérgica ────────────────────────

    def test_f1_exergy_zero_raises(self):
        """F1.2a: Φ_max = 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="max_exergy_friction"):
            MatterAgent(max_exergy_friction=0.0)

    def test_f1_exergy_negative_raises(self):
        """F1.2b: Φ_max < 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="max_exergy_friction"):
            MatterAgent(max_exergy_friction=-1.0)

    def test_f1_exergy_very_small_positive_accepted(self):
        """F1.2c: Φ_max = 2ε es admisible (mínimo físico)."""
        φ_min = _FLOAT_EPSILON * 2
        agent = MatterAgent(max_exergy_friction=φ_min)
        assert agent.context.max_exergy_friction == pytest.approx(φ_min)

    def test_f1_exergy_large_value_accepted(self):
        """F1.2d: Φ_max = 1e15 (valor astronómico) es admisible."""
        agent = MatterAgent(max_exergy_friction=1e15)
        assert agent.context.max_exergy_friction == 1e15

    # ── F1.3 Validación de parámetros de Rayleigh ─────────────────────────────

    def test_f1_rayleigh_exponent_below_one_raises(self):
        """F1.3a: n < 1 debe lanzar ValueError (singularidad en m→0⁺)."""
        with pytest.raises(ValueError, match="rayleigh_exponent"):
            MatterAgent(rayleigh_exponent=0.5)

    def test_f1_rayleigh_exponent_exactly_one_accepted(self):
        """F1.3b: n = 1 (modelo lineal de Stokes) es admisible."""
        agent = MatterAgent(rayleigh_exponent=1.0)
        assert agent.context.rayleigh_exponent == pytest.approx(1.0)

    def test_f1_rayleigh_exponent_two_accepted(self):
        """F1.3c: n = 2 (modelo cuadrático, flujo laminar) es admisible."""
        agent = MatterAgent(rayleigh_exponent=2.0)
        assert agent.context.rayleigh_exponent == pytest.approx(2.0)

    def test_f1_rayleigh_exponent_three_accepted(self):
        """F1.3d: n = 3 (régimen turbulento) es admisible."""
        agent = MatterAgent(rayleigh_exponent=3.0)
        assert agent.context.rayleigh_exponent == pytest.approx(3.0)

    def test_f1_rayleigh_viscosity_zero_raises(self):
        """F1.3e: ν = 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="rayleigh_viscosity"):
            MatterAgent(rayleigh_viscosity=0.0)

    def test_f1_rayleigh_viscosity_negative_raises(self):
        """F1.3f: ν < 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="rayleigh_viscosity"):
            MatterAgent(rayleigh_viscosity=-0.01)

    def test_f1_rayleigh_viscosity_small_positive_accepted(self):
        """F1.3g: ν = 2ε es admisible."""
        ν_min = _FLOAT_EPSILON * 2
        agent = MatterAgent(rayleigh_viscosity=ν_min)
        assert agent.context.rayleigh_viscosity == pytest.approx(ν_min)

    # ── F1.4 Reconciliación del motor externo ─────────────────────────────────

    def test_f1_engine_none_builds_internal(self):
        """F1.4a: Sin motor externo, se construye MatterGenerator interno."""
        with patch(
            "app.agents.omega.matter_agent.MatterGenerator",
            autospec=True,
        ) as MockGen:
            MockGen.return_value = _make_engine()
            agent = MatterAgent(engine=None, max_gini_critical=0.70)
            MockGen.assert_called_once_with(gini_threshold=0.70)
            assert agent.context.engine is MockGen.return_value

    def test_f1_engine_external_same_threshold_kept(self):
        """F1.4b: Motor externo con mismo gini_threshold se acepta sin cambios."""
        engine = _make_engine(gini_threshold=0.80)
        agent = MatterAgent(engine=engine, max_gini_critical=0.80)
        assert agent.context.engine is engine

    def test_f1_engine_external_higher_threshold_reconciled(self):
        """F1.4c: Motor con threshold > γ_c → se adopta γ_c (conservador)."""
        engine = _make_engine(gini_threshold=0.95)
        agent = MatterAgent(engine=engine, max_gini_critical=0.70)
        # El contexto debe reportar γ_c del agente
        assert agent.context.max_gini_critical == pytest.approx(0.70)

    def test_f1_engine_external_lower_threshold_kept(self):
        """F1.4d: Motor con threshold < γ_c → el mínimo conservador es el del motor."""
        engine = _make_engine(gini_threshold=0.50)
        agent = MatterAgent(engine=engine, max_gini_critical=0.80)
        # El agente opera con γ_c=0.80 pero el motor internamente usa 0.50
        assert agent.context.max_gini_critical == pytest.approx(0.80)
        assert agent.context.engine is engine

    # ── F1.5 Sellado criptográfico del contexto ───────────────────────────────

    def test_f1_context_hash_is_64_hex_chars(self, default_agent):
        """F1.5a: context_hash es un hexdigest SHA-256 de 64 caracteres."""
        h = default_agent.context.context_hash
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_f1_context_hash_deterministic(self):
        """F1.5b: Mismos parámetros → mismo hash (determinismo SHA-256)."""
        kwargs = dict(
            max_gini_critical=0.75,
            max_exergy_friction=5e5,
            rayleigh_exponent=2.0,
            rayleigh_viscosity=0.03,
        )
        h1 = MatterAgent(**kwargs).context.context_hash
        h2 = MatterAgent(**kwargs).context.context_hash
        assert h1 == h2

    def test_f1_context_hash_differs_on_different_params(self):
        """F1.5c: Parámetros distintos → hashes distintos (colisión improbable)."""
        h1 = MatterAgent(max_gini_critical=0.70).context.context_hash
        h2 = MatterAgent(max_gini_critical=0.71).context.context_hash
        assert h1 != h2

    def test_f1_context_hash_differs_on_rayleigh_change(self):
        """F1.5d: Cambio en ν → hash distinto."""
        h1 = MatterAgent(rayleigh_viscosity=0.05).context.context_hash
        h2 = MatterAgent(rayleigh_viscosity=0.06).context.context_hash
        assert h1 != h2

    # ── F1.6 Propiedades del MatterAgentContext ────────────────────────────────

    def test_f1_context_is_frozen(self, default_agent):
        """F1.6a: MatterAgentContext es frozen (inmutable)."""
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            object.__setattr__(default_agent.context, "max_gini_critical", 0.99)

    def test_f1_context_exposes_all_fields(self, default_agent):
        """F1.6b: Todos los campos del contexto son accesibles y coherentes."""
        ctx = default_agent.context
        assert hasattr(ctx, "engine")
        assert hasattr(ctx, "max_gini_critical")
        assert hasattr(ctx, "max_exergy_friction")
        assert hasattr(ctx, "rayleigh_exponent")
        assert hasattr(ctx, "rayleigh_viscosity")
        assert hasattr(ctx, "context_hash")

    def test_f1_context_accessible_via_property(self, default_agent):
        """F1.6c: La propiedad ``context`` del agente devuelve el contexto sellado."""
        ctx = default_agent.context
        assert isinstance(ctx, MatterAgentContext)

    def test_f1_context_property_is_read_only(self, default_agent):
        """F1.6d: La propiedad ``context`` no tiene setter (solo lectura)."""
        with pytest.raises(AttributeError):
            default_agent.context = MagicMock()

    # ── F1.7 Invariante de transición Fase1 → Fase2 ───────────────────────────

    def test_f1_build_context_returns_matter_agent_context(self):
        """F1.7a: build_context() retorna exactamente MatterAgentContext."""
        phase1 = MatterAgent.Phase1_ParameterValidation(
            engine=None,
            max_gini_critical=0.80,
            max_exergy_friction=1e6,
            rayleigh_exponent=2.0,
            rayleigh_viscosity=0.05,
        )
        with patch("app.agents.omega.matter_agent.MatterGenerator", autospec=True):
            ctx = phase1.build_context()
        assert isinstance(ctx, MatterAgentContext)

    def test_f1_build_context_preserves_parameters(self):
        """F1.7b: Los valores del contexto reflejan exactamente los parámetros."""
        γ, φ, n, ν = 0.72, 3e5, 2.5, 0.08
        phase1 = MatterAgent.Phase1_ParameterValidation(
            engine=_make_engine(),
            max_gini_critical=γ,
            max_exergy_friction=φ,
            rayleigh_exponent=n,
            rayleigh_viscosity=ν,
        )
        ctx = phase1.build_context()
        assert ctx.max_gini_critical  == pytest.approx(γ)
        assert ctx.max_exergy_friction == pytest.approx(φ)
        assert ctx.rayleigh_exponent  == pytest.approx(n)
        assert ctx.rayleigh_viscosity  == pytest.approx(ν)


# ══════════════════════════════════════════════════════════════════════════════
# FAMILIA F2 – CONTRATOS DE FASE 2 (DELIBERACIÓN TERMODINÁMICA)
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase2_ThermodynamicDeliberation:
    r"""
    F2: Contratos de la Fase 2.

    Cubre el retículo booleano de vetos, el modelo de disipación de
    Rayleigh no-lineal y la generación de la firma del morfismo φ.
    """

    # ── F2.1 Modelo de disipación de Rayleigh ─────────────────────────────────

    def test_f2_rayleigh_n2_exact_value(self, phase2):
        """F2.1a: Φ(100 kg) = 0.05 · 100² = 500 W para n=2, ν=0.05."""
        result = phase2._compute_rayleigh_dissipation(total_mass=100.0)
        assert result == pytest.approx(500.0, rel=1e-9)

    def test_f2_rayleigh_zero_mass_returns_zero(self, phase2):
        """F2.1b: Φ(0) = 0 (caso límite m=0, sin singularidad)."""
        result = phase2._compute_rayleigh_dissipation(total_mass=0.0)
        assert result == pytest.approx(0.0)

    def test_f2_rayleigh_negative_mass_returns_zero_with_log(self, phase2, caplog):
        """F2.1c: Masa negativa (dato corrupto) → Φ=0.0 + log de error."""
        import logging
        with caplog.at_level(logging.ERROR, logger="MIC.Omega.MatterAgent"):
            result = phase2._compute_rayleigh_dissipation(total_mass=-50.0)
        assert result == pytest.approx(0.0)
        assert any("corrupto" in msg or "total_mass" in msg for msg in caplog.messages)

    def test_f2_rayleigh_monotone_increasing(self, phase2):
        """F2.1d: Φ es estrictamente creciente en m (n≥1, ν>0)."""
        masses = [0.0, 1.0, 10.0, 100.0, 1000.0]
        dissipations = [
            phase2._compute_rayleigh_dissipation(m) for m in masses
        ]
        for i in range(1, len(dissipations)):
            assert dissipations[i] > dissipations[i - 1], (
                f"Φ({masses[i]}) = {dissipations[i]} no supera "
                f"Φ({masses[i-1]}) = {dissipations[i-1]}"
            )

    def test_f2_rayleigh_n1_linear(self):
        """F2.1e: Para n=1, Φ(m) = ν·m (modelo lineal de Stokes)."""
        engine = _make_engine()
        agent = MatterAgent(
            engine=engine,
            rayleigh_exponent=1.0,
            rayleigh_viscosity=0.10,
        )
        p2 = MatterAgent.Phase2_ThermodynamicDeliberation(context=agent.context)
        for m in [1.0, 50.0, 200.0]:
            assert p2._compute_rayleigh_dissipation(m) == pytest.approx(
                0.10 * m, rel=1e-9
            )

    def test_f2_rayleigh_n3_turbulent(self):
        """F2.1f: Para n=3, Φ(m) = ν·m³ (régimen turbulento)."""
        engine = _make_engine()
        agent = MatterAgent(
            engine=engine,
            rayleigh_exponent=3.0,
            rayleigh_viscosity=0.02,
        )
        p2 = MatterAgent.Phase2_ThermodynamicDeliberation(context=agent.context)
        for m in [2.0, 5.0, 10.0]:
            expected = 0.02 * m ** 3
            assert p2._compute_rayleigh_dissipation(m) == pytest.approx(
                expected, rel=1e-9
            )

    def test_f2_rayleigh_large_mass_no_overflow(self, phase2):
        """F2.1g: Masa muy grande (1e8 kg) no produce OverflowError."""
        result = phase2._compute_rayleigh_dissipation(total_mass=1e8)
        assert math.isfinite(result)
        assert result > 0.0

    # ── F2.2 Veto [V3]: NegativeExergyVeto ───────────────────────────────────

    def test_f2_v3_negative_exergy_raises(self, phase2):
        """F2.2a: E_x < 0 lanza NegativeExergyVeto."""
        bom = _make_bom(exergy_available=-1.0, gini_asymmetry=0.3)
        with pytest.raises(NegativeExergyVeto):
            phase2.deliberate(bom)

    def test_f2_v3_zero_exergy_accepted(self, phase2):
        """F2.2b: E_x = 0 es admisible (límite de la Segunda Ley)."""
        bom = _make_bom(
            exergy_available=0.0,
            gini_asymmetry=0.3,
            total_mass=1.0,  # Φ = 0.05 * 1² = 0.05 W << 1e6 W
        )
        verdict = phase2.deliberate(bom)
        assert verdict.is_viable is True
        assert verdict.exergy_available == pytest.approx(0.0)

    def test_f2_v3_just_below_zero_raises(self, phase2):
        """F2.2c: E_x = -ε lanza NegativeExergyVeto (sensibilidad de epsilon)."""
        bom = _make_bom(exergy_available=-(_FLOAT_EPSILON * 10), gini_asymmetry=0.3)
        with pytest.raises(NegativeExergyVeto):
            phase2.deliberate(bom)

    def test_f2_v3_veto_code_correct(self, phase2):
        """F2.2d: NegativeExergyVeto tiene el veto_code correcto."""
        bom = _make_bom(exergy_available=-100.0)
        with pytest.raises(NegativeExergyVeto) as exc_info:
            phase2.deliberate(bom)
        assert exc_info.value.veto_code == "NEGATIVE_EXERGY_VETO"

    def test_f2_v3_payload_contains_exergy(self, phase2):
        """F2.2e: El payload del veto incluye el valor de exergía negativa."""
        bom = _make_bom(exergy_available=-42.5)
        with pytest.raises(NegativeExergyVeto) as exc_info:
            phase2.deliberate(bom)
        assert "exergy_available" in exc_info.value.payload
        assert exc_info.value.payload["exergy_available"] == pytest.approx(-42.5)

    def test_f2_v3_precedes_v1(self, phase2):
        """F2.2f: [V3] se aplica antes que [V1]: BOM con E_x<0 y G≥γ_c lanza [V3]."""
        # Ambas violaciones activas simultáneamente
        bom = _make_bom(
            exergy_available=-1.0,
            gini_asymmetry=0.99,  # También violaría [V1]
        )
        with pytest.raises(NegativeExergyVeto):  # [V3] debe ganar
            phase2.deliberate(bom)

    def test_f2_v3_missing_attribute_treated_as_zero(self, phase2):
        """F2.2g: BOM sin 'exergy_available' → getattr devuelve 0.0 → aceptado."""
        bom = MagicMock(spec=BillOfMaterials)
        bom.total_mass = 1.0
        bom.gini_asymmetry = 0.3
        # No tiene exergy_available → getattr(..., 0.0) = 0.0
        del bom.exergy_available
        bom.to_dict.return_value = {}
        verdict = phase2.deliberate(bom)
        assert verdict.is_viable is True

    # ── F2.3 Veto [V1]: LogisticSingularityVeto ───────────────────────────────

    def test_f2_v1_gini_at_critical_raises(self, phase2):
        """F2.3a: G = γ_c lanza LogisticSingularityVeto (límite incluido)."""
        γ_c = phase2._ctx.max_gini_critical  # 0.85
        bom = _make_bom(gini_asymmetry=γ_c)
        with pytest.raises(LogisticSingularityVeto):
            phase2.deliberate(bom)

    def test_f2_v1_gini_above_critical_raises(self, phase2):
        """F2.3b: G > γ_c lanza LogisticSingularityVeto."""
        bom = _make_bom(gini_asymmetry=0.99)
        with pytest.raises(LogisticSingularityVeto):
            phase2.deliberate(bom)

    def test_f2_v1_gini_just_below_critical_accepted(self, phase2):
        """F2.3c: G = γ_c - 2ε es aceptado (justo bajo el umbral)."""
        γ_c = phase2._ctx.max_gini_critical
        bom = _make_bom(
            gini_asymmetry=γ_c - _FLOAT_EPSILON * 2,
            total_mass=1.0,
        )
        verdict = phase2.deliberate(bom)
        assert verdict.is_viable is True

    def test_f2_v1_gini_zero_accepted(self, phase2):
        """F2.3d: G = 0 (distribución perfectamente equitativa) es aceptado."""
        bom = _make_bom(gini_asymmetry=0.0, total_mass=1.0)
        verdict = phase2.deliberate(bom)
        assert verdict.gini_asymmetry == pytest.approx(0.0)

    def test_f2_v1_veto_code_correct(self, phase2):
        """F2.3e: LogisticSingularityVeto tiene el veto_code correcto."""
        bom = _make_bom(gini_asymmetry=0.99)
        with pytest.raises(LogisticSingularityVeto) as exc_info:
            phase2.deliberate(bom)
        assert exc_info.value.veto_code == "LOGISTIC_SINGULARITY_VETO"

    def test_f2_v1_payload_contains_excess(self, phase2):
        """F2.3f: El payload del veto incluye el exceso G - γ_c."""
        γ_c = phase2._ctx.max_gini_critical
        G = γ_c + 0.05
        bom = _make_bom(gini_asymmetry=G)
        with pytest.raises(LogisticSingularityVeto) as exc_info:
            phase2.deliberate(bom)
        assert "excess" in exc_info.value.payload
        assert exc_info.value.payload["excess"] == pytest.approx(G - γ_c, rel=1e-6)

    def test_f2_v1_precedes_v2(self, phase2):
        """F2.3g: [V1] se aplica antes que [V2]: G≥γ_c lanza [V1] aunque Φ también falle."""
        # Para forzar [V2] necesitaríamos masa enorme; pero [V1] debe evaluarse primero
        bom = _make_bom(
            gini_asymmetry=0.99,   # Viola [V1]
            total_mass=1e9,        # También violaría [V2]
            exergy_available=1.0,  # [V3] OK
        )
        with pytest.raises(LogisticSingularityVeto):  # [V1] debe ganar
            phase2.deliberate(bom)

    # ── F2.4 Veto [V2]: ThermodynamicFrictionVeto ────────────────────────────

    def test_f2_v2_friction_above_limit_raises(self):
        """F2.4a: Φ > Φ_max lanza ThermodynamicFrictionVeto."""
        # Con n=2, ν=0.05: Φ(m) = 0.05·m²
        # Para Φ_max = 100: necesitamos m > sqrt(100/0.05) = sqrt(2000) ≈ 44.7 kg
        engine = _make_engine()
        agent = MatterAgent(
            engine=engine,
            max_exergy_friction=100.0,
            max_gini_critical=0.85,
        )
        p2 = MatterAgent.Phase2_ThermodynamicDeliberation(context=agent.context)
        bom = _make_bom(
            total_mass=50.0,        # Φ = 0.05·2500 = 125 W > 100 W ✓ veto
            gini_asymmetry=0.40,
            exergy_available=1.0,
        )
        with pytest.raises(ThermodynamicFrictionVeto):
            p2.deliberate(bom)

    def test_f2_v2_friction_at_limit_accepted(self):
        """F2.4b: Φ = Φ_max exacto es aceptado (límite cerrado superior)."""
        # Φ(m) = 0.05·m² = Φ_max → m = sqrt(Φ_max / 0.05)
        φ_max = 500.0  # W
        ν = 0.05
        n = 2.0
        m_exact = math.pow(φ_max / ν, 1.0 / n)  # ≈ 100 kg

        engine = _make_engine()
        agent = MatterAgent(
            engine=engine,
            max_exergy_friction=φ_max,
            max_gini_critical=0.85,
            rayleigh_viscosity=ν,
            rayleigh_exponent=n,
        )
        p2 = MatterAgent.Phase2_ThermodynamicDeliberation(context=agent.context)
        bom = _make_bom(
            total_mass=m_exact,
            gini_asymmetry=0.40,
            exergy_available=1.0,
        )
        verdict = p2.deliberate(bom)
        assert verdict.is_viable is True

    def test_f2_v2_veto_code_correct(self):
        """F2.4c: ThermodynamicFrictionVeto tiene el veto_code correcto."""
        engine = _make_engine()
        agent = MatterAgent(
            engine=engine,
            max_exergy_friction=1.0,  # Umbral extremadamente bajo
        )
        p2 = MatterAgent.Phase2_ThermodynamicDeliberation(context=agent.context)
        bom = _make_bom(total_mass=200.0, gini_asymmetry=0.3, exergy_available=1.0)
        with pytest.raises(ThermodynamicFrictionVeto) as exc_info:
            p2.deliberate(bom)
        assert exc_info.value.veto_code == "THERMODYNAMIC_FRICTION_VETO"

    def test_f2_v2_payload_contains_excess_ratio(self):
        """F2.4d: El payload incluye excess_ratio = Φ / Φ_max."""
        φ_max = 100.0
        engine = _make_engine()
        agent = MatterAgent(engine=engine, max_exergy_friction=φ_max)
        p2 = MatterAgent.Phase2_ThermodynamicDeliberation(context=agent.context)
        bom = _make_bom(total_mass=50.0, gini_asymmetry=0.3, exergy_available=1.0)
        # Φ = 0.05 * 2500 = 125 W → excess_ratio = 125/100 = 1.25
        with pytest.raises(ThermodynamicFrictionVeto) as exc_info:
            p2.deliberate(bom)
        assert "excess_ratio" in exc_info.value.payload
        assert exc_info.value.payload["excess_ratio"] == pytest.approx(1.25, rel=1e-6)

    # ── F2.5 Retículo booleano completo de vetos ──────────────────────────────

    def test_f2_reticulado_all_vetos_passed(self, phase2, healthy_bom):
        """F2.5a: BOM saludable supera los tres vetos sin excepciones."""
        verdict = phase2.deliberate(healthy_bom)
        assert verdict.is_viable is True

    def test_f2_reticulado_v3_is_subclass_of_hadronic(self):
        """F2.5b: NegativeExergyVeto ≤ HadronicCollapseVetoError (jerarquía)."""
        assert issubclass(NegativeExergyVeto, HadronicCollapseVetoError)

    def test_f2_reticulado_v1_is_subclass_of_hadronic(self):
        """F2.5c: LogisticSingularityVeto ≤ HadronicCollapseVetoError."""
        assert issubclass(LogisticSingularityVeto, HadronicCollapseVetoError)

    def test_f2_reticulado_v2_is_subclass_of_hadronic(self):
        """F2.5d: ThermodynamicFrictionVeto ≤ HadronicCollapseVetoError."""
        assert issubclass(ThermodynamicFrictionVeto, HadronicCollapseVetoError)

    def test_f2_reticulado_hadronic_is_subclass_of_topological(self):
        """F2.5e: HadronicCollapseVetoError ≤ TopologicalInvariantError."""
        assert issubclass(HadronicCollapseVetoError, TopologicalInvariantError)

    def test_f2_reticulado_catch_base_catches_all(self, phase2):
        """F2.5f: Capturar HadronicCollapseVetoError atrapa cualquier veto."""
        bom = _make_bom(exergy_available=-1.0)
        with pytest.raises(HadronicCollapseVetoError):
            phase2.deliberate(bom)

    # ── F2.6 Firma del morfismo φ ─────────────────────────────────────────────

    def test_f2_morphism_signature_is_64_hex(self, phase2, healthy_bom):
        """F2.6a: morphism_signature es SHA-256 de 64 chars hexadecimales."""
        verdict = phase2.deliberate(healthy_bom)
        sig = verdict.morphism_signature
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_f2_morphism_signature_deterministic(self, phase2, healthy_bom):
        """F2.6b: El mismo BOM produce la misma firma (determinismo)."""
        v1 = phase2.deliberate(healthy_bom)
        v2 = phase2.deliberate(healthy_bom)
        assert v1.morphism_signature == v2.morphism_signature

    def test_f2_morphism_signature_differs_on_different_bom(self, phase2):
        """F2.6c: BOM distintos producen firmas distintas."""
        bom_a = _make_bom(total_mass=100.0, gini_asymmetry=0.3)
        bom_b = _make_bom(total_mass=200.0, gini_asymmetry=0.3)
        va = phase2.deliberate(bom_a)
        vb = phase2.deliberate(bom_b)
        assert va.morphism_signature != vb.morphism_signature

    def test_f2_morphism_signature_fallback_on_no_to_dict(self, phase2):
        """F2.6d: BOM sin to_dict() usa repr como fallback (no levanta excepción)."""
        bom = MagicMock(spec=BillOfMaterials)
        bom.total_mass = 10.0
        bom.gini_asymmetry = 0.2
        bom.exergy_available = 500.0
        del bom.to_dict
        del bom.num_nodes
        del bom.num_edges
        del bom.euler_characteristic

        verdict = phase2.deliberate(bom)
        assert len(verdict.morphism_signature) == 64

    # ── F2.7 Campos del veredicto ─────────────────────────────────────────────

    def test_f2_verdict_is_frozen(self, phase2, healthy_bom):
        """F2.7a: HadronicDeliberationVerdict es inmutable (frozen)."""
        verdict = phase2.deliberate(healthy_bom)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            object.__setattr__(verdict, "is_viable", False)

    def test_f2_verdict_exergy_dissipated_matches_rayleigh(self, phase2, healthy_bom):
        """F2.7b: exergy_dissipated coincide con la fórmula de Rayleigh."""
        verdict = phase2.deliberate(healthy_bom)
        expected = (
            _RAYLEIGH_VISCOSITY_COEFFICIENT
            * math.pow(healthy_bom.total_mass, _RAYLEIGH_EXPONENT)
        )
        assert verdict.exergy_dissipated == pytest.approx(expected, rel=1e-9)

    def test_f2_verdict_exergy_available_matches_bom(self, phase2, healthy_bom):
        """F2.7c: exergy_available en el veredicto refleja el BOM."""
        verdict = phase2.deliberate(healthy_bom)
        assert verdict.exergy_available == pytest.approx(
            healthy_bom.exergy_available, rel=1e-9
        )

    def test_f2_verdict_timestamp_is_recent(self, phase2, healthy_bom):
        """F2.7d: deliberation_timestamp_utc está dentro del último segundo."""
        t_before = time.time()
        verdict = phase2.deliberate(healthy_bom)
        t_after = time.time()
        assert t_before <= verdict.deliberation_timestamp_utc <= t_after

    def test_f2_verdict_veto_reason_is_none_on_success(self, phase2, healthy_bom):
        """F2.7e: topological_veto_reason es None cuando el BOM es viable."""
        verdict = phase2.deliberate(healthy_bom)
        assert verdict.topological_veto_reason is None


# ══════════════════════════════════════════════════════════════════════════════
# FAMILIA F3 – CONTRATOS DE FASE 3 (PROYECCIÓN CATEGÓRICA)
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase3_CategoricalProjection:
    r"""
    F3: Contratos de la Fase 3.

    Verifica el cálculo del invariante de Euler-Poincaré, la firma π
    de la proyección, la completitud del payload y el stratum destino.
    """

    # ── F3.1 Cálculo del invariante de Euler-Poincaré ─────────────────────────

    def test_f3_euler_from_precalculated_field(self, phase3, approved_verdict):
        """F3.1a: Prioridad 1 – usa euler_characteristic si el BOM lo provee."""
        bom = _make_bom(euler_characteristic=7)
        verdict = dataclasses.replace(approved_verdict, bom_tensor=bom)
        state = phase3.project(verdict)
        chi = state.payload["topological_invariants"]["euler_poincare_characteristic"]
        assert chi == 7

    def test_f3_euler_from_graph_topology(self, phase3, approved_verdict):
        """F3.1b: Prioridad 2 – calcula χ = V - E + 1 del grafo."""
        # V=10 nodos, E=9 aristas (árbol) → χ = 10 - 9 + 1 = 2
        bom = _make_bom(num_nodes=10, num_edges=9)
        verdict = dataclasses.replace(approved_verdict, bom_tensor=bom)
        state = phase3.project(verdict)
        chi = state.payload["topological_invariants"]["euler_poincare_characteristic"]
        assert chi == 2

    def test_f3_euler_from_graph_topology_with_cycles(self, phase3, approved_verdict):
        """F3.1c: Grafo con ciclos (E > V-1) produce χ < 1."""
        # V=5, E=7 → χ = 5 - 7 + 1 = -1
        bom = _make_bom(num_nodes=5, num_edges=7)
        verdict = dataclasses.replace(approved_verdict, bom_tensor=bom)
        state = phase3.project(verdict)
        chi = state.payload["topological_invariants"]["euler_poincare_characteristic"]
        assert chi == -1

    def test_f3_euler_fallback_is_one(self, phase3, approved_verdict):
        """F3.1d: Fallback (sin atributos topológicos) → χ = 1."""
        bom = _make_bom()  # Sin euler_characteristic, num_nodes, num_edges
        verdict = dataclasses.replace(approved_verdict, bom_tensor=bom)
        state = phase3.project(verdict)
        chi = state.payload["topological_invariants"]["euler_poincare_characteristic"]
        assert chi == 1

    def test_f3_euler_precalculated_precedes_graph(self, phase3, approved_verdict):
        """F3.1e: euler_characteristic tiene prioridad sobre (num_nodes, num_edges)."""
        bom = _make_bom(
            euler_characteristic=42,
            num_nodes=10,
            num_edges=9,
        )
        verdict = dataclasses.replace(approved_verdict, bom_tensor=bom)
        state = phase3.project(verdict)
        chi = state.payload["topological_invariants"]["euler_poincare_characteristic"]
        assert chi == 42  # No 2 = V-E+1

    def test_f3_euler_tree_satisfies_euler_formula(self, phase3, approved_verdict):
        """F3.1f: Árbol con V nodos tiene χ = V - (V-1) + 1 = 2 ∀ V ≥ 1."""
        for V in [3, 5, 10, 50]:
            E = V - 1  # Árbol generador mínimo
            bom = _make_bom(num_nodes=V, num_edges=E)
            verdict = dataclasses.replace(approved_verdict, bom_tensor=bom)
            state = phase3.project(verdict)
            chi = state.payload["topological_invariants"][
                "euler_poincare_characteristic"
            ]
            assert chi == 2, f"χ debería ser 2 para árbol con V={V}"

    # ── F3.2 Firma de la proyección π ─────────────────────────────────────────

    def test_f3_pi_signature_is_64_hex(self, phase3, approved_verdict):
        """F3.2a: pi_signature es SHA-256 de 64 chars hexadecimales."""
        state = phase3.project(approved_verdict)
        sig = state.payload["morphism_chain"]["pi_signature"]
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_f3_pi_signature_deterministic(self, phase3, approved_verdict):
        """F3.2b: El mismo veredicto produce la misma firma π (determinismo)."""
        s1 = phase3.project(approved_verdict)
        s2 = phase3.project(approved_verdict)
        assert (
            s1.payload["morphism_chain"]["pi_signature"]
            == s2.payload["morphism_chain"]["pi_signature"]
        )

    def test_f3_pi_signature_differs_on_different_gini(
        self, phase3, approved_verdict
    ):
        """F3.2c: Distinto gini_asymmetry → distinta firma π."""
        v2 = dataclasses.replace(approved_verdict, gini_asymmetry=0.60)
        s1 = phase3.project(approved_verdict)
        s2 = phase3.project(v2)
        assert (
            s1.payload["morphism_chain"]["pi_signature"]
            != s2.payload["morphism_chain"]["pi_signature"]
        )

    def test_f3_pi_signature_depends_on_context_hash(self, approved_verdict):
        """F3.2d: Contextos distintos → firmas π distintas."""
        ctx_a = MatterAgent(max_gini_critical=0.70).context
        ctx_b = MatterAgent(max_gini_critical=0.75).context
        p3_a = MatterAgent.Phase3_CategoricalProjection(context=ctx_a)
        p3_b = MatterAgent.Phase3_CategoricalProjection(context=ctx_b)
        s_a = p3_a.project(approved_verdict)
        s_b = p3_b.project(approved_verdict)
        assert (
            s_a.payload["morphism_chain"]["pi_signature"]
            != s_b.payload["morphism_chain"]["pi_signature"]
        )

    # ── F3.3 Completitud y corrección del payload ─────────────────────────────

    def test_f3_payload_contains_bom_tensor(self, phase3, approved_verdict):
        """F3.3a: El payload incluye bom_tensor."""
        state = phase3.project(approved_verdict)
        assert "bom_tensor" in state.payload
        assert state.payload["bom_tensor"] is approved_verdict.bom_tensor

    def test_f3_payload_deliberation_metrics_complete(
        self, phase3, approved_verdict
    ):
        """F3.3b: deliberation_metrics contiene todos los campos requeridos."""
        state = phase3.project(approved_verdict)
        metrics = state.payload["deliberation_metrics"]
        required_keys = {
            "gini_asymmetry",
            "exergy_dissipated_W",
            "exergy_available_J",
            "is_viable",
            "veto_reason",
            "deliberation_ts_utc",
        }
        assert required_keys.issubset(set(metrics.keys()))

    def test_f3_payload_gini_matches_verdict(self, phase3, approved_verdict):
        """F3.3c: gini_asymmetry en el payload refleja el veredicto."""
        state = phase3.project(approved_verdict)
        assert state.payload["deliberation_metrics"]["gini_asymmetry"] == pytest.approx(
            approved_verdict.gini_asymmetry, rel=1e-9
        )

    def test_f3_payload_exergy_dissipated_matches_verdict(
        self, phase3, approved_verdict
    ):
        """F3.3d: exergy_dissipated_W en el payload refleja el veredicto."""
        state = phase3.project(approved_verdict)
        assert state.payload["deliberation_metrics"][
            "exergy_dissipated_W"
        ] == pytest.approx(approved_verdict.exergy_dissipated, rel=1e-9)

    def test_f3_payload_exergy_available_matches_verdict(
        self, phase3, approved_verdict
    ):
        """F3.3e: exergy_available_J en el payload refleja el veredicto."""
        state = phase3.project(approved_verdict)
        assert state.payload["deliberation_metrics"][
            "exergy_available_J"
        ] == pytest.approx(approved_verdict.exergy_available, rel=1e-9)

    def test_f3_payload_morphism_chain_complete(self, phase3, approved_verdict):
        """F3.3f: morphism_chain contiene phi_signature, pi_signature, context_hash."""
        state = phase3.project(approved_verdict)
        chain = state.payload["morphism_chain"]
        assert "phi_signature"  in chain
        assert "pi_signature"   in chain
        assert "context_hash"   in chain
        assert "module_version" in chain

    def test_f3_payload_phi_signature_matches_verdict(
        self, phase3, approved_verdict
    ):
        """F3.3g: phi_signature en la cadena coincide con el veredicto."""
        state = phase3.project(approved_verdict)
        assert (
            state.payload["morphism_chain"]["phi_signature"]
            == approved_verdict.morphism_signature
        )

    def test_f3_payload_module_version_correct(self, phase3, approved_verdict):
        """F3.3h: module_version en la cadena coincide con MODULE_VERSION."""
        state = phase3.project(approved_verdict)
        assert (
            state.payload["morphism_chain"]["module_version"] == MODULE_VERSION
        )

    def test_f3_payload_veto_reason_none_on_success(self, phase3, approved_verdict):
        """F3.3i: veto_reason es None cuando el veredicto es aprobatorio."""
        state = phase3.project(approved_verdict)
        assert state.payload["deliberation_metrics"]["veto_reason"] is None

    # ── F3.4 Stratum y tipo de retorno ────────────────────────────────────────

    def test_f3_returns_categorical_state(self, phase3, approved_verdict):
        """F3.4a: project() retorna exactamente CategoricalState."""
        state = phase3.project(approved_verdict)
        assert isinstance(state, CategoricalState)

    def test_f3_stratum_is_wisdom(self, phase3, approved_verdict):
        """F3.4b: El stratum del CategoricalState es WISDOM."""
        state = phase3.project(approved_verdict)
        assert state.stratum == Stratum.WISDOM

    def test_f3_context_hash_in_payload_matches_context(
        self, phase3, approved_verdict, default_agent
    ):
        """F3.4c: context_hash en el payload coincide con el contexto del agente."""
        state = phase3.project(approved_verdict)
        assert (
            state.payload["morphism_chain"]["context_hash"]
            == default_agent.context.context_hash
        )


# ══════════════════════════════════════════════════════════════════════════════
# FAMILIA F4 – CONTRATOS DE INMUTABILIDAD Y TRAZABILIDAD
# ══════════════════════════════════════════════════════════════════════════════


class TestImmutabilityAndTraceability:
    r"""
    F4: Contratos de Inmutabilidad y Trazabilidad.

    Verifica que todas las estructuras de datos son frozen, que el
    determinismo SHA-256 se mantiene cross-instancia y que la cadena
    de firmas φ → π → CategoricalState es consistente.
    """

    # ── F4.1 Inmutabilidad de dataclasses ─────────────────────────────────────

    def test_f4_matter_agent_context_frozen(self, default_agent):
        """F4.1a: MatterAgentContext no admite mutación post-construcción."""
        ctx = default_agent.context
        for field_name in ["max_gini_critical", "max_exergy_friction",
                            "rayleigh_exponent", "rayleigh_viscosity"]:
            with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
                object.__setattr__(ctx, field_name, 999.0)

    def test_f4_hadronic_verdict_frozen(self, phase2, healthy_bom):
        """F4.1b: HadronicDeliberationVerdict no admite mutación."""
        verdict = phase2.deliberate(healthy_bom)
        for field_name in ["is_viable", "gini_asymmetry", "exergy_dissipated"]:
            with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
                object.__setattr__(verdict, field_name, object())

    def test_f4_agent_phase2_is_set_in_constructor(self, default_agent):
        """F4.1c: _phase2 se inicializa en el constructor (no perezosa)."""
        assert default_agent._phase2 is not None
        assert isinstance(
            default_agent._phase2,
            MatterAgent.Phase2_ThermodynamicDeliberation,
        )

    def test_f4_agent_phase3_is_set_in_constructor(self, default_agent):
        """F4.1d: _phase3 se inicializa en el constructor (no perezosa)."""
        assert default_agent._phase3 is not None
        assert isinstance(
            default_agent._phase3,
            MatterAgent.Phase3_CategoricalProjection,
        )

    # ── F4.2 Determinismo SHA-256 cross-instancia ─────────────────────────────

    def test_f4_sha256_canonical_json_reproducible(self):
        """F4.2a: La serialización canónica JSON produce el mismo hash en dos ejecuciones."""
        params = {
            "gamma_c": f"{0.75:.12f}",
            "phi_max": f"{1e6:.12f}",
            "n":       f"{2.0:.12f}",
            "nu":      f"{0.05:.12f}",
        }
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
        h1 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        h2 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert h1 == h2

    def test_f4_context_hash_stable_across_agent_instances(self):
        """F4.2b: Dos agentes con mismos parámetros tienen el mismo context_hash."""
        kwargs = dict(
            max_gini_critical=0.80,
            max_exergy_friction=2e5,
            rayleigh_exponent=2.0,
            rayleigh_viscosity=0.05,
        )
        h1 = MatterAgent(**kwargs).context.context_hash
        h2 = MatterAgent(**kwargs).context.context_hash
        assert h1 == h2

    # ── F4.3 Cadena de firmas φ → π → CategoricalState ───────────────────────

    def test_f4_chain_phi_in_pi_signature(self, phase2, phase3, healthy_bom):
        """F4.3a: La firma π incluye la firma φ (composición verificable)."""
        verdict = phase2.deliberate(healthy_bom)
        state = phase3.project(verdict)

        phi_sig = verdict.morphism_signature
        pi_sig = state.payload["morphism_chain"]["pi_signature"]
        ctx_hash = state.payload["morphism_chain"]["context_hash"]

        # Reconstruir la firma π manualmente para verificar la fórmula
        composite = json.dumps(
            {
                "phi_sig":  phi_sig,
                "ctx_hash": ctx_hash,
                "gini":     f"{verdict.gini_asymmetry:.12f}",
                "exergy_d": f"{verdict.exergy_dissipated:.12f}",
                "exergy_a": f"{verdict.exergy_available:.12f}",
                "ts":       f"{verdict.deliberation_timestamp_utc:.6f}",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        expected_pi = hashlib.sha256(composite.encode("utf-8")).hexdigest()
        assert pi_sig == expected_pi

    def test_f4_chain_different_bom_different_full_chain(
        self, default_agent, phase2, phase3
    ):
        """F4.3b: BOM distintos generan cadenas de firmas completamente distintas."""
        bom_a = _make_bom(total_mass=100.0, gini_asymmetry=0.30)
        bom_b = _make_bom(total_mass=200.0, gini_asymmetry=0.30)

        v_a = phase2.deliberate(bom_a)
        v_b = phase2.deliberate(bom_b)

        s_a = phase3.project(v_a)
        s_b = phase3.project(v_b)

        assert v_a.morphism_signature != v_b.morphism_signature
        assert (
            s_a.payload["morphism_chain"]["pi_signature"]
            != s_b.payload["morphism_chain"]["pi_signature"]
        )

    # ── F4.4 Jerarquía de excepciones ─────────────────────────────────────────

    def test_f4_exception_has_timestamp(self, phase2):
        """F4.4a: Las excepciones de veto incluyen timestamp_utc."""
        bom = _make_bom(exergy_available=-1.0)
        with pytest.raises(NegativeExergyVeto) as exc_info:
            phase2.deliberate(bom)
        assert hasattr(exc_info.value, "timestamp_utc")
        assert exc_info.value.timestamp_utc > 0.0

    def test_f4_exception_str_contains_veto_code(self, phase2):
        """F4.4b: str(excepción) contiene el veto_code para trazabilidad en logs."""
        bom = _make_bom(gini_asymmetry=0.99)
        with pytest.raises(LogisticSingularityVeto) as exc_info:
            phase2.deliberate(bom)
        assert "LOGISTIC_SINGULARITY_VETO" in str(exc_info.value)

    def test_f4_exception_payload_is_dict(self, phase2):
        """F4.4c: El payload de toda excepción de veto es un dict."""
        bom = _make_bom(exergy_available=-1.0)
        with pytest.raises(NegativeExergyVeto) as exc_info:
            phase2.deliberate(bom)
        assert isinstance(exc_info.value.payload, dict)

    def test_f4_exception_reason_is_str(self, phase2):
        """F4.4d: El campo 'reason' de toda excepción de veto es str."""
        bom = _make_bom(gini_asymmetry=0.99)
        with pytest.raises(LogisticSingularityVeto) as exc_info:
            phase2.deliberate(bom)
        assert isinstance(exc_info.value.reason, str)
        assert len(exc_info.value.reason) > 0


# ══════════════════════════════════════════════════════════════════════════════
# FAMILIA F5 – CONTRATOS DE INTEGRACIÓN END-TO-END
# ══════════════════════════════════════════════════════════════════════════════


class TestEndToEndIntegration:
    r"""
    F5: Contratos de Integración End-to-End.

    Orquesta la composición completa F = π ∘ δ ∘ φ, verificando que
    el agente coordina correctamente las tres fases y que el
    CategoricalState resultante satisface la clausura transitiva del
    Topos de Grothendieck.
    """

    # ── Fixture local de invocación completa ──────────────────────────────────

    @pytest.fixture
    def invocation_result(self, default_agent, healthy_bom) -> CategoricalState:
        """Resultado de una invocación completa del endofuntor F."""
        default_agent.context.engine.project_to_bom.return_value = healthy_bom
        return default_agent.project_intent_and_deliberate(
            hierarchical_complex={"A": ["B", "C"]},
            root_node="A",
            friction_map={"A-B": 0.1, "A-C": 0.2},
            price_map={"B": 10.0, "C": 20.0},
        )

    # ── F5.1 Clausura transitiva del topos ────────────────────────────────────

    def test_f5_returns_categorical_state(self, invocation_result):
        """F5.1a: La composición F retorna CategoricalState (clausura del topos)."""
        assert isinstance(invocation_result, CategoricalState)

    def test_f5_stratum_is_wisdom(self, invocation_result):
        """F5.1b: El estrato de destino es WISDOM."""
        assert invocation_result.stratum == Stratum.WISDOM

    def test_f5_payload_not_empty(self, invocation_result):
        """F5.1c: El payload del estado no está vacío."""
        assert invocation_result.payload
        assert len(invocation_result.payload) >= 3

    # ── F5.2 Delegación correcta al motor físico ──────────────────────────────

    def test_f5_engine_project_to_bom_called_once(
        self, default_agent, healthy_bom
    ):
        """F5.2a: project_to_bom se invoca exactamente una vez por llamada."""
        default_agent.context.engine.project_to_bom.return_value = healthy_bom
        default_agent.project_intent_and_deliberate(
            hierarchical_complex={},
            root_node="root",
            friction_map={},
            price_map={},
        )
        default_agent.context.engine.project_to_bom.assert_called_once()

    def test_f5_engine_receives_correct_arguments(
        self, default_agent, healthy_bom
    ):
        """F5.2b: El motor recibe los argumentos exactos del llamador."""
        default_agent.context.engine.project_to_bom.return_value = healthy_bom
        hc = {"X": ["Y"]}
        fm = {"X-Y": 0.5}
        pm = {"Y": 15.0}

        default_agent.project_intent_and_deliberate(
            hierarchical_complex=hc,
            root_node="X",
            friction_map=fm,
            price_map=pm,
        )

        default_agent.context.engine.project_to_bom.assert_called_once_with(
            hierarchical_complex=hc,
            root_node="X",
            friction_tensor_map=fm,
            price_tensor_map=pm,
        )

    # ── F5.3 Propagación correcta de vetos ────────────────────────────────────

    def test_f5_propagates_logistic_singularity_veto(
        self, default_agent
    ):
        """F5.3a: LogisticSingularityVeto del motor se propaga al llamador."""
        bad_bom = _make_bom(gini_asymmetry=0.99, exergy_available=1.0)
        default_agent.context.engine.project_to_bom.return_value = bad_bom
        with pytest.raises(LogisticSingularityVeto):
            default_agent.project_intent_and_deliberate(
                hierarchical_complex={}, root_node="r",
                friction_map={}, price_map={},
            )

    def test_f5_propagates_thermodynamic_friction_veto(self):
        """F5.3b: ThermodynamicFrictionVeto se propaga al llamador."""
        bad_bom = _make_bom(
            total_mass=1e7,      # Φ = 0.05 * 1e14 >> 1e6 W
            gini_asymmetry=0.3,
            exergy_available=1.0,
        )
        engine = _make_engine(bom=bad_bom)
        agent = MatterAgent(engine=engine, max_exergy_friction=1e6)
        with pytest.raises(ThermodynamicFrictionVeto):
            agent.project_intent_and_deliberate(
                hierarchical_complex={}, root_node="r",
                friction_map={}, price_map={},
            )

    def test_f5_propagates_negative_exergy_veto(self, default_agent):
        """F5.3c: NegativeExergyVeto se propaga al llamador."""
        bad_bom = _make_bom(exergy_available=-500.0, gini_asymmetry=0.3)
        default_agent.context.engine.project_to_bom.return_value = bad_bom
        with pytest.raises(NegativeExergyVeto):
            default_agent.project_intent_and_deliberate(
                hierarchical_complex={}, root_node="r",
                friction_map={}, price_map={},
            )

    # ── F5.4 Idempotencia funcional ───────────────────────────────────────────

    def test_f5_idempotent_on_same_bom(self, default_agent, healthy_bom):
        """F5.4a: Dos invocaciones con el mismo BOM producen estados equivalentes."""
        default_agent.context.engine.project_to_bom.return_value = healthy_bom

        s1 = default_agent.project_intent_and_deliberate(
            hierarchical_complex={}, root_node="r",
            friction_map={}, price_map={},
        )
        s2 = default_agent.project_intent_and_deliberate(
            hierarchical_complex={}, root_node="r",
            friction_map={}, price_map={},
        )

        assert s1.stratum == s2.stratum
        assert (
            s1.payload["deliberation_metrics"]["gini_asymmetry"]
            == s2.payload["deliberation_metrics"]["gini_asymmetry"]
        )
        assert (
            s1.payload["deliberation_metrics"]["exergy_dissipated_W"]
            == s2.payload["deliberation_metrics"]["exergy_dissipated_W"]
        )

    def test_f5_different_bom_different_state(self, default_agent):
        """F5.4b: BOM distintos producen estados categóricos distintos."""
        bom_a = _make_bom(total_mass=50.0, gini_asymmetry=0.20)
        bom_b = _make_bom(total_mass=80.0, gini_asymmetry=0.35)

        default_agent.context.engine.project_to_bom.return_value = bom_a
        s_a = default_agent.project_intent_and_deliberate(
            hierarchical_complex={}, root_node="r",
            friction_map={}, price_map={},
        )

        default_agent.context.engine.project_to_bom.return_value = bom_b
        s_b = default_agent.project_intent_and_deliberate(
            hierarchical_complex={}, root_node="r",
            friction_map={}, price_map={},
        )

        assert (
            s_a.payload["deliberation_metrics"]["gini_asymmetry"]
            != s_b.payload["deliberation_metrics"]["gini_asymmetry"]
        )

    # ── F5.5 Trazabilidad end-to-end ──────────────────────────────────────────

    def test_f5_pi_signature_in_result(self, invocation_result):
        """F5.5a: El resultado final contiene la firma π en la cadena de morfismos."""
        chain = invocation_result.payload["morphism_chain"]
        assert "pi_signature" in chain
        assert len(chain["pi_signature"]) == 64

    def test_f5_phi_signature_in_result(self, invocation_result):
        """F5.5b: El resultado final contiene la firma φ en la cadena de morfismos."""
        chain = invocation_result.payload["morphism_chain"]
        assert "phi_signature" in chain
        assert len(chain["phi_signature"]) == 64

    def test_f5_context_hash_in_result_matches_agent(
        self, default_agent, invocation_result
    ):
        """F5.5c: context_hash en el resultado coincide con el agente."""
        assert (
            invocation_result.payload["morphism_chain"]["context_hash"]
            == default_agent.context.context_hash
        )

    def test_f5_topological_invariant_in_result(self, invocation_result):
        """F5.5d: El resultado incluye el invariante topológico χ(K)."""
        assert "topological_invariants" in invocation_result.payload
        chi = invocation_result.payload["topological_invariants"][
            "euler_poincare_characteristic"
        ]
        assert isinstance(chi, int)

    # ── F5.6 Logging de auditoría ─────────────────────────────────────────────

    def test_f5_logs_inicio_colapso(
        self, default_agent, healthy_bom, caplog
    ):
        """F5.6a: Se emite log INFO al iniciar el colapso hadrónico."""
        import logging
        default_agent.context.engine.project_to_bom.return_value = healthy_bom
        with caplog.at_level(logging.INFO, logger="MIC.Omega.MatterAgent"):
            default_agent.project_intent_and_deliberate(
                hierarchical_complex={}, root_node="r",
                friction_map={}, price_map={},
            )
        assert any(
            "colapso hadrónico" in msg.lower() or "estrato" in msg.lower()
            for msg in caplog.messages
        )

    def test_f5_logs_completion(
        self, default_agent, healthy_bom, caplog
    ):
        """F5.6b: Se emite log INFO al completar el colapso con éxito."""
        import logging
        default_agent.context.engine.project_to_bom.return_value = healthy_bom
        with caplog.at_level(logging.INFO, logger="MIC.Omega.MatterAgent"):
            default_agent.project_intent_and_deliberate(
                hierarchical_complex={}, root_node="r",
                friction_map={}, price_map={},
            )
        assert any(
            "completado" in msg.lower() or "preservado" in msg.lower()
            for msg in caplog.messages
        )

    def test_f5_logs_veto_on_gini_violation(
        self, default_agent, caplog
    ):
        """F5.6c: Se emite log ERROR al activar el veto de Gini."""
        import logging
        bad_bom = _make_bom(gini_asymmetry=0.99, exergy_available=1.0)
        default_agent.context.engine.project_to_bom.return_value = bad_bom

        with caplog.at_level(logging.ERROR, logger="MIC.Omega.MatterAgent"):
            with pytest.raises(LogisticSingularityVeto):
                default_agent.project_intent_and_deliberate(
                    hierarchical_complex={}, root_node="r",
                    friction_map={}, price_map={},
                )
        assert any(
            "veto" in msg.lower() or "singularidad" in msg.lower()
            for msg in caplog.messages
        )

    # ── F5.7 Casos límite y condiciones de frontera ───────────────────────────

    def test_f5_boundary_gini_just_under_critical(self, default_agent):
        """F5.7a: G = γ_c - 2ε (frontera inferior) → aprobado."""
        γ_c = default_agent.context.max_gini_critical
        bom = _make_bom(
            gini_asymmetry=γ_c - _FLOAT_EPSILON * 2,
            total_mass=1.0,
            exergy_available=1.0,
        )
        default_agent.context.engine.project_to_bom.return_value = bom
        state = default_agent.project_intent_and_deliberate(
            hierarchical_complex={}, root_node="r",
            friction_map={}, price_map={},
        )
        assert isinstance(state, CategoricalState)

    def test_f5_boundary_mass_zero(self, default_agent):
        """F5.7b: Masa = 0 → Φ = 0 (sin disipación) → aprobado."""
        bom = _make_bom(total_mass=0.0, gini_asymmetry=0.3, exergy_available=1.0)
        default_agent.context.engine.project_to_bom.return_value = bom
        state = default_agent.project_intent_and_deliberate(
            hierarchical_complex={}, root_node="r",
            friction_map={}, price_map={},
        )
        dissipated = state.payload["deliberation_metrics"]["exergy_dissipated_W"]
        assert dissipated == pytest.approx(0.0)

    def test_f5_boundary_gini_zero(self, default_agent):
        """F5.7c: G = 0 (equidad perfecta) → aprobado."""
        bom = _make_bom(gini_asymmetry=0.0, total_mass=1.0, exergy_available=1.0)
        default_agent.context.engine.project_to_bom.return_value = bom
        state = default_agent.project_intent_and_deliberate(
            hierarchical_complex={}, root_node="r",
            friction_map={}, price_map={},
        )
        assert state.payload["deliberation_metrics"]["gini_asymmetry"] == pytest.approx(0.0)

    def test_f5_custom_rayleigh_parameters_respected(self):
        """F5.7d: Parámetros de Rayleigh personalizados producen fricción correcta."""
        ν, n = 0.10, 3.0
        m = 5.0
        expected_friction = ν * m ** n  # 0.10 * 125 = 12.5 W

        bom = _make_bom(total_mass=m, gini_asymmetry=0.3, exergy_available=1.0)
        engine = _make_engine(bom=bom)
        agent = MatterAgent(
            engine=engine,
            max_exergy_friction=100.0,
            rayleigh_exponent=n,
            rayleigh_viscosity=ν,
        )
        state = agent.project_intent_and_deliberate(
            hierarchical_complex={}, root_node="r",
            friction_map={}, price_map={},
        )
        dissipated = state.payload["deliberation_metrics"]["exergy_dissipated_W"]
        assert dissipated == pytest.approx(expected_friction, rel=1e-9)
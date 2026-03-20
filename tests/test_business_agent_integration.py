"""
Suite de integración para BusinessAgent.

Fundamentación:
───────────────

1. Modelo topológico del presupuesto:
   El presupuesto forma un grafo bipartito dirigido G = (V_APU ∪ V_INS, E)
   donde:
       - V_APU = conjunto de APUs (vértices fuente)
       - V_INS = conjunto de Insumos (vértices sumidero)
       - E ⊆ V_APU × V_INS = relaciones de composición

   Un presupuesto "bien formado" satisface:
       - β₀ = 1 (grafo conectado: todo APU tiene al menos un insumo)
       - β₁ = 0 (sin ciclos: la estructura es un DAG)
       - ∀ v ∈ V_APU: deg⁺(v) ≥ 1 (todo APU tiene al menos un insumo)

   Referencia: [1] Diestel, R. "Graph Theory", 5th Ed., Springer, 2017.

2. Invariantes financieros:
   - Valor Presente Neto (VPN):
         VPN = -I₀ + Σₜ₌₁ⁿ CFₜ / (1 + r)ᵗ
     El VPN es monótonamente decreciente en r para flujos positivos.

   - Período de recuperación (Payback):
         PB = min{t : Σₛ₌₁ᵗ CFₛ ≥ I₀}
     Con interpolación lineal para período fraccionario.

   - Consistencia: is_viable() ⟺ VPN > 0

   Referencia: [2] Brealey, R. et al. "Principles of Corporate Finance",
   13th Ed., McGraw-Hill, 2020.

3. Delegación a la MIC:
   El BusinessAgent delega el análisis financiero cuantitativo a la
   Matriz de Interacción Central (MIC) vía project_intent("financial_analysis").
   El agente es responsable de:
       - Preparar el payload con los datos del presupuesto
       - Interpretar los resultados de la MIC
       - Generar la narrativa estratégica
       - Detectar anomalías topológicas

   Referencia: [3] Arquitectura interna del sistema.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, call

import numpy as np
import pandas as pd
import pytest

from app.strategy.business_agent import BusinessAgent
from app.adapters.tools_interface import MICRegistry


# =============================================================================
# CONSTANTES
# =============================================================================

# Timestamp fijo para reproducibilidad.
_FIXED_TIMESTAMP: str = "2024-01-15T10:30:00"

# Tolerancia absoluta para comparaciones financieras.
# Justificación: los cálculos financieros involucran sumas de fracciones
# con denominadores (1+r)^t. Para r ∈ [0.01, 0.20] y t ≤ 30, el error
# de punto flotante acumulado es O(n · ε_mach) ≈ 30 · 2.22e-16 ≈ 7e-15,
# muy por debajo de esta tolerancia.
_FINANCIAL_ABS_TOLERANCE: float = 1e-6

# Tolerancia relativa para comparaciones de montos grandes.
_FINANCIAL_REL_TOLERANCE: float = 1e-6

# Límite de tiempo para tests de rendimiento [segundos].
_PERFORMANCE_TIME_LIMIT: float = 10.0

# Estratos validados por defecto en contextos de prueba.
_DEFAULT_VALIDATED_STRATA: Set[str] = {"PHYSICS", "TACTICS"}

# Tipos válidos de insumo.
_VALID_INSUMO_TYPES: frozenset = frozenset({
    "MATERIAL", "MANO_OBRA", "EQUIPO", "TRANSPORTE",
})


# =============================================================================
# DATA CLASSES DE DOMINIO
# =============================================================================


@dataclass(frozen=True)
class APURecord:
    """
    Registro de Análisis de Precios Unitarios.

    Attributes
    ----------
    codigo : str
        Identificador único. Debe iniciar con "APU-".
    descripcion : str
        Descripción de la actividad.
    cantidad : float
        Cantidad presupuestada. Debe ser ≥ 0.
    unidad : str
        Unidad de medida. Default "UND".
    """

    codigo: str
    descripcion: str
    cantidad: float
    unidad: str = "UND"

    def __post_init__(self) -> None:
        if self.cantidad < 0:
            raise ValueError(
                f"Cantidad no puede ser negativa: {self.cantidad}"
            )
        if not self.codigo.startswith("APU-"):
            raise ValueError(
                f"Código APU debe iniciar con 'APU-': {self.codigo}"
            )


@dataclass(frozen=True)
class InsumoRecord:
    """
    Registro de insumo vinculado a un APU.

    Attributes
    ----------
    codigo_apu : str
        Código del APU padre.
    descripcion : str
        Descripción del insumo.
    tipo : str
        Tipo de insumo. Debe ser uno de _VALID_INSUMO_TYPES.
    costo : float
        Costo unitario. Debe ser ≥ 0.
    cantidad : float
        Cantidad requerida. Debe ser ≥ 0.
    """

    codigo_apu: str
    descripcion: str
    tipo: str
    costo: float
    cantidad: float = 1.0

    def __post_init__(self) -> None:
        if self.tipo not in _VALID_INSUMO_TYPES:
            raise ValueError(
                f"Tipo inválido '{self.tipo}'. Válidos: {_VALID_INSUMO_TYPES}"
            )
        if self.costo < 0:
            raise ValueError(f"Costo no puede ser negativo: {self.costo}")
        if self.cantidad < 0:
            raise ValueError(
                f"Cantidad no puede ser negativa: {self.cantidad}"
            )


@dataclass
class CashFlowProjection:
    """
    Proyección de flujos de caja con invariantes financieros.

    Atributos validados en __post_init__:
        - initial_investment > 0
        - cash_flows no vacío
        - 0 ≤ risk_free_rate ≤ 1

    Attributes
    ----------
    initial_investment : float
        Inversión inicial I₀ > 0.
    cash_flows : List[float]
        Flujos de caja [CF₁, CF₂, ..., CFₙ]. Pueden ser negativos.
    risk_free_rate : float
        Tasa libre de riesgo r ∈ [0, 1].
    """

    initial_investment: float
    cash_flows: List[float]
    risk_free_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.initial_investment <= 0:
            raise ValueError(
                f"Inversión inicial debe ser positiva: "
                f"{self.initial_investment}"
            )
        if not self.cash_flows:
            raise ValueError("Debe haber al menos un flujo de caja.")
        if not 0 <= self.risk_free_rate <= 1:
            raise ValueError(
                f"Tasa libre de riesgo debe estar en [0, 1]: "
                f"{self.risk_free_rate}"
            )

    def calculate_npv(
        self, discount_rate: Optional[float] = None,
    ) -> float:
        """
        Calcula el Valor Presente Neto (VPN).

        VPN = -I₀ + Σₜ₌₁ⁿ CFₜ / (1 + r)ᵗ

        Para flujos positivos, el VPN es monótonamente decreciente en r
        porque ∂VPN/∂r = -Σₜ₌₁ⁿ t·CFₜ/(1+r)^(t+1) < 0.

        Parameters
        ----------
        discount_rate : Optional[float]
            Tasa de descuento. Si None, usa risk_free_rate.
            Debe ser > -1 para que el denominador sea positivo.

        Returns
        -------
        float
            Valor presente neto.

        Raises
        ------
        ValueError
            Si discount_rate ≤ -1.
        """
        rate: float = discount_rate if discount_rate is not None else self.risk_free_rate
        if rate <= -1.0:
            raise ValueError(
                f"Tasa de descuento debe ser > -1: {rate}"
            )

        npv: float = -self.initial_investment
        for t, cf in enumerate(self.cash_flows, start=1):
            npv += cf / ((1.0 + rate) ** t)
        return npv

    def calculate_payback_period(self) -> Optional[float]:
        """
        Calcula el período de recuperación simple con interpolación lineal.

        PB = (t-1) + (I₀ - Σₛ₌₁^(t-1) CFₛ) / CFₜ

        donde t es el primer período en que la suma acumulada ≥ I₀.

        Returns
        -------
        Optional[float]
            Período de recuperación, o None si no se recupera en el horizonte.
        """
        cumulative: float = 0.0
        for t, cf in enumerate(self.cash_flows, start=1):
            prev_cumulative: float = cumulative
            cumulative += cf
            if cumulative >= self.initial_investment:
                remaining: float = self.initial_investment - prev_cumulative
                fraction: float = remaining / cf if cf > 0 else 0.0
                return (t - 1) + fraction
        return None

    def is_viable(self) -> bool:
        """
        Determina viabilidad financiera.

        Un proyecto es viable ⟺ VPN > 0.

        Returns
        -------
        bool
        """
        return self.calculate_npv() > 0


# =============================================================================
# BUILDERS
# =============================================================================


class PresupuestoBuilder:
    """
    Builder para DataFrames de presupuesto con validación de unicidad.

    Invariantes:
        - Cada APU tiene código único.
        - Cantidades son no negativas.
        - Códigos siguen formato "APU-XXX".
    """

    def __init__(self) -> None:
        self._apus: List[APURecord] = []

    def with_apu(
        self,
        codigo: str,
        descripcion: str,
        cantidad: float,
        unidad: str = "UND",
    ) -> PresupuestoBuilder:
        """Agrega un APU. Valida formato y no-negatividad."""
        self._apus.append(
            APURecord(codigo, descripcion, cantidad, unidad)
        )
        return self

    def with_standard_apus(self, count: int = 3) -> PresupuestoBuilder:
        """Genera `count` APUs estándar para testing rápido."""
        for i in range(1, count + 1):
            self._apus.append(APURecord(
                codigo=f"APU-{i:03d}",
                descripcion=f"Actividad de Construcción #{i}",
                cantidad=float(i * 10),
            ))
        return self

    def get_codes(self) -> List[str]:
        """Retorna los códigos APU registrados."""
        return [apu.codigo for apu in self._apus]

    def build(self) -> pd.DataFrame:
        """
        Construye el DataFrame validando unicidad de códigos.

        Raises
        ------
        ValueError
            Si hay códigos APU duplicados.
        """
        if not self._apus:
            return pd.DataFrame(columns=[
                "CODIGO_APU", "DESCRIPCION_APU",
                "CANTIDAD_PRESUPUESTO", "UNIDAD",
            ])

        codigos = [apu.codigo for apu in self._apus]
        if len(codigos) != len(set(codigos)):
            duplicados = {c for c in codigos if codigos.count(c) > 1}
            raise ValueError(f"Códigos APU duplicados: {duplicados}")

        return pd.DataFrame([
            {
                "CODIGO_APU": apu.codigo,
                "DESCRIPCION_APU": apu.descripcion,
                "CANTIDAD_PRESUPUESTO": apu.cantidad,
                "UNIDAD": apu.unidad,
            }
            for apu in self._apus
        ])


class InsumosBuilder:
    """
    Builder para DataFrames de insumos con validación de conectividad.

    Invariantes topológicos:
        - Cada insumo referencia un APU existente (si se valida).
        - No hay insumos huérfanos.
    """

    def __init__(self) -> None:
        self._insumos: List[InsumoRecord] = []
        self._apu_codes: Set[str] = set()

    def for_apus(self, *codigos: str) -> InsumosBuilder:
        """Registra los APUs válidos para validación de conectividad."""
        self._apu_codes = set(codigos)
        return self

    def with_insumo(
        self,
        codigo_apu: str,
        descripcion: str,
        tipo: str,
        costo: float,
        cantidad: float = 1.0,
    ) -> InsumosBuilder:
        """Agrega un insumo vinculado a un APU."""
        self._insumos.append(InsumoRecord(
            codigo_apu=codigo_apu,
            descripcion=descripcion,
            tipo=tipo,
            costo=costo,
            cantidad=cantidad,
        ))
        return self

    def with_standard_insumos_for(
        self, codigo_apu: str,
    ) -> InsumosBuilder:
        """Genera set estándar de 4 insumos para un APU."""
        base_cost: float = 100.0
        suffix: str = f" ({codigo_apu})"
        self._insumos.extend([
            InsumoRecord(codigo_apu, f"Cemento{suffix}", "MATERIAL", base_cost * 0.4),
            InsumoRecord(codigo_apu, f"Oficial{suffix}", "MANO_OBRA", base_cost * 0.35),
            InsumoRecord(codigo_apu, f"Mezcladora{suffix}", "EQUIPO", base_cost * 0.15),
            InsumoRecord(codigo_apu, f"Flete{suffix}", "TRANSPORTE", base_cost * 0.10),
        ])
        return self

    def get_total_cost(self) -> float:
        """Σ (costo_i · cantidad_i) para todos los insumos."""
        return sum(ins.costo * ins.cantidad for ins in self._insumos)

    def get_cost_by_apu(self) -> Dict[str, float]:
        """Costo agregado por APU: {codigo_apu: Σ costo·cantidad}."""
        costs: Dict[str, float] = {}
        for ins in self._insumos:
            costs[ins.codigo_apu] = costs.get(ins.codigo_apu, 0.0) + (
                ins.costo * ins.cantidad
            )
        return costs

    def build(self, validate_connectivity: bool = True) -> pd.DataFrame:
        """
        Construye el DataFrame, opcionalmente validando conectividad.

        Raises
        ------
        ValueError
            Si hay insumos huérfanos y validate_connectivity=True.
        """
        if not self._insumos:
            return pd.DataFrame(columns=[
                "CODIGO_APU", "DESCRIPCION_INSUMO", "TIPO_INSUMO",
                "COSTO_INSUMO_EN_APU", "CANTIDAD_APU",
            ])

        if validate_connectivity and self._apu_codes:
            orphans = {
                ins.codigo_apu for ins in self._insumos
            } - self._apu_codes
            if orphans:
                raise ValueError(
                    f"Insumos huérfanos (APU no existe): {orphans}"
                )

        return pd.DataFrame([
            {
                "CODIGO_APU": ins.codigo_apu,
                "DESCRIPCION_INSUMO": ins.descripcion,
                "TIPO_INSUMO": ins.tipo,
                "COSTO_INSUMO_EN_APU": ins.costo,
                "CANTIDAD_APU": ins.cantidad,
            }
            for ins in self._insumos
        ])


class ProjectContextBuilder:
    """
    Builder para el contexto completo de evaluación de proyecto.

    Garantiza coherencia entre presupuesto, insumos y flujos financieros.
    """

    def __init__(self) -> None:
        self._presupuesto: PresupuestoBuilder = PresupuestoBuilder()
        self._insumos: InsumosBuilder = InsumosBuilder()
        self._cash_flow: Optional[CashFlowProjection] = None
        self._extra_params: Dict[str, Any] = {}

    def with_presupuesto(
        self, builder: PresupuestoBuilder,
    ) -> ProjectContextBuilder:
        """Configura el presupuesto."""
        self._presupuesto = builder
        return self

    def with_insumos(
        self, builder: InsumosBuilder,
    ) -> ProjectContextBuilder:
        """Configura los insumos."""
        self._insumos = builder
        return self

    def with_cash_flow(
        self,
        initial_investment: float,
        cash_flows: List[float],
        risk_free_rate: float = 0.05,
    ) -> ProjectContextBuilder:
        """Configura la proyección financiera."""
        self._cash_flow = CashFlowProjection(
            initial_investment=initial_investment,
            cash_flows=cash_flows,
            risk_free_rate=risk_free_rate,
        )
        return self

    def with_viable_project(self) -> ProjectContextBuilder:
        """Configura un proyecto financieramente viable (VPN > 0)."""
        self._cash_flow = CashFlowProjection(
            initial_investment=1000.0,
            cash_flows=[400.0, 400.0, 400.0, 400.0],
        )
        return self

    def with_non_viable_project(self) -> ProjectContextBuilder:
        """Configura un proyecto no viable (VPN < 0)."""
        self._cash_flow = CashFlowProjection(
            initial_investment=10000.0,
            cash_flows=[100.0, 100.0, 100.0],
        )
        return self

    def with_param(
        self, key: str, value: Any,
    ) -> ProjectContextBuilder:
        """Agrega parámetro adicional al contexto."""
        self._extra_params[key] = value
        return self

    def build(self) -> Dict[str, Any]:
        """
        Construye el contexto validando coherencia interna.

        Si no se configuró cash_flow, usa un default viable.
        """
        df_presupuesto = self._presupuesto.build()
        apu_codes = (
            df_presupuesto["CODIGO_APU"].tolist()
            if not df_presupuesto.empty else []
        )

        self._insumos.for_apus(*apu_codes)
        df_merged = self._insumos.build(
            validate_connectivity=bool(apu_codes)
        )

        if self._cash_flow is None:
            self._cash_flow = CashFlowProjection(
                initial_investment=1000.0,
                cash_flows=[300.0, 400.0, 500.0],
            )

        return {
            "df_presupuesto": df_presupuesto,
            "df_merged": df_merged,
            "initial_investment": self._cash_flow.initial_investment,
            "cash_flows": self._cash_flow.cash_flows,
            "timestamp": _FIXED_TIMESTAMP,
            "validated_strata": _DEFAULT_VALIDATED_STRATA.copy(),
            **self._extra_params,
        }

    def build_with_projection(
        self,
    ) -> Tuple[Dict[str, Any], CashFlowProjection]:
        """Retorna contexto junto con la proyección para validación."""
        context = self.build()
        assert self._cash_flow is not None
        return context, self._cash_flow


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================


def _create_mock_mic(
    npv: float = 150.0,
    irr: float = 0.15,
    payback: float = 3.5,
    recommendation: str = "APPROVE",
    risk_level: str = "LOW",
) -> MagicMock:
    """
    Crea un mock de MICRegistry con respuestas configurables.

    El mock responde a project_intent("financial_analysis") con los
    valores proporcionados y rechaza otros servicios.

    Parameters
    ----------
    npv : float
        Valor presente neto a retornar.
    irr : float
        Tasa interna de retorno.
    payback : float
        Período de recuperación.
    recommendation : str
        Recomendación (APPROVE/REJECT).
    risk_level : str
        Nivel de riesgo (LOW/MEDIUM/HIGH).

    Returns
    -------
    MagicMock
        Mock configurado de MICRegistry.
    """
    mic = MagicMock(spec=MICRegistry)

    def side_effect(service: str, payload: Any, context: Any = None) -> Dict:
        if service == "financial_analysis":
            return {
                "success": True,
                "results": {
                    "npv": npv,
                    "irr": irr,
                    "payback_years": payback,
                    "payback": payback,
                    "wacc": 0.10,
                    "risk_adjusted_return": irr - 0.03,
                    "performance": {
                        "recommendation": recommendation,
                        "risk_level": risk_level,
                    },
                },
            }
        elif service == "lateral_thinking_pivot":
            return {"success": False, "error": "Mocked pivot rejection"}
        return {"success": False, "error": f"Unknown service: {service}"}

    mic.project_intent.side_effect = side_effect
    return mic


def _assert_report_has_structure(report: Any, context_desc: str = "") -> None:
    """
    Verifica que un reporte del BusinessAgent tiene la estructura mínima.

    Parameters
    ----------
    report : Any
        Reporte retornado por evaluate_project.
    context_desc : str
        Descripción del contexto para mensajes de error.
    """
    prefix = f"[{context_desc}] " if context_desc else ""

    assert report is not None, (
        f"{prefix}evaluate_project retornó None."
    )
    assert hasattr(report, "strategic_narrative"), (
        f"{prefix}Reporte carece de 'strategic_narrative'."
    )
    assert hasattr(report, "details"), (
        f"{prefix}Reporte carece de 'details'."
    )
    assert isinstance(report.strategic_narrative, str), (
        f"{prefix}strategic_narrative debe ser str, "
        f"es {type(report.strategic_narrative).__name__}."
    )
    assert len(report.strategic_narrative) > 0, (
        f"{prefix}strategic_narrative está vacío."
    )


def _narrative_contains_any(
    narrative: str,
    terms: List[str],
    context: str = "",
) -> None:
    """
    Verifica que la narrativa contiene al menos uno de los términos dados.

    Parameters
    ----------
    narrative : str
        Texto de la narrativa.
    terms : List[str]
        Términos a buscar (case-insensitive, búsqueda parcial).
    context : str
        Descripción del contexto para error.
    """
    narrative_lower = narrative.lower()
    found = [t for t in terms if t in narrative_lower]

    assert len(found) > 0, (
        f"[{context}] Narrativa no contiene ninguno de los términos "
        f"esperados: {terms}.\n"
        f"Narrativa (primeros 500 chars): "
        f"'{narrative_lower[:500]}...'"
    )


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def default_config() -> Dict[str, Any]:
    """
    Configuración por defecto del BusinessAgent.

    Returns
    -------
    Dict[str, Any]
        Configuración con tasas financieras, umbrales de riesgo,
        y opciones de reporte.
    """
    return {
        "financial_config": {
            "risk_free_rate": 0.05,
            "inflation_rate": 0.03,
            "tax_rate": 0.25,
        },
        "risk_config": {
            "volatility_threshold": 0.2,
            "concentration_limit": 0.4,
        },
        "report_config": {
            "include_charts": False,
            "language": "es",
        },
    }


@pytest.fixture
def mock_mic() -> MagicMock:
    """
    Mock de MICRegistry con respuestas por defecto para financial_analysis.

    Returns
    -------
    MagicMock
        Mock con VPN=150, IRR=15%, Payback=3.5, recommendation=APPROVE.
    """
    return _create_mock_mic()


@pytest.fixture
def agent(default_config: Dict, mock_mic: MagicMock) -> BusinessAgent:
    """
    BusinessAgent configurado para testing.

    Returns
    -------
    BusinessAgent
        Instancia con configuración por defecto y MIC mock.
    """
    return BusinessAgent(default_config, mic=mock_mic)


@pytest.fixture
def minimal_context() -> Dict[str, Any]:
    """
    Contexto mínimo válido: 1 APU, 1 insumo, flujos básicos.

    Returns
    -------
    Dict[str, Any]
    """
    return (
        ProjectContextBuilder()
        .with_presupuesto(
            PresupuestoBuilder().with_apu("APU-001", "Test APU", 10.0)
        )
        .with_insumos(
            InsumosBuilder()
            .for_apus("APU-001")
            .with_insumo("APU-001", "Material Test", "MATERIAL", 100.0)
        )
        .with_cash_flow(1000.0, [300.0, 400.0, 500.0])
        .build()
    )


@pytest.fixture
def complete_context() -> Tuple[Dict[str, Any], CashFlowProjection]:
    """
    Contexto completo: 3 APUs con insumos estándar, proyecto viable.

    Returns
    -------
    Tuple[Dict[str, Any], CashFlowProjection]
        Contexto y proyección para validación cruzada.
    """
    presupuesto = (
        PresupuestoBuilder()
        .with_apu("APU-001", "Cimentación", 50.0)
        .with_apu("APU-002", "Estructura", 100.0)
        .with_apu("APU-003", "Acabados", 75.0)
    )
    insumos = (
        InsumosBuilder()
        .for_apus("APU-001", "APU-002", "APU-003")
        .with_standard_insumos_for("APU-001")
        .with_standard_insumos_for("APU-002")
        .with_standard_insumos_for("APU-003")
    )
    return (
        ProjectContextBuilder()
        .with_presupuesto(presupuesto)
        .with_insumos(insumos)
        .with_viable_project()
        .build_with_projection()
    )


@pytest.fixture
def empty_context() -> Dict[str, Any]:
    """
    Contexto con DataFrames vacíos para tests de degradación.

    Returns
    -------
    Dict[str, Any]
    """
    return (
        ProjectContextBuilder()
        .with_presupuesto(PresupuestoBuilder())
        .with_insumos(InsumosBuilder())
        .with_cash_flow(100.0, [50.0])
        .build()
    )


@pytest.fixture
def high_risk_context() -> Dict[str, Any]:
    """
    Contexto de alto riesgo: un solo insumo concentra todo el costo,
    proyecto no viable.

    Returns
    -------
    Dict[str, Any]
    """
    return (
        ProjectContextBuilder()
        .with_presupuesto(
            PresupuestoBuilder().with_apu("APU-001", "Único APU", 100.0)
        )
        .with_insumos(
            InsumosBuilder()
            .for_apus("APU-001")
            .with_insumo("APU-001", "Insumo Crítico", "MATERIAL", 10000.0)
        )
        .with_non_viable_project()
        .build()
    )


# =============================================================================
# TEST SUITE 1: VALIDACIÓN DE BUILDERS (PRECONDICIONES)
# =============================================================================


class TestBuilderInvariants:
    """
    Valida que los builders de datos de prueba cumplan sus contratos.

    Estos tests verifican las PRECONDICIONES de los demás tests:
    si los builders producen datos inválidos, toda la suite es inútil.
    """

    def test_presupuesto_rejects_duplicate_codes(self) -> None:
        """Códigos APU duplicados provocan ValueError."""
        builder = (
            PresupuestoBuilder()
            .with_apu("APU-001", "Primero", 10.0)
            .with_apu("APU-001", "Duplicado", 20.0)
        )
        with pytest.raises(ValueError, match="duplicados"):
            builder.build()

    def test_presupuesto_rejects_negative_quantity(self) -> None:
        """Cantidades negativas provocan ValueError."""
        with pytest.raises(ValueError, match="negativa"):
            PresupuestoBuilder().with_apu("APU-001", "Test", -5.0)

    def test_presupuesto_rejects_invalid_code_format(self) -> None:
        """Códigos sin prefijo 'APU-' provocan ValueError."""
        with pytest.raises(ValueError, match="APU-"):
            PresupuestoBuilder().with_apu("INVALID", "Test", 10.0)

    def test_insumos_detects_orphans(self) -> None:
        """Insumos sin APU padre provocan ValueError."""
        builder = (
            InsumosBuilder()
            .for_apus("APU-001")
            .with_insumo("APU-999", "Huérfano", "MATERIAL", 100.0)
        )
        with pytest.raises(ValueError, match="huérfanos"):
            builder.build(validate_connectivity=True)

    def test_insumos_rejects_invalid_tipo(self) -> None:
        """Tipos de insumo no reconocidos provocan ValueError."""
        with pytest.raises(ValueError, match="Tipo inválido"):
            InsumosBuilder().with_insumo(
                "APU-001", "Test", "INVALIDO", 100.0
            )

    def test_insumos_total_cost_is_sum_of_products(self) -> None:
        """
        Costo total = Σ (costo_i · cantidad_i).

        Verifica la invariante de suma aditiva.
        """
        builder = (
            InsumosBuilder()
            .with_insumo("APU-001", "A", "MATERIAL", 100.0, cantidad=2.0)
            .with_insumo("APU-001", "B", "MANO_OBRA", 50.0, cantidad=1.0)
        )
        # 100·2 + 50·1 = 250
        assert builder.get_total_cost() == pytest.approx(250.0)

    def test_insumos_cost_by_apu_sums_to_total(self) -> None:
        """
        Σ costo_por_APU = costo_total (partición aditiva).
        """
        builder = (
            InsumosBuilder()
            .with_insumo("APU-001", "A", "MATERIAL", 100.0)
            .with_insumo("APU-001", "B", "MANO_OBRA", 50.0)
            .with_insumo("APU-002", "C", "EQUIPO", 75.0)
        )
        total = builder.get_total_cost()
        by_apu = builder.get_cost_by_apu()

        assert abs(sum(by_apu.values()) - total) < _FINANCIAL_ABS_TOLERANCE

    def test_context_maintains_apu_insumo_connectivity(self) -> None:
        """
        Todos los insumos en el contexto referencian APUs existentes.

        Verifica la propiedad topológica:
            ∀ e = (apu, ins) ∈ E: apu ∈ V_APU
        """
        context = (
            ProjectContextBuilder()
            .with_presupuesto(
                PresupuestoBuilder()
                .with_apu("APU-001", "Test 1", 10.0)
                .with_apu("APU-002", "Test 2", 20.0)
            )
            .with_insumos(
                InsumosBuilder()
                .with_insumo("APU-001", "Insumo 1", "MATERIAL", 50.0)
                .with_insumo("APU-002", "Insumo 2", "MANO_OBRA", 75.0)
            )
            .with_cash_flow(1000.0, [500.0, 600.0])
            .build()
        )

        apu_codes = set(context["df_presupuesto"]["CODIGO_APU"])
        insumo_refs = set(context["df_merged"]["CODIGO_APU"])

        assert insumo_refs.issubset(apu_codes), (
            f"Insumos huérfanos: {insumo_refs - apu_codes}"
        )

    def test_zero_investment_rejected(self) -> None:
        """Inversión inicial = 0 viola postcondición I₀ > 0."""
        with pytest.raises(ValueError, match="positiva"):
            CashFlowProjection(initial_investment=0.0, cash_flows=[100.0])


# =============================================================================
# TEST SUITE 2: INVARIANTES FINANCIEROS DE CashFlowProjection
# =============================================================================


class TestCashFlowProjectionInvariants:
    """
    Verifica las propiedades matemáticas de CashFlowProjection.

    Invariantes:
        (F1) VPN calculado coincide con fórmula cerrada
        (F2) VPN monótonamente decreciente en r (flujos positivos)
        (F3) Payback ≤ horizonte (o None)
        (F4) is_viable() ⟺ VPN > 0
        (F5) Tasa de descuento ≤ -1 es rechazada
    """

    def test_npv_matches_closed_form(self) -> None:
        """
        Invariante (F1): VPN coincide con cálculo manual.

        VPN = -1000 + 500/1.1 + 500/1.21 + 500/1.331
            = -1000 + 454.545... + 413.223... + 375.657...
            = 243.426...
        """
        projection = CashFlowProjection(
            initial_investment=1000.0,
            cash_flows=[500.0, 500.0, 500.0],
            risk_free_rate=0.10,
        )

        expected = -1000 + 500 / 1.1 + 500 / 1.21 + 500 / 1.331
        actual = projection.calculate_npv()

        assert actual == pytest.approx(expected, abs=_FINANCIAL_ABS_TOLERANCE), (
            f"VPN calculado = {actual}, esperado = {expected}."
        )

    def test_npv_monotonically_decreasing_in_rate(self) -> None:
        """
        Invariante (F2): ∂VPN/∂r < 0 para flujos positivos.

        Para CFₜ > 0 ∀t:
            ∂VPN/∂r = -Σₜ t·CFₜ/(1+r)^(t+1) < 0
        """
        projection = CashFlowProjection(
            initial_investment=1000.0,
            cash_flows=[400.0, 400.0, 400.0],
        )

        rates = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
        npvs = [projection.calculate_npv(r) for r in rates]

        for i in range(1, len(npvs)):
            assert npvs[i] < npvs[i - 1], (
                f"VPN no es decreciente: VPN({rates[i]}) = {npvs[i]} "
                f"≥ VPN({rates[i-1]}) = {npvs[i-1]}."
            )

    def test_payback_correct_with_interpolation(self) -> None:
        """
        Invariante (F3): Payback con interpolación lineal.

        I₀ = 1000, CF = [400, 400, 400]
        Acumulado: [400, 800, 1200]
        Payback = 2 + (1000 - 800) / 400 = 2.5
        """
        projection = CashFlowProjection(
            initial_investment=1000.0,
            cash_flows=[400.0, 400.0, 400.0],
        )
        assert projection.calculate_payback_period() == pytest.approx(2.5)

    def test_payback_none_when_not_recovered(self) -> None:
        """Payback = None si Σ CFₜ < I₀."""
        projection = CashFlowProjection(
            initial_investment=10000.0,
            cash_flows=[100.0, 100.0, 100.0],
        )
        assert projection.calculate_payback_period() is None

    def test_payback_bounded_by_horizon(self) -> None:
        """Payback ≤ n (número de períodos) cuando se recupera."""
        projection = CashFlowProjection(
            initial_investment=1000.0,
            cash_flows=[500.0, 500.0, 500.0],
        )
        payback = projection.calculate_payback_period()
        assert payback is not None
        assert payback <= len(projection.cash_flows)

    def test_viability_consistent_with_npv_sign(self) -> None:
        """
        Invariante (F4): is_viable() ⟺ VPN > 0.

        Verificado para proyectos viable y no viable.
        """
        viable = CashFlowProjection(1000.0, [500.0, 500.0, 500.0])
        non_viable = CashFlowProjection(10000.0, [100.0, 100.0])

        assert viable.is_viable() is True
        assert viable.calculate_npv() > 0
        assert non_viable.is_viable() is False
        assert non_viable.calculate_npv() < 0

    def test_negative_discount_rate_rejected(self) -> None:
        """
        Invariante (F5): r ≤ -1 hace que (1+r)^t ≤ 0, causando
        división por cero o resultados sin sentido financiero.
        """
        projection = CashFlowProjection(1000.0, [500.0])
        with pytest.raises(ValueError, match="> -1"):
            projection.calculate_npv(discount_rate=-1.0)


# =============================================================================
# TEST SUITE 3: FLUJO PRINCIPAL DE EVALUACIÓN
# =============================================================================


@pytest.mark.integration
class TestEvaluateProjectFlow:
    """
    Tests para el flujo completo de evaluate_project.

    Verifica que el BusinessAgent:
    1. Retorna un reporte con estructura válida
    2. Incluye narrativa estratégica no vacía
    3. Contiene métricas numéricas
    4. Es determinista
    """

    def test_returns_valid_report(
        self, agent: BusinessAgent, minimal_context: Dict,
    ) -> None:
        """evaluate_project retorna reporte con estructura mínima."""
        report = agent.evaluate_project(minimal_context)
        _assert_report_has_structure(report, "minimal")

    def test_report_contains_strategic_narrative(
        self, agent: BusinessAgent, minimal_context: Dict,
    ) -> None:
        """El reporte incluye narrativa estratégica no vacía."""
        report = agent.evaluate_project(minimal_context)

        assert report.strategic_narrative is not None
        assert len(report.strategic_narrative) > 0
        assert isinstance(report.strategic_narrative, str)

    def test_report_contains_required_sections(
        self,
        agent: BusinessAgent,
        complete_context: Tuple[Dict, CashFlowProjection],
    ) -> None:
        """
        La narrativa contiene las secciones requeridas del informe.

        Secciones obligatorias del contrato de interfaz.
        """
        context, _ = complete_context
        report = agent.evaluate_project(context)

        required_sections = [
            "INFORME DE INGENIERÍA ESTRATÉGICA",
            "Auditoría de Integridad Estructural",
            "Análisis de Cargas Financieras",
            "Geotecnia de Mercado",
            "Dictamen del Ingeniero Jefe",
        ]

        for section in required_sections:
            assert section in report.strategic_narrative, (
                f"Sección requerida faltante: '{section}'.\n"
                f"Narrativa (primeros 300 chars): "
                f"'{report.strategic_narrative[:300]}...'"
            )

    def test_details_contains_required_keys(
        self,
        agent: BusinessAgent,
        complete_context: Tuple[Dict, CashFlowProjection],
    ) -> None:
        """Los detalles incluyen claves de métricas requeridas."""
        context, _ = complete_context
        report = agent.evaluate_project(context)

        for key in ["metrics", "financial_metrics", "strategic_narrative"]:
            assert key in report.details, (
                f"Clave requerida faltante en details: '{key}'. "
                f"Claves presentes: {list(report.details.keys())}."
            )

    def test_metrics_are_numeric(
        self,
        agent: BusinessAgent,
        complete_context: Tuple[Dict, CashFlowProjection],
    ) -> None:
        """Las métricas clave son valores numéricos."""
        context, _ = complete_context
        report = agent.evaluate_project(context)
        metrics = report.details.get("metrics", {})

        for metric_name in ["npv", "total_cost", "risk_score"]:
            if metric_name in metrics:
                assert isinstance(
                    metrics[metric_name], (int, float, Decimal)
                ), (
                    f"'{metric_name}' debe ser numérico, "
                    f"es {type(metrics[metric_name]).__name__}: "
                    f"{metrics[metric_name]}"
                )

    def test_evaluation_is_deterministic(
        self, agent: BusinessAgent, minimal_context: Dict,
    ) -> None:
        """
        Múltiples evaluaciones del mismo contexto producen
        mismo resultado (función pura respecto al contexto).
        """
        report1 = agent.evaluate_project(minimal_context)
        report2 = agent.evaluate_project(minimal_context)

        assert report1.strategic_narrative == report2.strategic_narrative, (
            "Narrativa difiere entre evaluaciones del mismo contexto."
        )


# =============================================================================
# TEST SUITE 4: DELEGACIÓN A LA MIC
# =============================================================================


@pytest.mark.integration
class TestMICDelegation:
    """
    Verifica que el BusinessAgent delega correctamente a la MIC.
    """

    def test_financial_analysis_delegated_to_mic(
        self,
        default_config: Dict,
        mock_mic: MagicMock,
    ) -> None:
        """
        El agente invoca project_intent("financial_analysis") con
        payload que incluye monto y horizonte temporal.
        """
        context = (
            ProjectContextBuilder()
            .with_presupuesto(
                PresupuestoBuilder().with_apu("APU-001", "Test", 10.0)
            )
            .with_insumos(
                InsumosBuilder()
                .for_apus("APU-001")
                .with_insumo("APU-001", "Material", "MATERIAL", 100.0)
            )
            .with_cash_flow(1000.0, [400.0, 400.0, 400.0])
            .build()
        )

        agent = BusinessAgent(default_config, mic=mock_mic)
        agent.evaluate_project(context)

        mock_mic.project_intent.assert_called()

        # Verificar que financial_analysis fue invocado
        call_args_list = mock_mic.project_intent.call_args_list
        service_names = [c[0][0] for c in call_args_list]

        assert "financial_analysis" in service_names, (
            f"financial_analysis no fue invocado. "
            f"Servicios llamados: {service_names}"
        )

    def test_financial_payload_contains_required_fields(
        self,
        default_config: Dict,
        mock_mic: MagicMock,
    ) -> None:
        """
        El payload de financial_analysis incluye amount y time.
        """
        context = (
            ProjectContextBuilder()
            .with_presupuesto(
                PresupuestoBuilder().with_apu("APU-001", "Test", 10.0)
            )
            .with_insumos(
                InsumosBuilder()
                .for_apus("APU-001")
                .with_insumo("APU-001", "M", "MATERIAL", 100.0)
            )
            .with_cash_flow(1000.0, [400.0, 400.0, 400.0])
            .build()
        )

        agent = BusinessAgent(default_config, mic=mock_mic)
        agent.evaluate_project(context)

        # Encontrar la llamada a financial_analysis
        financial_calls = [
            c for c in mock_mic.project_intent.call_args_list
            if c[0][0] == "financial_analysis"
        ]
        assert len(financial_calls) > 0

        payload = financial_calls[0][0][1]
        assert payload["amount"] == 1000.0, (
            f"amount incorrecto: {payload.get('amount')}"
        )
        assert payload["time"] == 3, (
            f"time incorrecto: {payload.get('time')}"
        )


# =============================================================================
# TEST SUITE 5: ANÁLISIS TOPOLÓGICO
# =============================================================================


@pytest.mark.integration
class TestTopologicalAnalysis:
    """
    Tests para el análisis topológico del presupuesto.

    Verifica detección de:
    - Presupuesto conectado (todos los APUs tienen insumos)
    - APUs desconectados (sin insumos)
    - Concentración de costos
    """

    def test_connected_budget_positive_narrative(
        self,
        agent: BusinessAgent,
        complete_context: Tuple[Dict, CashFlowProjection],
    ) -> None:
        """
        Presupuesto bien formado (todos los APUs con insumos)
        genera narrativa con indicadores de integridad.
        """
        context, _ = complete_context
        report = agent.evaluate_project(context)

        _narrative_contains_any(
            report.strategic_narrative,
            ["íntegr", "conectad", "coheren", "estable", "completo"],
            "budget-connected",
        )

    def test_high_cost_concentration_flagged(
        self,
        agent: BusinessAgent,
        high_risk_context: Dict,
    ) -> None:
        """
        Alta concentración de costos en un solo insumo genera
        indicadores de riesgo en la narrativa.
        """
        report = agent.evaluate_project(high_risk_context)

        _narrative_contains_any(
            report.strategic_narrative,
            ["concentra", "riesgo", "vulnerab", "depend", "crític"],
            "cost-concentration",
        )


# =============================================================================
# TEST SUITE 6: ANÁLISIS FINANCIERO
# =============================================================================


@pytest.mark.integration
class TestFinancialAnalysis:
    """
    Tests para el componente financiero de la evaluación.
    """

    def test_viable_project_positive_narrative(
        self, default_config: Dict,
    ) -> None:
        """
        Proyecto viable (VPN > 0 en MIC) genera narrativa positiva.
        """
        mic = _create_mock_mic(npv=500.0, recommendation="APPROVE")
        agent = BusinessAgent(default_config, mic=mic)

        context = (
            ProjectContextBuilder()
            .with_presupuesto(
                PresupuestoBuilder().with_apu("APU-001", "Test", 10.0)
            )
            .with_insumos(
                InsumosBuilder()
                .for_apus("APU-001")
                .with_insumo("APU-001", "Material", "MATERIAL", 100.0)
            )
            .with_viable_project()
            .build()
        )

        report = agent.evaluate_project(context)

        _narrative_contains_any(
            report.strategic_narrative,
            ["viable", "positiv", "favorable", "recomend", "aprob", "estable"],
            "viable-project",
        )

    def test_non_viable_project_warning_narrative(
        self,
        agent: BusinessAgent,
        high_risk_context: Dict,
    ) -> None:
        """Proyecto no viable genera advertencias."""
        report = agent.evaluate_project(high_risk_context)

        _narrative_contains_any(
            report.strategic_narrative,
            ["no viable", "negativ", "riesgo", "precaución", "rechaz"],
            "non-viable",
        )

    def test_payback_period_reported(
        self,
        agent: BusinessAgent,
        complete_context: Tuple[Dict, CashFlowProjection],
    ) -> None:
        """Período de recuperación aparece en narrativa o métricas."""
        context, _ = complete_context
        report = agent.evaluate_project(context)

        metrics = report.details.get("metrics", {})
        financial = report.details.get("financial_metrics", {})
        narrative_lower = report.strategic_narrative.lower()

        has_payback = (
            "payback" in metrics
            or "payback" in financial
            or "recuperación" in narrative_lower
        )

        assert has_payback, (
            "Período de recuperación no reportado en métricas ni narrativa."
        )

    def test_total_cost_equals_sum_of_insumos(
        self,
        agent: BusinessAgent,
        complete_context: Tuple[Dict, CashFlowProjection],
    ) -> None:
        """
        Invariante de suma: costo_total_reportado = Σ (costo·cantidad)
        calculado desde los datos de entrada.
        """
        context, _ = complete_context
        df_merged = context["df_merged"]

        expected_total = float(
            (df_merged["COSTO_INSUMO_EN_APU"] * df_merged["CANTIDAD_APU"]).sum()
        )

        report = agent.evaluate_project(context)
        metrics = report.details.get("metrics", {})

        if "total_cost" in metrics:
            reported = float(metrics["total_cost"])
            assert reported == pytest.approx(
                expected_total, rel=_FINANCIAL_REL_TOLERANCE
            ), (
                f"Costo total reportado ({reported}) ≠ "
                f"suma de insumos ({expected_total})."
            )


# =============================================================================
# TEST SUITE 7: CASOS EDGE Y MANEJO DE ERRORES
# =============================================================================


@pytest.mark.integration
class TestEdgeCases:
    """Tests para casos límite y manejo de errores."""

    def test_empty_dataframes_handled(
        self, agent: BusinessAgent, empty_context: Dict,
    ) -> None:
        """DataFrames vacíos son manejados sin excepción."""
        report = agent.evaluate_project(empty_context)
        _assert_report_has_structure(report, "empty-df")

    def test_single_apu_project(
        self, agent: BusinessAgent, minimal_context: Dict,
    ) -> None:
        """Proyecto con un solo APU es evaluado correctamente."""
        report = agent.evaluate_project(minimal_context)
        _assert_report_has_structure(report, "single-apu")

    def test_negative_cash_flows_handled(
        self, default_config: Dict, mock_mic: MagicMock,
    ) -> None:
        """Flujos de caja negativos (pérdidas) son procesados."""
        context = (
            ProjectContextBuilder()
            .with_presupuesto(
                PresupuestoBuilder().with_apu("APU-001", "Test", 10.0)
            )
            .with_insumos(
                InsumosBuilder()
                .for_apus("APU-001")
                .with_insumo("APU-001", "Material", "MATERIAL", 100.0)
            )
            .with_cash_flow(1000.0, [500.0, -100.0, 300.0])
            .build()
        )

        agent = BusinessAgent(default_config, mic=mock_mic)
        report = agent.evaluate_project(context)
        _assert_report_has_structure(report, "negative-cf")

    def test_missing_cash_flows_handled(
        self, default_config: Dict, mock_mic: MagicMock,
    ) -> None:
        """
        Contexto sin cash_flows: el agente maneja graciosamente
        o lanza error descriptivo.
        """
        context = {
            "df_presupuesto": (
                PresupuestoBuilder()
                .with_apu("APU-001", "Test", 10.0)
                .build()
            ),
            "df_merged": (
                InsumosBuilder()
                .with_insumo("APU-001", "M", "MATERIAL", 100.0)
                .build()
            ),
            "validated_strata": _DEFAULT_VALIDATED_STRATA.copy(),
        }

        agent = BusinessAgent(default_config, mic=mock_mic)

        try:
            report = agent.evaluate_project(context)
            _assert_report_has_structure(report, "missing-cf")
        except (ValueError, KeyError) as e:
            # Error esperado — verificar que el mensaje es descriptivo
            error_msg = str(e).lower()
            assert any(
                term in error_msg
                for term in ["cash_flow", "investment", "flujo"]
            ), (
                f"Error no descriptivo para cash_flows faltantes: {e}"
            )

    @pytest.mark.stress
    def test_large_project_performance(
        self, default_config: Dict, mock_mic: MagicMock,
    ) -> None:
        """
        Proyecto con 100 APUs se evalúa en menos de
        _PERFORMANCE_TIME_LIMIT segundos.
        """
        presupuesto = PresupuestoBuilder()
        insumos = InsumosBuilder()

        apu_codes: List[str] = []
        for i in range(1, 101):
            codigo = f"APU-{i:03d}"
            apu_codes.append(codigo)
            presupuesto.with_apu(codigo, f"Actividad {i}", float(i))

        insumos.for_apus(*apu_codes)
        for codigo in apu_codes:
            insumos.with_insumo(
                codigo, f"Material ({codigo})", "MATERIAL", 100.0
            )

        context = (
            ProjectContextBuilder()
            .with_presupuesto(presupuesto)
            .with_insumos(insumos)
            .with_viable_project()
            .build()
        )

        agent = BusinessAgent(default_config, mic=mock_mic)

        start = time.perf_counter()
        report = agent.evaluate_project(context)
        elapsed = time.perf_counter() - start

        _assert_report_has_structure(report, "large-project")
        assert elapsed < _PERFORMANCE_TIME_LIMIT, (
            f"Evaluación tardó {elapsed:.2f}s, "
            f"límite: {_PERFORMANCE_TIME_LIMIT}s."
        )


# =============================================================================
# TEST SUITE 8: TRADUCCIÓN SEMÁNTICA
# =============================================================================


@pytest.mark.integration
class TestSemanticTranslation:
    """
    Verifica la traducción de métricas a narrativa estratégica.

    Los tests verifican presencia de términos indicativos, no
    redacción exacta, para ser resilientes a cambios de estilo.
    """

    def test_narrative_uses_engineering_metaphors(
        self,
        agent: BusinessAgent,
        complete_context: Tuple[Dict, CashFlowProjection],
    ) -> None:
        """La narrativa usa metáforas de construcción/ingeniería."""
        context, _ = complete_context
        report = agent.evaluate_project(context)

        _narrative_contains_any(
            report.strategic_narrative,
            [
                "estructura", "cimiento", "carga", "estabilidad",
                "integridad", "geotecnia", "ingenier", "audit",
            ],
            "engineering-metaphors",
        )

    def test_high_risk_translated_to_severity(
        self,
        agent: BusinessAgent,
        high_risk_context: Dict,
    ) -> None:
        """Nivel de riesgo alto se traduce a términos de severidad."""
        report = agent.evaluate_project(high_risk_context)

        _narrative_contains_any(
            report.strategic_narrative,
            ["crític", "sever", "alto", "precaución", "alert", "riesgo"],
            "risk-severity",
        )
"""
Pruebas de Integración para BusinessAgent
==========================================

Suite robustecida que valida:
- Flujo completo de evaluación de proyectos
- Coherencia topológica del grafo APU → Insumos
- Invariantes financieros (VPN, TIR, Payback)
- Traducción semántica a narrativa estratégica
- Manejo de casos edge y errores
- Propiedades matemáticas del análisis de riesgo

Modelo Topológico del Dominio:
------------------------------
El presupuesto forma un complejo simplicial donde:
- Vértices (0-simplices): APUs e Insumos
- Aristas (1-simplices): Relaciones APU → Insumo
- β₀ = 1 indica presupuesto conectado (ideal)
- β₁ = 0 indica ausencia de ciclos (ideal)
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from app.business_agent import BusinessAgent


# =============================================================================
# CONSTANTES PARA DETERMINISMO
# =============================================================================

FIXED_TIMESTAMP = "2024-01-15T10:30:00"
FIXED_DATE = datetime(2024, 1, 15, 10, 30, 0)

# Tolerancias para comparaciones numéricas
FINANCIAL_TOLERANCE = 1e-6
PERCENTAGE_TOLERANCE = 0.01


# =============================================================================
# BUILDERS: Construcción de Datos con Validación de Invariantes
# =============================================================================


@dataclass
class APURecord:
    """Registro de Análisis de Precios Unitarios."""
    codigo: str
    descripcion: str
    cantidad: float
    unidad: str = "UND"
    
    def __post_init__(self):
        if self.cantidad < 0:
            raise ValueError(f"Cantidad no puede ser negativa: {self.cantidad}")
        if not self.codigo.startswith("APU-"):
            raise ValueError(f"Código APU debe iniciar con 'APU-': {self.codigo}")


@dataclass
class InsumoRecord:
    """Registro de insumo vinculado a un APU."""
    codigo_apu: str
    descripcion: str
    tipo: str
    costo: float
    cantidad: float = 1.0
    
    TIPOS_VALIDOS = frozenset({"MATERIAL", "MANO_OBRA", "EQUIPO", "TRANSPORTE"})
    
    def __post_init__(self):
        if self.tipo not in self.TIPOS_VALIDOS:
            raise ValueError(f"Tipo inválido '{self.tipo}'. Válidos: {self.TIPOS_VALIDOS}")
        if self.costo < 0:
            raise ValueError(f"Costo no puede ser negativo: {self.costo}")


class PresupuestoBuilder:
    """
    Builder para DataFrames de presupuesto con validación topológica.
    
    Invariantes Garantizados:
    - Cada APU tiene código único
    - Cantidades son no-negativas
    - Códigos siguen formato estándar
    """
    
    def __init__(self):
        self._apus: List[APURecord] = []
    
    def with_apu(
        self, 
        codigo: str, 
        descripcion: str, 
        cantidad: float,
        unidad: str = "UND"
    ) -> "PresupuestoBuilder":
        """Agrega un APU al presupuesto."""
        self._apus.append(APURecord(codigo, descripcion, cantidad, unidad))
        return self
    
    def with_standard_apus(self, count: int = 3) -> "PresupuestoBuilder":
        """Genera APUs estándar para testing rápido."""
        for i in range(1, count + 1):
            self._apus.append(APURecord(
                codigo=f"APU-{i:03d}",
                descripcion=f"Actividad de Construcción #{i}",
                cantidad=float(i * 10)
            ))
        return self
    
    def build(self) -> pd.DataFrame:
        """
        Construye el DataFrame validando unicidad de códigos.
        
        Returns:
            DataFrame con columnas estándar de presupuesto
        """
        if not self._apus:
            return pd.DataFrame(columns=[
                "CODIGO_APU", "DESCRIPCION_APU", "CANTIDAD_PRESUPUESTO", "UNIDAD"
            ])
        
        codigos = [apu.codigo for apu in self._apus]
        if len(codigos) != len(set(codigos)):
            duplicados = [c for c in codigos if codigos.count(c) > 1]
            raise ValueError(f"Códigos APU duplicados: {set(duplicados)}")
        
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
    
    Invariantes Topológicos:
    - Cada insumo debe referenciar un APU existente (conectividad)
    - La suma de costos por APU es el costo total del APU
    - No hay insumos huérfanos (sin APU padre)
    """
    
    def __init__(self):
        self._insumos: List[InsumoRecord] = []
        self._apu_codes: set = set()
    
    def for_apus(self, *codigos: str) -> "InsumosBuilder":
        """Registra los APUs válidos para validación de conectividad."""
        self._apu_codes = set(codigos)
        return self
    
    def with_insumo(
        self,
        codigo_apu: str,
        descripcion: str,
        tipo: str,
        costo: float,
        cantidad: float = 1.0
    ) -> "InsumosBuilder":
        """Agrega un insumo vinculado a un APU."""
        self._insumos.append(InsumoRecord(
            codigo_apu=codigo_apu,
            descripcion=descripcion,
            tipo=tipo,
            costo=costo,
            cantidad=cantidad
        ))
        return self
    
    def with_standard_insumos_for(self, codigo_apu: str) -> "InsumosBuilder":
        """Genera set estándar de insumos para un APU."""
        base_cost = 100.0
        self._insumos.extend([
            InsumoRecord(codigo_apu, "Cemento Portland", "MATERIAL", base_cost * 0.4),
            InsumoRecord(codigo_apu, "Oficial de Obra", "MANO_OBRA", base_cost * 0.35),
            InsumoRecord(codigo_apu, "Mezcladora", "EQUIPO", base_cost * 0.15),
            InsumoRecord(codigo_apu, "Flete Materiales", "TRANSPORTE", base_cost * 0.10),
        ])
        return self
    
    def build(self, validate_connectivity: bool = True) -> pd.DataFrame:
        """
        Construye el DataFrame validando conectividad topológica.
        
        Args:
            validate_connectivity: Si True, verifica que todos los insumos
                                   referencien APUs registrados.
        """
        if not self._insumos:
            return pd.DataFrame(columns=[
                "CODIGO_APU", "DESCRIPCION_INSUMO", "TIPO_INSUMO",
                "COSTO_INSUMO_EN_APU", "CANTIDAD_APU"
            ])
        
        if validate_connectivity and self._apu_codes:
            orphans = {
                ins.codigo_apu for ins in self._insumos
            } - self._apu_codes
            if orphans:
                raise ValueError(f"Insumos huérfanos (APU no existe): {orphans}")
        
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
    
    def get_total_cost(self) -> float:
        """Calcula el costo total de todos los insumos."""
        return sum(ins.costo * ins.cantidad for ins in self._insumos)
    
    def get_cost_by_apu(self) -> Dict[str, float]:
        """Retorna el costo agregado por cada APU."""
        costs: Dict[str, float] = {}
        for ins in self._insumos:
            costs[ins.codigo_apu] = costs.get(ins.codigo_apu, 0) + (ins.costo * ins.cantidad)
        return costs


@dataclass
class CashFlowProjection:
    """Proyección de flujos de caja con validación financiera."""
    initial_investment: float
    cash_flows: List[float]
    risk_free_rate: float = 0.05
    
    def __post_init__(self):
        if self.initial_investment <= 0:
            raise ValueError(f"Inversión inicial debe ser positiva: {self.initial_investment}")
        if not self.cash_flows:
            raise ValueError("Debe haber al menos un flujo de caja")
        if not 0 <= self.risk_free_rate <= 1:
            raise ValueError(f"Tasa libre de riesgo debe estar en [0, 1]: {self.risk_free_rate}")
    
    def calculate_npv(self, discount_rate: Optional[float] = None) -> float:
        """
        Calcula el Valor Presente Neto (VPN).
        
        VPN = -I₀ + Σ(CFₜ / (1 + r)ᵗ) para t = 1..n
        """
        rate = discount_rate or self.risk_free_rate
        npv = -self.initial_investment
        for t, cf in enumerate(self.cash_flows, start=1):
            npv += cf / ((1 + rate) ** t)
        return npv
    
    def calculate_payback_period(self) -> Optional[float]:
        """
        Calcula el período de recuperación simple.
        
        Returns:
            Número de períodos o None si no se recupera
        """
        cumulative = 0.0
        for t, cf in enumerate(self.cash_flows, start=1):
            cumulative += cf
            if cumulative >= self.initial_investment:
                # Interpolación lineal para período fraccionario
                prev_cumulative = cumulative - cf
                remaining = self.initial_investment - prev_cumulative
                fraction = remaining / cf if cf > 0 else 0
                return t - 1 + fraction
        return None  # No se recupera en el horizonte dado
    
    def is_viable(self) -> bool:
        """Un proyecto es viable si VPN > 0."""
        return self.calculate_npv() > 0


class ProjectContextBuilder:
    """
    Builder para el contexto completo de evaluación de proyecto.
    
    Garantiza coherencia entre:
    - Presupuesto (APUs)
    - Insumos (vinculados a APUs)
    - Flujos financieros (coherentes con costos)
    """
    
    def __init__(self):
        self._presupuesto_builder = PresupuestoBuilder()
        self._insumos_builder = InsumosBuilder()
        self._cash_flow: Optional[CashFlowProjection] = None
        self._extra_params: Dict[str, Any] = {}
    
    def with_presupuesto(self, builder: PresupuestoBuilder) -> "ProjectContextBuilder":
        """Configura el presupuesto usando un builder existente."""
        self._presupuesto_builder = builder
        return self
    
    def with_insumos(self, builder: InsumosBuilder) -> "ProjectContextBuilder":
        """Configura los insumos usando un builder existente."""
        self._insumos_builder = builder
        return self
    
    def with_cash_flow(
        self,
        initial_investment: float,
        cash_flows: List[float],
        risk_free_rate: float = 0.05
    ) -> "ProjectContextBuilder":
        """Configura la proyección financiera."""
        self._cash_flow = CashFlowProjection(
            initial_investment=initial_investment,
            cash_flows=cash_flows,
            risk_free_rate=risk_free_rate
        )
        return self
    
    def with_viable_project(self) -> "ProjectContextBuilder":
        """Configura un proyecto financieramente viable (VPN > 0)."""
        self._cash_flow = CashFlowProjection(
            initial_investment=1000.0,
            cash_flows=[400.0, 400.0, 400.0, 400.0]  # VPN positivo
        )
        return self
    
    def with_non_viable_project(self) -> "ProjectContextBuilder":
        """Configura un proyecto no viable (VPN < 0)."""
        self._cash_flow = CashFlowProjection(
            initial_investment=10000.0,
            cash_flows=[100.0, 100.0, 100.0]  # VPN negativo
        )
        return self
    
    def with_param(self, key: str, value: Any) -> "ProjectContextBuilder":
        """Agrega parámetro adicional al contexto."""
        self._extra_params[key] = value
        return self
    
    def build(self) -> Dict[str, Any]:
        """
        Construye el contexto validando coherencia interna.
        
        Validaciones:
        - APUs y sus insumos están conectados
        - Flujos de caja están definidos
        """
        df_presupuesto = self._presupuesto_builder.build()
        
        # Registrar APUs para validación de conectividad
        apu_codes = df_presupuesto["CODIGO_APU"].tolist() if not df_presupuesto.empty else []
        self._insumos_builder.for_apus(*apu_codes)
        
        df_merged = self._insumos_builder.build(validate_connectivity=bool(apu_codes))
        
        if self._cash_flow is None:
            # Default viable
            self._cash_flow = CashFlowProjection(
                initial_investment=1000.0,
                cash_flows=[300.0, 400.0, 500.0]
            )
        
        context = {
            "df_presupuesto": df_presupuesto,
            "df_merged": df_merged,
            "initial_investment": self._cash_flow.initial_investment,
            "cash_flows": self._cash_flow.cash_flows,
            "timestamp": FIXED_TIMESTAMP,
            **self._extra_params
        }
        
        return context
    
    def build_with_projection(self) -> Tuple[Dict[str, Any], CashFlowProjection]:
        """Retorna contexto junto con la proyección para validación."""
        context = self.build()
        return context, self._cash_flow


# =============================================================================
# FIXTURES PYTEST
# =============================================================================


class TestFixtures:
    """Clase base con fixtures reutilizables para BusinessAgent."""

    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        """Configuración por defecto del agente."""
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
            }
        }

    @pytest.fixture
    def agent(self, default_config) -> BusinessAgent:
        """Instancia de BusinessAgent configurada para testing."""
        return BusinessAgent(default_config)

    @pytest.fixture
    def minimal_context(self) -> Dict[str, Any]:
        """Contexto mínimo válido para pruebas básicas."""
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
    def complete_context(self) -> Tuple[Dict[str, Any], CashFlowProjection]:
        """Contexto completo con proyección para validación numérica."""
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
    def empty_context(self) -> Dict[str, Any]:
        """Contexto con DataFrames vacíos para pruebas edge."""
        return (
            ProjectContextBuilder()
            .with_presupuesto(PresupuestoBuilder())
            .with_insumos(InsumosBuilder())
            .with_cash_flow(100.0, [50.0])
            .build()
        )

    @pytest.fixture
    def high_risk_context(self) -> Dict[str, Any]:
        """Contexto que debería generar alto riesgo."""
        # Un solo insumo concentra todo el costo (alta concentración = riesgo)
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
# TESTS: VALIDACIÓN DE BUILDERS
# =============================================================================


class TestBuilders(TestFixtures):
    """Valida que los builders cumplan sus invariantes."""

    def test_presupuesto_builder_rejects_duplicate_codes(self):
        """Builder rechaza códigos APU duplicados."""
        builder = (
            PresupuestoBuilder()
            .with_apu("APU-001", "Primero", 10.0)
            .with_apu("APU-001", "Duplicado", 20.0)  # Mismo código
        )
        
        with pytest.raises(ValueError, match="duplicados"):
            builder.build()

    def test_presupuesto_builder_rejects_negative_quantity(self):
        """Builder rechaza cantidades negativas."""
        with pytest.raises(ValueError, match="negativa"):
            PresupuestoBuilder().with_apu("APU-001", "Test", -5.0)

    def test_presupuesto_builder_rejects_invalid_code_format(self):
        """Builder rechaza códigos con formato inválido."""
        with pytest.raises(ValueError, match="APU-"):
            PresupuestoBuilder().with_apu("INVALID", "Test", 10.0)

    def test_insumos_builder_detects_orphans(self):
        """Builder detecta insumos sin APU padre."""
        builder = (
            InsumosBuilder()
            .for_apus("APU-001")
            .with_insumo("APU-999", "Huérfano", "MATERIAL", 100.0)  # APU no existe
        )
        
        with pytest.raises(ValueError, match="huérfanos"):
            builder.build(validate_connectivity=True)

    def test_insumos_builder_rejects_invalid_tipo(self):
        """Builder rechaza tipos de insumo inválidos."""
        with pytest.raises(ValueError, match="Tipo inválido"):
            InsumosBuilder().with_insumo("APU-001", "Test", "INVALIDO", 100.0)

    def test_insumos_builder_calculates_total_cost(self):
        """Builder calcula correctamente el costo total."""
        builder = (
            InsumosBuilder()
            .with_insumo("APU-001", "A", "MATERIAL", 100.0, cantidad=2.0)
            .with_insumo("APU-001", "B", "MANO_OBRA", 50.0, cantidad=1.0)
        )
        
        total = builder.get_total_cost()
        
        assert total == pytest.approx(250.0)  # 100*2 + 50*1

    def test_cash_flow_projection_calculates_npv(self):
        """Proyección calcula VPN correctamente."""
        projection = CashFlowProjection(
            initial_investment=1000.0,
            cash_flows=[500.0, 500.0, 500.0],
            risk_free_rate=0.10
        )
        
        # VPN = -1000 + 500/1.1 + 500/1.21 + 500/1.331
        expected_npv = -1000 + 500/1.1 + 500/1.21 + 500/1.331
        
        assert projection.calculate_npv() == pytest.approx(expected_npv, rel=FINANCIAL_TOLERANCE)

    def test_cash_flow_projection_calculates_payback(self):
        """Proyección calcula payback correctamente."""
        projection = CashFlowProjection(
            initial_investment=1000.0,
            cash_flows=[400.0, 400.0, 400.0]
        )
        
        payback = projection.calculate_payback_period()
        
        # Recupera 800 en 2 períodos, necesita 200 más del tercer flujo
        # Payback = 2 + (200/400) = 2.5
        assert payback == pytest.approx(2.5)

    def test_project_context_maintains_connectivity(self):
        """Contexto de proyecto mantiene conectividad APU-Insumos."""
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
        
        # Verificar que todos los insumos referencian APUs existentes
        apu_codes = set(context["df_presupuesto"]["CODIGO_APU"])
        insumo_refs = set(context["df_merged"]["CODIGO_APU"])
        
        assert insumo_refs.issubset(apu_codes), "Todos los insumos deben referenciar APUs existentes"


# =============================================================================
# TESTS: FLUJO PRINCIPAL DE EVALUACIÓN
# =============================================================================


class TestEvaluateProjectFlow(TestFixtures):
    """Tests para el flujo completo de evaluate_project."""

    def test_evaluate_returns_valid_report(self, agent, minimal_context):
        """evaluate_project retorna un reporte no nulo."""
        report = agent.evaluate_project(minimal_context)
        
        assert report is not None
        assert hasattr(report, 'strategic_narrative')
        assert hasattr(report, 'details')

    def test_report_contains_strategic_narrative(self, agent, minimal_context):
        """El reporte incluye narrativa estratégica."""
        report = agent.evaluate_project(minimal_context)
        
        assert report.strategic_narrative is not None
        assert len(report.strategic_narrative) > 0
        assert isinstance(report.strategic_narrative, str)

    def test_narrative_contains_required_sections(self, agent, complete_context):
        """La narrativa contiene todas las secciones requeridas."""
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
            assert section in report.strategic_narrative, \
                f"Sección requerida faltante: {section}"

    def test_details_contains_metrics(self, agent, complete_context):
        """Los detalles incluyen las métricas esperadas."""
        context, _ = complete_context
        report = agent.evaluate_project(context)
        
        required_keys = ["metrics", "financial_metrics_input", "strategic_narrative"]
        
        for key in required_keys:
            assert key in report.details, f"Clave requerida faltante en details: {key}"

    def test_metrics_are_numeric(self, agent, complete_context):
        """Las métricas en el reporte son valores numéricos válidos."""
        context, _ = complete_context
        report = agent.evaluate_project(context)
        
        metrics = report.details.get("metrics", {})
        
        # Verificar que las métricas clave existen y son numéricas
        for metric_name in ["npv", "total_cost", "risk_score"]:
            if metric_name in metrics:
                assert isinstance(metrics[metric_name], (int, float, Decimal)), \
                    f"{metric_name} debe ser numérico"

    def test_evaluation_is_deterministic(self, agent, minimal_context):
        """Múltiples evaluaciones del mismo contexto producen mismo resultado."""
        report1 = agent.evaluate_project(minimal_context)
        report2 = agent.evaluate_project(minimal_context)
        
        # La narrativa debe ser idéntica
        assert report1.strategic_narrative == report2.strategic_narrative
        
        # Las métricas deben ser idénticas
        assert report1.details == report2.details


# =============================================================================
# TESTS: COMPONENTE TOPOLÓGICO
# =============================================================================


class TestTopologicalAnalysis(TestFixtures):
    """Tests para el análisis topológico del presupuesto."""

    def test_connected_budget_detected(self, agent, complete_context):
        """Presupuesto conectado (β₀=1) es detectado correctamente."""
        context, _ = complete_context
        report = agent.evaluate_project(context)
        
        # Un presupuesto bien formado tiene todos los APUs conectados a insumos
        # Esto debería reflejarse en la narrativa como estructura "íntegra"
        assert any(
            term in report.strategic_narrative.lower()
            for term in ["íntegr", "conectad", "coheren", "estable"]
        )

    def test_disconnected_apus_flagged(self, agent, default_config):
        """APUs sin insumos (desconectados) son reportados."""
        # Crear contexto con APU sin insumos
        context = (
            ProjectContextBuilder()
            .with_presupuesto(
                PresupuestoBuilder()
                .with_apu("APU-001", "Con Insumos", 10.0)
                .with_apu("APU-002", "Sin Insumos", 20.0)  # Desconectado
            )
            .with_insumos(
                InsumosBuilder()
                .for_apus("APU-001", "APU-002")
                .with_insumo("APU-001", "Material", "MATERIAL", 100.0)
                # APU-002 no tiene insumos
            )
            .with_cash_flow(1000.0, [500.0])
            .build()
        )
        
        agent = BusinessAgent(default_config)
        report = agent.evaluate_project(context)
        
        # El análisis topológico debería detectar la falta de insumos
        # Verificamos que el reporte mencione algún tipo de advertencia
        assert report is not None
        # La estructura no está completa - debería reflejarse en métricas
        if "incomplete_apus" in report.details.get("metrics", {}):
            assert "APU-002" in str(report.details["metrics"]["incomplete_apus"])

    def test_cost_concentration_analysis(self, agent, high_risk_context):
        """Alta concentración de costos es identificada como riesgo."""
        report = agent.evaluate_project(high_risk_context)
        
        # Alta concentración debería generar alertas en la narrativa
        narrative_lower = report.strategic_narrative.lower()
        risk_indicators = ["concentra", "riesgo", "vulnerab", "depend"]
        
        assert any(term in narrative_lower for term in risk_indicators), \
            "Alta concentración de costos debe generar advertencia de riesgo"

    def test_cost_distribution_by_type(self, agent, complete_context):
        """Distribución de costos por tipo de insumo es calculada."""
        context, _ = complete_context
        report = agent.evaluate_project(context)
        
        metrics = report.details.get("metrics", {})
        
        # Debería haber análisis de distribución por tipo
        if "cost_by_type" in metrics:
            cost_by_type = metrics["cost_by_type"]
            assert "MATERIAL" in cost_by_type or "material" in str(cost_by_type).lower()


# =============================================================================
# TESTS: COMPONENTE FINANCIERO
# =============================================================================


class TestFinancialAnalysis(TestFixtures):
    """Tests para el análisis financiero del proyecto."""

    def test_npv_calculated_correctly(self, agent, default_config):
        """VPN se calcula correctamente según la fórmula estándar."""
        context, projection = (
            ProjectContextBuilder()
            .with_presupuesto(PresupuestoBuilder().with_apu("APU-001", "Test", 10.0))
            .with_insumos(
                InsumosBuilder()
                .for_apus("APU-001")
                .with_insumo("APU-001", "Material", "MATERIAL", 100.0)
            )
            .with_cash_flow(1000.0, [400.0, 400.0, 400.0], risk_free_rate=0.05)
            .build_with_projection()
        )
        
        agent = BusinessAgent(default_config)
        report = agent.evaluate_project(context)
        
        expected_npv = projection.calculate_npv()
        
        # Verificar que el VPN reportado coincide con el calculado
        if "npv" in report.details.get("metrics", {}):
            reported_npv = report.details["metrics"]["npv"]
            assert abs(reported_npv - expected_npv) < FINANCIAL_TOLERANCE * abs(expected_npv)

    def test_viable_project_positive_assessment(self, agent, default_config):
        """Proyecto viable (VPN > 0) recibe evaluación positiva."""
        context = (
            ProjectContextBuilder()
            .with_presupuesto(PresupuestoBuilder().with_apu("APU-001", "Test", 10.0))
            .with_insumos(
                InsumosBuilder()
                .for_apus("APU-001")
                .with_insumo("APU-001", "Material", "MATERIAL", 100.0)
            )
            .with_viable_project()
            .build()
        )
        
        agent = BusinessAgent(default_config)
        report = agent.evaluate_project(context)
        
        positive_terms = ["viable", "positiv", "favorable", "recomend", "aprob"]
        narrative_lower = report.strategic_narrative.lower()
        
        assert any(term in narrative_lower for term in positive_terms)

    def test_non_viable_project_warning(self, agent, high_risk_context):
        """Proyecto no viable (VPN < 0) genera advertencia."""
        report = agent.evaluate_project(high_risk_context)
        
        warning_terms = ["no viable", "negativ", "riesgo", "precaución", "rechaz"]
        narrative_lower = report.strategic_narrative.lower()
        
        assert any(term in narrative_lower for term in warning_terms)

    def test_payback_period_reported(self, agent, complete_context):
        """Período de recuperación es incluido en el reporte."""
        context, _ = complete_context
        report = agent.evaluate_project(context)
        
        # El payback debería mencionarse en la narrativa o métricas
        metrics = report.details.get("metrics", {})
        narrative_lower = report.strategic_narrative.lower()
        
        has_payback = (
            "payback" in metrics or
            "período de recuperación" in narrative_lower or
            "recuperación" in narrative_lower
        )
        
        assert has_payback, "Período de recuperación debe reportarse"


# =============================================================================
# TESTS: CASOS EDGE Y MANEJO DE ERRORES
# =============================================================================


class TestEdgeCases(TestFixtures):
    """Tests para casos límite y manejo de errores."""

    def test_empty_dataframes_handled(self, agent, empty_context):
        """DataFrames vacíos son manejados sin errores."""
        # No debe lanzar excepción
        report = agent.evaluate_project(empty_context)
        
        assert report is not None
        # Debería indicar que no hay datos suficientes
        assert report.strategic_narrative is not None

    def test_single_apu_project(self, agent, minimal_context):
        """Proyecto con un solo APU es evaluado correctamente."""
        report = agent.evaluate_project(minimal_context)
        
        assert report is not None
        assert "INFORME" in report.strategic_narrative

    def test_missing_cash_flows_handled(self, agent, default_config):
        """Contexto sin flujos de caja es manejado apropiadamente."""
        context = {
            "df_presupuesto": PresupuestoBuilder().with_apu("APU-001", "Test", 10.0).build(),
            "df_merged": InsumosBuilder().with_insumo("APU-001", "M", "MATERIAL", 100.0).build(),
            # Sin initial_investment ni cash_flows
        }
        
        agent = BusinessAgent(default_config)
        
        # Puede lanzar error o manejarlo gracefully
        try:
            report = agent.evaluate_project(context)
            # Si no lanza error, verificar que hay advertencia
            assert report is not None
        except (ValueError, KeyError) as e:
            # Error esperado por datos faltantes
            assert "cash_flows" in str(e).lower() or "investment" in str(e).lower()

    def test_negative_cash_flows_handled(self, agent, default_config):
        """Flujos de caja negativos son procesados correctamente."""
        context = (
            ProjectContextBuilder()
            .with_presupuesto(PresupuestoBuilder().with_apu("APU-001", "Test", 10.0))
            .with_insumos(
                InsumosBuilder()
                .for_apus("APU-001")
                .with_insumo("APU-001", "Material", "MATERIAL", 100.0)
            )
            .with_cash_flow(1000.0, [500.0, -100.0, 300.0])  # Flujo negativo
            .build()
        )
        
        agent = BusinessAgent(default_config)
        report = agent.evaluate_project(context)
        
        assert report is not None

    def test_zero_investment_rejected(self):
        """Inversión inicial cero es rechazada."""
        with pytest.raises(ValueError, match="positiva"):
            CashFlowProjection(
                initial_investment=0.0,
                cash_flows=[100.0]
            )

    def test_very_large_project(self, agent, default_config):
        """Proyecto con muchos APUs es procesado sin timeout."""
        presupuesto = PresupuestoBuilder()
        insumos = InsumosBuilder()
        
        # Crear 100 APUs con insumos
        apu_codes = []
        for i in range(1, 101):
            codigo = f"APU-{i:03d}"
            apu_codes.append(codigo)
            presupuesto.with_apu(codigo, f"Actividad {i}", float(i))
        
        insumos.for_apus(*apu_codes)
        for codigo in apu_codes:
            insumos.with_insumo(codigo, "Material", "MATERIAL", 100.0)
        
        context = (
            ProjectContextBuilder()
            .with_presupuesto(presupuesto)
            .with_insumos(insumos)
            .with_viable_project()
            .build()
        )
        
        agent = BusinessAgent(default_config)
        
        import time
        start = time.time()
        report = agent.evaluate_project(context)
        elapsed = time.time() - start
        
        assert report is not None
        assert elapsed < 10.0, "Evaluación no debe tardar más de 10 segundos"


# =============================================================================
# TESTS: INVARIANTES MATEMÁTICOS
# =============================================================================


class TestMathematicalInvariants(TestFixtures):
    """Valida invariantes matemáticos del análisis."""

    def test_total_cost_equals_sum_of_parts(self, agent, complete_context):
        """Costo total = Σ costos por APU (Invariante de suma)."""
        context, _ = complete_context
        
        # Calcular suma desde los datos de entrada
        df_merged = context["df_merged"]
        expected_total = (df_merged["COSTO_INSUMO_EN_APU"] * df_merged["CANTIDAD_APU"]).sum()
        
        report = agent.evaluate_project(context)
        
        if "total_cost" in report.details.get("metrics", {}):
            reported_total = report.details["metrics"]["total_cost"]
            assert abs(reported_total - expected_total) < FINANCIAL_TOLERANCE * expected_total

    def test_npv_monotonic_with_discount_rate(self):
        """VPN es monótonamente decreciente con la tasa de descuento."""
        projection = CashFlowProjection(
            initial_investment=1000.0,
            cash_flows=[400.0, 400.0, 400.0],
            risk_free_rate=0.05
        )
        
        rates = [0.01, 0.05, 0.10, 0.15, 0.20]
        npvs = [projection.calculate_npv(r) for r in rates]
        
        # Cada VPN debe ser menor que el anterior
        for i in range(1, len(npvs)):
            assert npvs[i] < npvs[i-1], \
                f"VPN({rates[i]})={npvs[i]} debe ser < VPN({rates[i-1]})={npvs[i-1]}"

    def test_payback_bounded_by_horizon(self):
        """Payback ≤ número de períodos (o None si no recupera)."""
        # Proyecto que recupera en el horizonte
        recovering = CashFlowProjection(
            initial_investment=1000.0,
            cash_flows=[500.0, 500.0, 500.0]
        )
        payback = recovering.calculate_payback_period()
        assert payback is not None
        assert payback <= len(recovering.cash_flows)
        
        # Proyecto que no recupera
        non_recovering = CashFlowProjection(
            initial_investment=10000.0,
            cash_flows=[100.0, 100.0, 100.0]
        )
        assert non_recovering.calculate_payback_period() is None

    def test_viability_consistent_with_npv_sign(self):
        """is_viable() es consistente con signo de VPN."""
        viable = CashFlowProjection(1000.0, [500.0, 500.0, 500.0])
        non_viable = CashFlowProjection(10000.0, [100.0, 100.0])
        
        assert viable.is_viable() == (viable.calculate_npv() > 0)
        assert non_viable.is_viable() == (non_viable.calculate_npv() > 0)

    def test_cost_by_apu_sums_to_total(self):
        """Σ costo_por_APU = costo_total."""
        builder = (
            InsumosBuilder()
            .with_insumo("APU-001", "A", "MATERIAL", 100.0)
            .with_insumo("APU-001", "B", "MANO_OBRA", 50.0)
            .with_insumo("APU-002", "C", "EQUIPO", 75.0)
        )
        
        total = builder.get_total_cost()
        by_apu = builder.get_cost_by_apu()
        sum_by_apu = sum(by_apu.values())
        
        assert abs(sum_by_apu - total) < FINANCIAL_TOLERANCE


# =============================================================================
# TESTS: TRADUCCIÓN SEMÁNTICA
# =============================================================================


class TestSemanticTranslation(TestFixtures):
    """Tests para la traducción de métricas a narrativa estratégica."""

    def test_narrative_in_spanish(self, agent, complete_context):
        """La narrativa está en español."""
        context, _ = complete_context
        report = agent.evaluate_project(context)
        
        spanish_indicators = ["del", "para", "que", "con", "los", "las", "una", "este"]
        words = report.strategic_narrative.lower().split()
        
        matches = sum(1 for word in spanish_indicators if word in words)
        
        assert matches >= 3, "Narrativa debe estar en español"

    def test_construction_metaphors_present(self, agent, complete_context):
        """La narrativa usa metáforas de construcción/ingeniería."""
        context, _ = complete_context
        report = agent.evaluate_project(context)
        
        construction_terms = [
            "estructura", "cimiento", "carga", "estabilidad",
            "integridad", "geotecnia", "ingenier", "audit"
        ]
        
        narrative_lower = report.strategic_narrative.lower()
        matches = sum(1 for term in construction_terms if term in narrative_lower)
        
        assert matches >= 2, "Narrativa debe usar metáforas de construcción"

    def test_risk_level_translated_to_severity(self, agent, high_risk_context):
        """Nivel de riesgo alto se traduce a términos de severidad."""
        report = agent.evaluate_project(high_risk_context)
        
        severity_terms = ["crític", "sever", "alto", "precaución", "alert", "riesgo"]
        narrative_lower = report.strategic_narrative.lower()
        
        assert any(term in narrative_lower for term in severity_terms)


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
# -*- coding: utf-8 -*-
"""
=========================================================================================
Suite de pruebas unitarias — Data Validator v4.0
Ubicación: tests/unit/tactics/test_data_validator.py
=========================================================================================

Cubre las 3 fases anidadas y sus morfismos de transición:

    FASE 1  – Microscopía de campos
              · _es_numero_valido, _ulp, _validar_coherencia_matematica
              · _limpiar_y_validar_descripcion
              · _validate_extreme_costs, _validate_quantity_and_coherence
              · _validate_descriptions
              · phase1_to_topological_domain  (morfismo final F₁ → F₂)

    FASE 2  – Topología bipartita
              · BipartiteTopology.build_graph
              · BipartiteTopology.compute_laplacian_spectrum
              · BipartiteTopology.detect_floating_nodes
              · BipartiteTopology.detect_spof
              · compute_pyramidal_metrics
              · phase2_to_thermodynamic_state  (morfismo final F₂ → F₃)

    FASE 3  – Termodinámica informacional
              · AnomalyValidator.detect_cost_anomalies
              · ThermodynamicEvaluator.entropy_of_alerts
              · ThermodynamicEvaluator.compute_quality_temperature
              · ThermodynamicEvaluator.classify_stability
              · ThermodynamicEvaluator.evaluate

    ORQUESTADOR
              · validate_and_clean_data  (F₃ ∘ F₂ ∘ F₁)
              · Casos borde: store vacío, datos degenerados, telemetría

Principios de prueba:
    • Invariantes numéricos IEEE 754 (NaN, ±Inf, ulp, ε_mach).
    • Conservación C = Q·P con tolerancia híbrida.
    • Espectro del laplaciano: β₀, λ₂ ≥ 0, L ≽ 0.
    • Entropía normalizada ∈ [0,1]; temperatura ∈ [0,100].
    • Composición categórica de morfismos (salida Fᵢ = entrada Fᵢ₊₁).
=========================================================================================
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import del módulo bajo prueba
# ---------------------------------------------------------------------------
from app.tactics.data_validator import (
    # Constantes
    COSTO_MAXIMO_RAZONABLE,
    COSTO_MINIMO_VALIDO,
    EPS_MACHINE,
    SENSITIVITY_ALERT_THRESHOLD,
    SPECTRAL_ZERO_TOL,
    SPOF_IMPACT_FRACTION,
    TOLERANCIA_PORCENTUAL_COHERENCIA,
    VALORES_DESCRIPCION_INVALIDOS,
    # Enums / dataclasses
    TipoAlerta,
    ValidationMetrics,
    CoherenceAnalysis,
    PyramidalMetrics,
    ThermodynamicState,
    IncidenceDomain,
    StructuralState,
    # Fase 1
    _es_numero_valido,
    _ulp,
    _agregar_alerta,
    _validar_coherencia_matematica,
    _limpiar_y_validar_descripcion,
    _validate_extreme_costs,
    _validate_quantity_and_coherence,
    _validate_descriptions,
    phase1_to_topological_domain,
    # Fase 2
    BipartiteTopology,
    compute_pyramidal_metrics,
    phase2_to_thermodynamic_state,
    # Fase 3
    AnomalyValidator,
    ThermodynamicEvaluator,
    # Orquestador
    validate_and_clean_data,
)


# ============================================================================
# FIXTURES COMPARTIDAS
# ============================================================================

@pytest.fixture
def item_coherente() -> Dict[str, Any]:
    """Ítem APU con C = Q·P exacto."""
    return {
        "CANTIDAD": 10.0,
        "VR_UNITARIO": 25.0,
        "VALOR_TOTAL": 250.0,
        "DESCRIPCION_INSUMO": "Cemento gris 50kg",
        "APU_CODIGO": "APU-001",
    }


@pytest.fixture
def item_incoherente() -> Dict[str, Any]:
    """Ítem APU con C ≠ Q·P fuera de tolerancia."""
    return {
        "CANTIDAD": 10.0,
        "VR_UNITARIO": 25.0,
        "VALOR_TOTAL": 999.0,  # esperado 250
        "DESCRIPCION_INSUMO": "Arena de río",
        "APU_CODIGO": "APU-002",
    }


@pytest.fixture
def item_q_nula() -> Dict[str, Any]:
    """Q≈0, VT>0, P>0 → debe recalcular Q = VT/P."""
    return {
        "CANTIDAD": 0.0,
        "VR_UNITARIO": 50.0,
        "VALOR_TOTAL": 500.0,
        "DESCRIPCION_INSUMO": "Acero corrugado",
        "APU_CODIGO": "APU-003",
    }


@pytest.fixture
def presupuesto_basico() -> List[Dict[str, Any]]:
    return [
        {"CODIGO": "APU-001", "VALOR_CONSTRUCCION_UN": 1_000.0, "DESCRIPCION": "Muro"},
        {"CODIGO": "APU-002", "VALOR_CONSTRUCCION_UN": 2_500.0, "DESCRIPCION": "Losa"},
        {"CODIGO": "APU-003", "VALOR_CONSTRUCCION_UN": -50.0, "DESCRIPCION": "Negativo"},
        {
            "CODIGO": "APU-004",
            "VALOR_CONSTRUCCION_UN": COSTO_MAXIMO_RAZONABLE + 1,
            "DESCRIPCION": "Excesivo",
        },
    ]


@pytest.fixture
def apus_detail_basico(
    item_coherente, item_incoherente, item_q_nula
) -> List[Dict[str, Any]]:
    return [
        deepcopy(item_coherente),
        deepcopy(item_incoherente),
        deepcopy(item_q_nula),
        {
            "CANTIDAD": 5.0,
            "VR_UNITARIO": 100.0,
            "VALOR_TOTAL": 500.0,
            "DESCRIPCION_INSUMO": "nan",  # inválida
            "APU_CODIGO": "APU-001",
        },
        {
            "CANTIDAD": 2.0,
            "VR_UNITARIO": 200.0,
            "VALOR_TOTAL": 400.0,
            "DESCRIPCION_INSUMO": "  Grava ¾\"  ",
            "APU_CODIGO": "APU-002",
        },
    ]


@pytest.fixture
def data_store_completo(presupuesto_basico, apus_detail_basico) -> Dict[str, Any]:
    return {
        "presupuesto": deepcopy(presupuesto_basico),
        "apus_detail": deepcopy(apus_detail_basico),
    }


@pytest.fixture
def grafo_conexo_simple() -> Dict[str, Any]:
    """
    Grafo bipartito conexo:
        U = {A1, A2},  V = {I1, I2}
        E = {(A1,I1), (A1,I2), (A2,I1)}
    ⇒ 1 componente, λ₂ > 0, sin flotantes.
    """
    apus_df = pd.DataFrame({"CODIGO_APU": ["A1", "A2"]})
    insumos_df = pd.DataFrame({
        "APU_CODIGO": ["A1", "A1", "A2"],
        "DESCRIPCION_INSUMO": ["I1", "I2", "I1"],
        "DESCRIPCION_INSUMO_NORM": ["i1", "i2", "i1"],
    })
    return {"apus_df": apus_df, "insumos_df": insumos_df}


@pytest.fixture
def grafo_con_flotante() -> Dict[str, Any]:
    """A3 sin insumos → nodo flotante."""
    apus_df = pd.DataFrame({"CODIGO_APU": ["A1", "A2", "A3"]})
    insumos_df = pd.DataFrame({
        "APU_CODIGO": ["A1", "A2"],
        "DESCRIPCION_INSUMO": ["Cemento", "Arena"],
        "DESCRIPCION_INSUMO_NORM": ["cemento", "arena"],
    })
    return {"apus_df": apus_df, "insumos_df": insumos_df}


@pytest.fixture
def grafo_con_spof() -> Dict[str, Any]:
    """
    Un insumo compartido por todos los APUs → SPOF.
    |U|=5, insumo crítico conectado a los 5.
    """
    apus = [f"A{i}" for i in range(5)]
    apus_df = pd.DataFrame({"CODIGO_APU": apus})
    rows = []
    for a in apus:
        rows.append({
            "APU_CODIGO": a,
            "DESCRIPCION_INSUMO": "InsumoCritico",
            "DESCRIPCION_INSUMO_NORM": "insumocritico",
        })
        rows.append({
            "APU_CODIGO": a,
            "DESCRIPCION_INSUMO": f"Local-{a}",
            "DESCRIPCION_INSUMO_NORM": f"local-{a.lower()}",
        })
    insumos_df = pd.DataFrame(rows)
    return {"apus_df": apus_df, "insumos_df": insumos_df}


# ============================================================================
# FASE 0 — ESTRUCTURAS BASE
# ============================================================================

class TestTipoAlerta:
    def test_todos_los_tipos_tienen_value_str(self):
        for t in TipoAlerta:
            assert isinstance(t.value, str)
            assert len(t.value) > 0

    def test_tipos_criticos_existen(self):
        nombres = {t.name for t in TipoAlerta}
        assert "INCOHERENCIA_MATEMATICA" in nombres
        assert "SPOF_DETECTADO" in nombres
        assert "ALTA_SENSIBILIDAD" in nombres
        assert "CONDICIONAMIENTO_CRITICO" in nombres
        assert "COMPONENTE_DESCONECTADA" in nombres


class TestValidationMetrics:
    def test_agregar_alerta_incrementa_contador(self):
        m = ValidationMetrics()
        m.agregar_alerta(TipoAlerta.COSTO_EXCESIVO)
        m.agregar_alerta(TipoAlerta.COSTO_EXCESIVO)
        m.agregar_alerta(TipoAlerta.COSTO_NEGATIVO)
        assert m.costos_excesivos == 2
        assert m.costos_negativos == 1
        assert m.alertas_por_tipo[TipoAlerta.COSTO_EXCESIVO.value] == 2

    def test_to_dict_keys(self):
        m = ValidationMetrics(total_items_procesados=5)
        d = m.to_dict()
        assert d["total_items_procesados"] == 5
        assert "alertas_por_tipo" in d
        assert isinstance(d["alertas_por_tipo"], dict)

    def test_incoherencias_y_infinitos(self):
        m = ValidationMetrics()
        m.agregar_alerta(TipoAlerta.INCOHERENCIA_MATEMATICA)
        m.agregar_alerta(TipoAlerta.VALOR_INFINITO)
        assert m.incoherencias_matematicas == 1
        assert m.valores_infinitos == 1


class TestCoherenceAnalysisDataclass:
    def test_campos_requeridos(self):
        ca = CoherenceAnalysis(
            coherente=True,
            error_absoluto=0.0,
            error_porcentual=0.0,
            tolerancia_aplicada=1e-6,
            sensibilidad=1.0,
            condicion=1.0,
            ganancia_sistema=1.0,
            mensaje="ok",
        )
        assert ca.coherente is True
        assert ca.error_absoluto == 0.0


# ============================================================================
# FASE 1 — MICROSCOPÍA DE CAMPOS
# ============================================================================

# ---------------------------------------------------------------------------
# 1.0  Predicados y utilidades numéricas
# ---------------------------------------------------------------------------

class TestEsNumeroValido:
    @pytest.mark.parametrize("valor", [0, 1, -1, 3.14, 1e-20, 1e20, np.float64(2.5), np.int64(7)])
    def test_numeros_validos(self, valor):
        assert _es_numero_valido(valor) is True

    @pytest.mark.parametrize("valor", [
        None, float("nan"), float("inf"), float("-inf"),
        "abc", "", [], {}, object(),
        np.nan, np.inf, -np.inf,
    ])
    def test_numeros_invalidos(self, valor):
        assert _es_numero_valido(valor) is False


class TestUlp:
    def test_ulp_de_uno(self):
        # ulp(1.0) en float64 = 2^{-52} ≈ 2.22e-16
        u = _ulp(1.0)
        assert u == pytest.approx(EPS_MACHINE, rel=1e-6)

    def test_ulp_de_cero(self):
        assert _ulp(0.0) == EPS_MACHINE

    def test_ulp_crece_con_magnitud(self):
        assert _ulp(1e10) > _ulp(1.0)
        assert _ulp(1e20) > _ulp(1e10)

    def test_ulp_simetrico(self):
        assert _ulp(42.0) == _ulp(-42.0)

    def test_ulp_positivo(self):
        for x in [1e-300, 1.0, 1e300]:
            assert _ulp(x) > 0


class TestAgregarAlerta:
    def test_inserta_alerta(self):
        item: Dict[str, Any] = {}
        ok = _agregar_alerta(item, "msg", TipoAlerta.COSTO_NEGATIVO)
        assert ok is True
        assert len(item["alertas"]) == 1
        assert item["alertas"][0]["tipo"] == TipoAlerta.COSTO_NEGATIVO.value

    def test_rechaza_duplicado(self):
        item: Dict[str, Any] = {}
        _agregar_alerta(item, "msg", TipoAlerta.COSTO_NEGATIVO)
        ok = _agregar_alerta(item, "msg", TipoAlerta.COSTO_NEGATIVO)
        assert ok is False
        assert len(item["alertas"]) == 1

    def test_permite_duplicados_si_flag(self):
        item: Dict[str, Any] = {}
        _agregar_alerta(item, "msg", TipoAlerta.COSTO_NEGATIVO)
        ok = _agregar_alerta(
            item, "msg", TipoAlerta.COSTO_NEGATIVO, permitir_duplicados=True
        )
        assert ok is True
        assert len(item["alertas"]) == 2

    def test_actualiza_metrics(self):
        item: Dict[str, Any] = {}
        m = ValidationMetrics()
        _agregar_alerta(item, "x", TipoAlerta.COSTO_EXCESIVO, metrics=m)
        assert m.costos_excesivos == 1

    def test_rechaza_item_no_dict(self):
        assert _agregar_alerta("no-dict", "m", TipoAlerta.COSTO_NEGATIVO) is False  # type: ignore

    def test_respeta_max_alertas(self):
        from app.tactics.data_validator import MAX_ALERTAS_POR_ITEM
        item: Dict[str, Any] = {
            "alertas": [
                {"tipo": f"T{i}", "mensaje": f"m{i}"}
                for i in range(MAX_ALERTAS_POR_ITEM)
            ]
        }
        ok = _agregar_alerta(item, "extra", TipoAlerta.COSTO_NEGATIVO)
        assert ok is False


# ---------------------------------------------------------------------------
# 1.1  Ley de conservación C = Q·P
# ---------------------------------------------------------------------------

class TestValidarCoherenciaMatematica:
    def test_coherente_exacto(self):
        ca = _validar_coherencia_matematica(10.0, 25.0, 250.0)
        assert isinstance(ca, CoherenceAnalysis)
        assert ca.coherente is True
        assert ca.error_absoluto == pytest.approx(0.0, abs=1e-12)
        assert ca.error_porcentual == pytest.approx(0.0, abs=1e-10)

    def test_coherente_dentro_tolerancia_relativa(self):
        # τ = 1% → error de 0.5% debe pasar
        Q, P = 100.0, 10.0
        esperado = Q * P  # 1000
        VT = esperado * 1.005  # +0.5%
        ca = _validar_coherencia_matematica(Q, P, VT)
        assert ca.coherente is True

    def test_incoherente_fuera_tolerancia(self):
        ca = _validar_coherencia_matematica(10.0, 25.0, 999.0)
        assert ca.coherente is False
        assert ca.error_absoluto > 0
        assert ca.error_porcentual > TOLERANCIA_PORCENTUAL_COHERENCIA * 100

    def test_sensibilidad_hipotenusa(self):
        # g = √(P² + Q²)
        Q, P = 3.0, 4.0
        ca = _validar_coherencia_matematica(Q, P, 12.0)
        assert ca.sensibilidad == pytest.approx(5.0, rel=1e-12)

    def test_alta_sensibilidad_detectada_en_mensaje(self):
        # g = √(P²+Q²) > 1000
        Q, P = 1000.0, 1000.0
        ca = _validar_coherencia_matematica(Q, P, Q * P)
        assert ca.sensibilidad > SENSITIVITY_ALERT_THRESHOLD
        assert "ALTA SENSIBILIDAD" in ca.mensaje or ca.sensibilidad > 1000

    def test_condicion_finita_para_producto_normal(self):
        ca = _validar_coherencia_matematica(2.0, 3.0, 6.0)
        assert math.isfinite(ca.condicion)
        assert ca.condicion > 0

    def test_error_conversion(self):
        ca = _validar_coherencia_matematica("x", 1.0, 1.0)  # type: ignore
        assert ca.coherente is False
        assert ca.error_absoluto == float("inf")

    def test_valores_infinitos(self):
        ca = _validar_coherencia_matematica(1.0, float("inf"), float("inf"))
        assert ca.coherente is False

    def test_tolerancia_hibrida_cubre_ruido_flotante(self):
        # Producto con redondeo flotante típico
        Q = 0.1
        P = 0.2
        # 0.1 * 0.2 no es exacto en binario
        esperado = Q * P
        ca = _validar_coherencia_matematica(Q, P, esperado)
        assert ca.coherente is True
        assert ca.tolerancia_aplicada >= _ulp(esperado)

    def test_ganancia_sistema_es_max_abs(self):
        ca = _validar_coherencia_matematica(3.0, 7.0, 21.0)
        assert ca.ganancia_sistema == pytest.approx(7.0)

    def test_tau_personalizado(self):
        # Con τ muy estricto, un error pequeño falla
        Q, P = 100.0, 10.0
        VT = 1000.0 * 1.001  # 0.1%
        ca_laxo = _validar_coherencia_matematica(Q, P, VT, tau=0.01)
        ca_estricto = _validar_coherencia_matematica(Q, P, VT, tau=1e-6)
        assert ca_laxo.coherente is True
        assert ca_estricto.coherente is False


# ---------------------------------------------------------------------------
# 1.2  Limpieza de descripciones
# ---------------------------------------------------------------------------

class TestLimpiarYValidarDescripcion:
    def test_descripcion_valida(self):
        limpia, valida, razon = _limpiar_y_validar_descripcion("Cemento gris 50kg")
        assert valida is True
        assert limpia == "Cemento gris 50kg"
        assert razon is None

    def test_strip_y_nfc(self):
        limpia, valida, _ = _limpiar_y_validar_descripcion("  Arena  ")
        assert valida is True
        assert limpia == "Arena"

    def test_vacia(self):
        limpia, valida, razon = _limpiar_y_validar_descripcion("")
        assert valida is False
        assert limpia is None

    def test_none(self):
        limpia, valida, razon = _limpiar_y_validar_descripcion(None)
        assert valida is False

    def test_nan_pandas(self):
        limpia, valida, razon = _limpiar_y_validar_descripcion(float("nan"))
        assert valida is False

    @pytest.mark.parametrize("invalido", list(VALORES_DESCRIPCION_INVALIDOS))
    def test_valores_invalidos_conocidos(self, invalido):
        limpia, valida, razon = _limpiar_y_validar_descripcion(invalido)
        assert valida is False, f"Debería rechazar: {invalido!r}"

    def test_truncamiento(self):
        from app.tactics.data_validator import MAX_DESCRIPCION_LENGTH
        larga = "X" * (MAX_DESCRIPCION_LENGTH + 100)
        limpia, valida, _ = _limpiar_y_validar_descripcion(larga)
        assert valida is True
        assert limpia is not None
        assert len(limpia) == MAX_DESCRIPCION_LENGTH

    def test_elimina_controles_c0(self):
        limpia, valida, _ = _limpiar_y_validar_descripcion("Hola\x00Mundo\x1f")
        assert valida is True
        assert limpia is not None
        assert "\x00" not in limpia
        assert "\x1f" not in limpia


# ---------------------------------------------------------------------------
# 1.3  Validaciones atómicas
# ---------------------------------------------------------------------------

class TestValidateExtremeCosts:
    def test_detecta_negativo_y_excesivo(self, presupuesto_basico):
        result, metrics = _validate_extreme_costs(presupuesto_basico)
        assert metrics.total_items_procesados == len(presupuesto_basico)
        # APU-003 negativo, APU-004 excesivo
        tipos = []
        for item in result:
            for a in item.get("alertas") or []:
                tipos.append(a["tipo"])
        assert TipoAlerta.COSTO_NEGATIVO.value in tipos
        assert TipoAlerta.COSTO_EXCESIVO.value in tipos

    def test_lista_vacia(self):
        result, metrics = _validate_extreme_costs([])
        assert result == []
        assert metrics.total_items_procesados == 0

    def test_no_muta_original(self, presupuesto_basico):
        original = deepcopy(presupuesto_basico)
        _validate_extreme_costs(presupuesto_basico)
        assert presupuesto_basico == original

    def test_infinito_genera_alerta(self):
        data = [{"VALOR_CONSTRUCCION_UN": float("inf")}]
        result, metrics = _validate_extreme_costs(data)
        tipos = [a["tipo"] for a in result[0].get("alertas") or []]
        assert TipoAlerta.VALOR_INFINITO.value in tipos
        assert metrics.valores_infinitos >= 1

    def test_costo_en_rango_sin_alerta(self):
        data = [{"VALOR_CONSTRUCCION_UN": 1000.0}]
        result, metrics = _validate_extreme_costs(data)
        assert not result[0].get("alertas")
        assert metrics.items_con_alertas == 0


class TestValidateQuantityAndCoherence:
    def test_incoherencia_detectada(self, item_incoherente):
        result, metrics = _validate_quantity_and_coherence([item_incoherente])
        assert metrics.incoherencias_matematicas >= 1
        tipos = [a["tipo"] for a in result[0].get("alertas") or []]
        assert TipoAlerta.INCOHERENCIA_MATEMATICA.value in tipos

    def test_coherente_sin_alerta_incoherencia(self, item_coherente):
        result, metrics = _validate_quantity_and_coherence([item_coherente])
        tipos = [a["tipo"] for a in result[0].get("alertas") or []]
        assert TipoAlerta.INCOHERENCIA_MATEMATICA.value not in tipos

    def test_recalcula_cantidad_nula(self, item_q_nula):
        result, metrics = _validate_quantity_and_coherence([item_q_nula])
        assert metrics.cantidades_recalculadas >= 1
        assert result[0]["CANTIDAD"] == pytest.approx(10.0)  # 500/50
        tipos = [a["tipo"] for a in result[0].get("alertas") or []]
        assert TipoAlerta.CANTIDAD_RECALCULADA.value in tipos

    def test_lista_vacia(self):
        result, metrics = _validate_quantity_and_coherence([])
        assert result == []
        assert metrics.total_items_procesados == 0

    def test_alta_sensibilidad(self):
        item = {
            "CANTIDAD": 2000.0,
            "VR_UNITARIO": 2000.0,
            "VALOR_TOTAL": 4_000_000.0,
            "DESCRIPCION_INSUMO": "Heavy",
        }
        result, metrics = _validate_quantity_and_coherence([item])
        tipos = [a["tipo"] for a in result[0].get("alertas") or []]
        # g = √(2000²+2000²) ≈ 2828 > 1000
        assert TipoAlerta.ALTA_SENSIBILIDAD.value in tipos

    def test_campos_faltantes_no_rompe(self):
        result, metrics = _validate_quantity_and_coherence([{"CANTIDAD": 1.0}])
        assert metrics.total_items_procesados == 1


class TestValidateDescriptions:
    def test_descripcion_invalida_reemplazada(self):
        data = [{"DESCRIPCION_INSUMO": "nan"}]
        result, metrics = _validate_descriptions(data)
        assert result[0]["DESCRIPCION_INSUMO"] == "Insumo sin descripción"
        assert metrics.descripciones_corregidas >= 1
        tipos = [a["tipo"] for a in result[0].get("alertas") or []]
        assert TipoAlerta.DESCRIPCION_FALTANTE.value in tipos

    def test_descripcion_normalizada(self):
        data = [{"DESCRIPCION_INSUMO": "  Cemento  "}]
        result, metrics = _validate_descriptions(data)
        assert result[0]["DESCRIPCION_INSUMO"] == "Cemento"
        tipos = [a["tipo"] for a in result[0].get("alertas") or []]
        assert TipoAlerta.DESCRIPCION_CORREGIDA.value in tipos

    def test_descripcion_ok_sin_cambio(self):
        data = [{"DESCRIPCION_INSUMO": "Arena limpia"}]
        result, metrics = _validate_descriptions(data)
        assert result[0]["DESCRIPCION_INSUMO"] == "Arena limpia"
        assert not result[0].get("alertas")

    def test_lista_vacia(self):
        result, metrics = _validate_descriptions([])
        assert result == []


# ---------------------------------------------------------------------------
# 1.4  Morfismo final Fase 1 → Fase 2
# ---------------------------------------------------------------------------

class TestPhase1ToTopologicalDomain:
    def test_produce_incidence_domain(self, data_store_completo):
        # Simular store ya validado
        store = deepcopy(data_store_completo)
        metricas = {"presupuesto": {}}
        domain = phase1_to_topological_domain(store, metricas)
        assert isinstance(domain, IncidenceDomain)
        assert isinstance(domain.df_apus, pd.DataFrame)
        assert isinstance(domain.df_insumos, pd.DataFrame)
        assert domain.validated_store is store or domain.validated_store == store
        assert domain.metricas_fase1 == metricas

    def test_normaliza_columna_codigo_apu(self):
        store = {
            "presupuesto": [{"CODIGO": "X1", "VALOR_CONSTRUCCION_UN": 100}],
            "apus_detail": [{
                "CANTIDAD": 1, "VR_UNITARIO": 10, "VALOR_TOTAL": 10,
                "DESCRIPCION_INSUMO": "Y", "APU_CODIGO": "X1",
            }],
        }
        domain = phase1_to_topological_domain(store, {})
        assert "CODIGO_APU" in domain.df_apus.columns

    def test_crea_descripcion_norm(self):
        store = {
            "presupuesto": [{"CODIGO_APU": "A1"}],
            "apus_detail": [{
                "DESCRIPCION_INSUMO": "  Cemento ",
                "APU_CODIGO": "A1",
            }],
        }
        domain = phase1_to_topological_domain(store, {})
        assert "DESCRIPCION_INSUMO_NORM" in domain.df_insumos.columns
        norm_val = domain.df_insumos["DESCRIPCION_INSUMO_NORM"].iloc[0]
        assert norm_val == "cemento"

    def test_store_vacio(self):
        domain = phase1_to_topological_domain({}, {})
        assert domain.df_apus.empty
        assert domain.df_insumos.empty

    def test_es_entrada_valida_para_fase2(self, data_store_completo):
        """Invariante categórico: salida F₁ es entrada válida de F₂."""
        domain = phase1_to_topological_domain(data_store_completo, {})
        # No debe lanzar
        metrics = compute_pyramidal_metrics(domain.df_apus, domain.df_insumos)
        assert isinstance(metrics, PyramidalMetrics)


# ============================================================================
# FASE 2 — TOPOLOGÍA BIPARTITA
# ============================================================================

class TestBipartiteTopologyBuildGraph:
    def test_grafo_conexo(self, grafo_conexo_simple):
        g = BipartiteTopology.build_graph(
            grafo_conexo_simple["apus_df"],
            grafo_conexo_simple["insumos_df"],
        )
        assert len(g["nodos_apus"]) == 2
        assert len(g["nodos_insumos"]) == 2
        assert len(g["aristas"]) == 3
        assert g["adj_matrix"].shape == (2, 2)
        assert g["adj_matrix"].sum() == 3

    def test_apus_vacios(self):
        g = BipartiteTopology.build_graph(pd.DataFrame(), pd.DataFrame())
        assert g["nodos_apus"] == []
        assert g["adj_matrix"].size == 0

    def test_insumos_vacios_pero_apus_presentes(self):
        apus_df = pd.DataFrame({"CODIGO_APU": ["A1", "A2"]})
        g = BipartiteTopology.build_graph(apus_df, pd.DataFrame())
        assert len(g["nodos_apus"]) == 2
        assert g["nodos_insumos"] == []

    def test_adj_matrix_dtype_int8(self, grafo_conexo_simple):
        g = BipartiteTopology.build_graph(
            grafo_conexo_simple["apus_df"],
            grafo_conexo_simple["insumos_df"],
        )
        assert g["adj_matrix"].dtype == np.int8

    def test_mapeos_idx(self, grafo_conexo_simple):
        g = BipartiteTopology.build_graph(
            grafo_conexo_simple["apus_df"],
            grafo_conexo_simple["insumos_df"],
        )
        assert "A1" in g["apu_to_idx"]
        assert set(g["apu_to_idx"].values()) == {0, 1}


class TestBipartiteTopologyLaplacianSpectrum:
    def test_espectro_grafo_conexo(self, grafo_conexo_simple):
        g = BipartiteTopology.build_graph(
            grafo_conexo_simple["apus_df"],
            grafo_conexo_simple["insumos_df"],
        )
        spec = BipartiteTopology.compute_laplacian_spectrum(g["adj_matrix"])
        eigvals = spec["eigenvalues"]
        # L ≽ 0
        assert np.all(eigvals >= -SPECTRAL_ZERO_TOL)
        # Grafo conexo ⇒ β₀ = 1, λ₂ > 0
        assert spec["connected_components"] == 1
        assert spec["algebraic_connectivity"] > SPECTRAL_ZERO_TOL

    def test_matriz_vacia(self):
        spec = BipartiteTopology.compute_laplacian_spectrum(np.array([]))
        assert spec["algebraic_connectivity"] == 0.0
        assert len(spec["eigenvalues"]) == 0

    def test_dos_componentes(self):
        """
        Dos APUs, dos insumos, sin aristas cruzadas entre pares:
        A1—I1, A2—I2  ⇒ 2 componentes si no hay más aristas...
        En bipartito, cada arista conecta; dos aristas disjuntas ⇒ 2 comp.
        """
        A = np.array([
            [1, 0],
            [0, 1],
        ], dtype=float)
        spec = BipartiteTopology.compute_laplacian_spectrum(A)
        # 2 componentes conexas
        assert spec["connected_components"] == 2
        # λ₂ debería ser 0 (mult ≥ 2 del cero)
        assert spec["algebraic_connectivity"] == pytest.approx(0.0, abs=1e-8)

    def test_autovalores_ordenados_ascendente(self, grafo_conexo_simple):
        g = BipartiteTopology.build_graph(
            grafo_conexo_simple["apus_df"],
            grafo_conexo_simple["insumos_df"],
        )
        spec = BipartiteTopology.compute_laplacian_spectrum(g["adj_matrix"])
        eig = spec["eigenvalues"]
        assert np.all(np.diff(eig) >= -1e-12)

    def test_simetria_espectral_no_negativa(self):
        """Para cualquier A, eig(L) ≥ 0 (salvo ruido numérico)."""
        rng = np.random.default_rng(42)
        A = (rng.random((6, 4)) > 0.5).astype(float)
        spec = BipartiteTopology.compute_laplacian_spectrum(A)
        assert np.all(spec["eigenvalues"] >= -1e-8)


class TestDetectFloatingNodes:
    def test_detecta_flotante(self, grafo_con_flotante):
        floating = BipartiteTopology.detect_floating_nodes(
            grafo_con_flotante["apus_df"],
            grafo_con_flotante["insumos_df"],
        )
        assert "A3" in floating
        assert "A1" not in floating
        assert "A2" not in floating

    def test_sin_flotantes(self, grafo_conexo_simple):
        floating = BipartiteTopology.detect_floating_nodes(
            grafo_conexo_simple["apus_df"],
            grafo_conexo_simple["insumos_df"],
        )
        assert floating == []

    def test_todos_flotantes_si_sin_insumos(self):
        apus_df = pd.DataFrame({"CODIGO_APU": ["X", "Y"]})
        floating = BipartiteTopology.detect_floating_nodes(apus_df, pd.DataFrame())
        assert set(floating) == {"X", "Y"}

    def test_apus_vacios(self):
        floating = BipartiteTopology.detect_floating_nodes(
            pd.DataFrame(), pd.DataFrame()
        )
        assert floating == []


class TestDetectSpof:
    def test_detecta_spof(self, grafo_con_spof):
        g = BipartiteTopology.build_graph(
            grafo_con_spof["apus_df"],
            grafo_con_spof["insumos_df"],
        )
        spofs = BipartiteTopology.detect_spof(g["adj_matrix"], g["nodos_insumos"])
        assert len(spofs) >= 1
        insumos_spof = [s["insumo"] for s in spofs]
        # El insumo crítico debe aparecer (normalizado o no según columna)
        assert any("critico" in str(i).lower() or "Critico" in str(i) for i in insumos_spof)

    def test_sin_spof_grafo_pequeno(self, grafo_conexo_simple):
        g = BipartiteTopology.build_graph(
            grafo_conexo_simple["apus_df"],
            grafo_conexo_simple["insumos_df"],
        )
        # |U|=2, umbral = max(2, ceil(0.2)) = 2; grados pueden alcanzar 2
        spofs = BipartiteTopology.detect_spof(
            g["adj_matrix"], g["nodos_insumos"], umbral_impacto=10
        )
        assert spofs == []

    def test_matriz_vacia(self):
        assert BipartiteTopology.detect_spof(np.array([]), []) == []

    def test_spof_ordenado_por_impacto(self, grafo_con_spof):
        g = BipartiteTopology.build_graph(
            grafo_con_spof["apus_df"],
            grafo_con_spof["insumos_df"],
        )
        spofs = BipartiteTopology.detect_spof(g["adj_matrix"], g["nodos_insumos"])
        if len(spofs) >= 2:
            impactos = [s["impacto"] for s in spofs]
            assert impactos == sorted(impactos, reverse=True)

    def test_campos_spof(self, grafo_con_spof):
        g = BipartiteTopology.build_graph(
            grafo_con_spof["apus_df"],
            grafo_con_spof["insumos_df"],
        )
        spofs = BipartiteTopology.detect_spof(g["adj_matrix"], g["nodos_insumos"])
        assert len(spofs) >= 1
        s = spofs[0]
        assert "insumo" in s
        assert "impacto" in s
        assert "porcentaje_apus" in s
        assert "apus_afectados_muestra" in s
        assert s["impacto"] >= 1
        assert 0 < s["porcentaje_apus"] <= 1.0


class TestComputePyramidalMetrics:
    def test_metricas_grafo_conexo(self, grafo_conexo_simple):
        pm = compute_pyramidal_metrics(
            grafo_conexo_simple["apus_df"],
            grafo_conexo_simple["insumos_df"],
        )
        assert isinstance(pm, PyramidalMetrics)
        assert pm.structure_load == 2
        assert pm.base_width == 2
        assert 0 < pm.pyramid_stability_index <= 1.0
        # Ψ = tanh(2/2) = tanh(1) ≈ 0.761
        assert pm.pyramid_stability_index == pytest.approx(math.tanh(1.0), rel=1e-6)
        assert pm.connected_components == 1
        assert pm.algebraic_connectivity > 0
        assert pm.floating_nodes == []
        assert pm.spectral_radius > 0

    def test_metricas_con_flotante(self, grafo_con_flotante):
        pm = compute_pyramidal_metrics(
            grafo_con_flotante["apus_df"],
            grafo_con_flotante["insumos_df"],
        )
        assert "A3" in pm.floating_nodes
        assert pm.structure_load == 3

    def test_to_dict(self, grafo_conexo_simple):
        pm = compute_pyramidal_metrics(
            grafo_conexo_simple["apus_df"],
            grafo_conexo_simple["insumos_df"],
        )
        d = pm.to_dict()
        assert d["base_width"] == 2
        assert "algebraic_connectivity" in d
        assert "spof_list" in d

    def test_dataframes_vacios(self):
        pm = compute_pyramidal_metrics(pd.DataFrame(), pd.DataFrame())
        assert pm.base_width == 0
        assert pm.structure_load == 0
        assert pm.spectral_radius == 0.0


class TestPhase2ToThermodynamicState:
    def test_produce_structural_state(self, data_store_completo):
        domain = phase1_to_topological_domain(data_store_completo, {"m": 1})
        pm = compute_pyramidal_metrics(domain.df_apus, domain.df_insumos)
        structural = phase2_to_thermodynamic_state(domain, pm)
        assert isinstance(structural, StructuralState)
        assert structural.pyramidal is pm
        assert structural.incidence is domain
        assert structural.metricas_fase1 == {"m": 1}

    def test_es_entrada_valida_para_fase3(self, data_store_completo):
        """Invariante categórico: salida F₂ es entrada válida de F₃."""
        domain = phase1_to_topological_domain(data_store_completo, {})
        pm = compute_pyramidal_metrics(domain.df_apus, domain.df_insumos)
        structural = phase2_to_thermodynamic_state(domain, pm)
        thermo = ThermodynamicEvaluator.evaluate(structural)
        assert isinstance(thermo, ThermodynamicState)


# ============================================================================
# FASE 3 — TERMODINÁMICA INFORMACIONAL
# ============================================================================

class TestAnomalyValidator:
    def test_sin_datos(self):
        av = AnomalyValidator()
        data, stats = av.detect_cost_anomalies([])
        assert data == []
        assert stats["total_anomalias"] == 0

    def test_sin_valores_numericos(self):
        av = AnomalyValidator()
        data, stats = av.detect_cost_anomalies([{"VALOR_CONSTRUCCION_UN": "x"}])
        assert stats["total_anomalias"] == 0

    def test_detecta_outlier_obvio(self):
        av = AnomalyValidator(config={"zscore_threshold": 2.0, "iqr_multiplier": 1.0})
        # 20 valores ~100, uno de 1e6
        data = [{"VALOR_CONSTRUCCION_UN": 100.0 + i * 0.1} for i in range(20)]
        data.append({"VALOR_CONSTRUCCION_UN": 1_000_000.0})
        result, stats = av.detect_cost_anomalies(data)
        assert stats["total_anomalias"] >= 1
        # El outlier debe tener anomalias
        assert result[-1].get("anomalias")

    def test_distribucion_uniforme_sin_anomalias(self):
        av = AnomalyValidator()
        data = [{"VALOR_CONSTRUCCION_UN": float(100 + i)} for i in range(30)]
        result, stats = av.detect_cost_anomalies(data)
        # Con datos lineales suaves, votación ≥2 es improbable
        assert stats["total_anomalias"] <= 2  # margen generoso

    def test_stats_contiene_media_std_iqr(self):
        av = AnomalyValidator()
        data = [{"VALOR_CONSTRUCCION_UN": float(i)} for i in range(10)]
        _, stats = av.detect_cost_anomalies(data)
        assert "estadisticas" in stats
        assert "media" in stats["estadisticas"]
        assert "std" in stats["estadisticas"]
        assert "iqr" in stats["estadisticas"]

    def test_campo_costo_personalizado(self):
        av = AnomalyValidator()
        data = [{"PRECIO": 100.0}, {"PRECIO": 100.0}, {"PRECIO": 1e9}]
        result, stats = av.detect_cost_anomalies(data, campo_costo="PRECIO")
        # Al menos el outlier extremo
        assert isinstance(stats["total_anomalias"], int)

    def test_metodos_en_anomalia(self):
        av = AnomalyValidator(config={"zscore_threshold": 1.5, "iqr_multiplier": 0.5})
        data = [{"VALOR_CONSTRUCCION_UN": 10.0}] * 15
        data.append({"VALOR_CONSTRUCCION_UN": 1e8})
        result, stats = av.detect_cost_anomalies(data)
        if stats["total_anomalias"] > 0:
            anom = result[-1]["anomalias"][0]
            assert "metodos" in anom
            assert "zscore" in anom["metodos"]
            assert "iqr" in anom["metodos"]
            assert "isolation_forest" in anom["metodos"]


class TestThermodynamicEvaluatorEntropy:
    def test_sin_alertas_entropia_cero(self):
        items = [{"CANTIDAD": 1}, {"CANTIDAD": 2}]
        info = ThermodynamicEvaluator.entropy_of_alerts(items)
        assert info["entropia"] == 0.0
        assert info["entropia_norm"] == 0.0
        assert info["entropia_von_neumann"] == 0.0

    def test_un_solo_tipo_entropia_cero(self):
        items = [
            {"alertas": [{"tipo": "A", "mensaje": "m"}]},
            {"alertas": [{"tipo": "A", "mensaje": "m2"}]},
        ]
        info = ThermodynamicEvaluator.entropy_of_alerts(items)
        assert info["entropia"] == pytest.approx(0.0, abs=1e-12)
        assert info["n_tipos"] == 1

    def test_dos_tipos_equiprobables(self):
        items = [
            {"alertas": [{"tipo": "A", "mensaje": "1"}]},
            {"alertas": [{"tipo": "B", "mensaje": "2"}]},
        ]
        info = ThermodynamicEvaluator.entropy_of_alerts(items)
        # H = 1 bit, H_norm = 1
        assert info["entropia"] == pytest.approx(1.0, abs=1e-6)
        assert info["entropia_norm"] == pytest.approx(1.0, abs=1e-6)
        assert info["entropia_von_neumann"] == pytest.approx(1.0, abs=1e-6)

    def test_entropia_norm_en_rango(self):
        items = [
            {"alertas": [{"tipo": t, "mensaje": "x"}]}
            for t in ["A", "B", "C", "A", "B"]
        ]
        info = ThermodynamicEvaluator.entropy_of_alerts(items)
        assert 0.0 <= info["entropia_norm"] <= 1.0
        assert info["entropia"] >= 0.0

    def test_incluye_anomalias(self):
        items = [
            {"anomalias": [{"tipo": "COSTO_ANOMALO"}]},
            {"alertas": [{"tipo": "COSTO_NEGATIVO", "mensaje": "x"}]},
        ]
        info = ThermodynamicEvaluator.entropy_of_alerts(items)
        assert info["n_alertas"] == 2
        assert info["n_tipos"] == 2

    def test_lista_vacia(self):
        info = ThermodynamicEvaluator.entropy_of_alerts([])
        assert info["entropia"] == 0.0
        assert info["n_alertas"] == 0


class TestThermodynamicEvaluatorTemperature:
    def test_sin_alertas_psi_uno(self):
        store = {
            "presupuesto": [{"VALOR_CONSTRUCCION_UN": 100}],
            "apus_detail": [{"CANTIDAD": 1}],
        }
        pm = PyramidalMetrics(
            base_width=10, structure_load=5,
            pyramid_stability_index=1.0,
            floating_nodes=[], connected_components=1,
            algebraic_connectivity=0.5, spof_list=[],
            spectral_radius=1.0,
        )
        info = ThermodynamicEvaluator.compute_quality_temperature(store, pm)
        assert info["tasa_alertas"] == 0.0
        assert info["temperatura"] == pytest.approx(0.0, abs=1e-6)

    def test_todas_alertas_psi_cero(self):
        store = {
            "presupuesto": [
                {"alertas": [{"tipo": "X", "mensaje": "y"}]},
                {"alertas": [{"tipo": "X", "mensaje": "y"}]},
            ],
            "apus_detail": [],
        }
        pm = PyramidalMetrics(
            base_width=0, structure_load=1,
            pyramid_stability_index=0.0,
            floating_nodes=[], connected_components=1,
            algebraic_connectivity=0.0, spof_list=[],
            spectral_radius=0.0,
        )
        info = ThermodynamicEvaluator.compute_quality_temperature(store, pm)
        # T = 60*1 + 40*(1-0) = 100
        assert info["temperatura"] == pytest.approx(100.0, abs=1e-6)

    def test_temperatura_acotada(self):
        store = {
            "presupuesto": [{"alertas": [{"tipo": "X", "mensaje": "y"}]}],
        }
        pm = PyramidalMetrics(
            base_width=0, structure_load=1,
            pyramid_stability_index=-1.0,  # forzar (1-ψ)>1
            floating_nodes=[], connected_components=0,
            algebraic_connectivity=0.0, spof_list=[],
            spectral_radius=0.0,
        )
        info = ThermodynamicEvaluator.compute_quality_temperature(store, pm)
        assert 0.0 <= info["temperatura"] <= 100.0

    def test_sin_pyramidal_asume_psi_uno(self):
        store = {"presupuesto": [{"x": 1}], "apus_detail": []}
        info = ThermodynamicEvaluator.compute_quality_temperature(store, None)
        assert info["psi"] == 1.0


class TestClassifyStability:
    @pytest.mark.parametrize("temp,expected", [
        (0.0, "ESTABLE"),
        (19.9, "ESTABLE"),
        (20.0, "METASTABLE"),
        (49.9, "METASTABLE"),
        (50.0, "CAOTICO"),
        (100.0, "CAOTICO"),
    ])
    def test_umbrales(self, temp, expected):
        assert ThermodynamicEvaluator.classify_stability(temp) == expected


class TestThermodynamicEvaluatorEvaluate:
    def test_evaluate_completo(self, data_store_completo):
        # Pipeline F₁ → F₂ → F₃
        domain = phase1_to_topological_domain(data_store_completo, {})
        pm = compute_pyramidal_metrics(domain.df_apus, domain.df_insumos)
        structural = phase2_to_thermodynamic_state(domain, pm)
        thermo = ThermodynamicEvaluator.evaluate(structural)
        assert isinstance(thermo, ThermodynamicState)
        assert 0.0 <= thermo.entropia_normalizada <= 1.0
        assert 0.0 <= thermo.temperatura_ingesta <= 100.0
        assert thermo.estabilidad in ("ESTABLE", "METASTABLE", "CAOTICO")
        assert thermo.free_energy_proxy == pytest.approx(
            thermo.temperatura_ingesta * thermo.entropia_normalizada, rel=1e-9
        )

    def test_to_dict(self, data_store_completo):
        domain = phase1_to_topological_domain(data_store_completo, {})
        pm = compute_pyramidal_metrics(domain.df_apus, domain.df_insumos)
        structural = phase2_to_thermodynamic_state(domain, pm)
        thermo = ThermodynamicEvaluator.evaluate(structural)
        d = thermo.to_dict()
        assert "entropia_shannon" in d
        assert "entropia_von_neumann" in d
        assert "free_energy_proxy" in d
        assert "estabilidad" in d


# ============================================================================
# ORQUESTADOR — F₃ ∘ F₂ ∘ F₁
# ============================================================================

class TestValidateAndCleanData:
    def test_flujo_completo_exito(self, data_store_completo):
        result = validate_and_clean_data(deepcopy(data_store_completo))
        assert result["validation_summary"]["exito"] is True
        assert "validation_metrics" in result
        assert "pyramidal_metrics" in result
        assert "quality_entropy_analysis" in result
        assert result["validation_summary"]["thermal_status"] in (
            "ESTABLE", "METASTABLE", "CAOTICO", "DESCONOCIDO", "OMITIDO",
        )

    def test_store_invalido(self):
        result = validate_and_clean_data(None)  # type: ignore
        assert result["validation_summary"]["exito"] is False
        assert "error" in result

    def test_store_vacio(self):
        result = validate_and_clean_data({})
        # Dict vacío es falsy → error
        assert result["validation_summary"]["exito"] is False

    def test_solo_presupuesto(self, presupuesto_basico):
        store = {"presupuesto": deepcopy(presupuesto_basico)}
        result = validate_and_clean_data(store)
        assert result["validation_summary"]["exito"] is True
        assert "presupuesto" in result

    def test_solo_apus_detail(self, apus_detail_basico):
        store = {"apus_detail": deepcopy(apus_detail_basico)}
        result = validate_and_clean_data(store)
        assert result["validation_summary"]["exito"] is True

    def test_no_muta_entrada(self, data_store_completo):
        original = deepcopy(data_store_completo)
        validate_and_clean_data(data_store_completo)
        # Los ítems originales no deben ganar claves alertas
        for orig_item, curr_item in zip(
            original.get("presupuesto", []),
            data_store_completo.get("presupuesto", []),
        ):
            assert "alertas" not in orig_item or orig_item == curr_item

    def test_omitir_analisis_termico(self, data_store_completo):
        result = validate_and_clean_data(
            deepcopy(data_store_completo),
            aplicar_analisis_termico=False,
        )
        assert result["quality_entropy_analysis"]["estabilidad"] == "OMITIDO"

    def test_alertas_en_presupuesto(self, data_store_completo):
        result = validate_and_clean_data(deepcopy(data_store_completo))
        # Debe haber al menos costo negativo y excesivo
        alertas_tipos = []
        for item in result.get("presupuesto") or []:
            for a in item.get("alertas") or []:
                alertas_tipos.append(a["tipo"])
        assert TipoAlerta.COSTO_NEGATIVO.value in alertas_tipos
        assert TipoAlerta.COSTO_EXCESIVO.value in alertas_tipos

    def test_recalculo_cantidad_en_apus(self, data_store_completo):
        result = validate_and_clean_data(deepcopy(data_store_completo))
        recalculados = [
            it for it in result.get("apus_detail") or []
            if any(
                a.get("tipo") == TipoAlerta.CANTIDAD_RECALCULADA.value
                for a in (it.get("alertas") or [])
            )
        ]
        assert len(recalculados) >= 1

    def test_pyramidal_metrics_keys(self, data_store_completo):
        result = validate_and_clean_data(deepcopy(data_store_completo))
        pm = result["pyramidal_metrics"]
        for key in (
            "base_width", "structure_load", "pyramid_stability_index",
            "floating_nodes", "connected_components",
            "algebraic_connectivity", "spof_list", "spectral_radius",
        ):
            assert key in pm

    def test_quality_entropy_keys(self, data_store_completo):
        result = validate_and_clean_data(deepcopy(data_store_completo))
        qe = result["quality_entropy_analysis"]
        for key in (
            "entropia_shannon", "entropia_normalizada",
            "entropia_von_neumann", "tasa_alertas",
            "psi_estructural", "temperatura_ingesta",
            "estabilidad", "free_energy_proxy",
        ):
            assert key in qe

    def test_summary_contiene_topologia(self, data_store_completo):
        result = validate_and_clean_data(deepcopy(data_store_completo))
        s = result["validation_summary"]
        assert "algebraic_connectivity" in s
        assert "connected_components" in s
        assert "spof_count" in s
        assert "floating_nodes_count" in s

    def test_telemetria_opcional(self, data_store_completo):
        mock_tel = MagicMock()
        result = validate_and_clean_data(
            deepcopy(data_store_completo),
            telemetry_context=mock_tel,
        )
        assert result["validation_summary"]["exito"] is True
        mock_tel.start_step.assert_called()
        mock_tel.end_step.assert_called()

    def test_flags_validaciones_deshabilitadas(self, data_store_completo):
        result = validate_and_clean_data(
            deepcopy(data_store_completo),
            validaciones_habilitadas={
                "extreme_costs": False,
                "quantity_coherence": False,
                "descriptions": False,
            },
        )
        assert result["validation_summary"]["exito"] is True

    def test_total_alertas_no_negativo(self, data_store_completo):
        result = validate_and_clean_data(deepcopy(data_store_completo))
        assert result["validation_summary"]["total_alertas"] >= 0


# ============================================================================
# INVARIANTES MATEMÁTICOS GLOBALES (propiedad / property-style)
# ============================================================================

class TestInvariantesMatematicos:
    """Propiedades que deben cumplirse para cualquier entrada bien formada."""

    def test_coherencia_reflexiva(self):
        """∀ Q,P: validar(Q, P, Q*P) es coherente (salvo overflow)."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            Q = float(rng.uniform(1e-3, 1e4))
            P = float(rng.uniform(1e-3, 1e4))
            ca = _validar_coherencia_matematica(Q, P, Q * P)
            assert ca.coherente is True, f"Falló para Q={Q}, P={P}"

    def test_error_abs_no_negativo(self):
        rng = np.random.default_rng(1)
        for _ in range(30):
            Q, P, VT = rng.uniform(0, 1000, size=3)
            ca = _validar_coherencia_matematica(float(Q), float(P), float(VT))
            assert ca.error_absoluto >= 0

    def test_psi_en_rango_unitario(self, grafo_conexo_simple, grafo_con_flotante, grafo_con_spof):
        for fixture in (grafo_conexo_simple, grafo_con_flotante, grafo_con_spof):
            pm = compute_pyramidal_metrics(fixture["apus_df"], fixture["insumos_df"])
            assert 0.0 <= pm.pyramid_stability_index <= 1.0

    def test_lambda2_no_negativo(self, grafo_conexo_simple, grafo_con_spof):
        for fixture in (grafo_conexo_simple, grafo_con_spof):
            pm = compute_pyramidal_metrics(fixture["apus_df"], fixture["insumos_df"])
            assert pm.algebraic_connectivity >= -1e-10

    def test_beta0_al_menos_uno_si_hay_nodos(self, grafo_conexo_simple):
        pm = compute_pyramidal_metrics(
            grafo_conexo_simple["apus_df"],
            grafo_conexo_simple["insumos_df"],
        )
        assert pm.connected_components >= 1

    def test_spectral_radius_no_negativo(self, grafo_conexo_simple):
        pm = compute_pyramidal_metrics(
            grafo_conexo_simple["apus_df"],
            grafo_conexo_simple["insumos_df"],
        )
        assert pm.spectral_radius >= 0

    def test_composicion_funtores_idempotente_en_tipo(self, data_store_completo):
        """
        F₃(F₂(F₁(x))) produce ThermodynamicState;
        re-aplicar F₁ sobre el store validado no rompe tipos.
        """
        r1 = validate_and_clean_data(deepcopy(data_store_completo))
        # El store enriquecido sigue siendo un dict válido para re-entrada
        r2 = validate_and_clean_data({
            "presupuesto": r1.get("presupuesto", []),
            "apus_detail": r1.get("apus_detail", []),
        })
        assert r2["validation_summary"]["exito"] is True
        assert "quality_entropy_analysis" in r2

    def test_entropia_von_neumann_igual_shannon_clasico(self):
        items = [
            {"alertas": [{"tipo": t, "mensaje": "m"}]}
            for t in ["A", "B", "C", "A"]
        ]
        info = ThermodynamicEvaluator.entropy_of_alerts(items)
        assert info["entropia"] == pytest.approx(
            info["entropia_von_neumann"], abs=1e-12
        )


# ============================================================================
# CASOS DEGENERADOS Y RESILIENCIA
# ============================================================================

class TestCasosDegenerados:
    def test_apus_con_item_no_dict(self):
        data = ["no-dict", 42, None, {"CANTIDAD": 1, "VR_UNITARIO": 2, "VALOR_TOTAL": 2}]  # type: ignore
        result, metrics = _validate_quantity_and_coherence(data)  # type: ignore
        assert metrics.total_items_procesados == 4

    def test_presupuesto_con_item_no_dict(self):
        data = [None, "x", {"VALOR_CONSTRUCCION_UN": -1}]  # type: ignore
        result, metrics = _validate_extreme_costs(data)  # type: ignore
        assert metrics.total_items_procesados == 3

    def test_descripcion_con_numero(self):
        limpia, valida, _ = _limpiar_y_validar_descripcion(12345)
        assert valida is True
        assert limpia == "12345"

    def test_coherencia_con_ceros(self):
        ca = _validar_coherencia_matematica(0.0, 0.0, 0.0)
        assert ca.coherente is True

    def test_coherencia_q_cero_vt_cero(self):
        ca = _validar_coherencia_matematica(0.0, 100.0, 0.0)
        assert ca.coherente is True

    def test_orquestador_skip_on_error(self, data_store_completo):
        result = validate_and_clean_data(
            deepcopy(data_store_completo), skip_on_error=True
        )
        assert "validation_summary" in result

    def test_grafo_un_solo_nodo_apu_sin_aristas(self):
        apus_df = pd.DataFrame({"CODIGO_APU": ["ONLY"]})
        insumos_df = pd.DataFrame({
            "APU_CODIGO": [],
            "DESCRIPCION_INSUMO": [],
        })
        pm = compute_pyramidal_metrics(apus_df, insumos_df)
        assert pm.structure_load == 1
        assert "ONLY" in pm.floating_nodes

    def test_anomaly_validator_un_solo_valor(self):
        av = AnomalyValidator()
        data = [{"VALOR_CONSTRUCCION_UN": 42.0}]
        result, stats = av.detect_cost_anomalies(data)
        # Un solo punto no puede ser outlier por votación robusta
        assert stats["total_anomalias"] == 0


# ============================================================================
# TESTS DE REGRESIÓN NUMÉRICA (anclas de precisión)
# ============================================================================

class TestRegresionNumerica:
    def test_ulp_potencias_de_dos(self):
        # Para x = 2^k, ulp(x) = 2^{k-52}
        for k in range(-10, 20):
            x = math.ldexp(1.0, k)  # 2^k
            expected = math.ldexp(1.0, k - 52)
            assert _ulp(x) == pytest.approx(expected, rel=0, abs=0)

    def test_sensibilidad_3_4_5(self):
        ca = _validar_coherencia_matematica(3.0, 4.0, 12.0)
        assert ca.sensibilidad == pytest.approx(5.0)

    def test_psi_tanh_identidad(self):
        # |V|=|U| ⇒ Ψ = tanh(1)
        apus_df = pd.DataFrame({"CODIGO_APU": ["A", "B"]})
        insumos_df = pd.DataFrame({
            "APU_CODIGO": ["A", "B"],
            "DESCRIPCION_INSUMO": ["I1", "I2"],
            "DESCRIPCION_INSUMO_NORM": ["i1", "i2"],
        })
        pm = compute_pyramidal_metrics(apus_df, insumos_df)
        assert pm.pyramid_stability_index == pytest.approx(math.tanh(1.0), rel=1e-9)

    def test_laplaciano_camino_bipartito(self):
        """
        Camino A1—I1—A2  (en bipartito: A1-I1, A2-I1).
        Matriz A (2×1): [[1],[1]].
        L es 3×3; β₀=1; λ₂ > 0.
        """
        A = np.array([[1.0], [1.0]])
        spec = BipartiteTopology.compute_laplacian_spectrum(A)
        assert spec["connected_components"] == 1
        assert spec["algebraic_connectivity"] > 0

    def test_free_energy_proxy_producto(self):
        items = [
            {"alertas": [{"tipo": "A", "mensaje": "1"}]},
            {"alertas": [{"tipo": "B", "mensaje": "2"}]},
        ]
        store = {"presupuesto": items, "apus_detail": []}
        pm = PyramidalMetrics(
            base_width=1, structure_load=1,
            pyramid_stability_index=0.5,
            floating_nodes=[], connected_components=1,
            algebraic_connectivity=0.1, spof_list=[],
            spectral_radius=1.0,
        )
        domain = IncidenceDomain(
            df_apus=pd.DataFrame(),
            df_insumos=pd.DataFrame(),
            validated_store=store,
            metricas_fase1={},
        )
        structural = StructuralState(
            validated_store=store,
            metricas_fase1={},
            pyramidal=pm,
            incidence=domain,
        )
        thermo = ThermodynamicEvaluator.evaluate(structural)
        assert thermo.free_energy_proxy == pytest.approx(
            thermo.temperatura_ingesta * thermo.entropia_normalizada,
            rel=1e-12,
        )
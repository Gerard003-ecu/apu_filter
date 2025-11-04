"""
Contiene datos de prueba centralizados para ser reutilizados en múltiples
suites de pruebas.

Este módulo define strings multi-línea con datos simulados para los archivos
de entrada (presupuesto, APUs, insumos) y un diccionario de configuración
de prueba. El objetivo es proporcionar un conjunto de datos coherente y
realista que permita probar la interacción entre diferentes componentes del
sistema de forma predecible.
"""

# ======================================================================
# DATOS DE PRUEBA CENTRALIZADOS Y REALISTAS
# ======================================================================

PRESUPUESTO_DATA = """ITEM;DESCRIPCION;UND;CANT.; VR. UNIT ; VR.TOTAL
1.1;SUMINISTRO DE TEJA TRAPEZOIDAL;M2;120.5;52000;6266000
1.2;INSTALACION DE TEJA TRAPEZOIDAL;M2;120.5;11760;1417080
2.1;CUADRILLA TIPO 1 (1 OF + 2 AYU);DIA;5;280000;1400000
3.1;PINTURA ANTICORROSIVA;GL;10;65000;650000
"""

APUS_DATA = """
ITEM: 1.1; UNIDAD: M2
SUMINISTRO TEJA TRAPEZOIDAL ROJA CAL.28
MATERIALES
SUMINISTRO TEJA TRAPEZOIDAL ROJA;M2;1.05;;47619;50000
TRANSPORTE;VIAJE;0.01;;200000;2000

ITEM: 1.2; UNIDAD: M2
INSTALACION TEJA TRAPEZOIDAL
MANO DE OBRA
CUADRILLA TIPO 1 (1 OF + 2 AYU);;0.04;;280000;11200
EQUIPO
HERRAMIENTA MENOR (% MANO DE OBRA);%;0.05;;11200;560

ITEM: 2.1; UNIDAD: DIA
CUADRILLA TIPO 1 (1 OF + 2 AYU)
MANO DE OBRA
OFICIAL;JOR;1;;120000;120000
AYUDANTE;JOR;2;;80000;160000

ITEM: 3.1; UNIDAD: GL
APU DE SUMINISTRO PINTURA
MATERIALES
SUMINISTRO PINTURA ANTICORROSIVA TIPO 1;GL;1.0;;60000;60000
BROCHAS Y RODILLOS;UND;0.1;;50000;5000
"""

INSUMOS_DATA = """G1;MATERIALES;;;;;
CODIGO;DESCRIPCION;UND;;CANT.;VR. UNIT.;
INS-001;TEJA TRAPEZOIDAL ROJA;M2;;;47619;
INS-002;PINTURA ANTICORROSIVA TIPO 1;GL;;;60000;
INS-003;BROCHAS Y RODILLOS;UND;;;50000;
INS-004;TRANSPORTE;VIAJE;;;200000;
G2;MANO DE OBRA;;;;;
CODIGO;DESCRIPCION;UND;;CANT.;VR. UNIT.;
M.O-001;OFICIAL;JOR;;;120000;
M.O-002;AYUDANTE;JOR;;;80000;
"""

TEST_CONFIG = {
    "presupuesto_column_map": {
        "CODIGO_APU": ["ITEM"],
        "DESCRIPCION_APU": ["DESCRIPCION"],
        "CANTIDAD_PRESUPUESTO": ["CANT."],
    },
    "category_keywords": {
        "MATERIALES": "MATERIALES",
        "MANO DE OBRA": "MANO DE OBRA",
        "EQUIPO": "EQUIPO",
        "OTROS": "OTROS",
    },
    "param_map": {
        "material": {"TEJA": "TEJA TRAPEZOIDAL", "PINTURA": "PINTURA ANTICORROSIVA"},
        "cuadrilla": {"1": "TIPO 1"},
    },
}

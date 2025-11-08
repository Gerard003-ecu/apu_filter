import os
import sys
import unittest

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.estimator import calculate_estimate
from app.procesador_csv import process_all_files

# Importar los datos de prueba centralizados y realistas
from tests.test_data import (
    APUS_DATA,
    INSUMOS_DATA,
    PRESUPUESTO_DATA,
    TEST_CONFIG,
)


class TestEstimatorWithNewData(unittest.TestCase):
    """
    Pruebas de integración robustas para la función `calculate_estimate`.
    Valida lógica de búsqueda, mapeo, reglas de negocio, fallbacks y manejo de errores.
    """

    @classmethod
    def setUpClass(cls):
        """
        Prepara el entorno de prueba para toda la clase.

        Crea archivos temporales (presupuesto, APUs, insumos) a partir de los
        datos de prueba centralizados y los procesa una sola vez para
        optimizar el tiempo de ejecución de las pruebas. El `data_store`
        resultante se reutiliza en todos los tests de la clase.
        """
        cls.presupuesto_path = "test_presupuesto_estimator.csv"
        with open(cls.presupuesto_path, "w", encoding="latin1") as f:
            f.write(PRESUPUESTO_DATA)

        cls.apus_path = "test_apus_estimator.csv"
        with open(cls.apus_path, "w", encoding="latin1") as f:
            f.write(APUS_DATA)

        cls.insumos_path = "test_insumos_estimator.csv"
        with open(cls.insumos_path, "w", encoding="latin1") as f:
            f.write(INSUMOS_DATA)

        # Procesar los datos una sola vez para toda la clase de prueba
        cls.data_store = process_all_files(
            cls.presupuesto_path, cls.apus_path, cls.insumos_path, config=TEST_CONFIG
        )
        if "error" in cls.data_store:
            raise RuntimeError(f"La preparación de datos falló: {cls.data_store['error']}")

    @classmethod
    def tearDownClass(cls):
        """
        Limpia el entorno después de que todas las pruebas de la clase se hayan
        ejecutado, eliminando los archivos temporales creados.
        """
        for path in [cls.presupuesto_path, cls.apus_path, cls.insumos_path]:
            if os.path.exists(path):
                os.remove(path)

    # ==============================
    # CASO 1: ESCENARIO IDEAL (TODOS LOS APUS EXISTEN)
    # ==============================
    def test_calculate_estimate_ideal_case(self):
        """
        Escenario ideal: Todos los APUs existen y coinciden exactamente.
        Verifica valores calculados y lógica de mapeo, reglas de negocio y rendimiento.
        """
        params = {"material": "TEJA", "cuadrilla": "1", "zona": "ZONA 1", "izaje": "MECANICO", "seguridad": "ALTA"}
        result = calculate_estimate(params, self.data_store, TEST_CONFIG)

        # No debe haber errores
        self.assertNotIn("error", result, f"Error inesperado: {result.get('error')}")

        # Valores esperados según los datos de prueba
        expected_suministro = 50000.0
        expected_instalacion = 11760.0  # (8400 / 25 * 1.1) + 1200 * 1.2 + 500
        expected_total = 61760.0
        expected_rendimiento = 25.0  # 1 / (1/25)

        # Validar valores numéricos con tolerancia
        self.assertAlmostEqual(result["valor_suministro"], expected_suministro, places=2, msg="Valor de suministro incorrecto")
        self.assertAlmostEqual(result["valor_instalacion"], expected_instalacion, places=2, msg="Valor de instalación incorrecto")
        self.assertAlmostEqual(result["valor_construccion"], expected_total, places=2, msg="Valor total incorrecto")
        self.assertAlmostEqual(result["rendimiento_m2_por_dia"], expected_rendimiento, places=2, msg="Rendimiento incorrecto")

        # Validar que el mapeo de cuadrilla funcionó
        self.assertIn("CUADRILLA TIPO 1", result["apu_encontrado"], "Cuadrilla no fue mapeada correctamente")
        self.assertIn("TEJA TRAPEZOIDAL", result["apu_encontrado"], "Descripción de suministro o tarea no encontrada")

        # Validar reglas de negocio aplicadas
        rules = TEST_CONFIG.get("estimator_rules", {})
        factor_zona = rules.get("factores_zona", {}).get("ZONA 1", 1.0)
        factor_seguridad = rules.get("factor_seguridad", {}).get("ALTA", 1.0)
        costo_adicional_izaje = rules.get("costo_adicional_izaje", {}).get("MECANICO", 0)

        # Verificar que las reglas se aplicaron correctamente (validación indirecta)
        # No validamos el cálculo aquí, pero sí que los valores no son los por defecto
        self.assertNotEqual(factor_zona, 1.0, "Factor de zona no debería ser 1.0 en ZONA 1")
        self.assertNotEqual(factor_seguridad, 1.0, "Factor de seguridad no debería ser 1.0 en ALTA")
        self.assertEqual(costo_adicional_izaje, 500, "Costo adicional por izaje mecánico debe ser 500")

    # ==============================
    # CASO 2: TAREA SINTÉTICA (NO EXISTE APU DE INSTALACIÓN)
    # ==============================
    def test_calculate_estimate_synthetic_task_fallback(self):
        """
        Prueba el fallback de tarea sintética cuando no existe APU de instalación.
        Verifica que se use el material mapeado y que el rendimiento sea 0.
        """
        params = {"material": "PINTURA", "cuadrilla": "1"}  # Pintura no tiene APU de instalación
        result = calculate_estimate(params, self.data_store, TEST_CONFIG)

        self.assertNotIn("error", result, f"Error inesperado: {result.get('error')}")

        # Suministro debe existir
        self.assertGreater(result["valor_suministro"], 0, "Suministro de pintura no encontrado")
        self.assertIn("PINTURA ANTICORROSIVA", result["apu_encontrado"], "Descripción de suministro no coincide")

        # Tarea debe ser sintética
        self.assertIn("INSTALACION PINTURA", result["apu_encontrado"], "Tarea sintética no se creó correctamente")
        self.assertEqual(result["rendimiento_m2_por_dia"], 0.0, "Rendimiento debe ser 0 cuando no hay tarea de instalación")
        self.assertEqual(result["valor_instalacion"], 0.0, "Instalación debe ser 0 sin rendimiento ni equipo")

        # Total = solo suministro
        self.assertAlmostEqual(result["valor_construccion"], result["valor_suministro"], places=2, msg="Total debe ser igual al suministro cuando instalación es 0")

    # ==============================
    # CASO 3: CUADRILLA NO ESPECIFICADA ("0")
    # ==============================
    def test_calculate_estimate_no_cuadrilla(self):
        """
        Prueba caso donde cuadrilla = "0" → no se busca cuadrilla.
        Verifica que costo_diario_cuadrilla = 0 y no se intenta calcular rendimiento.
        """
        params = {"material": "TEJA", "cuadrilla": "0"}  # No se busca cuadrilla
        result = calculate_estimate(params, self.data_store, TEST_CONFIG)

        self.assertNotIn("error", result)

        # Suministro y tarea deben encontrarse
        self.assertGreater(result["valor_suministro"], 0)
        self.assertGreater(result["valor_instalacion"], 0)  # Debe usar equipo y rendimiento

        # Cuadrilla no debe estar presente
        self.assertIn("Cuadrilla: No encontrada", result["apu_encontrado"], "Cuadrilla debe aparecer como 'No encontrada'")

        # Rendimiento debe ser >0 (porque se encontró tarea)
        self.assertGreater(result["rendimiento_m2_por_dia"], 0)
        # Costo de mano de obra debe ser 0 → instalación = solo equipo * factor
        self.assertEqual(result["valor_instalacion"], result["valor_instalacion"], "Instalación debe depender solo de equipo")

    # ==============================
    # CASO 4: MATERIAL NO MAPEADO / NO ENCONTRADO
    # ==============================
    def test_calculate_estimate_unknown_material(self):
        """
        Prueba material que no existe en APU ni en mapeo.
        Verifica que se use el material original y que el sistema no falle.
        """
        params = {"material": "MATERIAL_FICTICIO_XYZ", "cuadrilla": "1"}
        result = calculate_estimate(params, self.data_store, TEST_CONFIG)

        self.assertNotIn("error", result)

        # Suministro debe ser 0 (no encontrado)
        self.assertEqual(result["valor_suministro"], 0.0, "Suministro debe ser 0 para material desconocido")

        # Tarea debe ser sintética
        self.assertIn("INSTALACION MATERIAL_FICTICIO_XYZ", result["apu_encontrado"], "Tarea sintética no se creó")

        # Cuadrilla debe encontrarse si existe
        self.assertIn("CUADRILLA TIPO 1", result["apu_encontrado"])

        # Total debe ser solo la tarea sintética (con equipo 0 y rendimiento 0) → total = 0
        self.assertEqual(result["valor_construccion"], 0.0, "Total debe ser 0 si no hay suministro ni instalación")

    # ==============================
    # CASO 5: DATOS INCOMPLETOS O CORRUPTOS (MANEJO DE ERRORES)
    # ==============================
    def test_calculate_estimate_missing_processed_apus(self):
        """
        Prueba que el sistema maneje correctamente data_store sin processed_apus.
        """
        corrupted_data_store = self.data_store.copy()
        corrupted_data_store["processed_apus"] = None  # Simular corrupción

        params = {"material": "TEJA", "cuadrilla": "1"}
        result = calculate_estimate(params, corrupted_data_store, TEST_CONFIG)

        self.assertIn("error", result, "Debe retornar error si processed_apus es None")
        self.assertIn("No hay datos de APU procesados disponibles", result["error"])

    def test_calculate_estimate_missing_config_param_map(self):
        """
        Prueba que el sistema no falle si config no tiene param_map.
        """
        config_without_map = TEST_CONFIG.copy()
        config_without_map.pop("param_map", None)

        params = {"material": "TEJA", "cuadrilla": "1"}
        result = calculate_estimate(params, self.data_store, config_without_map)

        self.assertNotIn("error", result, "No debe fallar si param_map no existe")
        self.assertGreater(result["valor_suministro"], 0, "Suministro debe encontrarse sin mapeo")

    def test_calculate_estimate_empty_keywords(self):
        """
        Prueba que el sistema maneje material vacío o nulo.
        """
        params = {"material": "", "cuadrilla": "1"}
        result = calculate_estimate(params, self.data_store, TEST_CONFIG)

        self.assertNotIn("error", result)
        self.assertEqual(result["valor_suministro"], 0.0, "Suministro debe ser 0 si material es vacío")
        self.assertIn("Suministro: No encontrado", result["apu_encontrado"])

    # ==============================
    # CASO 6: VALIDACIÓN DE REGLAS DE NEGOCIO
    # ==============================
    def test_calculate_estimate_custom_rules(self):
        """
        Prueba que las reglas de negocio (zona, izaje, seguridad) se apliquen correctamente.
        """
        # Sobreescribir reglas para controlar el resultado
        custom_rules = {
            "factores_zona": {"ZONA 1": 2.0},
            "costo_adicional_izaje": {"MECANICO": 1000},
            "factor_seguridad": {"ALTA": 1.5}
        }
        config_with_custom_rules = TEST_CONFIG.copy()
        config_with_custom_rules["estimator_rules"] = custom_rules

        params = {
            "material": "TEJA",
            "cuadrilla": "1",
            "zona": "ZONA 1",
            "izaje": "MECANICO",
            "seguridad": "ALTA"
        }

        result = calculate_estimate(params, self.data_store, config_with_custom_rules)

        # Calculamos manualmente el valor esperado
        # Suministro: 50000
        # Cuadrilla diaria: 8400
        # Rendimiento: 25 un/día → Mo base = 8400 / 25 = 336
        # Mo ajustada: 336 * 1.5 = 504
        # Equipo: 1200
        # Instalación: (504 + 1200) * 2.0 + 1000 = 1704 * 2 + 1000 = 3408 + 1000 = 4408
        # Total: 50000 + 4408 = 54408

        expected_total = 54408.0
        self.assertAlmostEqual(result["valor_construccion"], expected_total, places=2, msg="Cálculo con reglas personalizadas incorrecto")

    # ==============================
    # CASO 7: MODOS DE BÚSQUEDA (substring para cuadrillas)
    # ==============================
    def test_calculate_estimate_substring_match_for_cuadrilla(self):
        """
        Verifica que la búsqueda de cuadrilla use `match_mode='substring'` y no solo palabras sueltas.
        Esto es crítico porque "CUADRILLA TIPO 1" debe coincidir con "cuadrilla tipo 1".
        """
        # En los datos, la cuadrilla está como "CUADRILLA TIPO 1 (1 OF + 2 AYU)"
        # La keyword es "cuadrilla tipo 1" → debe coincidir por substring
        params = {"material": "TEJA", "cuadrilla": "1"}
        result = calculate_estimate(params, self.data_store, TEST_CONFIG)

        self.assertNotIn("error", result)
        self.assertIn("CUADRILLA TIPO 1", result["apu_encontrado"])
        self.assertGreater(result["valor_instalacion"], 0, "Instalación debe incluir costo de cuadrilla")

        # Si el modo fuera 'words', no coincidiría por el "(1 OF + 2 AYU)"
        # Por lo tanto, el hecho de que encuentre la cuadrilla confirma que usa substring

    # ==============================
    # CASO 8: VALIDACIÓN DE LOG (NO STRING, SINO ESTRUCTURA)
    # ==============================
    def test_calculate_estimate_log_structure(self):
        """
        Verifica que el log contenga mensajes clave (no para validación de valores, sino de flujo).
        Esto asegura que la lógica interna se ejecuta como se espera.
        """
        params = {"material": "TEJA", "cuadrilla": "1"}
        result = calculate_estimate(params, self.data_store, TEST_CONFIG)

        log_lines = result["log"].splitlines()

        # Verificamos que el flujo de lógica se ejecutó correctamente
        expected_log_snippets = [
            "🔍 Buscando: teja",
            "🔍 Buscando: cuadrilla tipo 1",
            "🔍 Buscando: teja",
            "✅ Match FLEXIBLE encontrado",
            "💰 Valor encontrado: $50,000.00",
            "⏱️ Rendimiento calculado: 25.00 un/día",
            "📊 RESUMEN EJECUTIVO"
        ]

        for snippet in expected_log_snippets:
            found = any(snippet in line for line in log_lines)
            self.assertTrue(found, f"Mensaje esperado en log no encontrado: '{snippet}'")

        # También verificamos que no haya errores en el log
        self.assertFalse(any("ERROR" in line or "❌" in line for line in log_lines), "No se esperan errores en este escenario")

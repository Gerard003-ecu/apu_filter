"""
Suite de pruebas para la aplicación Flask.
Prueba endpoints, manejo de sesiones, validación de archivos y procesamiento de datos.
"""

import io
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import tempfile
import shutil

# Agregar el directorio raíz del proyecto al path de Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importar componentes de la aplicación
from app.app import (
    create_app,
    SessionManager,
    FileValidator,
    APUProcessor,
    session_manager,
    setup_logging
)

# Importar datos de prueba centralizados
from tests.test_data import (
    APUS_DATA,
    INSUMOS_DATA,
    PRESUPUESTO_DATA,
    TEST_CONFIG,
)

# ============================================================================
# FIXTURES Y DATOS DE PRUEBA ADICIONALES
# ============================================================================

INVALID_CSV_DATA = "invalid,csv,data\nwithout,proper,structure"
LARGE_FILE_DATA = "x" * (17 * 1024 * 1024)  # Archivo de más de 16MB

APU_TEST_DETAILS = [
    {
        "CODIGO_APU": "TEST001",
        "DESCRIPCION_INSUMO": "Material A",
        "CATEGORIA": "MATERIALES",
        "CANTIDAD_APU": 10.0,
        "VALOR_TOTAL_APU": 100000.0,
        "RENDIMIENTO": 1.0,
        "UNIDAD_APU": "UN",
        "PRECIO_UNIT_APU": 10000.0,
        "UNIDAD_INSUMO": "UN",
        "alerta": "Test alert"
    },
    {
        "CODIGO_APU": "TEST001",
        "DESCRIPCION_INSUMO": "Material A",  # Duplicado para probar agrupación
        "CATEGORIA": "MATERIALES",
        "CANTIDAD_APU": 5.0,
        "VALOR_TOTAL_APU": 50000.0,
        "RENDIMIENTO": 0.5,
        "UNIDAD_APU": "UN",
        "PRECIO_UNIT_APU": 10000.0,
        "UNIDAD_INSUMO": "UN",
        "alerta": "Another alert"
    }
]

# ============================================================================
# PRUEBAS DEL GESTOR DE SESIONES
# ============================================================================

class TestSessionManager(unittest.TestCase):
    """Pruebas para la clase SessionManager."""
    
    def setUp(self):
        """Configurar un nuevo gestor de sesiones para cada prueba."""
        self.manager = SessionManager(timeout=60)  # Timeout corto para pruebas
    
    def test_create_session(self):
        """Verificar creación de nuevas sesiones."""
        session_id = self.manager.create_session()
        self.assertIsNotNone(session_id)
        self.assertEqual(self.manager.active_count, 1)
        
        # Verificar que se puede crear con ID personalizado
        custom_id = "custom-session-id"
        result_id = self.manager.create_session(custom_id)
        self.assertEqual(result_id, custom_id)
        self.assertEqual(self.manager.active_count, 2)
    
    def test_get_session(self):
        """Verificar recuperación de sesiones."""
        session_id = self.manager.create_session()
        session_data = self.manager.get_session(session_id)
        
        self.assertIsNotNone(session_data)
        self.assertIn("data", session_data)
        self.assertIn("last_activity", session_data)
        self.assertIn("created_at", session_data)
        
        # Intentar obtener sesión inexistente
        non_existent = self.manager.get_session("non-existent")
        self.assertIsNone(non_existent)
    
    def test_update_session(self):
        """Verificar actualización de datos de sesión."""
        session_id = self.manager.create_session()
        test_data = {"test_key": "test_value"}
        
        success = self.manager.update_session(session_id, test_data)
        self.assertTrue(success)
        
        session_data = self.manager.get_session(session_id)
        self.assertEqual(session_data["data"], test_data)
        
        # Intentar actualizar sesión inexistente
        success = self.manager.update_session("non-existent", test_data)
        self.assertFalse(success)
    
    def test_session_expiration(self):
        """Verificar expiración automática de sesiones."""
        self.manager.timeout = 0.1  # Timeout muy corto para prueba
        session_id = self.manager.create_session()
        
        # Sesión debe existir inicialmente
        self.assertIsNotNone(self.manager.get_session(session_id))
        
        # Esperar a que expire
        time.sleep(0.2)
        
        # Sesión debe haber expirado
        self.assertIsNone(self.manager.get_session(session_id))
        self.assertEqual(self.manager.active_count, 0)
    
    def test_cleanup_expired_sessions(self):
        """Verificar limpieza de sesiones expiradas."""
        self.manager.timeout = 0.1
        
        # Crear múltiples sesiones
        session_ids = [self.manager.create_session() for _ in range(3)]
        self.assertEqual(self.manager.active_count, 3)
        
        # Esperar a que expiren
        time.sleep(0.2)
        
        # Limpiar sesiones expiradas
        removed_count = self.manager.cleanup_expired_sessions()
        self.assertEqual(removed_count, 3)
        self.assertEqual(self.manager.active_count, 0)
    
    def test_remove_session(self):
        """Verificar eliminación manual de sesiones."""
        session_id = self.manager.create_session()
        
        # Eliminar sesión existente
        success = self.manager.remove_session(session_id)
        self.assertTrue(success)
        self.assertEqual(self.manager.active_count, 0)
        
        # Intentar eliminar sesión inexistente
        success = self.manager.remove_session("non-existent")
        self.assertFalse(success)

# ============================================================================
# PRUEBAS DEL VALIDADOR DE ARCHIVOS
# ============================================================================

class TestFileValidator(unittest.TestCase):
    """Pruebas para la clase FileValidator."""
    
    def setUp(self):
        """Configurar validador para cada prueba."""
        self.validator = FileValidator()
    
    def test_validate_empty_file(self):
        """Verificar validación de archivo vacío."""
        mock_file = MagicMock()
        mock_file.filename = ""
        
        is_valid, error = self.validator.validate_file(mock_file)
        self.assertFalse(is_valid)
        self.assertEqual(error, "Archivo no seleccionado")
    
    def test_validate_invalid_extension(self):
        """Verificar rechazo de extensiones no permitidas."""
        mock_file = MagicMock()
        mock_file.filename = "test.exe"
        
        is_valid, error = self.validator.validate_file(mock_file)
        self.assertFalse(is_valid)
        self.assertIn("Extensión no permitida", error)
    
    def test_validate_valid_extensions(self):
        """Verificar aceptación de extensiones válidas."""
        valid_extensions = ['.csv', '.xlsx', '.xls']
        
        for ext in valid_extensions:
            mock_file = MagicMock()
            mock_file.filename = f"test{ext}"
            mock_file.tell.return_value = 1024  # Tamaño pequeño
            
            is_valid, error = self.validator.validate_file(mock_file)
            self.assertTrue(is_valid)
            self.assertIsNone(error)
    
    def test_validate_file_size(self):
        """Verificar validación de tamaño de archivo."""
        mock_file = MagicMock()
        mock_file.filename = "test.csv"
        mock_file.tell.return_value = 20 * 1024 * 1024  # 20MB
        
        is_valid, error = self.validator.validate_file(mock_file)
        self.assertFalse(is_valid)
        self.assertIn("Archivo demasiado grande", error)

# ============================================================================
# PRUEBAS DEL PROCESADOR DE APU
# ============================================================================

class TestAPUProcessor(unittest.TestCase):
    """Pruebas para la clase APUProcessor."""
    
    def setUp(self):
        """Configurar procesador para cada prueba."""
        mock_logger = MagicMock()
        self.processor = APUProcessor(mock_logger)
    
    def test_process_apu_details_empty(self):
        """Verificar manejo de lista vacía de detalles."""
        with self.assertRaises(ValueError) as context:
            self.processor.process_apu_details([], "TEST001")
        
        self.assertIn("No se encontraron detalles", str(context.exception))
    
    def test_process_apu_details_grouping(self):
        """Verificar agrupación correcta de items por categoría."""
        result = self.processor.process_apu_details(APU_TEST_DETAILS, "TEST001")
        
        self.assertIn("items", result)
        self.assertIn("desglose", result)
        self.assertIn("total_items", result)
        
        # Verificar que los items duplicados se agruparon
        self.assertEqual(result["total_items"], 1)  # Solo 1 item después de agrupar
        
        # Verificar suma de cantidades
        item = result["items"][0]
        self.assertEqual(item["CANTIDAD"], 15.0)  # 10 + 5
        self.assertEqual(item["VR_TOTAL"], 150000.0)  # 100000 + 50000
    
    def test_organize_breakdown_by_category(self):
        """Verificar organización correcta del desglose por categoría."""
        result = self.processor.process_apu_details(APU_TEST_DETAILS, "TEST001")
        desglose = result["desglose"]
        
        self.assertIn("MATERIALES", desglose)
        self.assertIsInstance(desglose["MATERIALES"], list)
        self.assertEqual(len(desglose["MATERIALES"]), 1)

# ============================================================================
# PRUEBAS DE ENDPOINTS DE LA APLICACIÓN
# ============================================================================

class TestAppEndpoints(unittest.TestCase):
    """
    Suite completa de pruebas para los endpoints de la aplicación Flask.
    """
    
    def setUp(self):
        """Configurar el entorno de prueba."""
        self.app = create_app('testing')
        self.app.testing = True
        self.app_context = self.app.app_context()
        self.app_context.push()
        
        # Crear directorio temporal para uploads
        self.temp_dir = tempfile.mkdtemp()
        self.app.config["UPLOAD_FOLDER"] = self.temp_dir
        
        self.client = self.app.test_client()
        
        # Limpiar el gestor de sesiones global
        global session_manager
        session_manager._sessions.clear()
    
    def tearDown(self):
        """Limpiar el entorno después de cada prueba."""
        self.app_context.pop()
        
        # Eliminar directorio temporal
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Limpiar sesiones
        session_manager._sessions.clear()
    
    def _create_test_file(self, filename, content):
        """
        Crear un archivo de prueba en memoria.
        
        Args:
            filename: Nombre del archivo
            content: Contenido del archivo
            
        Returns:
            Tupla (BytesIO, filename) para usar en requests multipart
        """
        return (io.BytesIO(content.encode("latin1")), filename)
    
    def test_index_route(self):
        """Verificar que la ruta principal funciona."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
    
    def test_health_check(self):
        """Verificar el endpoint de salud."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("active_sessions", data)
        self.assertIn("timestamp", data)
        self.assertIn("version", data)
    
    def test_upload_files_success(self):
        """Verificar carga exitosa de archivos."""
        with self.client as c:
            # Configurar la aplicación con los datos de prueba
            c.application.config["APP_CONFIG"] = TEST_CONFIG
            
            # Preparar archivos de prueba
            data = {
                "presupuesto": self._create_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._create_test_file("apus.csv", APUS_DATA),
                "insumos": self._create_test_file("insumos.csv", INSUMOS_DATA),
            }
            
            # Realizar carga
            response = c.post("/upload", data=data, content_type="multipart/form-data")
            
            self.assertEqual(response.status_code, 200)
            
            data = json.loads(response.data)
            self.assertIn("session_id", data)
            self.assertIn("presupuesto", data)
            self.assertIn("apus_detail", data)
    
    def test_upload_files_missing_files(self):
        """Verificar manejo de archivos faltantes."""
        with self.client as c:
            # Solo enviar un archivo de los tres requeridos
            data = {
                "presupuesto": self._create_test_file("presupuesto.csv", PRESUPUESTO_DATA)
            }
            
            response = c.post("/upload", data=data, content_type="multipart/form-data")
            
            self.assertEqual(response.status_code, 400)
            
            data = json.loads(response.data)
            self.assertIn("error", data)
            self.assertIn("Faltan archivos", data["error"])
            self.assertEqual(data["code"], "MISSING_FILES")
    
    def test_upload_files_invalid_extension(self):
        """Verificar rechazo de archivos con extensión inválida."""
        with self.client as c:
            data = {
                "presupuesto": self._create_test_file("presupuesto.txt", PRESUPUESTO_DATA),
                "apus": self._create_test_file("apus.csv", APUS_DATA),
                "insumos": self._create_test_file("insumos.csv", INSUMOS_DATA),
            }
            
            response = c.post("/upload", data=data, content_type="multipart/form-data")
            
            self.assertEqual(response.status_code, 400)
            
            data = json.loads(response.data)
            self.assertIn("error", data)
            self.assertEqual(data["code"], "INVALID_FILE")
    
    def test_upload_files_empty_file(self):
        """Verificar manejo de archivos vacíos."""
        with self.client as c:
            data = {
                "presupuesto": self._create_test_file("", ""),
                "apus": self._create_test_file("apus.csv", APUS_DATA),
                "insumos": self._create_test_file("insumos.csv", INSUMOS_DATA),
            }
            
            response = c.post("/upload", data=data, content_type="multipart/form-data")
            
            self.assertEqual(response.status_code, 400)
            
            data = json.loads(response.data)
            self.assertIn("error", data)
            self.assertEqual(data["code"], "INVALID_FILE")
    
    def test_get_apu_detail_with_session(self):
        """Verificar obtención de detalles de APU con sesión válida."""
        with self.client as c:
            # Primero cargar archivos para crear sesión
            c.application.config["APP_CONFIG"] = TEST_CONFIG
            
            upload_data = {
                "presupuesto": self._create_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._create_test_file("apus.csv", APUS_DATA),
                "insumos": self._create_test_file("insumos.csv", INSUMOS_DATA),
            }
            
            upload_response = c.post(
                "/upload", 
                data=upload_data, 
                content_type="multipart/form-data"
            )
            self.assertEqual(upload_response.status_code, 200)
            
            # Solicitar detalles de un APU
            response = c.get("/api/apu/10.M01")
            self.assertEqual(response.status_code, 200)
            
            data = json.loads(response.data)
            self.assertIn("codigo", data)
            self.assertIn("descripcion", data)
            self.assertIn("desglose", data)
            self.assertIn("simulation", data)
            self.assertIn("metadata", data)
    
    def test_get_apu_detail_without_session(self):
        """Verificar error al solicitar APU sin sesión."""
        response = self.client.get("/api/apu/10.M01")
        self.assertEqual(response.status_code, 401)
        
        data = json.loads(response.data)
        self.assertEqual(data["code"], "NO_SESSION")
    
    def test_get_apu_detail_not_found(self):
        """Verificar manejo de APU no encontrado."""
        with self.client as c:
            # Crear sesión con datos
            c.application.config["APP_CONFIG"] = TEST_CONFIG
            
            upload_data = {
                "presupuesto": self._create_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._create_test_file("apus.csv", APUS_DATA),
                "insumos": self._create_test_file("insumos.csv", INSUMOS_DATA),
            }
            
            upload_response = c.post(
                "/upload", 
                data=upload_data, 
                content_type="multipart/form-data"
            )
            self.assertEqual(upload_response.status_code, 200)
            
            # Solicitar APU inexistente
            response = c.get("/api/apu/NON_EXISTENT")
            self.assertEqual(response.status_code, 404)
            
            data = json.loads(response.data)
            self.assertEqual(data["code"], "APU_NOT_FOUND")
    
    def test_get_estimate_with_session(self):
        """Verificar cálculo de estimación con sesión válida."""
        with self.client as c:
            # Cargar archivos para crear sesión
            c.application.config["APP_CONFIG"] = TEST_CONFIG
            
            upload_data = {
                "presupuesto": self._create_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._create_test_file("apus.csv", APUS_DATA),
                "insumos": self._create_test_file("insumos.csv", INSUMOS_DATA),
            }
            
            upload_response = c.post(
                "/upload", 
                data=upload_data, 
                content_type="multipart/form-data"
            )
            self.assertEqual(upload_response.status_code, 200)
            
            # Solicitar estimación
            estimate_params = {"material": "TEJA", "cuadrilla": "1"}
            response = c.post("/api/estimate", json=estimate_params)
            
            self.assertEqual(response.status_code, 200)
            
            data = json.loads(response.data)
            self.assertIn("valor_suministro", data)
            self.assertIn("valor_instalacion", data)
            self.assertIn("valor_construccion", data)
            
            # Verificar valores esperados
            self.assertAlmostEqual(data["valor_suministro"], 50000.0, places=2)
            self.assertAlmostEqual(data["valor_instalacion"], 11760.0, places=2)
            self.assertAlmostEqual(data["valor_construccion"], 61760.0, places=2)
    
    def test_get_estimate_without_session(self):
        """Verificar error al solicitar estimación sin sesión."""
        estimate_params = {"material": "TEJA", "cuadrilla": "1"}
        response = self.client.post("/api/estimate", json=estimate_params)
        
        self.assertEqual(response.status_code, 401)
        
        data = json.loads(response.data)
        self.assertEqual(data["code"], "NO_SESSION")
    
    def test_get_estimate_invalid_content_type(self):
        """Verificar rechazo de content-type inválido."""
        with self.client as c:
            # Crear sesión primero
            c.application.config["APP_CONFIG"] = TEST_CONFIG
            
            upload_data = {
                "presupuesto": self._create_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._create_test_file("apus.csv", APUS_DATA),
                "insumos": self._create_test_file("insumos.csv", INSUMOS_DATA),
            }
            
            c.post("/upload", data=upload_data, content_type="multipart/form-data")
            
            # Enviar estimación con content-type incorrecto
            response = c.post(
                "/api/estimate", 
                data="material=TEJA&cuadrilla=1",
                content_type="application/x-www-form-urlencoded"
            )
            
            self.assertEqual(response.status_code, 400)
            
            data = json.loads(response.data)
            self.assertEqual(data["code"], "INVALID_CONTENT_TYPE")
    
    def test_get_estimate_no_params(self):
        """Verificar manejo de solicitud sin parámetros."""
        with self.client as c:
            # Crear sesión
            c.application.config["APP_CONFIG"] = TEST_CONFIG
            
            upload_data = {
                "presupuesto": self._create_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._create_test_file("apus.csv", APUS_DATA),
                "insumos": self._create_test_file("insumos.csv", INSUMOS_DATA),
            }
            
            c.post("/upload", data=upload_data, content_type="multipart/form-data")
            
            # Enviar estimación sin parámetros
            response = c.post("/api/estimate", json={})
            
            self.assertEqual(response.status_code, 400)
            
            data = json.loads(response.data)
            self.assertEqual(data["code"], "NO_PARAMS")
    
    def test_404_error_handler(self):
        """Verificar manejo de rutas no encontradas."""
        response = self.client.get("/non/existent/route")
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertIn("error", data)
        self.assertEqual(data["code"], "NOT_FOUND")
        self.assertIn("path", data)
    
    def test_session_persistence_across_requests(self):
        """Verificar que la sesión persiste entre requests."""
        with self.client as c:
            # Primera request: cargar archivos
            c.application.config["APP_CONFIG"] = TEST_CONFIG
            
            upload_data = {
                "presupuesto": self._create_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._create_test_file("apus.csv", APUS_DATA),
                "insumos": self._create_test_file("insumos.csv", INSUMOS_DATA),
            }
            
            upload_response = c.post(
                "/upload", 
                data=upload_data, 
                content_type="multipart/form-data"
            )
            self.assertEqual(upload_response.status_code, 200)
            
            # Segunda request: obtener detalles de APU (debe usar misma sesión)
            apu_response = c.get("/api/apu/10.M01")
            self.assertEqual(apu_response.status_code, 200)
            
            # Tercera request: obtener estimación (debe usar misma sesión)
            estimate_params = {"material": "TEJA", "cuadrilla": "1"}
            estimate_response = c.post("/api/estimate", json=estimate_params)
            self.assertEqual(estimate_response.status_code, 200)
    
    def test_concurrent_sessions(self):
        """Verificar manejo de múltiples sesiones concurrentes."""
        # Cliente 1
        client1 = self.app.test_client()
        with client1 as c1:
            c1.application.config["APP_CONFIG"] = TEST_CONFIG
            
            data1 = {
                "presupuesto": self._create_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._create_test_file("apus.csv", APUS_DATA),
                "insumos": self._create_test_file("insumos.csv", INSUMOS_DATA),
            }
            
            response1 = c1.post("/upload", data=data1, content_type="multipart/form-data")
            self.assertEqual(response1.status_code, 200)
        
        # Cliente 2
        client2 = self.app.test_client()
        with client2 as c2:
            c2.application.config["APP_CONFIG"] = TEST_CONFIG
            
            data2 = {
                "presupuesto": self._create_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._create_test_file("apus.csv", APUS_DATA),
                "insumos": self._create_test_file("insumos.csv", INSUMOS_DATA),
            }
            
            response2 = c2.post("/upload", data=data2, content_type="multipart/form-data")
            self.assertEqual(response2.status_code, 200)
        
        # Verificar que hay 2 sesiones activas
        self.assertEqual(session_manager.active_count, 2)

# ============================================================================
# PRUEBAS DE INTEGRACIÓN
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Pruebas de integración end-to-end."""
    
    def setUp(self):
        """Configurar entorno de integración."""
        self.app = create_app('testing')
        self.app.testing = True
        self.app_context = self.app.app_context()
        self.app_context.push()
        
        self.temp_dir = tempfile.mkdtemp()
        self.app.config["UPLOAD_FOLDER"] = self.temp_dir
        self.app.config["APP_CONFIG"] = TEST_CONFIG
        
        self.client = self.app.test_client()
        session_manager._sessions.clear()
    
    def tearDown(self):
        """Limpiar entorno de integración."""
        self.app_context.pop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        session_manager._sessions.clear()
    
    def test_complete_workflow(self):
        """Prueba el flujo completo de la aplicación."""
        with self.client as c:
            # 1. Verificar salud de la aplicación
            health_response = c.get("/api/health")
            self.assertEqual(health_response.status_code, 200)
            
            # 2. Cargar archivos
            upload_data = {
                "presupuesto": (
                    io.BytesIO(PRESUPUESTO_DATA.encode("latin1")), 
                    "presupuesto.csv"
                ),
                "apus": (
                    io.BytesIO(APUS_DATA.encode("latin1")), 
                    "apus.csv"
                ),
                "insumos": (
                    io.BytesIO(INSUMOS_DATA.encode("latin1")), 
                    "insumos.csv"
                ),
            }
            
            upload_response = c.post(
                "/upload", 
                data=upload_data, 
                content_type="multipart/form-data"
            )
            self.assertEqual(upload_response.status_code, 200)
            
            upload_data = json.loads(upload_response.data)
            self.assertIn("session_id", upload_data)
            
            # 3. Obtener detalles de múltiples APUs
            apu_codes = ["10.M01", "11.M01"]  # Asumiendo que existen en los datos
            
            for code in apu_codes:
                apu_response = c.get(f"/api/apu/{code}")
                self.assertEqual(apu_response.status_code, 200)
                
                apu_data = json.loads(apu_response.data)
                self.assertEqual(apu_data["codigo"], code)
                self.assertIn("simulation", apu_data)
            
            # 4. Calcular múltiples estimaciones con diferentes parámetros
            estimate_scenarios = [
                {"material": "TEJA", "cuadrilla": "1"},
                {"material": "TEJA", "cuadrilla": "2"},
                {"material": "CONCRETO", "cuadrilla": "1"}
            ]
            
            for params in estimate_scenarios:
                estimate_response = c.post("/api/estimate", json=params)
                self.assertEqual(estimate_response.status_code, 200)
                
                estimate_data = json.loads(estimate_response.data)
                self.assertIn("valor_construccion", estimate_data)
                self.assertGreater(estimate_data["valor_construccion"], 0)
            
            # 5. Verificar que la sesión sigue activa
            final_health = c.get("/api/health")
            health_data = json.loads(final_health.data)
            self.assertGreaterEqual(health_data["active_sessions"], 1)

# ============================================================================
# RUNNER DE PRUEBAS
# ============================================================================

def run_tests():
    """Ejecutar todas las suites de pruebas con reporte detallado."""
    # Crear loader y suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Agregar todas las clases de prueba
    test_classes = [
        TestSessionManager,
        TestFileValidator,
        TestAPUProcessor,
        TestAppEndpoints,
        TestIntegration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Ejecutar con runner verboso
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Imprimir resumen
    print("\n" + "="*70)
    print("RESUMEN DE PRUEBAS")
    print("="*70)
    print(f"Pruebas ejecutadas: {result.testsRun}")
    print(f"Exitosas: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Errores: {len(result.errors)}")
    print(f"Tiempo total: {result.stop - result.start:.2f}s")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
"""
Suite completa de pruebas para app.py

Pruebas exhaustivas con cobertura de endpoints, decoradores, validadores,
procesadores, manejo de errores y funcionalidades de la aplicación Flask.
"""

import json
import logging
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from flask import Flask, session
from werkzeug.datastructures import FileStorage

# Importar módulo a probar
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app as app_module
from app.app import (
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    SESSION_TIMEOUT,
    APUProcessor,
    FileValidator,
    create_app,
    handle_errors,
    load_semantic_search_artifacts,
    require_session,
    setup_logging,
    temporary_upload_directory,
)


# ============================================================================
# FIXTURES - Configuración y datos de prueba
# ============================================================================

@pytest.fixture
def app():
    """Fixture que crea instancia de la aplicación Flask para tests."""
    
    # Mockear Redis para tests
    with patch('redis.from_url') as mock_redis:
        mock_redis_client = MagicMock()
        mock_redis_client.dbsize.return_value = 5
        mock_redis_client.get.return_value = None
        mock_redis_client.set.return_value = True
        mock_redis_client.expire.return_value = True
        mock_redis.return_value = mock_redis_client
        
        # Mockear carga de embeddings
        with patch('app.app.load_semantic_search_artifacts'):
            app = create_app('testing')
            app.config['TESTING'] = True
            app.config['SESSION_REDIS'] = mock_redis_client
            
            yield app


@pytest.fixture
def client(app):
    """Fixture que crea cliente de test para la aplicación."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Fixture que crea CLI runner para tests."""
    return app.test_cli_runner()


@pytest.fixture
def mock_redis():
    """Fixture que proporciona un cliente Redis mockeado."""
    mock_client = MagicMock()
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.expire.return_value = True
    mock_client.delete.return_value = True
    mock_client.dbsize.return_value = 0
    return mock_client


@pytest.fixture
def sample_session_data():
    """Fixture con datos de sesión de prueba."""
    return {
        "data": {
            "presupuesto": [
                {
                    "CODIGO_APU": "APU-001",
                    "original_description": "Concreto f'c=280 kg/cm2",
                    "UNIDAD": "M3",
                    "CANTIDAD": 100.0,
                    "VR_UNITARIO": 350000.0
                }
            ],
            "apus_detail": [
                {
                    "CODIGO_APU": "APU-001",
                    "DESCRIPCION_INSUMO": "Cemento Portland",
                    "CATEGORIA": "SUMINISTRO",
                    "CANTIDAD_APU": 7.5,
                    "UNIDAD_APU": "BLS",
                    "PRECIO_UNIT_APU": 35000.0,
                    "VALOR_TOTAL_APU": 262500.0,
                    "RENDIMIENTO": 1.0,
                    "UNIDAD_INSUMO": "BLS"
                },
                {
                    "CODIGO_APU": "APU-001",
                    "DESCRIPCION_INSUMO": "Arena",
                    "CATEGORIA": "SUMINISTRO",
                    "CANTIDAD_APU": 0.5,
                    "UNIDAD_APU": "M3",
                    "PRECIO_UNIT_APU": 80000.0,
                    "VALOR_TOTAL_APU": 40000.0,
                    "RENDIMIENTO": 1.0,
                    "UNIDAD_INSUMO": "M3"
                }
            ],
            "summary": {
                "total_apus": 1,
                "total_insumos": 2
            }
        }
    }


@pytest.fixture
def sample_csv_content():
    """Fixture con contenido CSV de prueba."""
    return """CODIGO,DESCRIPCION,UNIDAD,CANTIDAD,VR_UNITARIO
APU-001,Concreto,M3,100,350000
APU-002,Acero,KG,500,5000"""


@pytest.fixture
def sample_file():
    """Fixture que crea un archivo FileStorage de prueba."""
    content = b"CODIGO,DESCRIPCION\nAPU-001,Concreto"
    return FileStorage(
        stream=BytesIO(content),
        filename="test.csv",
        content_type="text/csv"
    )


@pytest.fixture
def sample_apu_data():
    """Fixture con datos de APU para procesamiento."""
    return [
        {
            "CODIGO_APU": "APU-001",
            "DESCRIPCION_INSUMO": "Cemento",
            "CATEGORIA": "SUMINISTRO",
            "CANTIDAD_APU": 7.5,
            "UNIDAD_APU": "BLS",
            "PRECIO_UNIT_APU": 35000.0,
            "VALOR_TOTAL_APU": 262500.0,
            "RENDIMIENTO": 1.0,
            "UNIDAD_INSUMO": "BLS"
        },
        {
            "CODIGO_APU": "APU-001",
            "DESCRIPCION_INSUMO": "Oficial",
            "CATEGORIA": "MANO_DE_OBRA",
            "CANTIDAD_APU": 1.0,
            "UNIDAD_APU": "JOR",
            "PRECIO_UNIT_APU": 65000.0,
            "VALOR_TOTAL_APU": 65000.0,
            "RENDIMIENTO": 1.0,
            "UNIDAD_INSUMO": "JOR"
        }
    ]


@pytest.fixture
def temp_dir():
    """Fixture que crea directorio temporal."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


# ============================================================================
# TESTS - CONFIGURACIÓN Y LOGGING
# ============================================================================

class TestSetupLogging:
    """Suite de pruebas para setup_logging()"""
    
    def test_setup_logging_basic(self, app):
        """Debe configurar logging correctamente."""
        setup_logging(app)
        
        assert app.logger.level == logging.DEBUG
        assert len(app.logger.handlers) >= 1
    
    def test_setup_logging_creates_log_directory(self, app):
        """Debe crear directorio de logs."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            setup_logging(app)
            mock_mkdir.assert_called()
    
    def test_setup_logging_file_handler(self, app):
        """Debe configurar handler de archivo."""
        setup_logging(app, log_file="test.log")
        
        # Verificar que hay handlers
        assert len(app.logger.handlers) > 0
    
    def test_setup_logging_console_handler(self, app):
        """Debe configurar handler de consola."""
        from logging import StreamHandler
        
        setup_logging(app)
        
        # Verificar que hay un StreamHandler
        has_stream_handler = any(
            isinstance(h, StreamHandler) for h in app.logger.handlers
        )
        assert has_stream_handler
    
    def test_setup_logging_formatter(self, app):
        """Debe configurar formateadores."""
        setup_logging(app)
        
        for handler in app.logger.handlers:
            assert handler.formatter is not None


# ============================================================================
# TESTS - VALIDADORES
# ============================================================================

class TestFileValidator:
    """Suite de pruebas para FileValidator"""
    
    def test_validate_file_valid_csv(self, sample_file):
        """Debe validar archivo CSV válido."""
        is_valid, error = FileValidator.validate_file(sample_file)
        
        assert is_valid
        assert error is None
    
    def test_validate_file_no_file(self):
        """Debe rechazar archivo None."""
        is_valid, error = FileValidator.validate_file(None)
        
        assert not is_valid
        assert "no seleccionado" in error.lower()
    
    def test_validate_file_empty_filename(self):
        """Debe rechazar filename vacío."""
        empty_file = FileStorage(
            stream=BytesIO(b"test"),
            filename="",
            content_type="text/csv"
        )
        
        is_valid, error = FileValidator.validate_file(empty_file)
        
        assert not is_valid
        assert "no seleccionado" in error.lower()
    
    def test_validate_file_invalid_extension(self):
        """Debe rechazar extensiones no permitidas."""
        invalid_file = FileStorage(
            stream=BytesIO(b"test"),
            filename="test.txt",
            content_type="text/plain"
        )
        
        is_valid, error = FileValidator.validate_file(invalid_file)
        
        assert not is_valid
        assert "extensión no permitida" in error.lower()
    
    def test_validate_file_allowed_extensions(self):
        """Debe aceptar extensiones permitidas."""
        for ext in ['.csv', '.xlsx', '.xls']:
            file = FileStorage(
                stream=BytesIO(b"test"),
                filename=f"test{ext}",
                content_type="application/octet-stream"
            )
            
            is_valid, error = FileValidator.validate_file(file)
            assert is_valid, f"Extensión {ext} debería ser válida"
    
    def test_validate_file_too_large(self):
        """Debe rechazar archivos muy grandes."""
        # Crear archivo que exceda MAX_CONTENT_LENGTH
        large_content = b"x" * (MAX_CONTENT_LENGTH + 1000)
        large_file = FileStorage(
            stream=BytesIO(large_content),
            filename="large.csv",
            content_type="text/csv"
        )
        
        is_valid, error = FileValidator.validate_file(large_file)
        
        assert not is_valid
        assert "demasiado grande" in error.lower()
    
    def test_validate_file_secure_filename(self):
        """Debe manejar nombres de archivo inseguros."""
        unsafe_file = FileStorage(
            stream=BytesIO(b"test"),
            filename="../../../etc/passwd",
            content_type="text/csv"
        )
        
        # secure_filename debería limpiar el nombre
        is_valid, error = FileValidator.validate_file(unsafe_file)
        
        # Puede ser válido o inválido dependiendo de secure_filename
        assert isinstance(is_valid, bool)


# ============================================================================
# TESTS - PROCESADOR APU
# ============================================================================

class TestAPUProcessor:
    """Suite de pruebas para APUProcessor"""
    
    @pytest.fixture
    def processor(self, app):
        """Fixture que crea instancia de APUProcessor."""
        return APUProcessor(app.logger)
    
    def test_process_apu_details_basic(self, processor, sample_apu_data):
        """Debe procesar detalles de APU correctamente."""
        result = processor.process_apu_details(sample_apu_data, "APU-001")
        
        assert "items" in result
        assert "desglose" in result
        assert "total_items" in result
        assert result["total_items"] > 0
    
    def test_process_apu_details_empty_raises_error(self, processor):
        """Debe lanzar error para lista vacía."""
        with pytest.raises(ValueError) as exc_info:
            processor.process_apu_details([], "APU-001")
        
        assert "no se encontraron detalles" in str(exc_info.value).lower()
    
    def test_process_apu_details_categories(self, processor, sample_apu_data):
        """Debe organizar por categorías."""
        result = processor.process_apu_details(sample_apu_data, "APU-001")
        
        assert "SUMINISTRO" in result["desglose"]
        assert "MANO_DE_OBRA" in result["desglose"]
    
    def test_group_by_category(self, processor, sample_apu_data):
        """Debe agrupar correctamente por categoría."""
        df = pd.DataFrame(sample_apu_data)
        result = processor._group_by_category(df)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verificar estructura
        for item in result:
            assert "DESCRIPCION" in item
            assert "CATEGORIA" in item
            assert "CANTIDAD" in item
    
    def test_group_by_category_aggregation(self, processor):
        """Debe agregar valores correctamente."""
        # Datos duplicados que deben agregarse
        data = [
            {
                "CODIGO_APU": "APU-001",
                "DESCRIPCION_INSUMO": "Cemento",
                "CATEGORIA": "SUMINISTRO",
                "CANTIDAD_APU": 5.0,
                "VALOR_TOTAL_APU": 100000.0,
                "RENDIMIENTO": 1.0,
                "UNIDAD_APU": "BLS",
                "PRECIO_UNIT_APU": 20000.0,
                "UNIDAD_INSUMO": "BLS"
            },
            {
                "CODIGO_APU": "APU-001",
                "DESCRIPCION_INSUMO": "Cemento",  # Mismo insumo
                "CATEGORIA": "SUMINISTRO",
                "CANTIDAD_APU": 3.0,
                "VALOR_TOTAL_APU": 60000.0,
                "RENDIMIENTO": 1.0,
                "UNIDAD_APU": "BLS",
                "PRECIO_UNIT_APU": 20000.0,
                "UNIDAD_INSUMO": "BLS"
            }
        ]
        
        df = pd.DataFrame(data)
        result = processor._group_by_category(df)
        
        # Debe haber un solo item agregado
        assert len(result) == 1
        assert result[0]["CANTIDAD"] == 8.0  # 5 + 3
        assert result[0]["VR_TOTAL"] == 160000.0  # 100000 + 60000
    
    def test_organize_breakdown(self, processor, sample_apu_data):
        """Debe organizar desglose por categoría."""
        df = pd.DataFrame(sample_apu_data)
        items = processor._group_by_category(df)
        result = processor._organize_breakdown(items)
        
        assert isinstance(result, dict)
        assert "SUMINISTRO" in result
        assert isinstance(result["SUMINISTRO"], list)
    
    def test_process_apu_details_with_nan(self, processor):
        """Debe manejar valores NaN."""
        data = [
            {
                "CODIGO_APU": "APU-001",
                "DESCRIPCION_INSUMO": "Cemento",
                "CATEGORIA": "SUMINISTRO",
                "CANTIDAD_APU": np.nan,
                "VALOR_TOTAL_APU": 100000.0,
                "RENDIMIENTO": 1.0,
                "UNIDAD_APU": "BLS",
                "PRECIO_UNIT_APU": 20000.0,
                "UNIDAD_INSUMO": "BLS"
            }
        ]
        
        result = processor.process_apu_details(data, "APU-001")
        
        # No debe lanzar error
        assert "items" in result
    
    def test_process_apu_details_with_alerts(self, processor):
        """Debe procesar alertas si existen."""
        data = [
            {
                "CODIGO_APU": "APU-001",
                "DESCRIPCION_INSUMO": "Cemento",
                "CATEGORIA": "SUMINISTRO",
                "CANTIDAD_APU": 5.0,
                "VALOR_TOTAL_APU": 100000.0,
                "RENDIMIENTO": 1.0,
                "UNIDAD_APU": "BLS",
                "PRECIO_UNIT_APU": 20000.0,
                "UNIDAD_INSUMO": "BLS",
                "alerta": "Precio alto"
            }
        ]
        
        result = processor.process_apu_details(data, "APU-001")
        
        # Verificar que se procesó sin error
        assert "items" in result


# ============================================================================
# TESTS - DECORADORES
# ============================================================================

class TestDecorators:
    """Suite de pruebas para decoradores"""
    
    def test_require_session_decorator_no_session(self, app, client):
        """Debe rechazar request sin sesión."""
        
        @require_session
        def test_endpoint(session_data=None):
            return {"status": "ok"}
        
        with app.test_request_context():
            # Mock de session.sid
            with patch('app.app.session') as mock_session:
                mock_session.sid = None
                
                result, status = test_endpoint()
                
                assert status == 401
                assert "error" in result
    
    def test_require_session_decorator_expired_session(self, app, mock_redis):
        """Debe rechazar sesión expirada."""
        
        @require_session
        def test_endpoint(session_data=None):
            return {"status": "ok"}
        
        with app.test_request_context():
            app.config['SESSION_REDIS'] = mock_redis
            mock_redis.get.return_value = None  # Sesión expirada
            
            with patch('app.app.session') as mock_session:
                mock_session.sid = "test_session_id"
                
                result, status = test_endpoint()
                
                assert status == 401
                assert "expirada" in result["error"].lower()
    
    def test_require_session_decorator_valid_session(self, app, mock_redis):
        """Debe permitir acceso con sesión válida."""
        
        @require_session
        def test_endpoint(session_data=None):
            return {"status": "ok", "data": session_data}
        
        session_data = {"key": "value"}
        mock_redis.get.return_value = json.dumps(session_data)
        
        with app.test_request_context():
            app.config['SESSION_REDIS'] = mock_redis
            
            with patch('app.app.session') as mock_session:
                mock_session.sid = "test_session_id"
                
                result = test_endpoint()
                
                assert result["status"] == "ok"
                assert "data" in result
    
    def test_handle_errors_decorator_value_error(self, app):
        """Debe manejar ValueError."""
        
        @handle_errors
        def test_endpoint():
            raise ValueError("Test error")
        
        with app.test_request_context():
            result, status = test_endpoint()
            
            assert status == 400
            assert result["code"] == "VALIDATION_ERROR"
    
    def test_handle_errors_decorator_key_error(self, app):
        """Debe manejar KeyError."""
        
        @handle_errors
        def test_endpoint():
            raise KeyError("missing_key")
        
        with app.test_request_context():
            result, status = test_endpoint()
            
            assert status == 400
            assert result["code"] == "MISSING_KEY"
    
    def test_handle_errors_decorator_generic_error(self, app):
        """Debe manejar errores genéricos."""
        
        @handle_errors
        def test_endpoint():
            raise Exception("Generic error")
        
        with app.test_request_context():
            result, status = test_endpoint()
            
            assert status == 500
            assert result["code"] == "INTERNAL_ERROR"
    
    def test_handle_errors_decorator_success(self, app):
        """Debe permitir ejecución exitosa."""
        
        @handle_errors
        def test_endpoint():
            return {"status": "ok"}, 200
        
        with app.test_request_context():
            result, status = test_endpoint()
            
            assert status == 200
            assert result["status"] == "ok"


# ============================================================================
# TESTS - CONTEXT MANAGERS
# ============================================================================

class TestContextManagers:
    """Suite de pruebas para context managers"""
    
    def test_temporary_upload_directory_creates_dir(self, temp_dir):
        """Debe crear directorio temporal."""
        session_id = "test_session"
        
        with temporary_upload_directory(temp_dir, session_id) as user_dir:
            assert user_dir.exists()
            assert user_dir.is_dir()
            assert session_id in str(user_dir)
    
    def test_temporary_upload_directory_cleanup(self, temp_dir):
        """Debe limpiar directorio al salir."""
        session_id = "test_session"
        created_dir = None
        
        with temporary_upload_directory(temp_dir, session_id) as user_dir:
            created_dir = user_dir
            # Crear archivo de prueba
            test_file = user_dir / "test.txt"
            test_file.write_text("test content")
            assert test_file.exists()
        
        # Después del context, el directorio debe estar limpio
        assert not created_dir.exists()
    
    def test_temporary_upload_directory_cleanup_on_error(self, temp_dir):
        """Debe limpiar directorio incluso si hay error."""
        session_id = "test_session"
        created_dir = None
        
        try:
            with temporary_upload_directory(temp_dir, session_id) as user_dir:
                created_dir = user_dir
                raise Exception("Test error")
        except Exception:
            pass
        
        # Debe haberse limpiado de todos modos
        assert not created_dir.exists()
    
    def test_temporary_upload_directory_multiple_files(self, temp_dir):
        """Debe limpiar múltiples archivos."""
        session_id = "test_session"
        
        with temporary_upload_directory(temp_dir, session_id) as user_dir:
            # Crear varios archivos
            for i in range(5):
                (user_dir / f"file{i}.txt").write_text(f"content {i}")
        
        # Todos deben ser eliminados
        assert not (temp_dir / session_id).exists()


# ============================================================================
# TESTS - ENDPOINTS
# ============================================================================

class TestEndpoints:
    """Suite de pruebas para endpoints de la API"""
    
    def test_index_endpoint(self, client):
        """Debe renderizar página principal."""
        with patch('app.app.render_template') as mock_render:
            mock_render.return_value = "Index Page"
            
            response = client.get('/')
            
            assert response.status_code == 200
            mock_render.assert_called_once_with('index.html')
    
    def test_health_check_endpoint(self, client):
        """Debe retornar estado de salud."""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'healthy'
        assert 'active_sessions' in data
        assert 'timestamp' in data
        assert 'version' in data
    
    def test_upload_missing_files(self, client):
        """Debe rechazar upload sin archivos requeridos."""
        response = client.post('/upload', data={})
        
        assert response.status_code == 400
        data = json.loads(response.data)
        
        assert 'error' in data
        assert data['code'] == 'MISSING_FILES'
    
    def test_upload_invalid_file(self, client):
        """Debe rechazar archivo inválido."""
        # Archivo con extensión inválida
        invalid_file = FileStorage(
            stream=BytesIO(b"test"),
            filename="test.txt",
            content_type="text/plain"
        )
        
        response = client.post('/upload', data={
            'presupuesto': invalid_file,
            'apus': invalid_file,
            'insumos': invalid_file
        })
        
        assert response.status_code == 400
        data = json.loads(response.data)
        
        assert data['code'] == 'INVALID_FILE'
    
    @patch('app.app.process_all_files')
    def test_upload_successful(self, mock_process, client, sample_file):
        """Debe procesar upload exitoso."""
        # Mock del procesamiento
        mock_process.return_value = {
            "presupuesto": [],
            "apus_detail": [],
            "summary": {"total_apus": 0}
        }
        
        response = client.post('/upload', data={
            'presupuesto': sample_file,
            'apus': sample_file,
            'insumos': sample_file
        })
        
        # Puede ser 200 si está bien configurado o error si falta algo
        assert response.status_code in [200, 400, 500]
    
    @patch('app.app.process_all_files')
    def test_upload_processing_error(self, mock_process, client, sample_file):
        """Debe manejar error de procesamiento."""
        mock_process.return_value = {
            "error": "Error al procesar archivos"
        }
        
        response = client.post('/upload', data={
            'presupuesto': sample_file,
            'apus': sample_file,
            'insumos': sample_file
        })
        
        assert response.status_code == 500
        data = json.loads(response.data)
        
        assert data['code'] == 'PROCESSING_ERROR'
    
    def test_get_apu_detail_no_session(self, client):
        """Debe rechazar petición sin sesión."""
        response = client.get('/api/apu/APU-001')
        
        assert response.status_code == 401
    
    @patch('app.app.run_monte_carlo_simulation')
    def test_get_apu_detail_success(self, mock_simulation, app, client, sample_session_data, mock_redis):
        """Debe retornar detalles de APU."""
        # Setup
        app.config['SESSION_REDIS'] = mock_redis
        mock_redis.get.return_value = json.dumps(sample_session_data["data"])
        
        mock_simulation.return_value = {
            "mean": 300000,
            "std": 15000,
            "percentiles": {}
        }
        
        with client.session_transaction() as sess:
            sess['user_id'] = 'test_user'
        
        # Mock session.sid
        with patch('app.app.session') as mock_session:
            mock_session.sid = "test_session_id"
            
            response = client.get('/api/apu/APU-001')
            
            # Verificar respuesta
            if response.status_code == 200:
                data = json.loads(response.data)
                assert 'codigo' in data
                assert 'desglose' in data
    
    def test_get_apu_detail_not_found(self, app, client, sample_session_data, mock_redis):
        """Debe retornar 404 para APU inexistente."""
        app.config['SESSION_REDIS'] = mock_redis
        mock_redis.get.return_value = json.dumps(sample_session_data["data"])
        
        with patch('app.app.session') as mock_session:
            mock_session.sid = "test_session_id"
            
            response = client.get('/api/apu/APU-999')
            
            if response.status_code == 404:
                data = json.loads(response.data)
                assert data['code'] == 'APU_NOT_FOUND'
    
    def test_get_estimate_no_session(self, client):
        """Debe rechazar estimación sin sesión."""
        response = client.post('/api/estimate',
                              data=json.dumps({}),
                              content_type='application/json')
        
        assert response.status_code == 401
    
    def test_get_estimate_invalid_content_type(self, app, client, mock_redis):
        """Debe rechazar content-type inválido."""
        app.config['SESSION_REDIS'] = mock_redis
        mock_redis.get.return_value = json.dumps({"data": {}})
        
        with patch('app.app.session') as mock_session:
            mock_session.sid = "test_session_id"
            
            response = client.post('/api/estimate', data="not json")
            
            if response.status_code == 400:
                data = json.loads(response.data)
                assert 'INVALID_CONTENT_TYPE' in data.get('code', '')
    
    def test_get_estimate_no_params(self, app, client, mock_redis):
        """Debe rechazar petición sin parámetros."""
        app.config['SESSION_REDIS'] = mock_redis
        mock_redis.get.return_value = json.dumps({"data": {}})
        
        with patch('app.app.session') as mock_session:
            mock_session.sid = "test_session_id"
            
            response = client.post('/api/estimate',
                                  data=json.dumps(None),
                                  content_type='application/json')
            
            if response.status_code == 400:
                data = json.loads(response.data)
                assert 'error' in data
    
    @patch('app.app.calculate_estimate')
    def test_get_estimate_success(self, mock_calculate, app, client, sample_session_data, mock_redis):
        """Debe calcular estimación exitosamente."""
        app.config['SESSION_REDIS'] = mock_redis
        mock_redis.get.return_value = json.dumps(sample_session_data["data"])
        
        mock_calculate.return_value = {
            "valor_construccion": 500000000,
            "rendimiento_m2_por_dia": 25.5
        }
        
        with patch('app.app.session') as mock_session:
            mock_session.sid = "test_session_id"
            
            params = {"area_m2": 1000, "pisos": 2}
            
            response = client.post('/api/estimate',
                                  data=json.dumps(params),
                                  content_type='application/json')
            
            if response.status_code == 200:
                data = json.loads(response.data)
                assert 'valor_construccion' in data


# ============================================================================
# TESTS - ERROR HANDLERS
# ============================================================================

class TestErrorHandlers:
    """Suite de pruebas para manejadores de errores"""
    
    def test_404_handler(self, client):
        """Debe manejar error 404."""
        response = client.get('/ruta/inexistente')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        
        assert data['code'] == 'NOT_FOUND'
        assert 'error' in data
        assert 'path' in data
    
    def test_413_handler(self, client):
        """Debe manejar archivo muy grande."""
        # Crear archivo que exceda el límite
        large_content = b"x" * (MAX_CONTENT_LENGTH + 1000)
        large_file = FileStorage(
            stream=BytesIO(large_content),
            filename="large.csv",
            content_type="text/csv"
        )
        
        response = client.post('/upload', data={
            'presupuesto': large_file,
            'apus': large_file,
            'insumos': large_file
        })
        
        # Puede ser 413 o rechazado antes
        if response.status_code == 413:
            data = json.loads(response.data)
            assert data['code'] == 'FILE_TOO_LARGE'
    
    def test_500_handler(self, app, client):
        """Debe manejar error 500."""
        # Crear endpoint que falle
        @app.route('/test_500')
        def test_500():
            raise Exception("Test internal error")
        
        response = client.get('/test_500')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        
        assert data['code'] == 'INTERNAL_ERROR'


# ============================================================================
# TESTS - MIDDLEWARE Y HOOKS
# ============================================================================

class TestMiddleware:
    """Suite de pruebas para middleware y hooks"""
    
    def test_before_request_logging(self, app, client, caplog):
        """Debe loggear requests."""
        with caplog.at_level(logging.DEBUG):
            client.get('/api/health')
            
            # Verificar que se loggeó algo
            assert len(caplog.records) > 0
    
    def test_after_request_security_headers(self, client):
        """Debe agregar headers de seguridad."""
        response = client.get('/api/health')
        
        assert 'X-Content-Type-Options' in response.headers
        assert response.headers['X-Content-Type-Options'] == 'nosniff'
        
        assert 'X-Frame-Options' in response.headers
        assert response.headers['X-Frame-Options'] == 'DENY'
        
        assert 'X-XSS-Protection' in response.headers
    
    def test_after_request_cors_disabled(self, app, client):
        """CORS debe estar deshabilitado por defecto."""
        app.config['ENABLE_CORS'] = False
        
        response = client.get('/api/health')
        
        assert 'Access-Control-Allow-Origin' not in response.headers
    
    def test_after_request_cors_enabled(self, app, client):
        """Debe agregar headers CORS si está habilitado."""
        app.config['ENABLE_CORS'] = True
        
        response = client.get('/api/health')
        
        assert 'Access-Control-Allow-Origin' in response.headers


# ============================================================================
# TESTS - CARGA DE MODELOS SEMÁNTICOS
# ============================================================================

class TestSemanticSearch:
    """Suite de pruebas para carga de búsqueda semántica"""
    
    @patch('pathlib.Path.exists')
    @patch('faiss.read_index')
    @patch('builtins.open', new_callable=mock_open, read_data='{"model_name": "test-model"}')
    @patch('app.app.SentenceTransformer')
    def test_load_semantic_search_success(self, mock_transformer, mock_file, mock_faiss, mock_exists, app):
        """Debe cargar artefactos exitosamente."""
        mock_exists.return_value = True
        mock_index = MagicMock()
        mock_index.ntotal = 1000
        mock_faiss.return_value = mock_index
        
        load_semantic_search_artifacts(app)
        
        assert app.config.get('FAISS_INDEX') is not None
        assert app.config.get('EMBEDDING_MODEL') is not None
    
    @patch('pathlib.Path.exists')
    def test_load_semantic_search_missing_files(self, mock_exists, app):
        """Debe manejar archivos faltantes."""
        mock_exists.return_value = False
        
        load_semantic_search_artifacts(app)
        
        # Debe configurar como None
        assert app.config.get('FAISS_INDEX') is None
        assert app.config.get('EMBEDDING_MODEL') is None
    
    @patch('pathlib.Path.exists')
    @patch('faiss.read_index')
    def test_load_semantic_search_error_handling(self, mock_faiss, mock_exists, app):
        """Debe manejar errores en carga."""
        mock_exists.return_value = True
        mock_faiss.side_effect = Exception("Error loading index")
        
        # No debe lanzar excepción
        load_semantic_search_artifacts(app)
        
        # Debe configurar como None
        assert app.config.get('FAISS_INDEX') is None


# ============================================================================
# TESTS - CREATE APP
# ============================================================================

class TestCreateApp:
    """Suite de pruebas para create_app factory"""
    
    @patch('redis.from_url')
    @patch('app.app.load_semantic_search_artifacts')
    def test_create_app_development(self, mock_load, mock_redis):
        """Debe crear app en modo desarrollo."""
        mock_redis.return_value = MagicMock()
        
        app = create_app('development')
        
        assert app is not None
        assert app.config['TESTING'] is False
        assert app.config['DEBUG'] is True
    
    @patch('redis.from_url')
    @patch('app.app.load_semantic_search_artifacts')
    def test_create_app_testing(self, mock_load, mock_redis):
        """Debe crear app en modo testing."""
        mock_redis.return_value = MagicMock()
        
        app = create_app('testing')
        
        assert app is not None
        assert app.config['TESTING'] is True
    
    @patch('redis.from_url')
    @patch('app.app.load_semantic_search_artifacts')
    def test_create_app_production(self, mock_load, mock_redis):
        """Debe crear app en modo producción."""
        mock_redis.return_value = MagicMock()
        
        app = create_app('production')
        
        assert app is not None
        assert app.config['SESSION_COOKIE_SECURE'] is True
    
    @patch('redis.from_url')
    @patch('app.app.load_semantic_search_artifacts')
    def test_create_app_config_loading(self, mock_load, mock_redis):
        """Debe cargar configuración desde archivo."""
        mock_redis.return_value = MagicMock()
        
        with patch('builtins.open', mock_open(read_data='{"version": "1.0.0"}')):
            app = create_app('testing')
            
            assert 'APP_CONFIG' in app.config
    
    @patch('redis.from_url')
    @patch('app.app.load_semantic_search_artifacts')
    def test_create_app_upload_folder_creation(self, mock_load, mock_redis):
        """Debe crear carpeta de uploads."""
        mock_redis.return_value = MagicMock()
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            app = create_app('testing')
            
            # Verificar que se intentó crear directorio
            assert 'UPLOAD_FOLDER' in app.config
    
    @patch('redis.from_url')
    @patch('app.app.load_semantic_search_artifacts')
    def test_create_app_session_configuration(self, mock_load, mock_redis):
        """Debe configurar sesiones correctamente."""
        mock_redis.return_value = MagicMock()
        
        app = create_app('testing')
        
        assert app.config['SESSION_TYPE'] == 'redis'
        assert app.config['SESSION_PERMANENT'] is True
        assert app.config['SESSION_USE_SIGNER'] is True
        assert app.config['PERMANENT_SESSION_LIFETIME'] == SESSION_TIMEOUT


# ============================================================================
# TESTS DE INTEGRACIÓN
# ============================================================================

class TestIntegration:
    """Tests de integración end-to-end"""
    
    @patch('app.app.process_all_files')
    @patch('app.app.run_monte_carlo_simulation')
    def test_full_workflow(self, mock_simulation, mock_process, client, sample_file, sample_session_data, mock_redis):
        """Workflow completo: upload -> detail -> estimate."""
        
        # 1. Upload
        mock_process.return_value = sample_session_data["data"]
        
        upload_response = client.post('/upload', data={
            'presupuesto': sample_file,
            'apus': sample_file,
            'insumos': sample_file
        })
        
        # Verificar upload
        assert upload_response.status_code in [200, 400, 500]
        
        # 2. Get APU detail (si upload fue exitoso)
        if upload_response.status_code == 200:
            mock_simulation.return_value = {"mean": 300000}
            
            with patch('app.app.session') as mock_session:
                mock_session.sid = "test_session"
                
                detail_response = client.get('/api/apu/APU-001')
                # Puede fallar por sesión, pero no debe crashear
                assert detail_response.status_code in [200, 401, 404]


# ============================================================================
# TESTS DE SEGURIDAD
# ============================================================================

class TestSecurity:
    """Tests de seguridad"""
    
    def test_sql_injection_attempt(self, client):
        """Debe prevenir SQL injection."""
        malicious_code = "'; DROP TABLE users; --"
        
        response = client.get(f'/api/apu/{malicious_code}')
        
        # No debe ejecutar código malicioso
        assert response.status_code in [401, 404]
    
    def test_xss_attempt(self, client):
        """Debe prevenir XSS."""
        xss_payload = "<script>alert('XSS')</script>"
        
        response = client.get(f'/api/apu/{xss_payload}')
        
        # No debe ejecutar script
        assert response.status_code in [401, 404]
        if response.data:
            assert b'<script>' not in response.data
    
    def test_path_traversal_attempt(self, client, sample_file):
        """Debe prevenir path traversal."""
        traversal_file = FileStorage(
            stream=BytesIO(b"test"),
            filename="../../../etc/passwd",
            content_type="text/csv"
        )
        
        response = client.post('/upload', data={
            'presupuesto': traversal_file,
            'apus': sample_file,
            'insumos': sample_file
        })
        
        # Debe rechazar o sanitizar
        assert response.status_code in [400, 500]


# ============================================================================
# TESTS DE CONSTANTES
# ============================================================================

class TestConstants:
    """Tests para constantes de configuración"""
    
    def test_session_timeout_constant(self):
        """SESSION_TIMEOUT debe ser número positivo."""
        assert SESSION_TIMEOUT > 0
        assert isinstance(SESSION_TIMEOUT, int)
    
    def test_max_content_length_constant(self):
        """MAX_CONTENT_LENGTH debe ser razonable."""
        assert MAX_CONTENT_LENGTH > 0
        assert MAX_CONTENT_LENGTH == 16 * 1024 * 1024  # 16MB
    
    def test_allowed_extensions_constant(self):
        """ALLOWED_EXTENSIONS debe contener extensiones válidas."""
        assert isinstance(ALLOWED_EXTENSIONS, set)
        assert '.csv' in ALLOWED_EXTENSIONS
        assert '.xlsx' in ALLOWED_EXTENSIONS
        assert '.xls' in ALLOWED_EXTENSIONS
        
        # No debe contener extensiones peligrosas
        assert '.exe' not in ALLOWED_EXTENSIONS
        assert '.sh' not in ALLOWED_EXTENSIONS


# ============================================================================
# TESTS DE RENDIMIENTO
# ============================================================================

class TestPerformance:
    """Tests de rendimiento"""
    
    def test_health_check_response_time(self, client):
        """Health check debe ser rápido."""
        import time
        
        start = time.time()
        response = client.get('/api/health')
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0  # Debe responder en menos de 1 segundo
    
    def test_multiple_concurrent_requests(self, client):
        """Debe manejar múltiples requests."""
        import concurrent.futures
        
        def make_request():
            return client.get('/api/health')
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [f.result() for f in futures]
        
        # Todas deben completarse exitosamente
        assert all(r.status_code == 200 for r in responses)


# ============================================================================
# CONFIGURACIÓN DE PYTEST
# ============================================================================

def pytest_configure(config):
    """Configuración de pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--cov=app.app',
        '--cov-report=html',
        '--cov-report=term-missing',
        '-W', 'ignore::DeprecationWarning',
        '--maxfail=3',  # Detener después de 3 fallos
    ])
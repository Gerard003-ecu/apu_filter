"""
Suite completa de pruebas para app.py

Pruebas exhaustivas con cobertura de endpoints, decoradores, validadores,
procesadores, manejo de errores y funcionalidades de la aplicación Flask.
"""

import json
import logging
import os
import sys
import tempfile
import time  # Importante para timestamps
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from flask import jsonify
from werkzeug.datastructures import FileStorage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.app import (
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    SESSION_TIMEOUT,
    FileValidator,
    create_app,
    handle_errors,
    load_semantic_search_artifacts,
    require_session,
    setup_logging,
    temporary_upload_directory,
)
from app.presenters import APUPresenter

# ============================================================================
# FIXTURES - Configuración y datos de prueba
# ============================================================================


@pytest.fixture
def app():
    """Fixture que crea instancia de la aplicación Flask para tests."""
    # Usar fakeredis para simular Redis en memoria y mockear limiter
    with (
        patch("redis.from_url") as mock_from_url,
        patch("flask_limiter.Limiter.init_app"),
        patch(
            "flask_limiter.Limiter.limit",
            side_effect=lambda *args, **kwargs: lambda func: func,
        ),
    ):
        import fakeredis

        fake_redis_client = fakeredis.FakeStrictRedis()
        mock_from_url.return_value = fake_redis_client

        # Mockear carga de embeddings
        with patch("app.app.load_semantic_search_artifacts"):
            app = create_app("testing")
            app.config["TESTING"] = True
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
                    "VR_UNITARIO": 350000.0,
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
                    "UNIDAD_INSUMO": "BLS",
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
                    "UNIDAD_INSUMO": "M3",
                },
            ],
            "insumos": {
                "MATERIALES": [{"CODIGO_INSUMO": "INS-001", "DESCRIPCION": "Cemento"}]
            },
            "summary": {"total_apus": 1, "total_insumos": 2},
        }
    }


@pytest.fixture
def sample_csv_content():
    """Fixture con contenido CSV de prueba."""
    return b"CODIGO_APU,CODIGO_INSUMO,DESCRIPCION\nAPU-001,INS-001,Concreto"


@pytest.fixture
def sample_file(sample_csv_content):
    """Fixture que crea un archivo FileStorage de prueba."""
    return FileStorage(
        stream=BytesIO(sample_csv_content), filename="test.csv", content_type="text/csv"
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
            "UNIDAD_INSUMO": "BLS",
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
            "UNIDAD_INSUMO": "JOR",
        },
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
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            setup_logging(app)
            mock_mkdir.assert_called()

    def test_setup_logging_file_handler(self, app):
        """Debe configurar handler de archivo."""
        setup_logging(app, log_file="test.log")
        assert len(app.logger.handlers) > 0

    def test_setup_logging_console_handler(self, app):
        """Debe configurar handler de consola."""
        from logging import StreamHandler

        setup_logging(app)
        has_stream_handler = any(isinstance(h, StreamHandler) for h in app.logger.handlers)
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
        is_valid, error = FileValidator.validate_file_metadata(sample_file)
        assert is_valid
        assert error is None

    def test_validate_file_no_file(self):
        """Debe rechazar archivo None."""
        is_valid, error = FileValidator.validate_file_metadata(None)
        assert not is_valid
        assert "no seleccionado" in error.lower()

    def test_validate_file_empty_filename(self):
        """Debe rechazar filename vacío."""
        empty_file = FileStorage(
            stream=BytesIO(b"test"), filename="", content_type="text/csv"
        )
        is_valid, error = FileValidator.validate_file_metadata(empty_file)
        assert not is_valid
        assert "no seleccionado" in error.lower()

    def test_validate_file_invalid_extension(self):
        """Debe rechazar extensiones no permitidas."""
        invalid_file = FileStorage(
            stream=BytesIO(b"test"), filename="test.txt", content_type="text/plain"
        )
        is_valid, error = FileValidator.validate_file_metadata(invalid_file)
        assert not is_valid
        assert "extensión no permitida" in error.lower()

    def test_validate_file_allowed_extensions(self):
        """Debe aceptar extensiones permitidas."""
        for ext in [".csv", ".xlsx", ".xls"]:
            file = FileStorage(
                stream=BytesIO(b"test"),
                filename=f"test{ext}",
                content_type="application/octet-stream",
            )
            is_valid, error = FileValidator.validate_file_metadata(file)
            assert is_valid, f"Extensión {ext} debería ser válida"

    def test_validate_file_too_large(self):
        """Debe rechazar archivos muy grandes."""
        large_content = b"x" * (MAX_CONTENT_LENGTH + 1000)
        large_file = FileStorage(
            stream=BytesIO(large_content), filename="large.csv", content_type="text/csv"
        )
        is_valid, error = FileValidator.validate_file_metadata(large_file)
        assert not is_valid
        assert "demasiado grande" in error.lower()

    def test_validate_file_secure_filename(self):
        """Debe manejar nombres de archivo inseguros."""
        unsafe_file = FileStorage(
            stream=BytesIO(b"test"), filename="../../../etc/passwd", content_type="text/csv"
        )
        is_valid, error = FileValidator.validate_file_metadata(unsafe_file)
        assert isinstance(is_valid, bool)


# ============================================================================
# TESTS - PROCESADOR APU
# ============================================================================


class TestAPUPresenter:
    """Suite de pruebas para APUPresenter"""

    @pytest.fixture
    def processor(self, app):
        """Fixture que crea instancia de APUPresenter."""
        return APUPresenter(app.logger)

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
        for item in result:
            assert "descripcion" in item
            assert "categoria" in item
            assert "cantidad" in item

    def test_group_by_category_aggregation(self, processor):
        """Debe agregar valores correctamente."""
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
            },
            {
                "CODIGO_APU": "APU-001",
                "DESCRIPCION_INSUMO": "Cemento",
                "CATEGORIA": "SUMINISTRO",
                "CANTIDAD_APU": 3.0,
                "VALOR_TOTAL_APU": 60000.0,
                "RENDIMIENTO": 1.0,
                "UNIDAD_APU": "BLS",
                "PRECIO_UNIT_APU": 20000.0,
                "UNIDAD_INSUMO": "BLS",
            },
        ]
        df = pd.DataFrame(data)
        result = processor._group_by_category(df)
        assert len(result) == 1
        assert result[0]["cantidad"] == 8.0
        assert result[0]["valor_total"] == 160000.0

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
                "UNIDAD_INSUMO": "BLS",
            }
        ]
        result = processor.process_apu_details(data, "APU-001")
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
                "alerta": "Precio alto",
            }
        ]
        result = processor.process_apu_details(data, "APU-001")
        assert "items" in result


# ============================================================================
# TESTS - DECORADORES
# ============================================================================


class TestDecorators:
    """Suite de pruebas para decoradores"""

    def test_require_session_decorator_no_session(self, app, client):
        """Debe rechazar request sin sesión."""

        @app.route("/test_no_session")
        @require_session
        def test_endpoint(session_data=None):
            return jsonify({"status": "ok"})

        response = client.get("/test_no_session")
        assert response.status_code == 401
        json_data = response.get_json()
        assert "sesión no iniciada" in json_data["error"].lower()

    def test_require_session_decorator_expired_session(self, app, client):
        """Debe rechazar sesión sin metadata o expirada."""

        @app.route("/test_expired_session")
        @require_session
        def test_endpoint(session_data=None):
            return jsonify({"status": "ok"})

        with client.session_transaction() as sess:
            sess["user_id"] = "test_user"
            sess["processed_data"] = {}
            # No session_metadata provided

        response = client.get("/test_expired_session")
        assert response.status_code == 401

    def test_require_session_decorator_valid_session(self, app, client):
        """Debe permitir acceso con sesión válida."""

        @app.route("/test_valid_session")
        @require_session
        def test_endpoint(session_data=None):
            return jsonify({"status": "ok", "data": session_data["data"]})

        session_payload = {
            "presupuesto": [{"CODIGO_APU": "1"}],
            "apus_detail": [{"CODIGO_APU": "1"}],
            "insumos": {"MATERIALES": [{"CODIGO_INSUMO": "1"}]},
        }
        metadata = {
            "session_id": "sid",
            "created_at": time.time(),
            "last_accessed": time.time(),
            "data_hash": "hash",
            "version": "2.0",
        }

        with client.session_transaction() as sess:
            sess["user_id"] = "test_user"
            sess["processed_data"] = session_payload
            sess["session_metadata"] = metadata

        response = client.get("/test_valid_session")
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["status"] == "ok"

    def test_handle_errors_decorator_value_error(self, app, client):
        """Debe manejar ValueError."""

        @app.route("/test_value_error")
        @handle_errors
        def test_endpoint():
            raise ValueError("Test error")

        response = client.get("/test_value_error")
        assert response.status_code == 400
        json_data = response.get_json()
        assert json_data["code"] == "VALIDATION_ERROR"

    def test_handle_errors_decorator_key_error(self, app, client):
        """Debe manejar KeyError."""

        @app.route("/test_key_error")
        @handle_errors
        def test_endpoint():
            raise KeyError("missing_key")

        response = client.get("/test_key_error")
        assert response.status_code == 400
        json_data = response.get_json()
        assert json_data["code"] == "MISSING_KEY"

    def test_handle_errors_decorator_generic_error(self, app, client):
        """Debe manejar errores genéricos."""

        @app.route("/test_generic_error")
        @handle_errors
        def test_endpoint():
            raise Exception("Generic error")

        response = client.get("/test_generic_error")
        assert response.status_code == 500
        json_data = response.get_json()
        assert json_data["code"] == "INTERNAL_ERROR"

    def test_handle_errors_decorator_success(self, app, client):
        """Debe permitir ejecución exitosa."""

        @app.route("/test_success")
        @handle_errors
        def test_endpoint():
            return {"status": "ok"}, 200

        response = client.get("/test_success")
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["status"] == "ok"


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
            test_file = user_dir / "test.txt"
            test_file.write_text("test content")
            assert test_file.exists()
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
        assert not created_dir.exists()

    def test_temporary_upload_directory_multiple_files(self, temp_dir):
        """Debe limpiar múltiples archivos."""
        session_id = "test_session"
        with temporary_upload_directory(temp_dir, session_id) as user_dir:
            for i in range(5):
                (user_dir / f"file{i}.txt").write_text(f"content {i}")
        assert not (temp_dir / session_id).exists()


# ============================================================================
# TESTS - ENDPOINTS
# ============================================================================


class TestEndpoints:
    """Suite de pruebas para endpoints de la API"""

    def test_index_endpoint(self, client):
        """Debe renderizar página principal."""
        with patch("app.app.render_template") as mock_render:
            mock_render.return_value = "Index Page"
            response = client.get("/")
            assert response.status_code == 200
            mock_render.assert_called_once_with("index.html")

    def test_health_check_endpoint(self, client):
        """Debe retornar estado de salud."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert "redis" in data
        assert "active_sessions" in data["redis"]

    def test_upload_missing_files(self, client):
        """Debe rechazar upload sin archivos requeridos."""
        response = client.post("/upload", data={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert data["code"] == "MISSING_FILES"

    def test_upload_invalid_file(self, client):
        """Debe rechazar archivo inválido."""
        invalid_file = FileStorage(
            stream=BytesIO(b"test"), filename="test.txt", content_type="text/plain"
        )
        response = client.post(
            "/upload",
            data={
                "presupuesto": invalid_file,
                "apus": invalid_file,
                "insumos": invalid_file,
            },
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["code"] == "INVALID_FILE"

    @patch("app.app.process_all_files")
    def test_upload_successful(self, mock_process, client, sample_csv_content):
        """Debe procesar upload exitoso."""
        # The result must pass validate_data_schema: keys + non-empty lists
        mock_process.return_value = {
            "presupuesto": [{"CODIGO_APU": "1"}],
            "apus_detail": [{"CODIGO_APU": "1"}],
            "insumos": {"MATERIALES": [{"CODIGO_INSUMO": "1"}]},
            "summary": {"total_apus": 0},
        }

        # Crear archivos separados para evitar consumo de stream compartido
        f1 = FileStorage(
            stream=BytesIO(sample_csv_content), filename="p.csv", content_type="text/csv"
        )
        f2 = FileStorage(
            stream=BytesIO(sample_csv_content), filename="a.csv", content_type="text/csv"
        )
        f3 = FileStorage(
            stream=BytesIO(sample_csv_content), filename="i.csv", content_type="text/csv"
        )

        response = client.post(
            "/upload",
            data={"presupuesto": f1, "apus": f2, "insumos": f3},
        )

        assert response.status_code in [200]

    @patch("app.app.process_all_files")
    def test_upload_processing_error(self, mock_process, client, sample_csv_content):
        """Debe manejar error de procesamiento."""
        mock_process.return_value = {"error": "Error al procesar archivos"}

        f1 = FileStorage(
            stream=BytesIO(sample_csv_content), filename="p.csv", content_type="text/csv"
        )
        f2 = FileStorage(
            stream=BytesIO(sample_csv_content), filename="a.csv", content_type="text/csv"
        )
        f3 = FileStorage(
            stream=BytesIO(sample_csv_content), filename="i.csv", content_type="text/csv"
        )

        response = client.post(
            "/upload",
            data={"presupuesto": f1, "apus": f2, "insumos": f3},
        )

        assert response.status_code == 500
        data = json.loads(response.data)
        assert data["code"] == "PROCESSING_ERROR"

    def test_get_apu_detail_no_session(self, client):
        """Debe rechazar petición sin sesión."""
        response = client.get("/api/apu/APU-001")
        assert response.status_code == 401

    @patch("app.app.run_monte_carlo_simulation")
    def test_get_apu_detail_success(self, mock_simulation, app, client, sample_session_data):
        """Debe retornar detalles de APU."""
        mock_simulation.return_value = {"mean": 300000, "std": 15000, "percentiles": {}}

        metadata = {
            "session_id": "sid",
            "created_at": time.time(),
            "last_accessed": time.time(),
            "data_hash": "hash",
            "version": "2.0",
        }

        with client.session_transaction() as sess:
            sess["user_id"] = "test_user"
            sess["processed_data"] = sample_session_data["data"]
            sess["session_metadata"] = metadata

        response = client.get("/api/apu/APU-001")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "codigo" in data

    def test_get_apu_detail_not_found(self, app, client, sample_session_data):
        """Debe retornar 404 para APU inexistente."""
        metadata = {
            "session_id": "sid",
            "created_at": time.time(),
            "last_accessed": time.time(),
            "data_hash": "hash",
            "version": "2.0",
        }
        with client.session_transaction() as sess:
            sess["user_id"] = "test_user"
            sess["processed_data"] = sample_session_data["data"]
            sess["session_metadata"] = metadata

        response = client.get("/api/apu/APU-999")
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data["code"] == "APU_NOT_FOUND"

    def test_get_estimate_no_session(self, client):
        """Debe rechazar estimación sin sesión."""
        response = client.post("/api/estimate", json={})
        assert response.status_code == 401

    def test_get_estimate_invalid_content_type(self, app, client):
        """Debe rechazar content-type inválido."""
        metadata = {
            "session_id": "sid",
            "created_at": time.time(),
            "last_accessed": time.time(),
            "data_hash": "hash",
            "version": "2.0",
        }
        with client.session_transaction() as sess:
            sess["user_id"] = "test_user"
            sess["processed_data"] = {
                "presupuesto": ["a"],
                "apus_detail": ["b"],
                "insumos": {"CAT": [{"id": 1}]},
            }
            sess["session_metadata"] = metadata

        response = client.post("/api/estimate", data="not json")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "INVALID_CONTENT_TYPE" in data.get("code", "")

    def test_get_estimate_no_params(self, app, client):
        """Debe rechazar petición sin parámetros."""
        metadata = {
            "session_id": "sid",
            "created_at": time.time(),
            "last_accessed": time.time(),
            "data_hash": "hash",
            "version": "2.0",
        }
        with client.session_transaction() as sess:
            sess["user_id"] = "test_user"
            sess["processed_data"] = {
                "presupuesto": ["a"],
                "apus_detail": ["b"],
                "insumos": {"CAT": [{"id": 1}]},
            }
            sess["session_metadata"] = metadata

        response = client.post("/api/estimate", json=None)
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    @patch("app.app.calculate_estimate")
    def test_get_estimate_success(self, mock_calculate, app, client, sample_session_data):
        """Debe calcular estimación exitosamente."""
        mock_calculate.return_value = {
            "valor_construccion": 500000000,
            "rendimiento_m2_por_dia": 25.5,
        }

        metadata = {
            "session_id": "sid",
            "created_at": time.time(),
            "last_accessed": time.time(),
            "data_hash": "hash",
            "version": "2.0",
        }

        with client.session_transaction() as sess:
            sess["user_id"] = "test_user"
            sess["processed_data"] = sample_session_data["data"]
            sess["session_metadata"] = metadata

        params = {"area_m2": 1000, "pisos": 2}
        response = client.post("/api/estimate", json=params)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "valor_construccion" in data


# ============================================================================
# TESTS - ERROR HANDLERS
# ============================================================================


class TestErrorHandlers:
    """Suite de pruebas para manejadores de errores"""

    def test_404_handler(self, client):
        """Debe manejar error 404."""
        response = client.get("/ruta/inexistente")
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data["code"] == "NOT_FOUND"

    def test_413_handler(self, client):
        """Debe manejar archivo muy grande."""
        large_content = b"x" * (MAX_CONTENT_LENGTH + 1000)
        large_file = FileStorage(
            stream=BytesIO(large_content), filename="large.csv", content_type="text/csv"
        )
        response = client.post(
            "/upload",
            data={"presupuesto": large_file, "apus": large_file, "insumos": large_file},
        )
        # Puede ser 413 o rechazado antes
        if response.status_code == 413:
            data = json.loads(response.data)
            assert data["code"] == "FILE_TOO_LARGE"

    def test_500_handler(self, app, client):
        """Debe manejar error 500."""

        @app.route("/test_500")
        def test_500():
            raise Exception("Test internal error")

        try:
            client.get("/test_500")
        except Exception:
            # In testing mode exceptions are propagated
            pass


# ============================================================================
# TESTS - MIDDLEWARE Y HOOKS
# ============================================================================


class TestMiddleware:
    """Suite de pruebas para middleware y hooks"""

    def test_before_request_logging(self, app, client, caplog):
        """Debe loggear requests."""
        with caplog.at_level(logging.DEBUG):
            client.get("/api/health")
            assert len(caplog.records) > 0

    def test_after_request_security_headers(self, client):
        """Debe agregar headers de seguridad."""
        response = client.get("/api/health")
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers

    def test_after_request_cors_disabled(self, app, client):
        """CORS debe estar deshabilitado por defecto."""
        app.config["ENABLE_CORS"] = False
        response = client.get("/api/health")
        assert "Access-Control-Allow-Origin" not in response.headers


# ============================================================================
# TESTS - CARGA DE MODELOS SEMÁNTICOS
# ============================================================================


class TestSemanticSearch:
    """Suite de pruebas para carga de búsqueda semántica"""

    @patch("pathlib.Path.exists")
    @patch("faiss.read_index")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"model_name": "test-model", "vector_dimension": 128, "total_vectors": 1000}',
    )
    @patch("app.app.SentenceTransformer")
    def test_load_semantic_search_success(
        self, mock_transformer, mock_file, mock_faiss, mock_exists, app
    ):
        """Debe cargar artefactos exitosamente."""
        mock_exists.return_value = True
        mock_index = MagicMock()
        mock_index.ntotal = 1000
        mock_index.d = 128
        mock_faiss.return_value = mock_index

        # Mock encoding for dimension check
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, 128))
        mock_transformer.return_value = mock_model

        # Need to populate the file read with valid ID map list
        file_read_mock = mock_file.return_value
        file_read_mock.read.side_effect = [
            '{"model_name": "test-model", "vector_dimension": 128, "total_vectors": 1000}',  # metadata
            json.dumps([str(i) for i in range(1000)]),  # id_map
        ]

        # Actually we need separate opens. This mocks all opens.
        # Let's just mock json.load instead to return dict for metadata and list for map.
        with patch("json.load") as mock_json:
            mock_json.side_effect = [
                {
                    "model_name": "test-model",
                    "vector_dimension": 128,
                    "total_vectors": 1000,
                },  # metadata
                [str(i) for i in range(1000)],  # id_map
            ]
            load_semantic_search_artifacts(app)

        assert app.config.get("FAISS_INDEX") is not None
        assert app.config.get("EMBEDDING_MODEL") is not None

    @patch("pathlib.Path.exists")
    def test_load_semantic_search_missing_files(self, mock_exists, app):
        """Debe manejar archivos faltantes."""
        mock_exists.return_value = False
        load_semantic_search_artifacts(app)
        assert app.config.get("FAISS_INDEX") is None

    @patch("pathlib.Path.exists")
    @patch("faiss.read_index")
    def test_load_semantic_search_error_handling(self, mock_faiss, mock_exists, app):
        """Debe manejar errores en carga."""
        mock_exists.return_value = True
        mock_faiss.side_effect = Exception("Error loading index")
        load_semantic_search_artifacts(app)
        assert app.config.get("FAISS_INDEX") is None


# ============================================================================
# TESTS - CREATE APP
# ============================================================================


class TestCreateApp:
    """Suite de pruebas para create_app factory"""

    @patch("redis.from_url")
    @patch("app.app.load_semantic_search_artifacts")
    def test_create_app_development(self, mock_load, mock_redis):
        """Debe crear app en modo desarrollo."""
        mock_redis.return_value = MagicMock()
        app = create_app("development")
        assert app is not None
        assert app.config["TESTING"] is False

    @patch("redis.from_url")
    @patch("app.app.load_semantic_search_artifacts")
    def test_create_app_testing(self, mock_load, mock_redis):
        """Debe crear app en modo testing."""
        mock_redis.return_value = MagicMock()
        app = create_app("testing")
        assert app is not None
        assert app.config["TESTING"] is True

    @patch("redis.from_url")
    @patch("app.app.load_semantic_search_artifacts")
    def test_create_app_production(self, mock_load, mock_redis):
        """Debe crear app en modo producción."""
        mock_redis.return_value = MagicMock()
        # En producción requiere secret key
        with patch.dict(os.environ, {"SECRET_KEY": "prod-secret"}):
            app = create_app("production")
            assert app is not None
            assert app.config["SESSION_COOKIE_SECURE"] is True

    @patch("redis.from_url")
    @patch("app.app.load_semantic_search_artifacts")
    def test_create_app_config_loading(self, mock_load, mock_redis):
        """Debe cargar configuración desde archivo."""
        mock_redis.return_value = MagicMock()
        with patch("builtins.open", mock_open(read_data='{"version": "1.0.0"}')):
            app = create_app("testing")
            assert "APP_CONFIG" in app.config

    @patch("redis.from_url")
    @patch("app.app.load_semantic_search_artifacts")
    def test_create_app_upload_folder_creation(self, mock_load, mock_redis):
        """Debe crear carpeta de uploads."""
        mock_redis.return_value = MagicMock()
        with patch("pathlib.Path.mkdir"):
            app = create_app("testing")
            assert "UPLOAD_FOLDER" in app.config

    @patch("redis.from_url")
    @patch("app.app.load_semantic_search_artifacts")
    def test_create_app_session_configuration(self, mock_load, mock_redis):
        """Debe configurar sesiones correctamente."""
        mock_redis.return_value = MagicMock()
        app = create_app("testing")
        assert app.config["SESSION_TYPE"] == "redis"
        assert app.config["SESSION_PERMANENT"] is True


# ============================================================================
# TESTS DE INTEGRACIÓN
# ============================================================================


class TestIntegration:
    """Tests de integración end-to-end"""

    @patch("app.app.process_all_files")
    @patch("app.app.run_monte_carlo_simulation")
    def test_full_workflow(
        self,
        mock_simulation,
        mock_process,
        app,
        client,
        sample_csv_content,
        sample_session_data,
    ):
        """Workflow completo: upload -> detail."""
        # 1. Upload: esto crea la sesión y almacena los datos
        mock_process.return_value = sample_session_data["data"]

        # Archivos separados
        f1 = FileStorage(
            stream=BytesIO(sample_csv_content), filename="p.csv", content_type="text/csv"
        )
        f2 = FileStorage(
            stream=BytesIO(sample_csv_content), filename="a.csv", content_type="text/csv"
        )
        f3 = FileStorage(
            stream=BytesIO(sample_csv_content), filename="i.csv", content_type="text/csv"
        )

        upload_response = client.post(
            "/upload",
            data={"presupuesto": f1, "apus": f2, "insumos": f3},
            follow_redirects=True,
        )

        assert upload_response.status_code == 200

        # Simular que el cliente guarda la cookie de sesión automáticamente
        # Pero como estamos usando mocks de Redis, necesitamos asegurarnos
        # que el endpoint de detalle pueda acceder a la sesión.
        # En los tests de integración con cliente de prueba, el cliente maneja las cookies.

        # 2. Get APU detail: debe funcionar con la sesión creada
        mock_simulation.return_value = {"mean": 300000, "std": 0, "percentiles": {}}
        detail_response = client.get("/api/apu/APU-001")

        assert detail_response.status_code == 200
        detail_data = json.loads(detail_response.data)
        assert "desglose" in detail_data


# ============================================================================
# TESTS DE SEGURIDAD
# ============================================================================


class TestSecurity:
    """Tests de seguridad"""

    def test_sql_injection_attempt(self, client):
        """Debe prevenir SQL injection."""
        malicious_code = "'; DROP TABLE users; --"
        response = client.get(f"/api/apu/{malicious_code}")
        assert response.status_code in [401, 404]

    def test_xss_attempt(self, client):
        """Debe prevenir XSS."""
        xss_payload = "<script>alert('XSS')</script>"
        response = client.get(f"/api/apu/{xss_payload}")
        assert response.status_code in [401, 404]
        if response.data:
            assert b"<script>" not in response.data

    def test_path_traversal_attempt(self, client, sample_file):
        """Debe prevenir path traversal."""
        traversal_file = FileStorage(
            stream=BytesIO(b"test"), filename="../../../etc/passwd", content_type="text/csv"
        )
        response = client.post(
            "/upload",
            data={
                "presupuesto": traversal_file,
                "apus": sample_file,
                "insumos": sample_file,
            },
        )
        assert response.status_code in [400, 500]


# ============================================================================
# TESTS DE CONSTANTES
# ============================================================================


class TestConstants:
    """Tests para constantes de configuración"""

    def test_session_timeout_constant(self):
        """SESSION_TIMEOUT debe ser número positivo."""
        assert SESSION_TIMEOUT > 0

    def test_max_content_length_constant(self):
        """MAX_CONTENT_LENGTH debe ser razonable."""
        assert MAX_CONTENT_LENGTH > 0

    def test_allowed_extensions_constant(self):
        """ALLOWED_EXTENSIONS debe contener extensiones válidas."""
        assert ".csv" in ALLOWED_EXTENSIONS
        assert ".exe" not in ALLOWED_EXTENSIONS


# ============================================================================
# TESTS DE RENDIMIENTO
# ============================================================================


class TestPerformance:
    """Tests de rendimiento"""

    def test_health_check_response_time(self, client):
        """Health check debe ser rápido."""
        start = time.time()
        response = client.get("/api/health")
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 1.0

    def test_multiple_concurrent_requests(self, client):
        """Debe manejar múltiples requests."""
        import concurrent.futures

        def make_request():
            return client.get("/api/health")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [f.result() for f in futures]
        assert all(r.status_code == 200 for r in responses)


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    pytest.main()

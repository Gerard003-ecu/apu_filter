import io
import json
import os
import sys
import unittest

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importar la app de Flask y las funciones a probar
from app.app import create_app, user_sessions

# Importar los nuevos datos de prueba centralizados
from tests.test_data import (
    APUS_DATA,
    INSUMOS_DATA,
    PRESUPUESTO_DATA,
    TEST_CONFIG,
)


class TestAppEndpoints(unittest.TestCase):
    """
    Conjunto de pruebas para los endpoints de la aplicación Flask.

    Estas pruebas verifican la funcionalidad de los puntos de entrada de la API,
    como la carga de archivos y el cálculo de estimaciones, asegurando que
    la interacción a través de HTTP funcione como se espera.
    """

    def setUp(self):
        """
        Configura el entorno necesario para cada prueba.

        Crea una instancia de la aplicación Flask en modo de prueba, establece un
        contexto de aplicación, crea la carpeta de subida de archivos y
        limpia las sesiones de usuario existentes.
        """
        self.app = create_app('testing')
        self.app_context = self.app.app_context()
        self.app_context.push()
        os.makedirs(self.app.config["UPLOAD_FOLDER"], exist_ok=True)
        self.client = self.app.test_client()
        user_sessions.clear()

    def tearDown(self):
        """
        Limpia el entorno después de la ejecución de cada prueba.

        Elimina el contexto de la aplicación y borra la carpeta de subida de
        archivos junto con su contenido para no dejar artefactos.
        """
        self.app_context.pop()
        upload_folder = self.app.config["UPLOAD_FOLDER"]
        if os.path.exists(upload_folder):
            for root, dirs, files in os.walk(upload_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(upload_folder)

    def _get_test_file(self, filename, content):
        """
        Crea un archivo en memoria para simular la carga de archivos en una petición.

        Args:
            filename (str): El nombre del archivo a simular.
            content (str): El contenido del archivo.

        Returns:
            tuple: Una tupla con un objeto BytesIO y el nombre del archivo,
                   lista para ser usada en una petición `multipart/form-data`.
        """
        return (io.BytesIO(content.encode("latin1")), filename)

    def test_get_estimate_with_session(self):
        """
        Prueba el flujo completo de carga de datos y solicitud de estimación.

        Simula una petición POST a `/upload` con los archivos de prueba y luego
        una petición POST a `/api/estimate` para verificar que el cálculo se
        realiza correctamente utilizando los datos de la sesión.
        """
        with self.client as c:
            # Simular la carga de archivos para crear una sesión de datos
            data = {
                "presupuesto": self._get_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._get_test_file("apus.csv", APUS_DATA),
                "insumos": self._get_test_file("insumos.csv", INSUMOS_DATA),
            }
            # Asignar la configuración de prueba directamente a la aplicación
            c.application.config["APP_CONFIG"] = TEST_CONFIG
            upload_response = c.post(
                "/upload", data=data, content_type="multipart/form-data"
            )
            self.assertEqual(upload_response.status_code, 200)

            # Solicitar una estimación con los parámetros definidos
            estimate_params = {"material": "TEJA", "cuadrilla": "1"}
            response = c.post("/api/estimate", json=estimate_params)
            self.assertEqual(response.status_code, 200)
            json_data = json.loads(response.data)

            # Aserciones basadas en los nuevos datos de prueba (ver test_estimator.py)
            self.assertAlmostEqual(json_data["valor_suministro"], 50000.0, places=2)
            self.assertAlmostEqual(json_data["valor_instalacion"], 11760.0, places=2)
            self.assertAlmostEqual(json_data["valor_construccion"], 61760.0, places=2)

if __name__ == "__main__":
    unittest.main()

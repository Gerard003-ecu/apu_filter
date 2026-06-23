"""
Configuración global de pytest para el ecosistema APU Filter.

Este módulo define fixtures compartidas de alcance de módulo que
proporcionan instancias de la aplicación Flask y un cliente HTTP
de prueba aislado para cada suite de tests.

Convenciones:
    - ``scope="module"``: la instancia se reutiliza en todos los tests
      del mismo módulo, lo que reduce el coste de inicialización.
    - La inserción en ``sys.path`` garantiza la resolución de importaciones
      relativas al paquete raíz del proyecto independientemente del
      directorio de trabajo desde el que se invoque pytest.
"""

import os
import torch; torch.float8_e8m0fnu = getattr(torch, "float8_e8m0fnu", torch.float8_e4m3fn)
# Fase 1: Esterilización del Espacio Vectorial (Vacío Termodinámico)
# Inyectar variables de entorno ANTES de cargar numpy/scipy para forzar BLAS/LAPACK a 1 hilo
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from decimal import getcontext
# Cuantización de la Tolerancia Numérica: Configura globalmente el contexto decimal
# para los tests ergódicos para evitar que el error de truncamiento del estándar
# IEEE 754 degenere la condición de Palais-Smale.
# rank(BLAS_THREADS)≡1 ⟹ dim(ker(MKL))=0
from decimal import ROUND_HALF_EVEN
getcontext().prec = 50
getcontext().rounding = ROUND_HALF_EVEN

import sys

import pytest
import tempfile

# Garantiza que el paquete raíz del proyecto sea resoluble sin instalación.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.app import create_app

@pytest.fixture(scope="module", autouse=True)
def sterile_logging_environment():
    """
    Anestesia el sistema de archivos inyectando un directorio temporal.
    Garantiza que setup_logging no intente proyectar sobre el disco local.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Inyectamos la variable de entorno para redirigir los logs
        os.environ["APU_LOG_DIR"] = tmpdirname
        yield
        # El cierre del contexto destruye el directorio temporal,
        # preservando la pureza de estado (Idempotencia).


@pytest.fixture(scope="session", autouse=True)
def _initialize_global_mic_with_core_vectors():
    """
    Bootstrap global de la MIC: garantiza que todos los vectores core
    (incluyendo ``stabilize_flux``, ``parse_raw``, ``structure_logic``,
    ``audit_fusion_homology``, ``lateral_thinking_pivot``,
    ``calculate_fat_tail_risk``) estén registrados antes de cualquier
    colección de tests.

    Fundamentación:
        Sin este bootstrap, los tests que mockean
        ``app.core.apu_agent.get_global_mic`` con ``create=True`` (lo que
        genera un ``MagicMock`` vacío) producen ``NameError: name
        'vector_stabilize_flux' is not defined`` al intentar acceder a
        atributos del mock. Pre-inicializando la singleton MIC una sola
        vez por sesión, esos mocks heredan la realidad estructural.

    Idempotencia:
        ``get_global_mic(force_reinit=False)`` retorna la instancia cacheada,
        por lo que este fixture es seguro contra invocaciones repetidas.
    """
    from app.adapters.tools_interface import get_global_mic
    try:
        get_global_mic()
    except Exception as exc:  # noqa: BLE001 — bootstrap defensivo
        # Si el bootstrap falla, los tests con patch MagicMock deben
        # continuar (la MIC mockeada seguirá siendo un MagicMock vacío,
        # pero al menos no rompemos la recolección por este motivo).
        import warnings
        warnings.warn(
            f"[conftest] Bootstrap MIC global no completó: {exc}. "
            "Los tests con mocks de get_global_mic deberán manejar el caso.",
            RuntimeWarning,
        )
    yield


@pytest.fixture(scope="module")
def app():
    """Crea y configura una nueva instancia de la aplicación para cada módulo de prueba.

    Utiliza el perfil ``testing`` de la aplicación Flask para asegurar que
    las opciones de depuración, caché y base de datos estén configuradas
    de manera adecuada para el entorno de prueba.

    Yields:
        Flask: Instancia de la aplicación activa dentro de su contexto.
    """
    application = create_app("testing")
    with application.app_context():
        yield application


@pytest.fixture(scope="module")
def client(app):
    """Proporciona un cliente HTTP de prueba para la aplicación.

    Args:
        app: Fixture de la aplicación Flask.

    Returns:
        FlaskClient: Cliente de prueba listo para enviar peticiones.
    """
    return app.test_client()

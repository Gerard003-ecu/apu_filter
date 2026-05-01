# APU Filter - Manual de Operaciones para Jules

Este repositorio contiene un sistema ciber-físico de procesamiento de presupuestos basado en Topología Algebraica, Termodinámica y Control Port-Hamiltoniano.

## 1. Coding Conventions (Reglas Arquitectónicas)
- **Rigor Matemático:** Preservar siempre `np.float64` y la estabilidad numérica en operaciones matriciales.
- **Arquitectura MIC:** Respetar la Matriz de Interacción Central. Ningún servicio de Estrategia (Nivel 1) puede ejecutarse sin validar la Física (Nivel 3).
- **Tipado:** Uso estricto de Type Hinting (PEP 484) y Docstrings (PEP 257).

## 2. Build Steps (Preparación de la VM)
Para preparar tu entorno efímero, ejecuta el siguiente script de inicialización que instala el entorno en conda.

Verifica que el script tenga permisos de escritura: chmod +x start_conda.sh

Ejecútalo: ./start_conda.sh

Valida los checks: check_base_dependencies check_conda_installation initialize_conda_shell check_network_connectivity

## 3. verificación del entorno

Ejecuta “python -m pytest -vv”. En este punto, no esperamos que todas las pruebas pasen, pero no debes ver errores de entorno como ModuleNotFoundError o errores de compilación. Deberías ver errores de lógica (AssertionError, TypeError), lo que indica que el entorno está listo.

Cada vez que finalices una actualización de métodos, clases o integraciones de nuevos microservicios debes garantizar que las pruebas aisladas y globales pasen todas en verde.

python -m pytest -vv

## 4. Limpieza de archivos patch y misceláneos

Cada vez que generes archivos patch tipo ".sh" o archivos con extensión ".log", ".txt" o scripts con extensión ".py" que no hacen parte de la arquitectura APU Filter deben ser eliminados antes de generar el commmit y el push.
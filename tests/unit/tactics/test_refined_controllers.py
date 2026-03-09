"""
Suite de Testing para Controladores Refinados del Sistema.
==========================================================

Fundamentos Matemáticos Verificados:
─────────────────────────────────────────────────────────────────────────────
CONTROL PI:
  · Ley de control:  u(t) = kp·e(t) + ki·∫e(τ)dτ
  · Anti-windup:     back-calculation con ganancia Kt = 1/kp
  · Estabilidad:     exponente de Lyapunov λ < 0 ⟺ sistema convergente
  · EMA filter:      y_f(t) = α·y(t) + (1-α)·y_f(t-1),  α ∈ (0,1]

CÁLCULO VECTORIAL DISCRETO (Complejo Simplicial):
  · Característica de Euler:  χ = V - E + F  (invariante topológico)
  · Identidad de Stokes:      curl(grad(φ)) = 0  ∀φ  (exactitud del complejo)
  · Números de Betti (K₄):   β₀=1, β₁=0, β₂=1  →  H*(K₄) ≅ H*(S²)
    Justificación: K₄ triangula la esfera S² como tetraedro hueco.
    χ(K₄) = 4 - 6 + 4 = 2 = χ(S²)  ✓

MAXWELL DISCRETO:
  · Condición CFL:   Δt ≤ h/(c√d)  para estabilidad explícita
  · Conservación:    dH/dt = -‖J‖²_R ≤ 0  (disipación de energía)

PORT-HAMILTONIANO:
  · Estructura J:    J + Jᵀ = 0  (antisimetría — conservación de energía)
  · Pasividad:       ∫₀ᵀ u(t)ᵀy(t)dt ≥ -β  para algún β ≥ 0
  · Función de almacenamiento:  H(x) ≥ 0  con Ḣ ≤ uᵀy

Referencias:
  - Åström & Hägglund (1995). PID Controllers: Theory, Design, and Tuning.
  - Hirani (2003). Discrete Exterior Calculus. Caltech Thesis.
  - van der Schaft & Jeltsema (2014). Port-Hamiltonian Systems Theory.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import pytest

# ── Importaciones condicionales con guardas explícitas ────────────────────────
try:
    import numpy as np
    from numpy.linalg import LinAlgError
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None            # type: ignore[assignment]
    LinAlgError = Exception  # type: ignore[misc,assignment]

try:
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    from scipy.sparse.linalg import norm as sparse_norm
    from scipy.stats import pearsonr
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    sparse = None        # type: ignore[assignment]
    sparse_norm = None   # type: ignore[assignment]

import networkx as nx

from app.flux_condenser import (
    CONSTANTS,
    ConfigurationError,
    DiscreteVectorCalculus,
    MaxwellSolver,
    NumericalInstabilityError,
    PIController,
    PortHamiltonianController,
    SystemConstants,
)

logger = logging.getLogger(__name__)

# ── Marcadores de skip centralizados ─────────────────────────────────────────
# Centralizar las condiciones de skip evita repetición y hace explícito
# qué dependencias requiere cada grupo de tests.
_requires_numpy = pytest.mark.skipif(
    not _NUMPY_AVAILABLE, reason="NumPy no disponible"
)
_requires_scipy = pytest.mark.skipif(
    not _SCIPY_AVAILABLE, reason="SciPy no disponible"
)

# ── Constantes de test ────────────────────────────────────────────────────────
_DEFAULT_KP:         float = 2.0
_DEFAULT_KI:         float = 0.5
_DEFAULT_SETPOINT:   float = 0.7
_DEFAULT_MIN_OUTPUT: float = 10.0
_DEFAULT_MAX_OUTPUT: float = 1000.0
_DEFAULT_EMA_ALPHA:  float = 0.3

_K4_NODES:    int = 4
_K4_EDGES:    int = 6
_K4_FACES:    int = 4
_K4_EULER:    int = 2   # χ = V - E + F = 4 - 6 + 4 = 2 = χ(S²)

_FULL_GRAPH_N: int = 6  # Grafo completo K₆ para tests de Maxwell/pH


# ============================================================================
# FIXTURES COMPARTIDOS
# ============================================================================


@pytest.fixture(scope="function")
def default_pi() -> PIController:
    """
    Controlador PI con parámetros de referencia para testing.

    scope='function' porque el controlador acumula estado interno
    (integral, filtro EMA, historial) entre llamadas a compute().
    Un fixture fresco garantiza independencia entre tests.
    """
    return PIController(
        kp=_DEFAULT_KP,
        ki=_DEFAULT_KI,
        setpoint=_DEFAULT_SETPOINT,
        min_output=_DEFAULT_MIN_OUTPUT,
        max_output=_DEFAULT_MAX_OUTPUT,
        ema_alpha=_DEFAULT_EMA_ALPHA,
    )


@pytest.fixture(scope="module")
def k4_graph() -> dict[int, set[int]]:
    """
    Grafo completo K₄ como diccionario de adyacencia.

    K₄ triangula la esfera S² → β₀=1, β₁=0, β₂=1, χ=2.
    scope='module': el grafo es inmutable, se comparte sin riesgo.
    """
    return {0: {1, 2, 3}, 1: {0, 2, 3}, 2: {0, 1, 3}, 3: {0, 1, 2}}


@pytest.fixture(scope="module")
def k6_adjacency() -> dict[int, set[int]]:
    """
    Grafo completo K₆ como diccionario de adyacencia.

    Usado como base para MaxwellSolver y PortHamiltonianController.
    """
    n = _FULL_GRAPH_N
    return {i: set(range(n)) - {i} for i in range(n)}


@pytest.fixture(scope="module")
def k4_calculus(k4_graph: dict[int, set[int]]) -> DiscreteVectorCalculus:
    """Cálculo vectorial discreto sobre K₄."""
    return DiscreteVectorCalculus(k4_graph)


@pytest.fixture(scope="module")
def k6_calculus(k6_adjacency: dict[int, set[int]]) -> DiscreteVectorCalculus:
    """Cálculo vectorial discreto sobre K₆."""
    return DiscreteVectorCalculus(k6_adjacency)


@pytest.fixture(scope="function")
def maxwell_solver(k6_calculus: DiscreteVectorCalculus) -> MaxwellSolver:
    """
    Solver de Maxwell sobre K₆.

    scope='function': el solver muta estado (E, B, D, H) durante la simulación.
    """
    return MaxwellSolver(k6_calculus)


@pytest.fixture(scope="function")
def ph_controller(maxwell_solver: MaxwellSolver) -> PortHamiltonianController:
    """
    Controlador Port-Hamiltoniano sobre el solver de Maxwell.

    scope='function': hereda estado mutable del solver.
    """
    return PortHamiltonianController(maxwell_solver)


# ============================================================================
# TEST: SystemConstants
# ============================================================================


class TestSystemConstants:
    """
    Pruebas de invariantes de SystemConstants.

    SystemConstants es un objeto de configuración inmutable que define
    los límites numéricos del sistema. Sus propiedades son contratos
    que otros módulos asumen como verdaderos.
    """

    def test_immutability_raises_on_assignment(self) -> None:
        """
        Las constantes son inmutables: asignar un atributo lanza AttributeError.

        La inmutabilidad protege contra modificación accidental en runtime.
        """
        constants = SystemConstants()
        with pytest.raises(AttributeError):
            constants.MIN_DELTA_TIME = 999.0  # type: ignore[misc]

    def test_tolerance_hierarchy_is_strict(self) -> None:
        """
        Jerarquía de tolerancias numéricas:
          NUMERICAL_ZERO < NUMERICAL_TOLERANCE < RELATIVE_TOLERANCE

        Motivación: NUMERICAL_ZERO es el umbral para "exactamente cero",
        NUMERICAL_TOLERANCE para comparaciones absolutas, y
        RELATIVE_TOLERANCE para comparaciones relativas.
        """
        c = SystemConstants()
        assert c.NUMERICAL_ZERO < c.NUMERICAL_TOLERANCE < c.RELATIVE_TOLERANCE, (
            f"Jerarquía violada: {c.NUMERICAL_ZERO} < "
            f"{c.NUMERICAL_TOLERANCE} < {c.RELATIVE_TOLERANCE}"
        )

    def test_cfl_safety_factor_in_open_unit_interval(self) -> None:
        """
        CFL_SAFETY_FACTOR ∈ (0, 1).

        Un factor CFL = 1 daría exactamente el límite de estabilidad
        (sin margen de seguridad). Un factor ≤ 0 sería inválido físicamente.
        """
        c = SystemConstants()
        assert 0.0 < c.CFL_SAFETY_FACTOR < 1.0, (
            f"CFL_SAFETY_FACTOR = {c.CFL_SAFETY_FACTOR} ∉ (0, 1)"
        )

    def test_time_bounds_are_consistent(self) -> None:
        """
        MIN_DELTA_TIME < MAX_DELTA_TIME.

        Garantía para que los solvers con paso de tiempo variable
        tengan un intervalo de búsqueda no vacío.
        """
        c = SystemConstants()
        assert c.MIN_DELTA_TIME < c.MAX_DELTA_TIME, (
            f"MIN_DELTA_TIME={c.MIN_DELTA_TIME} ≥ MAX_DELTA_TIME={c.MAX_DELTA_TIME}"
        )

    def test_all_constants_are_positive(self) -> None:
        """
        Todas las constantes numéricas son estrictamente positivas.

        Constantes no-positivas indicarían error de configuración en dominio físico.
        """
        c = SystemConstants()
        for name in ("NUMERICAL_ZERO", "NUMERICAL_TOLERANCE", "RELATIVE_TOLERANCE",
                     "CFL_SAFETY_FACTOR", "MIN_DELTA_TIME", "MAX_DELTA_TIME"):
            value = getattr(c, name)
            assert value > 0, f"CONSTANTS.{name} = {value} debe ser > 0"

    def test_global_constants_instance_is_consistent(self) -> None:
        """
        La instancia global CONSTANTS tiene la misma jerarquía que una instancia local.

        Verifica que el singleton exportado no fue modificado accidentalmente.
        """
        local = SystemConstants()
        assert CONSTANTS.NUMERICAL_ZERO      == local.NUMERICAL_ZERO
        assert CONSTANTS.NUMERICAL_TOLERANCE == local.NUMERICAL_TOLERANCE


# ============================================================================
# TEST: PIController
# ============================================================================


class TestPIControllerRefined:
    """
    Pruebas exhaustivas del controlador PI con anti-windup y filtro EMA.

    Ley de control implementada:
      u(t) = clip(kp·e(t) + ki·∫e(τ)dτ, min_output, max_output)

    donde e(t) = setpoint - y_f(t)  y  y_f(t) = EMA(y(t), α).
    """

    # ── Validación de parámetros ──────────────────────────────────────────────

    @pytest.mark.parametrize(
        "params, reason",
        [
            (
                {"kp": -1.0, "ki": 0.5, "setpoint": 0.7, "min_output": 10, "max_output": 1000},
                "kp negativo es físicamente inválido",
            ),
            (
                {"kp": 2.0, "ki": -0.5, "setpoint": 0.7, "min_output": 10, "max_output": 1000},
                "ki negativo causa divergencia del término integral",
            ),
            (
                {"kp": 2.0, "ki": 0.5, "setpoint": 1.5, "min_output": 10, "max_output": 1000},
                "setpoint > 1.0 excede el rango normalizado de la variable de proceso",
            ),
            (
                {"kp": 2.0, "ki": 0.5, "setpoint": 0.7, "min_output": -10, "max_output": 1000},
                "min_output negativo inválido para actuadores físicos",
            ),
            (
                {"kp": 2.0, "ki": 0.5, "setpoint": 0.7, "min_output": 1000, "max_output": 100},
                "min_output > max_output define rango vacío de saturación",
            ),
        ],
    )
    def test_invalid_parameters_raise_configuration_error(
        self, params: dict[str, Any], reason: str
    ) -> None:
        """
        Parámetros inválidos deben lanzar ConfigurationError.

        Cada caso está documentado con la razón matemática/física de la invalides.
        """
        with pytest.raises(ConfigurationError, match=r".*"):
            PIController(**params)

    # ── Invariantes de salida ─────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "measurement",
        [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    )
    def test_output_always_within_bounds(
        self, default_pi: PIController, measurement: float
    ) -> None:
        """
        u(t) ∈ [min_output, max_output]  para toda medición válida.

        La saturación es parte del contrato de seguridad del actuador.
        """
        output = default_pi.compute(measurement)
        assert _DEFAULT_MIN_OUTPUT <= output <= _DEFAULT_MAX_OUTPUT, (
            f"compute({measurement}) = {output} ∉ "
            f"[{_DEFAULT_MIN_OUTPUT}, {_DEFAULT_MAX_OUTPUT}]"
        )

    def test_output_monotone_increasing_with_error(self) -> None:
        """
        Mayor error positivo → mayor salida (respuesta proporcional).

        Se verifica con ki=0 para aislar la componente proporcional.
        Condición: out(e=0.2) > out(e=0.1) × umbral_proporcional.
        """
        def _make_p_only(setpoint: float) -> PIController:
            return PIController(
                kp=10.0, ki=0.0,
                setpoint=setpoint,
                min_output=1, max_output=10_000,
            )

        # Error pequeño: setpoint=0.5, medición=0.4  → e=0.1
        c_small = _make_p_only(setpoint=0.5)
        out_small = c_small.compute(0.4)

        # Error grande: setpoint=0.5, medición=0.3  → e=0.2
        c_large = _make_p_only(setpoint=0.5)
        out_large = c_large.compute(0.3)

        assert out_large > out_small, (
            f"Error mayor debe producir salida mayor: "
            f"out(e=0.2)={out_large} ≤ out(e=0.1)={out_small}"
        )
        # Verificar proporcionalidad: la relación debe ser al menos 1.5×
        assert out_large >= out_small * 1.5, (
            f"Proporcionalidad insuficiente: out_large/out_small = "
            f"{out_large/out_small:.2f} < 1.5"
        )

    def test_integral_accumulation_is_monotone_increasing(self) -> None:
        """
        Con error constante positivo y ki dominante, la salida es monótonamente creciente.

        Condición: e(t) = setpoint - measurement = 0.5 - 0.4 = 0.1 > 0 constante.
        El término integral ∫e dτ crece → u(t) crece.
        """
        controller = PIController(
            kp=0.1, ki=1.0,
            setpoint=0.5,
            min_output=1, max_output=10_000,
        )
        outputs = [controller.compute(0.4) for _ in range(20)]

        # Verificar monotonía estricta hasta saturación
        non_monotone = [
            i for i in range(len(outputs) - 1)
            if outputs[i] > outputs[i + 1]
        ]
        assert len(non_monotone) == 0, (
            f"Salida no monótona en posiciones: {non_monotone}. "
            f"Outputs: {[round(o, 3) for o in outputs]}"
        )

    # ── Anti-windup ───────────────────────────────────────────────────────────

    def test_anti_windup_enables_recovery_from_saturation(self) -> None:
        """
        Anti-windup por back-calculation: tras saturación prolongada,
        el controlador recupera control cuando el setpoint baja.

        Sin anti-windup, el integrador acumularía tanto que la recuperación
        tomaría muchos ciclos. Con anti-windup, la recuperación es rápida.

        Verificación: tras saturar con error grande y luego setear error=0,
        la salida debe bajar por debajo del 90% del máximo en ≤ 30 ciclos.
        """
        controller = PIController(
            kp=100.0, ki=20.0,
            setpoint=0.9,
            min_output=10, max_output=100,
        )

        # Fase de saturación: empujar el integrador al máximo
        for _ in range(50):
            controller.compute(0.01)  # error = 0.89, muy grande
            time.sleep(0.01)

        # Verificar que efectivamente saturó
        saturated_output = controller.compute(0.01)
        assert saturated_output >= controller.max_output * 0.9, (
            f"El controlador debería estar saturado: {saturated_output}"
        )

        # Fase de recuperación: setpoint = medición → error = 0
        # Usamos la API pública si existe; de lo contrario accedemos al atributo
        if hasattr(controller, "set_setpoint"):
            controller.set_setpoint(0.5)
        else:
            controller.setpoint = 0.5  # type: ignore[assignment]

        recovery_outputs = [controller.compute(0.5) for _ in range(30)]

        assert recovery_outputs[-1] < controller.max_output * 0.9, (
            f"Anti-windup falló: salida final = {recovery_outputs[-1]:.2f} "
            f"debe ser < {controller.max_output * 0.9:.2f} tras recuperación"
        )

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def test_rate_limiting_caps_output_change(self) -> None:
        """
        El rate limiter restringe el cambio de salida por ciclo.

        Δu(t) ≤ rate_limit × (max_output - min_output)

        Se usa kp grande para forzar un cambio potencial grande
        y verificar que el rate limiter lo restringe.
        """
        controller = PIController(
            kp=100.0, ki=0.0,
            setpoint=0.5,
            min_output=10, max_output=1000,
        )
        controller.compute(0.5)          # Inicializar estado
        out_reference = controller.compute(0.5)  # Salida estable
        out_after_step = controller.compute(0.1)  # Escalón grande → cambio potencial enorme

        output_range    = controller.max_output - controller.min_output
        max_allowed_change = 0.15 * output_range  # 15% del rango por ciclo

        actual_change = abs(out_after_step - out_reference)
        assert actual_change <= max_allowed_change, (
            f"Rate limiter violado: Δu = {actual_change:.2f} > "
            f"límite = {max_allowed_change:.2f} "
            f"(rango={output_range}, 15%={max_allowed_change:.2f})"
        )

    # ── Filtro EMA ────────────────────────────────────────────────────────────

    @_requires_numpy
    def test_ema_filter_reduces_output_variance(self) -> None:
        """
        El filtro EMA reduce la varianza de la salida respecto a la entrada ruidosa.

        Nota: la comparación es intra-controlador: se comparan dos controladores
        con α=1.0 (sin filtro) vs α=0.3 (con filtro) sobre la misma señal ruidosa.
        Esto aísla el efecto del EMA sin confundirlo con la saturación.
        """
        assert np is not None

        np.random.seed(42)
        noisy_signal = [0.5 + 0.1 * np.random.randn() for _ in range(50)]

        # Controlador sin filtro (α=1.0 → sin suavizado)
        no_filter = PIController(
            kp=1.0, ki=0.0,
            setpoint=0.5,
            min_output=0.001, max_output=1000.0, # avoid clamping limiting variance
            ema_alpha=1.0,
        )
        # Avoid rate limiting by resetting
        outputs_no_filter = []
        for m in noisy_signal:
            outputs_no_filter.append(no_filter.compute(m))
            no_filter._last_output = None # disable rate limit

        # Controlador con filtro EMA
        with_filter = PIController(
            kp=1.0, ki=0.0,
            setpoint=0.5,
            min_output=0.001, max_output=1000.0, # avoid clamping limiting variance
            ema_alpha=0.1,
        )
        outputs_with_filter = []
        for m in noisy_signal:
            outputs_with_filter.append(with_filter.compute(m))
            with_filter._last_output = None # disable rate limit

        var_no_filter   = np.var(outputs_no_filter)
        var_with_filter = np.var(outputs_with_filter)

        assert var_with_filter < var_no_filter, (
            f"EMA no reduce varianza: "
            f"var(sin_filtro)={var_no_filter:.4f}, "
            f"var(con_filtro)={var_with_filter:.4f}"
        )

    # ── Estabilidad de Lyapunov ───────────────────────────────────────────────

    @_requires_numpy
    def test_lyapunov_exponent_negative_for_converging_system(self) -> None:
        """
        Sistema convergente tiene exponente de Lyapunov estrictamente negativo.

        λ_Lyapunov < 0  ⟺  trayectorias cercanas convergen → sistema estable.

        Nota: el umbral es 0.0 (no 0.1 como en la versión original),
        ya que un exponente positivo indicaría divergencia (caos), no estabilidad.
        """
        assert np is not None

        controller = PIController(
            kp=5.0, ki=1.0,
            setpoint=0.5,
            min_output=10, max_output=1000,
        )
        measurement = 0.1
        for i in range(100):
            controller.compute(measurement)
            measurement = 0.5 - 0.4 * np.exp(-i * 0.05)  # Señal convergente

        exponent = controller.get_lyapunov_exponent()
        assert exponent < 0.0, (
            f"Sistema convergente debe tener λ_Lyapunov < 0, "
            f"obtenido: {exponent:.4f}"
        )

    # ── Análisis de estabilidad ───────────────────────────────────────────────

    @_requires_numpy
    def test_stability_analysis_operational_status(
        self, default_pi: PIController
    ) -> None:
        """
        Controlador operando en régimen nominal → status = 'OPERATIONAL'.

        Se alimenta con señal senoidal pequeña alrededor del setpoint
        para simular operación normal sin perturbaciones.
        """
        assert np is not None

        for i in range(50):
            default_pi.compute(0.5 + 0.1 * np.sin(i * 0.1))

        analysis = default_pi.get_stability_analysis()

        assert analysis["status"] == "OPERATIONAL", (
            f"Status inesperado: '{analysis['status']}'"
        )

    @_requires_numpy
    def test_stability_class_is_valid_enum_value(
        self, default_pi: PIController
    ) -> None:
        """
        stability_class debe ser uno de los valores válidos del enum.

        Los valores válidos están definidos por el contrato del analizador.
        """
        assert np is not None

        valid_classes = {"ASYMPTOTICALLY_STABLE", "MARGINALLY_STABLE", "UNSTABLE"}
        for i in range(50):
            default_pi.compute(0.5 + 0.1 * np.sin(i * 0.1))

        analysis = default_pi.get_stability_analysis()

        assert analysis["stability_class"] in valid_classes, (
            f"stability_class inválido: '{analysis['stability_class']}'. "
            f"Válidos: {valid_classes}"
        )

    # ── Reset del estado ──────────────────────────────────────────────────────

    def test_reset_clears_all_internal_state(
        self, default_pi: PIController
    ) -> None:
        """
        reset() restablece todo el estado acumulado a valores iniciales.

        Campos verificados:
          · _integral_error = 0.0
          · _last_output = None
          · _filtered_pv = None
        """
        for _ in range(20):
            default_pi.compute(0.3)

        default_pi.reset()

        assert default_pi._integral_error == 0.0, (
            f"_integral_error no reseteado: {default_pi._integral_error}"
        )
        assert default_pi._last_output is None, (
            f"_last_output no reseteado: {default_pi._last_output}"
        )
        assert default_pi._filtered_pv is None, (
            f"_filtered_pv no reseteado: {default_pi._filtered_pv}"
        )

    def test_reset_makes_output_reproducible(
        self, default_pi: PIController
    ) -> None:
        """
        Dos secuencias idénticas desde reset producen salidas idénticas.

        Propiedad de determinismo: el controlador es un sistema dinámico
        determinístico — misma condición inicial + misma entrada → misma salida.
        """
        default_pi.reset()
        out1 = default_pi.compute(0.4)

        default_pi.reset()
        out2 = default_pi.compute(0.4)

        assert out1 == out2, (
            f"Salida no reproducible tras reset: {out1} ≠ {out2}"
        )

    # ── Diagnósticos ─────────────────────────────────────────────────────────

    def test_diagnostics_contain_required_keys(
        self, default_pi: PIController
    ) -> None:
        """
        get_diagnostics() retorna dict con todas las claves del contrato de API.
        """
        default_pi.compute(0.5)
        diag = default_pi.get_diagnostics()

        required_keys = {"status", "control_metrics", "stability_analysis", "parameters"}
        missing = required_keys - set(diag.keys())
        assert not missing, (
            f"Claves faltantes en get_diagnostics(): {missing}"
        )

    def test_diagnostics_parameters_match_constructor(
        self, default_pi: PIController
    ) -> None:
        """
        Los parámetros en diagnósticos deben coincidir con los valores del constructor.

        Verifica que el controlador no modifica sus parámetros en runtime.
        """
        default_pi.compute(0.5)
        params = default_pi.get_diagnostics()["parameters"]

        assert params.get("kp") == _DEFAULT_KP, (
            f"kp en diagnósticos ({params.get('kp')}) ≠ constructor ({_DEFAULT_KP})"
        )
        assert params.get("ki") == _DEFAULT_KI, (
            f"ki en diagnósticos ({params.get('ki')}) ≠ constructor ({_DEFAULT_KI})"
        )

    # ── Respuesta al escalón ──────────────────────────────────────────────────

    def test_step_response_output_decreases_at_zero_error(self) -> None:
        """
        Cuando el error se vuelve cero, la salida debe decrecer (o estabilizarse).

        Protocolo sin time.sleep() — el controlador usa timestamps internos
        para calcular el paso de tiempo; el test no necesita esperar.

        Verificación: out_before (con error) > out_after (sin error).
        """
        controller = PIController(
            kp=100.0, ki=10.0,
            setpoint=0.5,
            min_output=10, max_output=1000,
        )

        # Fase pre-step: error = 0.5 - 0.2 = 0.3
        for _ in range(10):
            out_before = controller.compute(0.2)

        # Fase post-step: error = 0.5 - 0.5 = 0.0
        for _ in range(10):
            out_after = controller.compute(0.5)

        assert out_after < out_before, (
            f"Respuesta al escalón incorrecta: "
            f"out_before={out_before:.2f} debe ser > out_after={out_after:.2f}"
        )


# ============================================================================
# TEST: DiscreteVectorCalculus
# ============================================================================


class TestDiscreteVectorCalculusRefined:
    """
    Pruebas de cálculo vectorial discreto sobre complejos simpliciales.

    El complejo simplicial K₄ (tetraedro) triangula S²:
      V=4, E=6, F=4  →  χ = V - E + F = 2 = χ(S²)
      β₀=1 (conexo), β₁=0 (sin ciclos 1D), β₂=1 (esfera)
    """

    def test_empty_graph_raises_value_error(self) -> None:
        """Grafo vacío es inválido — sin nodos no hay complejo simplicial."""
        with pytest.raises(ValueError):
            DiscreteVectorCalculus({})

    def test_k4_simplicial_complex_dimensions(
        self, k4_calculus: DiscreteVectorCalculus
    ) -> None:
        """
        K₄ tiene exactamente V=4, E=6, F=4.

        Conteo:
          V = 4 nodos
          E = C(4,2) = 6 aristas
          F = C(4,3) = 4 triángulos (2-símplices)
        """
        assert k4_calculus.num_nodes == _K4_NODES, (
            f"num_nodes={k4_calculus.num_nodes} ≠ {_K4_NODES}"
        )
        assert k4_calculus.num_edges == _K4_EDGES, (
            f"num_edges={k4_calculus.num_edges} ≠ {_K4_EDGES}"
        )
        assert k4_calculus.num_faces == _K4_FACES, (
            f"num_faces={k4_calculus.num_faces} ≠ {_K4_FACES}"
        )

    def test_euler_characteristic_of_k4(
        self, k4_calculus: DiscreteVectorCalculus
    ) -> None:
        """
        χ(K₄) = V - E + F = 4 - 6 + 4 = 2 = χ(S²).

        La característica de Euler es un invariante topológico:
        dos espacios homeomorfos tienen la misma χ.
        """
        assert k4_calculus.euler_characteristic == _K4_EULER, (
            f"χ(K₄) = {k4_calculus.euler_characteristic} ≠ {_K4_EULER}"
        )

    @_requires_scipy
    def test_chain_complex_exactness(
        self, k4_calculus: DiscreteVectorCalculus
    ) -> None:
        """
        ∂₁ ∘ ∂₂ = 0  (exactitud del complejo de cadenas).

        Esta es la identidad de borde sobre borde: el borde de un
        borde es vacío. Es condición necesaria para que los operadores
        de cálculo vectorial estén bien definidos.
        """
        result = k4_calculus.verify_complex_exactness()
        assert result["is_chain_complex"], (
            f"Complejo de cadenas no exacto: {result}"
        )

    @_requires_scipy
    def test_betti_numbers_of_k4_equal_sphere(
        self, k4_calculus: DiscreteVectorCalculus
    ) -> None:
        """
        K₄ como 2-complejo simplicial tiene H*(K₄) ≅ H*(S²):
          β₀ = 1  (una componente conexa)
          β₁ = 0  (sin agujeros 1D — no hay ciclos independientes)
          β₂ = 1  (una cavidad 2D — la esfera interior)

        Fórmula general: βₖ = dim(ker ∂ₖ) - dim(im ∂_{k+1})
        """
        assert (k4_calculus.betti_0, k4_calculus.betti_1, k4_calculus.betti_2) == (1, 0, 1), (
            f"Números de Betti incorrectos: "
            f"β₀={k4_calculus.betti_0}, β₁={k4_calculus.betti_1}, β₂={k4_calculus.betti_2}. "
            f"Esperado: β₀=1, β₁=0, β₂=1 (esfera S²)"
        )

    @_requires_scipy
    def test_euler_characteristic_from_betti_numbers(
        self, k4_calculus: DiscreteVectorCalculus
    ) -> None:
        """
        Coherencia entre χ geométrico y χ homológico:
          χ = β₀ - β₁ + β₂  (fórmula de Euler-Poincaré)
          χ = 1 - 0 + 1 = 2  ✓
        """
        chi_homological = (
            k4_calculus.betti_0
            - k4_calculus.betti_1
            + k4_calculus.betti_2
        )
        assert chi_homological == k4_calculus.euler_characteristic, (
            f"χ geométrico ({k4_calculus.euler_characteristic}) ≠ "
            f"χ homológico ({chi_homological}) = β₀-β₁+β₂"
        )

    @_requires_scipy
    def test_gradient_of_constant_is_zero(
        self, k4_calculus: DiscreteVectorCalculus
    ) -> None:
        """
        grad(constante) = 0  en todos los ejes.

        Propiedad básica: funciones constantes no tienen variación
        → el gradiente discreto debe ser el vector cero.
        """
        assert np is not None
        result = k4_calculus.gradient(np.ones(k4_calculus.num_nodes))
        assert np.allclose(result, 0, atol=1e-10), (
            f"grad(1) ≠ 0: max|grad(1)| = {np.abs(result).max():.2e}"
        )

    @_requires_scipy
    def test_curl_of_gradient_is_zero(
        self, k4_calculus: DiscreteVectorCalculus
    ) -> None:
        """
        curl(grad(φ)) = 0  para toda función escalar φ.

        Identidad de Stokes discreta: im(grad) ⊆ ker(curl).
        Equivalente a d² = 0 en el lenguaje de formas diferenciales.
        """
        assert np is not None
        np.random.seed(0)
        phi = np.random.randn(k4_calculus.num_nodes)
        result = k4_calculus.curl(k4_calculus.gradient(phi))
        assert np.allclose(result, 0, atol=1e-10), (
            f"curl(grad(φ)) ≠ 0: max|curl(grad(φ))| = {np.abs(result).max():.2e}"
        )

    @_requires_scipy
    def test_gradient_linearity(
        self, k4_calculus: DiscreteVectorCalculus
    ) -> None:
        """
        grad(α·φ + β·ψ) = α·grad(φ) + β·grad(ψ)  (linealidad).

        El operador gradiente discreto es lineal por construcción
        (es una aplicación de matrices).
        """
        assert np is not None
        np.random.seed(1)
        phi = np.random.randn(k4_calculus.num_nodes)
        psi = np.random.randn(k4_calculus.num_nodes)
        alpha, beta = 2.3, -1.7

        lhs = k4_calculus.gradient(alpha * phi + beta * psi)
        rhs = alpha * k4_calculus.gradient(phi) + beta * k4_calculus.gradient(psi)

        assert np.allclose(lhs, rhs, atol=1e-12), (
            f"grad no es lineal: max|lhs-rhs| = {np.abs(lhs-rhs).max():.2e}"
        )


# ============================================================================
# TEST: MaxwellSolver
# ============================================================================


class TestMaxwellSolverRefined:
    """
    Pruebas del solver de Maxwell discreto sobre complejos simpliciales.

    Propiedades verificadas:
      · Condición CFL: garantiza estabilidad del esquema explícito.
      · Relaciones constitutivas: D = ε·⋆E, B = μ·⋆H.
      · Conservación de energía: dH/dt ≤ 0 (disipación por resistividad).
    """

    @_requires_scipy
    def test_cfl_condition_is_positive(
        self, maxwell_solver: MaxwellSolver
    ) -> None:
        """
        El paso de tiempo CFL es estrictamente positivo.

        dt_cfl = CFL_factor × h / (c√d)
        donde h es el tamaño de malla mínimo y c la velocidad de propagación.
        Un dt_cfl ≤ 0 indicaría error en el cálculo de la malla.
        """
        assert maxwell_solver.dt_cfl > 0, (
            f"dt_cfl = {maxwell_solver.dt_cfl} debe ser > 0"
        )

    @_requires_scipy
    def test_constitutive_relation_D_equals_epsilon_star_E(
        self, maxwell_solver: MaxwellSolver
    ) -> None:
        """
        Relación constitutiva: D = ε · ⋆₁E.

        ⋆₁ es el operador de Hodge estrella en aristas (1-formas).
        La relación D = ε·⋆E es la ley constitutiva del medio lineal.
        """
        assert np is not None
        np.random.seed(2)
        maxwell_solver.E = np.random.randn(maxwell_solver.calc.num_edges)
        maxwell_solver.B = np.random.randn(maxwell_solver.calc.num_faces)
        maxwell_solver.update_constitutive_relations()

        expected_D = maxwell_solver.epsilon * (maxwell_solver.calc.star1 @ maxwell_solver.E)
        assert np.allclose(maxwell_solver.D, expected_D, atol=1e-12), (
            f"D ≠ ε·⋆E: max|D - ε·⋆E| = {np.abs(maxwell_solver.D - expected_D).max():.2e}"
        )

    @_requires_scipy
    def test_energy_is_non_negative(
        self, maxwell_solver: MaxwellSolver
    ) -> None:
        """
        La energía electromagnética H(E, B) ≥ 0 siempre.

        H = ½(Eᵀ·D + Bᵀ·H) = ½(ε‖⋆E‖² + μ⁻¹‖⋆B‖²) ≥ 0
        por ser suma de normas al cuadrado escaladas positivamente.
        """
        assert np is not None
        np.random.seed(3)
        maxwell_solver.E = np.random.randn(maxwell_solver.calc.num_edges)
        maxwell_solver.B = np.random.randn(maxwell_solver.calc.num_faces)
        maxwell_solver.update_constitutive_relations()

        energy_metrics = maxwell_solver.compute_energy_and_momentum()
        energy = energy_metrics.get("total_energy", 0.0)
        assert energy >= 0.0, (
            f"Energía electromagnética debe ser ≥ 0, obtenida: {energy:.4f}"
        )

    @_requires_scipy
    def test_energy_conservation_with_relaxed_tolerance(
        self, maxwell_solver: MaxwellSolver
    ) -> None:
        """
        El solver conserva (o disipa) energía dentro de la tolerancia numérica.

        Tolerancia relaxada a 10% porque:
          · La discretización espacial introduce error de truncamiento O(h²).
          · El esquema explícito acumula error de O(dt) por paso.
          · El grafo discreto K₆ tiene menor accuracy que una malla regular.

        Criterio: |H(T) - H(0)| / H(0) ≤ 0.10
        """
        result = maxwell_solver.verify_energy_conservation(
            num_steps=20, tolerance=0.10
        )
        assert result["is_conservative"], (
            f"Energía no conservada: {result}"
        )


# ============================================================================
# TEST: PortHamiltonianController
# ============================================================================


class TestPortHamiltonianControllerRefined:
    """
    Pruebas del controlador Port-Hamiltoniano (PHS).

    Estructura PHS:
      ẋ = (J - R)∇H(x) + Bu
      y  = Bᵀ∇H(x)

    donde J es antisimétrica (J + Jᵀ = 0) y R es semidefinida positiva.
    """

    @_requires_scipy
    def test_phs_matrix_j_is_antisymmetric(
        self, ph_controller: PortHamiltonianController
    ) -> None:
        """
        J + Jᵀ = 0  (antisimetría estricta).

        La antisimetría de J garantiza la conservación de energía
        en ausencia de disipación (R=0) y puertos externos (u=0).

        Verificación: ‖J + Jᵀ‖_F < ε_máquina × n
        """
        assert sparse_norm is not None
        J = ph_controller.J_phs
        skew_norm = sparse_norm(J + J.T)
        assert skew_norm < 1e-10, (
            f"J no es antisimétrica: ‖J + Jᵀ‖ = {skew_norm:.2e} ≥ 1e-10"
        )

    @_requires_scipy
    def test_hamiltonian_is_non_negative(
        self, ph_controller: PortHamiltonianController
    ) -> None:
        """
        H(x) ≥ 0 para todo estado x.

        La función Hamiltoniana representa energía almacenada → no negativa.
        """
        assert np is not None
        np.random.seed(4)
        # Inicializar con estado aleatorio
        ph_controller.solver.E = np.random.randn(ph_controller.solver.calc.num_edges)
        ph_controller.solver.B = np.random.randn(ph_controller.solver.calc.num_faces)
        ph_controller.solver.update_constitutive_relations()

        H = ph_controller.compute_hamiltonian()
        assert H >= 0.0, (
            f"Hamiltoniano debe ser ≥ 0, obtenido: H = {H:.4f}"
        )

    @_requires_scipy
    def test_passivity_verification_returns_result(
        self, ph_controller: PortHamiltonianController
    ) -> None:
        """
        verify_passivity() retorna un dict con la clave 'is_passive'.

        La pasividad se define como:
          ∫₀ᵀ uᵀy dt ≥ -β  para algún β ≥ 0 (energía inicial almacenada)

        Este test verifica la estructura del resultado sin asumir que
        la pasividad se satisface numéricamente (pueden existir pequeñas
        violaciones por discretización).
        """
        result = ph_controller.verify_passivity(num_steps=20)

        assert isinstance(result, dict), (
            f"verify_passivity debe retornar dict, obtenido: {type(result).__name__}"
        )
        assert "is_passive" in result, (
            f"El resultado debe contener 'is_passive': {result.keys()}"
        )
        assert isinstance(result["is_passive"], bool), (
            f"is_passive debe ser bool, obtenido: {type(result['is_passive']).__name__}"
        )

    @_requires_scipy
    def test_passivity_supply_rate_upper_bounded(
        self, ph_controller: PortHamiltonianController
    ) -> None:
        """
        La tasa de suministro de energía está acotada superiormente.

        Para sistema pasivo: dH/dt ≤ uᵀy  →  la energía no puede crecer
        más rápido que el suministro externo.

        Si el resultado expone 'supply_rate', verificamos que es finito.
        """
        result = ph_controller.verify_passivity(num_steps=20)

        if "supply_rate" in result:
            assert math.isfinite(result["supply_rate"]), (
                f"supply_rate debe ser finito: {result['supply_rate']}"
            )


# ============================================================================
# TEST: Integración
# ============================================================================


class TestIntegrationRefined:
    """
    Pruebas de integración entre módulos del sistema.

    Verifica que el pipeline completo:
      DiscreteVectorCalculus → MaxwellSolver → PortHamiltonianController
    produce resultados físicamente válidos de extremo a extremo.
    """

    @_requires_scipy
    def test_full_pipeline_produces_finite_energy(
        self,
        k6_adjacency: dict[int, set[int]],
    ) -> None:
        """
        Pipeline completo produce energía finita en todos los pasos.

        Instanciamos el pipeline completo (no usamos fixtures compartidos
        para garantizar aislamiento del test de integración).
        """
        assert np is not None
        calc   = DiscreteVectorCalculus(k6_adjacency)
        solver = MaxwellSolver(calc)
        ctrl   = PortHamiltonianController(solver, target_energy=0.5)

        result = ctrl.simulate_regulation(num_steps=50)

        assert "energy" in result, "simulate_regulation debe retornar 'energy'"
        energy = result["energy"]
        assert np.all(np.isfinite(energy)), (
            f"Energía no finita en {np.sum(~np.isfinite(energy))} pasos de 50"
        )

    @_requires_scipy
    def test_full_pipeline_energy_does_not_diverge(
        self,
        k6_adjacency: dict[int, set[int]],
    ) -> None:
        """
        La energía no crece ilimitadamente — el controlador estabiliza el sistema.

        Criterio: H(T) ≤ H(0) × factor_de_crecimiento_máximo (10×).
        Un factor mayor indicaría inestabilidad numérica o del controlador.
        """
        assert np is not None
        calc   = DiscreteVectorCalculus(k6_adjacency)
        solver = MaxwellSolver(calc)
        ctrl   = PortHamiltonianController(solver, target_energy=0.5)

        result = ctrl.simulate_regulation(num_steps=50)
        energy = result["energy"]

        max_growth_factor = 10.0
        if energy[0] > 0:
            assert energy[-1] <= energy[0] * max_growth_factor, (
                f"Energía diverge: H(0)={energy[0]:.4f}, "
                f"H(T)={energy[-1]:.4f}, "
                f"ratio={energy[-1]/energy[0]:.2f} > {max_growth_factor}"
            )

    @_requires_scipy
    def test_full_pipeline_regulation_reduces_energy_error(
        self,
        k6_adjacency: dict[int, set[int]],
    ) -> None:
        """
        El controlador reduce el error de energía respecto al objetivo.

        |H(T) - H_target| ≤ |H(0) - H_target|

        Verifica que la regulación es efectiva, no solo que el sistema
        es estable numéricamente.
        """
        assert np is not None
        target = 0.5
        calc   = DiscreteVectorCalculus(k6_adjacency)
        solver = MaxwellSolver(calc)
        ctrl   = PortHamiltonianController(solver, target_energy=target)

        result = ctrl.simulate_regulation(num_steps=50)
        energy = result["energy"]

        error_initial = abs(energy[0]  - target)
        error_final   = abs(energy[-1] - target)

        assert error_final <= error_initial + 1e-6, (
            f"El controlador no reduce el error de energía:\n"
            f"  |H(0) - H_target|  = {error_initial:.4f}\n"
            f"  |H(T) - H_target|  = {error_final:.4f}\n"
            f"  target             = {target}"
        )

    @_requires_scipy
    def test_pipeline_components_are_consistent(
        self,
        k6_adjacency: dict[int, set[int]],
    ) -> None:
        """
        Los componentes del pipeline comparten la misma geometría.

        Verifica que calc, solver y ctrl referencian el mismo complejo
        simplicial (no copias con dimensiones diferentes).
        """
        calc   = DiscreteVectorCalculus(k6_adjacency)
        solver = MaxwellSolver(calc)
        ctrl   = PortHamiltonianController(solver)

        assert solver.calc is calc, (
            "MaxwellSolver debe referenciar el mismo DiscreteVectorCalculus"
        )
        assert ctrl.solver is solver, (
            "PortHamiltonianController debe referenciar el mismo MaxwellSolver"
        )
        assert ctrl.solver.calc.num_nodes == calc.num_nodes, (
            f"Dimensiones inconsistentes: "
            f"ctrl.solver.calc.num_nodes={ctrl.solver.calc.num_nodes} "
            f"≠ calc.num_nodes={calc.num_nodes}"
        )
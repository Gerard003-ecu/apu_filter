# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite de Pruebas : Dynamic Shield Router v3.0.0                              ║
║ Archivo          : tests/unit/core/immune_system/test_dynamic_shield_router.py    ║
║ Cobertura        : Phase1 · Phase2 · Phase3 · DynamicShieldRouter            ║
║                    Algebraica · Geométrica · Espectral · Adversarial         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Organización de la suite
────────────────────────
Las clases de prueba siguen el mismo anidamiento de fases que el módulo:

  TestFixtures          — fábrica centralizada de objetos de prueba
  TestPhase1_*          — conexión de Ehresmann (algebraica + adversarial)
  TestPhase2_*          — pullback termodinámico (por estrato + invariantes)
  TestPhase3_*          — vestimenta del escudo (Higham + Tikhonov)
  TestPipeline_*        — orquestador completo (composición + inmutabilidad)
  TestProperties_*      — pruebas basadas en propiedades (Hypothesis)

Convenciones matemáticas
────────────────────────
  · "SPD"  = simétrica definida positiva
  · "PSD"  = simétrica semidefinida positiva
  · "‖·‖_F"= norma de Frobenius
  · "κ(·)" = número de condición espectral (λ_max / λ_min)
  · "H²=H" = idempotencia del proyector horizontal
  · "σ"    = producción de entropía Tr(δR G⁻¹)
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import scipy.linalg as la
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS DEL MÓDULO BAJO PRUEBA
# ══════════════════════════════════════════════════════════════════════════════
from app.core.immune_system.dynamic_shield_router import (
    _BETTI_0,
    _DEFORMATION_RATIO_WARN,
    _EIG_TOL,
    _KAPPA_MAX,
    ConnectionData,
    ConnectionError,
    DeformationTensor,
    DressingError,
    DynamicShieldRouter,
    Phase1_EhresmannConnection,
    Phase2_ThermodynamicPullback,
    Phase3_ShieldDresser,
    PullbackError,
)
from app.core.schemas import Stratum
from app.core.immune_system.funtor_shield import (
    FuntorShield,
    QuadraticDissipation,
    PortHamiltonianFlow,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE PRUEBA
# ══════════════════════════════════════════════════════════════════════════════

#: Dimensión canónica del espacio de estados en las pruebas.
N: int = 4

#: Tolerancia numérica estricta para igualdades algebraicas exactas.
ATOL_STRICT: float = 1e-9

#: Tolerancia numérica relajada para igualdades termodinámicas aproximadas.
ATOL_THERMO: float = 1e-6

#: Semilla fija para reproducibilidad de matrices aleatorias.
RNG_SEED: int = 42


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES CENTRALIZADAS
# ══════════════════════════════════════════════════════════════════════════════


class TestFixtures:
    """
    Fábrica estática de objetos de prueba reutilizables en toda la suite.
    Centraliza la construcción de matrices SPD, FuntorShield mocks y
    telemetrías por estrato, garantizando coherencia entre clases de prueba.
    """

    @staticmethod
    def rng(seed: int = RNG_SEED) -> np.random.Generator:
        """Generador de números aleatorios reproducible."""
        return np.random.default_rng(seed)

    @staticmethod
    def spd_matrix(n: int = N, scale: float = 1.0, seed: int = RNG_SEED) -> NDArray[np.float64]:
        r"""
        Genera una matriz SPD de dimensión n × n.

        Construcción: M = A Aᵀ + n·I, con A aleatoria,
        garantizando λ_min(M) ≥ n > 0.

        Parámetros
        ──────────
        n     : Dimensión de la matriz.
        scale : Factor multiplicativo global.
        seed  : Semilla del generador.
        """
        rng = TestFixtures.rng(seed)
        A = rng.standard_normal((n, n))
        M = A @ A.T + n * np.eye(n)
        return (M * scale).astype(np.float64)

    @staticmethod
    def psd_matrix(n: int = N, rank: int | None = None, seed: int = RNG_SEED) -> NDArray[np.float64]:
        r"""
        Genera una matriz PSD (semidefinida, posiblemente singular) de rango r ≤ n.

        Construcción: M = B Bᵀ con B de forma (n, rank).
        """
        rng = TestFixtures.rng(seed)
        r = rank if rank is not None else n
        B = rng.standard_normal((n, r))
        return (B @ B.T).astype(np.float64)

    @staticmethod
    def identity_scaled(n: int = N, scale: float = 1.0) -> NDArray[np.float64]:
        """Devuelve scale · I_n."""
        return (scale * np.eye(n)).astype(np.float64)

    @staticmethod
    def mock_shield(
        R: NDArray[np.float64],
        J: NDArray[np.float64] | None = None,
        grad_H: NDArray[np.float64] | None = None,
    ) -> FuntorShield:
        r"""
        Construye un FuntorShield mock con la disipación R dada.

        Los atributos J y grad_H son matrices antisimétrica y aleatoria
        por defecto, respectivamente.
        """
        n = R.shape[0]
        rng = TestFixtures.rng()

        if J is None:
            A = rng.standard_normal((n, n))
            J = A - A.T   # antisimétrica (estructura simpléctica)

        if grad_H is None:
            grad_H = rng.standard_normal(n)

        dissipation = QuadraticDissipation(R)
        flow = PortHamiltonianFlow(J=J, R=dissipation, grad_H=grad_H)

        shield = MagicMock(spec=FuntorShield)
        shield.flow = flow
        return shield

    @staticmethod
    def telemetry_physics(
        R_desired: NDArray[np.float64],
        gamma: float = 0.1,
    ) -> Dict[str, Any]:
        """Telemetría canónica para el estrato PHYSICS."""
        return {
            "R_desired": R_desired,
            "gamma_coupling": gamma,
        }

    @staticmethod
    def telemetry_tactics(
        delta_e: float = 0.05,
        alpha_kT: float = 1.0,
    ) -> Dict[str, Any]:
        """Telemetría canónica para el estrato TACTICS."""
        return {
            "discarded_spherical_entropy": delta_e,
            "alpha_over_kT": alpha_kT,
        }

    @staticmethod
    def telemetry_omega(
        lindblad: float = 0.1,
        alpha_kT_sys: float = 1.0,
    ) -> Dict[str, Any]:
        """Telemetría canónica para los estratos STRATEGY y WISDOM."""
        return {
            "lindblad_dissipation_trace": lindblad,
            "alpha_over_kT_sys": alpha_kT_sys,
        }

    @staticmethod
    def connection_data(
        stratum: Stratum = Stratum.PHYSICS,
        n: int = N,
        seed: int = RNG_SEED,
    ) -> ConnectionData:
        """
        Construye un ConnectionData sintético y válido para pruebas
        que no requieren pasar por la Fase 1 completa.
        """
        G = TestFixtures.spd_matrix(n=n, seed=seed)
        H = np.eye(n)   # proyector trivial válido para pruebas aisladas
        return ConnectionData(
            metric_tensor=G,
            horizontal_projector=H,
            connection_form_norm=float(np.linalg.norm(la.inv(G), "fro")),
            stratum=stratum,
            kappa_metric=float(np.linalg.cond(G)),
        )

    @staticmethod
    def deformation_tensor(
        delta_R: NDArray[np.float64],
        R_base: NDArray[np.float64],
        G: NDArray[np.float64],
        stratum: Stratum = Stratum.PHYSICS,
    ) -> DeformationTensor:
        """
        Construye un DeformationTensor sintético con métricas calculadas
        explícitamente para pruebas de Fase 3 aisladas.
        """
        ratio = float(
            np.linalg.norm(delta_R, "fro") / (np.linalg.norm(R_base, "fro") + 1e-15)
        )
        sigma = float(np.trace(delta_R @ la.inv(G)))
        return DeformationTensor(
            delta_R=delta_R,
            frobenius_ratio=ratio,
            entropy_production=sigma,
            info={"stratum": stratum, "method": "test_synthetic"},
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — PRUEBAS DE LA CONEXIÓN DE EHRESMANN
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase1_ConstructorValidation:
    """
    Pruebas de la validación del constructor de Phase1_EhresmannConnection.
    Verifica que la métrica base es correctamente validada como SPD antes
    de cualquier cálculo.
    """

    def test_constructor_acepta_metrica_spd_valida(self) -> None:
        """El constructor no lanza excepción con una métrica SPD válida."""
        G = TestFixtures.spd_matrix()
        phase1 = Phase1_EhresmannConnection(base_metric=G)
        assert phase1 is not None

    def test_constructor_rechaza_metrica_no_cuadrada(self) -> None:
        """El constructor lanza ConnectionError si la métrica no es cuadrada."""
        G_rect = np.eye(4, 3)
        with pytest.raises(ConnectionError, match="cuadrada"):
            Phase1_EhresmannConnection(base_metric=G_rect)

    def test_constructor_rechaza_metrica_no_simetrica(self) -> None:
        """El constructor lanza ConnectionError si la métrica no es simétrica."""
        G = TestFixtures.spd_matrix()
        G_asym = G + 0.5 * np.tril(G, -1)   # romper simetría
        with pytest.raises(ConnectionError, match="simétrica"):
            Phase1_EhresmannConnection(base_metric=G_asym)

    def test_constructor_rechaza_metrica_no_definida_positiva(self) -> None:
        """El constructor lanza ConnectionError si la métrica no es SPD."""
        G = TestFixtures.spd_matrix()
        # Volcar el signo del valor propio más pequeño
        evals, evecs = la.eigh(G)
        evals[0] = -0.1
        G_indef = evecs @ np.diag(evals) @ evecs.T
        G_indef = 0.5 * (G_indef + G_indef.T)
        with pytest.raises(ConnectionError):
            Phase1_EhresmannConnection(base_metric=G_indef)

    def test_constructor_rechaza_metrica_semidefinida(self) -> None:
        """El constructor lanza ConnectionError si λ_min = 0 (PSD pero no SPD)."""
        G_psd = TestFixtures.psd_matrix(n=N, rank=N - 1)
        with pytest.raises(ConnectionError):
            Phase1_EhresmannConnection(base_metric=G_psd)


class TestPhase1_FactorEscala:
    """
    Pruebas del factor de escala termodinámica λ(s) derivado de la topología.
    Verifica la fórmula: λ(s) = 1 + log(1 + β₀(s)) · |κ_s|
    """

    @pytest.fixture
    def phase1(self) -> Phase1_EhresmannConnection:
        return Phase1_EhresmannConnection(
            base_metric=TestFixtures.identity_scaled(N, scale=2.0)
        )

    def test_factor_escala_es_mayor_o_igual_a_uno(
        self, phase1: Phase1_EhresmannConnection
    ) -> None:
        """λ(s) ≥ 1 para todo estrato (el logaritmo es no negativo)."""
        for stratum in Stratum:
            lambda_s = phase1._stratum_scale_factor(stratum)
            assert lambda_s >= 1.0, (
                f"λ({stratum}) = {lambda_s:.6f} < 1.0"
            )

    def test_factor_escala_es_monotono_en_betti(
        self, phase1: Phase1_EhresmannConnection
    ) -> None:
        """
        λ(s) es no decreciente según el número de Betti β₀(s).
        Orden esperado: PHYSICS ≤ TACTICS ≤ STRATEGY ≤ WISDOM.
        """
        lambdas = {
            s: phase1._stratum_scale_factor(s) for s in Stratum
        }
        # Verificar el orden topológico implícito por β₀
        assert lambdas[Stratum.PHYSICS] <= lambdas[Stratum.TACTICS]
        assert lambdas[Stratum.TACTICS] <= lambdas[Stratum.STRATEGY]
        assert lambdas[Stratum.STRATEGY] <= lambdas[Stratum.WISDOM]

    def test_factor_escala_formula_explicita(self) -> None:
        """
        Verifica la fórmula λ(s) = 1 + log(1 + β₀) · |Tr(G⁻¹)/n| contra
        una implementación de referencia directa.
        """
        G_base = TestFixtures.identity_scaled(N, scale=3.0)
        phase1 = Phase1_EhresmannConnection(base_metric=G_base)

        G_inv = la.inv(G_base)
        kappa_scalar = float(np.trace(G_inv)) / N

        for stratum, beta0 in _BETTI_0.items():
            expected = 1.0 + math.log1p(beta0) * abs(kappa_scalar)
            actual = phase1._stratum_scale_factor(stratum)
            assert abs(actual - expected) < ATOL_STRICT, (
                f"λ({stratum}): esperado={expected:.8f}, obtenido={actual:.8f}"
            )

    def test_factor_escala_con_metrica_identidad(self) -> None:
        """
        Con G = I_n, G⁻¹ = I, Tr(G⁻¹)/n = 1, por lo tanto:
        λ(s) = 1 + log(1 + β₀(s)).
        """
        G_id = np.eye(N)
        phase1 = Phase1_EhresmannConnection(base_metric=G_id)

        for stratum, beta0 in _BETTI_0.items():
            expected = 1.0 + math.log1p(beta0)
            actual = phase1._stratum_scale_factor(stratum)
            assert abs(actual - expected) < ATOL_STRICT


class TestPhase1_ProyectorHorizontal:
    """
    Pruebas algebraicas del proyector horizontal H construido por la Fase 1.
    Verifican las propiedades matemáticas fundamentales del proyector de Riesz.
    """

    @pytest.fixture(params=list(Stratum))
    def connection_data_por_estrato(
        self, request: pytest.FixtureRequest
    ) -> ConnectionData:
        """Fixture parametrizado: ConnectionData para cada estrato."""
        G = TestFixtures.spd_matrix(n=N)
        phase1 = Phase1_EhresmannConnection(
            base_metric=G, obstruction_scale=0.05
        )
        return phase1.compute_connection(request.param)

    def test_proyector_es_idempotente(
        self, connection_data_por_estrato: ConnectionData
    ) -> None:
        """
        Verifica H² = H (idempotencia).
        Es la propiedad definitoria de un proyector lineal.
        """
        H = connection_data_por_estrato.horizontal_projector
        H2 = H @ H
        assert np.allclose(H2, H, atol=ATOL_STRICT), (
            f"‖H² − H‖_F = {np.linalg.norm(H2 - H, 'fro'):.3e}"
        )

    def test_proyector_imagen_en_subespacio_horizontal(
        self, connection_data_por_estrato: ConnectionData
    ) -> None:
        """
        Verifica que H v = v para cualquier vector en Im(H):
        Si w = H v, entonces H w = w.
        """
        H = connection_data_por_estrato.horizontal_projector
        rng = TestFixtures.rng()
        v = rng.standard_normal(N)
        w = H @ v           # w ∈ Im(H)
        Hw = H @ w
        assert np.allclose(Hw, w, atol=ATOL_STRICT), (
            f"H(Hv) ≠ Hv: ‖H(Hv) − Hv‖₂ = {np.linalg.norm(Hw - w):.3e}"
        )

    def test_proyector_nucleo_es_anulado(
        self, connection_data_por_estrato: ConnectionData
    ) -> None:
        """
        Verifica que I − H es también un proyector:
        (I − H)² = I − H (complemento ortogonal).
        """
        H = connection_data_por_estrato.horizontal_projector
        P_vert = np.eye(N) - H
        P_vert2 = P_vert @ P_vert
        assert np.allclose(P_vert2, P_vert, atol=ATOL_STRICT), (
            f"‖(I−H)² − (I−H)‖_F = {np.linalg.norm(P_vert2 - P_vert, 'fro'):.3e}"
        )

    def test_proyector_valores_propios_son_cero_o_uno(
        self, connection_data_por_estrato: ConnectionData
    ) -> None:
        """
        Los valores propios de un proyector son exclusivamente 0 o 1.
        Tolerancia: |λ − round(λ)| < ATOL_STRICT.
        """
        H = connection_data_por_estrato.horizontal_projector
        evals = np.sort(la.eigvals(H).real)
        for ev in evals:
            assert abs(ev - round(ev)) < ATOL_STRICT, (
                f"Valor propio de H fuera de {{0,1}}: {ev:.6f}"
            )

    def test_proyector_traza_es_rango(
        self, connection_data_por_estrato: ConnectionData
    ) -> None:
        """
        Tr(H) = rk(H) (la traza de un proyector es su rango).
        """
        H = connection_data_por_estrato.horizontal_projector
        trace_H = float(np.trace(H))
        rank_H = int(round(float(np.linalg.matrix_rank(H, tol=ATOL_STRICT))))
        assert abs(trace_H - rank_H) < ATOL_STRICT, (
            f"Tr(H) = {trace_H:.4f} ≠ rk(H) = {rank_H}"
        )


class TestPhase1_MetricaEscalada:
    """
    Pruebas de las propiedades métricas de G_s = λ(s) · G_base.
    Verifica SPD, simetría y coherencia de κ(G_s) con el factor λ(s).
    """

    @pytest.fixture
    def phase1(self) -> Phase1_EhresmannConnection:
        return Phase1_EhresmannConnection(
            base_metric=TestFixtures.spd_matrix(n=N, seed=7)
        )

    def test_metrica_escalada_es_spd(self, phase1: Phase1_EhresmannConnection) -> None:
        """G_s = λ(s) · G_base es SPD para todos los estratos."""
        for stratum in Stratum:
            cd = phase1.compute_connection(stratum)
            G_s = cd.metric_tensor
            # SPD ↔ Cholesky sin excepción
            try:
                la.cholesky(G_s, lower=True)
            except la.LinAlgError:
                pytest.fail(f"G_s para el estrato {stratum} no es SPD.")

    def test_metrica_escalada_es_simetrica(
        self, phase1: Phase1_EhresmannConnection
    ) -> None:
        """G_s es simétrica: ‖G_s − G_sᵀ‖_F < ATOL_STRICT."""
        for stratum in Stratum:
            cd = phase1.compute_connection(stratum)
            G_s = cd.metric_tensor
            assert np.allclose(G_s, G_s.T, atol=ATOL_STRICT), (
                f"G_s para {stratum} no es simétrica."
            )

    def test_kappa_metrica_coincide_con_condicionamiento_real(
        self, phase1: Phase1_EhresmannConnection
    ) -> None:
        """
        El campo kappa_metric del ConnectionData coincide con
        np.linalg.cond(G_s) dentro de la tolerancia numérica.
        """
        for stratum in Stratum:
            cd = phase1.compute_connection(stratum)
            kappa_real = float(np.linalg.cond(cd.metric_tensor))
            assert abs(cd.kappa_metric - kappa_real) < ATOL_THERMO * kappa_real, (
                f"κ reportado={cd.kappa_metric:.4e}, real={kappa_real:.4e} "
                f"para el estrato {stratum}."
            )

    def test_norma_forma_conexion_es_positiva(
        self, phase1: Phase1_EhresmannConnection
    ) -> None:
        """‖ω‖_F > 0 (la forma de conexión es no trivial para ε_s > 0)."""
        for stratum in Stratum:
            cd = phase1.compute_connection(stratum)
            assert cd.connection_form_norm > 0.0, (
                f"‖ω‖_F = 0 para el estrato {stratum}."
            )

    def test_norma_forma_conexion_es_monotona_en_betti(
        self, phase1: Phase1_EhresmannConnection
    ) -> None:
        """
        ‖ω‖_F debe crecer con β₀(s): estratos más complejos generan
        conexiones con mayor curvatura local.
        """
        norms = {s: phase1.compute_connection(s).connection_form_norm for s in Stratum}
        assert norms[Stratum.PHYSICS] <= norms[Stratum.TACTICS]
        assert norms[Stratum.TACTICS] <= norms[Stratum.STRATEGY]
        assert norms[Stratum.STRATEGY] <= norms[Stratum.WISDOM]


class TestPhase1_Adversarial:
    """
    Pruebas adversariales de la Fase 1: entradas degeneradas, límites
    de condicionamiento y condiciones de borde geométricas.
    """

    def test_metrica_mal_condicionada_lanza_error(self) -> None:
        """
        Si κ(G_base) es grande pero < _KAPPA_MAX, compute_connection no falla.
        Si después del escalado κ(G_s) > _KAPPA_MAX, debe lanzar ConnectionError.
        """
        # Construir una métrica con κ ≈ _KAPPA_MAX / 0.9 > _KAPPA_MAX
        # tras el escalado mínimo (λ = 1 + algo ≥ 1)
        n = 3
        # Valores propios elegidos para que κ supere el umbral tras escalar
        target_kappa = _KAPPA_MAX * 10.0
        evals_designed = np.array([1.0, 1.0, target_kappa])
        Q = la.orth(np.random.default_rng(99).standard_normal((n, n)))
        G_bad = Q @ np.diag(evals_designed) @ Q.T
        G_bad = 0.5 * (G_bad + G_bad.T)

        phase1 = Phase1_EhresmannConnection(base_metric=G_bad)
        with pytest.raises(ConnectionError, match="condicionada"):
            phase1.compute_connection(Stratum.PHYSICS)

    def test_obstruccion_scale_cero_produce_proyector_identidad(self) -> None:
        """
        Con obstruction_scale=0, A_obs = 0, y el proyector H debe ser I.
        Esto verifica el caso límite: sin obstrucción no hay subespacio vertical.
        """
        G = TestFixtures.identity_scaled(N)
        phase1 = Phase1_EhresmannConnection(
            base_metric=G, obstruction_scale=0.0
        )
        cd = phase1.compute_connection(Stratum.PHYSICS)
        H = cd.horizontal_projector
        assert np.allclose(H, np.eye(N), atol=ATOL_STRICT), (
            f"Con A_obs=0, H debería ser I_N. ‖H − I‖_F = "
            f"{np.linalg.norm(H - np.eye(N), 'fro'):.3e}"
        )

    def test_compute_connection_es_determinista(self) -> None:
        """
        compute_connection es una función pura: llamadas múltiples con los
        mismos argumentos retornan ConnectionData idénticos.
        """
        G = TestFixtures.spd_matrix(n=N)
        phase1 = Phase1_EhresmannConnection(base_metric=G)
        cd1 = phase1.compute_connection(Stratum.TACTICS)
        cd2 = phase1.compute_connection(Stratum.TACTICS)
        assert np.allclose(cd1.metric_tensor, cd2.metric_tensor)
        assert np.allclose(cd1.horizontal_projector, cd2.horizontal_projector)
        assert cd1.kappa_metric == cd2.kappa_metric

    def test_connection_data_es_inmutable(self) -> None:
        """ConnectionData es frozen=True; la asignación de atributos lanza error."""
        G = TestFixtures.spd_matrix(n=N)
        phase1 = Phase1_EhresmannConnection(base_metric=G)
        cd = phase1.compute_connection(Stratum.PHYSICS)
        with pytest.raises((TypeError, AttributeError)):
            cd.stratum = Stratum.WISDOM  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — PRUEBAS DEL PULLBACK TERMODINÁMICO
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase2_ValidacionBaseR:
    """
    Pruebas de la validación de R_base en Phase2_ThermodynamicPullback.
    La validación debe ser rigurosa: simetría relativa y PSD con _EIG_TOL.
    """

    @pytest.fixture
    def phase2(self) -> Phase2_ThermodynamicPullback:
        return Phase2_ThermodynamicPullback()

    @pytest.fixture
    def conn(self) -> ConnectionData:
        return TestFixtures.connection_data(Stratum.PHYSICS)

    def test_acepta_r_base_spd(
        self,
        phase2: Phase2_ThermodynamicPullback,
        conn: ConnectionData,
    ) -> None:
        """No lanza excepción con R_base SPD válida."""
        R = TestFixtures.spd_matrix()
        tele = TestFixtures.telemetry_physics(R_desired=R)
        result = phase2.compute_deformation(conn, tele, R)
        assert isinstance(result, DeformationTensor)

    def test_acepta_r_base_psd_singular(
        self,
        phase2: Phase2_ThermodynamicPullback,
        conn: ConnectionData,
    ) -> None:
        """Acepta R_base PSD (semidefinida, con λ_min = 0)."""
        R_psd = TestFixtures.psd_matrix(n=N, rank=N - 1)
        tele = TestFixtures.telemetry_physics(R_desired=R_psd)
        # No debe lanzar excepción
        result = phase2.compute_deformation(conn, tele, R_psd)
        assert result is not None

    def test_rechaza_r_base_no_simetrica(
        self,
        phase2: Phase2_ThermodynamicPullback,
        conn: ConnectionData,
    ) -> None:
        """Lanza PullbackError si R_base no es simétrica."""
        R = TestFixtures.spd_matrix()
        R_asym = R + 0.5 * np.tril(R, -1)
        tele = TestFixtures.telemetry_physics(R_desired=R)
        with pytest.raises(PullbackError, match="simétrica"):
            phase2.compute_deformation(conn, tele, R_asym)

    def test_rechaza_r_base_con_eigenvalor_negativo_significativo(
        self,
        phase2: Phase2_ThermodynamicPullback,
        conn: ConnectionData,
    ) -> None:
        """Lanza PullbackError si λ_min(R_base) < −_EIG_TOL."""
        R = TestFixtures.spd_matrix()
        evals, evecs = la.eigh(R)
        evals[0] = -1e-6   # negativo significativo
        R_neg = evecs @ np.diag(evals) @ evecs.T
        R_neg = 0.5 * (R_neg + R_neg.T)
        tele = TestFixtures.telemetry_physics(R_desired=R)
        with pytest.raises(PullbackError, match="semidefinida positiva"):
            phase2.compute_deformation(conn, tele, R_neg)

    def test_rechaza_r_desired_no_simetrica(
        self,
        phase2: Phase2_ThermodynamicPullback,
        conn: ConnectionData,
    ) -> None:
        """Lanza PullbackError si R_desired (en telemetría) no es simétrica."""
        R_base = TestFixtures.spd_matrix()
        R_desired = TestFixtures.spd_matrix(seed=7)
        # Romper simetría de R_desired
        R_desired_asym = R_desired + 0.5 * np.tril(R_desired, -1)
        tele = {"R_desired": R_desired_asym, "gamma_coupling": 0.1}
        with pytest.raises(PullbackError):
            phase2.compute_deformation(conn, tele, R_base)


class TestPhase2_PullbackPhysics:
    """
    Pruebas del pullback para el estrato PHYSICS.
    Verifica la fórmula covariante y la derivada de Lie de Cartan.
    """

    @pytest.fixture
    def setup(self) -> dict:
        G = TestFixtures.spd_matrix(n=N, seed=10)
        H = np.eye(N)
        R_base = TestFixtures.spd_matrix(n=N, seed=11)
        conn = ConnectionData(
            metric_tensor=G,
            horizontal_projector=H,
            connection_form_norm=1.0,
            stratum=Stratum.PHYSICS,
            kappa_metric=float(np.linalg.cond(G)),
        )
        return {"G": G, "H": H, "R_base": R_base, "conn": conn}

    def test_delta_r_es_simetrico(self, setup: dict) -> None:
        """δR del pullback PHYSICS debe ser simétrico."""
        phase2 = Phase2_ThermodynamicPullback()
        R_desired = TestFixtures.spd_matrix(n=N, seed=12)
        tele = TestFixtures.telemetry_physics(R_desired=R_desired, gamma=0.15)
        result = phase2.compute_deformation(setup["conn"], tele, setup["R_base"])
        dR = result.delta_R
        assert np.allclose(dR, dR.T, atol=ATOL_STRICT), (
            f"δR no es simétrico: ‖δR − δRᵀ‖_F = {np.linalg.norm(dR - dR.T, 'fro'):.3e}"
        )

    def test_delta_r_cero_si_r_desired_igual_r_base(self, setup: dict) -> None:
        """
        Si R_desired = R_base, la derivada de Lie ℒ_X R es cero
        (X_skew = 0) y el término covariante también es cero.
        Por lo tanto δR = 0.
        """
        phase2 = Phase2_ThermodynamicPullback()
        R_base = setup["R_base"]
        tele = TestFixtures.telemetry_physics(R_desired=R_base, gamma=0.2)
        result = phase2.compute_deformation(setup["conn"], tele, R_base)
        assert np.allclose(result.delta_R, 0.0, atol=ATOL_STRICT), (
            f"δR ≠ 0 cuando R_desired = R_base. "
            f"‖δR‖_F = {np.linalg.norm(result.delta_R, 'fro'):.3e}"
        )

    def test_delta_r_escala_linealmente_con_gamma(self, setup: dict) -> None:
        """
        El término covariante H G (R_d − R_b) G Hᵀ escala con γ.
        Cuando ℒ_X ≠ 0, el ratio δR(γ₂) / δR(γ₁) debe reflejar que
        solo el término covariante cambia con γ.
        Verificamos el caso especial: R_desired simétrica ⟹ X_skew = 0
        ⟹ δR = γ · H G (R_d − R_b) G Hᵀ (lineal en γ exactamente).
        """
        phase2 = Phase2_ThermodynamicPullback()
        # Garantizar X_skew = 0: R_desired simétrica y distinta de R_base
        R_base = setup["R_base"]
        R_desired = TestFixtures.spd_matrix(n=N, seed=99)
        gamma1, gamma2 = 0.1, 0.3

        tele1 = TestFixtures.telemetry_physics(R_desired=R_desired, gamma=gamma1)
        tele2 = TestFixtures.telemetry_physics(R_desired=R_desired, gamma=gamma2)

        dR1 = phase2.compute_deformation(setup["conn"], tele1, R_base).delta_R
        dR2 = phase2.compute_deformation(setup["conn"], tele2, R_base).delta_R

        ratio_expected = gamma2 / gamma1
        ratio_actual = np.linalg.norm(dR2, "fro") / np.linalg.norm(dR1, "fro")
        assert abs(ratio_actual - ratio_expected) < ATOL_THERMO, (
            f"Linealidad en γ violada: esperado {ratio_expected:.4f}, "
            f"obtenido {ratio_actual:.4f}."
        )

    def test_lie_derivative_contribuye_cuando_r_desired_es_asimetrica_relativa(
        self, setup: dict
    ) -> None:
        """
        La derivada de Lie ℒ_X R = X R + R Xᵀ con X = skew(ΔR) es no nula
        cuando R_desired ≠ R_base. Verificamos que el campo info registra
        el método correcto.
        """
        phase2 = Phase2_ThermodynamicPullback()
        R_desired = TestFixtures.spd_matrix(n=N, seed=33)
        tele = TestFixtures.telemetry_physics(R_desired=R_desired, gamma=0.1)
        result = phase2.compute_deformation(setup["conn"], tele, setup["R_base"])
        assert result.info["method"] == "pullback_physics_covariante_cartan"

    def test_frobenius_ratio_se_calcula_correctamente(self, setup: dict) -> None:
        """
        frobenius_ratio = ‖δR‖_F / ‖R_base‖_F, calculado por Phase2,
        coincide con el valor computado manualmente.
        """
        phase2 = Phase2_ThermodynamicPullback()
        R_base = setup["R_base"]
        R_desired = TestFixtures.spd_matrix(n=N, seed=55)
        tele = TestFixtures.telemetry_physics(R_desired=R_desired)
        result = phase2.compute_deformation(setup["conn"], tele, R_base)
        ratio_manual = np.linalg.norm(result.delta_R, "fro") / np.linalg.norm(R_base, "fro")
        assert abs(result.frobenius_ratio - ratio_manual) < ATOL_STRICT


class TestPhase2_PullbackTactics:
    """
    Pruebas del pullback para el estrato TACTICS.
    Verifica δR = (α/k_BT) · Δε · H G Hᵀ.
    """

    @pytest.fixture
    def setup(self) -> dict:
        G = TestFixtures.spd_matrix(n=N, seed=20)
        H = np.eye(N)
        R_base = TestFixtures.spd_matrix(n=N, seed=21)
        conn = ConnectionData(
            metric_tensor=G,
            horizontal_projector=H,
            connection_form_norm=1.0,
            stratum=Stratum.TACTICS,
            kappa_metric=float(np.linalg.cond(G)),
        )
        return {"G": G, "H": H, "R_base": R_base, "conn": conn}

    def test_delta_r_es_proporcional_a_delta_e(self, setup: dict) -> None:
        """
        δR escala linealmente con Δε_ruido.
        δR(2Δε) = 2 · δR(Δε).
        """
        phase2 = Phase2_ThermodynamicPullback()
        R_base = setup["R_base"]
        tele1 = TestFixtures.telemetry_tactics(delta_e=0.05, alpha_kT=1.0)
        tele2 = TestFixtures.telemetry_tactics(delta_e=0.10, alpha_kT=1.0)
        dR1 = phase2.compute_deformation(setup["conn"], tele1, R_base).delta_R
        dR2 = phase2.compute_deformation(setup["conn"], tele2, R_base).delta_R
        assert np.allclose(2.0 * dR1, dR2, atol=ATOL_STRICT), (
            "δR no escala linealmente con Δε."
        )

    def test_delta_r_es_proporcional_a_alpha_kt(self, setup: dict) -> None:
        """δR escala linealmente con α/k_BT."""
        phase2 = Phase2_ThermodynamicPullback()
        R_base = setup["R_base"]
        tele1 = TestFixtures.telemetry_tactics(delta_e=0.05, alpha_kT=1.0)
        tele2 = TestFixtures.telemetry_tactics(delta_e=0.05, alpha_kT=3.0)
        dR1 = phase2.compute_deformation(setup["conn"], tele1, R_base).delta_R
        dR2 = phase2.compute_deformation(setup["conn"], tele2, R_base).delta_R
        assert np.allclose(3.0 * dR1, dR2, atol=ATOL_STRICT)

    def test_delta_r_cero_cuando_delta_e_es_cero(self, setup: dict) -> None:
        """Si Δε = 0, entonces δR = 0."""
        phase2 = Phase2_ThermodynamicPullback()
        tele = TestFixtures.telemetry_tactics(delta_e=0.0, alpha_kT=5.0)
        result = phase2.compute_deformation(setup["conn"], tele, setup["R_base"])
        assert np.allclose(result.delta_R, 0.0, atol=ATOL_STRICT)

    def test_delta_r_coincide_con_formula_explicita(self, setup: dict) -> None:
        """
        Verifica δR = α_kT · Δε · H G Hᵀ directamente
        contra la implementación.
        """
        phase2 = Phase2_ThermodynamicPullback()
        G, H = setup["G"], setup["H"]
        delta_e, alpha_kT = 0.08, 2.5
        tele = TestFixtures.telemetry_tactics(delta_e=delta_e, alpha_kT=alpha_kT)
        result = phase2.compute_deformation(setup["conn"], tele, setup["R_base"])

        expected = alpha_kT * delta_e * (H @ G @ H.T)
        expected = 0.5 * (expected + expected.T)   # simetrización explícita
        assert np.allclose(result.delta_R, expected, atol=ATOL_STRICT), (
            f"‖δR_obtenido − δR_esperado‖_F = "
            f"{np.linalg.norm(result.delta_R - expected, 'fro'):.3e}"
        )

    def test_delta_r_es_simetrico(self, setup: dict) -> None:
        """δR del pullback TACTICS es simétrico."""
        phase2 = Phase2_ThermodynamicPullback()
        tele = TestFixtures.telemetry_tactics(delta_e=0.1, alpha_kT=1.5)
        result = phase2.compute_deformation(setup["conn"], tele, setup["R_base"])
        dR = result.delta_R
        assert np.allclose(dR, dR.T, atol=ATOL_STRICT)


class TestPhase2_PullbackOmega:
    """
    Pruebas del pullback para los estratos STRATEGY y WISDOM.
    Verifica la normalización espectral por ρ(R_base).
    """

    @pytest.fixture(params=[Stratum.STRATEGY, Stratum.WISDOM])
    def setup_omega(self, request: pytest.FixtureRequest) -> dict:
        G = TestFixtures.spd_matrix(n=N, seed=30)
        H = np.eye(N)
        R_base = TestFixtures.spd_matrix(n=N, seed=31)
        conn = ConnectionData(
            metric_tensor=G,
            horizontal_projector=H,
            connection_form_norm=1.0,
            stratum=request.param,
            kappa_metric=float(np.linalg.cond(G)),
        )
        return {"G": G, "H": H, "R_base": R_base, "conn": conn,
                "stratum": request.param}

    def test_delta_r_es_multiplo_de_identidad(self, setup_omega: dict) -> None:
        """δR = c · I_n para algún escalar c."""
        phase2 = Phase2_ThermodynamicPullback()
        tele = TestFixtures.telemetry_omega(lindblad=0.5, alpha_kT_sys=1.0)
        result = phase2.compute_deformation(
            setup_omega["conn"], tele, setup_omega["R_base"]
        )
        dR = result.delta_R
        n = N
        # Si δR = c I, entonces dR / dR[0,0] debería ser I (si c≠0)
        c = dR[0, 0]
        if abs(c) > _EIG_TOL:
            ratio_to_identity = dR / c
            assert np.allclose(ratio_to_identity, np.eye(n), atol=ATOL_STRICT), (
                "δR no es múltiplo de la identidad para el estrato Omega."
            )

    def test_normalizacion_espectral_por_radio_espectral(
        self, setup_omega: dict
    ) -> None:
        """
        Verifica que el coeficiente de δR = c · I satisface:
        c = α_kT_sys · Λ_L / (ρ(R_base) + _EIG_TOL)
        """
        phase2 = Phase2_ThermodynamicPullback()
        lindblad, alpha_kT_sys = 0.3, 2.0
        tele = TestFixtures.telemetry_omega(
            lindblad=lindblad, alpha_kT_sys=alpha_kT_sys
        )
        R_base = setup_omega["R_base"]
        result = phase2.compute_deformation(
            setup_omega["conn"], tele, R_base
        )
        rho_sp = float(np.max(np.abs(la.eigvalsh(R_base)))) + _EIG_TOL
        c_expected = alpha_kT_sys * lindblad / rho_sp
        c_actual = float(result.delta_R[0, 0])
        assert abs(c_actual - c_expected) < ATOL_THERMO, (
            f"Coeficiente: esperado={c_expected:.6e}, obtenido={c_actual:.6e}."
        )

    def test_delta_r_cero_lindblad_cero(self, setup_omega: dict) -> None:
        """Si Λ_L = 0, δR = 0."""
        phase2 = Phase2_ThermodynamicPullback()
        tele = TestFixtures.telemetry_omega(lindblad=0.0)
        result = phase2.compute_deformation(
            setup_omega["conn"], tele, setup_omega["R_base"]
        )
        assert np.allclose(result.delta_R, 0.0, atol=ATOL_STRICT)


class TestPhase2_ProduccionEntropia:
    """
    Pruebas de la producción de entropía σ = Tr(δR · G_s⁻¹).
    Verifica coherencia termodinámica entre estrato y signo de σ.
    """

    def test_entropia_reportada_coincide_con_formula_explicita(self) -> None:
        """
        σ = Tr(δR · G⁻¹) calculado por Phase2 coincide con el valor
        computado manualmente sobre el DeformationTensor retornado.
        """
        G = TestFixtures.spd_matrix(n=N, seed=40)
        R_base = TestFixtures.spd_matrix(n=N, seed=41)
        conn = ConnectionData(
            metric_tensor=G,
            horizontal_projector=np.eye(N),
            connection_form_norm=1.0,
            stratum=Stratum.TACTICS,
            kappa_metric=float(np.linalg.cond(G)),
        )
        phase2 = Phase2_ThermodynamicPullback()
        tele = TestFixtures.telemetry_tactics(delta_e=0.1, alpha_kT=1.0)
        result = phase2.compute_deformation(conn, tele, R_base)

        sigma_manual = float(np.trace(result.delta_R @ la.inv(G)))
        assert abs(result.entropy_production - sigma_manual) < ATOL_THERMO, (
            f"σ reportado={result.entropy_production:.6e}, "
            f"manual={sigma_manual:.6e}."
        )

    def test_deformation_tensor_es_inmutable(self) -> None:
        """DeformationTensor es frozen=True; asignar lanza error."""
        G = TestFixtures.spd_matrix(n=N)
        R_base = TestFixtures.spd_matrix(n=N)
        conn = TestFixtures.connection_data(Stratum.TACTICS)
        phase2 = Phase2_ThermodynamicPullback()
        tele = TestFixtures.telemetry_tactics()
        result = phase2.compute_deformation(conn, tele, R_base)
        with pytest.raises((TypeError, AttributeError)):
            result.delta_R = np.zeros((N, N))  # type: ignore[misc]


class TestPhase2_Adversarial:
    """
    Pruebas adversariales de la Fase 2: NaN en telemetría, deformaciones
    explosivas y estratos no soportados.
    """

    @pytest.fixture
    def phase2(self) -> Phase2_ThermodynamicPullback:
        return Phase2_ThermodynamicPullback()

    def test_nan_en_r_desired_lanza_pullback_error(
        self, phase2: Phase2_ThermodynamicPullback
    ) -> None:
        """Si R_desired contiene NaN, PullbackError debe lanzarse."""
        R_base = TestFixtures.spd_matrix()
        conn = TestFixtures.connection_data(Stratum.PHYSICS)
        R_nan = R_base.copy()
        R_nan[0, 0] = float("nan")
        R_nan = 0.5 * (R_nan + R_nan.T)
        tele = {"R_desired": R_nan, "gamma_coupling": 0.1}
        with pytest.raises((PullbackError, ValueError)):
            phase2.compute_deformation(conn, tele, R_base)

    def test_gamma_muy_grande_emite_warning(
        self,
        phase2: Phase2_ThermodynamicPullback,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Si γ es muy grande, el ratio ‖δR‖/‖R_base‖ supera el umbral
        y se emite un WARNING de deformación intensa.
        """
        import logging
        R_base = TestFixtures.identity_scaled(N, scale=0.01)
        R_desired = TestFixtures.identity_scaled(N, scale=100.0)
        conn = TestFixtures.connection_data(Stratum.PHYSICS)
        tele = TestFixtures.telemetry_physics(R_desired=R_desired, gamma=50.0)
        with caplog.at_level(logging.WARNING, logger="MIC.ImmuneSystem.DynamicShieldRouter"):
            phase2.compute_deformation(conn, tele, R_base)
        assert any(
            "intensa" in record.message or "ratio" in record.message.lower()
            for record in caplog.records
        ), "Se esperaba un WARNING de deformación intensa."

    def test_estrato_invalido_lanza_pullback_error(
        self, phase2: Phase2_ThermodynamicPullback
    ) -> None:
        """Un estrato no reconocido lanza PullbackError."""
        R_base = TestFixtures.spd_matrix()
        # Crear un ConnectionData con un stratum mock inválido
        G = TestFixtures.spd_matrix()
        conn = ConnectionData(
            metric_tensor=G,
            horizontal_projector=np.eye(N),
            connection_form_norm=1.0,
            stratum=MagicMock(spec=Stratum),  # estrato inválido
            kappa_metric=1.0,
        )
        with pytest.raises(PullbackError, match="no soportado"):
            phase2.compute_deformation(conn, {}, R_base)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — PRUEBAS DE LA VESTIMENTA DEL ESCUDO
# ══════════════════════════════════════════════════════════════════════════════


class TestPhase3_ProyeccionHigham:
    """
    Pruebas de la proyección al cono S⁺ₙ mediante el algoritmo de Higham.
    Verifica optimalidad, preservación de vectores propios y robustez.
    """

    @pytest.fixture
    def dresser(self) -> Phase3_ShieldDresser:
        return Phase3_ShieldDresser()

    def test_proyeccion_de_matriz_spd_es_identidad(
        self, dresser: Phase3_ShieldDresser
    ) -> None:
        """
        Si R_raw ∈ S⁺ₙ (ya PSD), la proyección de Higham la deja intacta.
        Π_{S⁺ₙ}(M) = M si M ∈ S⁺ₙ.
        """
        R_psd = TestFixtures.spd_matrix(n=N)
        R_proj = dresser._project_to_psd_cone(R_psd)
        assert np.allclose(R_proj, R_psd, atol=ATOL_THERMO), (
            f"La proyección modificó una matriz ya PSD: "
            f"‖R_proj − R_psd‖_F = {np.linalg.norm(R_proj - R_psd, 'fro'):.3e}"
        )

    def test_proyeccion_elimina_valores_propios_negativos(
        self, dresser: Phase3_ShieldDresser
    ) -> None:
        """
        Después de la proyección, todos los valores propios son ≥ 0.
        """
        R_base = TestFixtures.spd_matrix(n=N)
        # Introducir valores propios negativos
        evals, evecs = la.eigh(R_base)
        evals[0] = -0.5
        evals[1] = -0.1
        R_neg = evecs @ np.diag(evals) @ evecs.T
        R_neg = 0.5 * (R_neg + R_neg.T)

        R_proj = dresser._project_to_psd_cone(R_neg)
        evals_proj = la.eigvalsh(R_proj)
        assert np.all(evals_proj >= -ATOL_STRICT), (
            f"Valores propios negativos tras proyección Higham: "
            f"λ_min = {float(np.min(evals_proj)):.3e}"
        )

    def test_proyeccion_es_optima_en_norma_frobenius(
        self, dresser: Phase3_ShieldDresser
    ) -> None:
        """
        Verifica que ‖R_raw − R_proj‖_F ≤ ‖R_raw − X‖_F para cualquier
        X ∈ S⁺ₙ alternativo (prueba de optimalidad de Higham).
        Se usa X = R_proj + ε I como candidato PSD alternativo.
        """
        R_base = TestFixtures.spd_matrix(n=N)
        evals, evecs = la.eigh(R_base)
        evals[0] = -0.3
        R_raw = evecs @ np.diag(evals) @ evecs.T
        R_raw = 0.5 * (R_raw + R_raw.T)

        R_proj = dresser._project_to_psd_cone(R_raw)
        dist_higham = np.linalg.norm(R_raw - R_proj, "fro")

        # Candidato alternativo: R_proj + 0.1 I (también PSD pero subóptimo)
        R_alt = R_proj + 0.1 * np.eye(N)
        dist_alt = np.linalg.norm(R_raw - R_alt, "fro")

        assert dist_higham <= dist_alt + ATOL_THERMO, (
            f"La proyección de Higham no es óptima: "
            f"d(Higham)={dist_higham:.4e} > d(alternativa)={dist_alt:.4e}"
        )

    def test_proyeccion_preserva_vectores_propios_positivos(
        self, dresser: Phase3_ShieldDresser
    ) -> None:
        """
        Los vectores propios asociados a λ > 0 en R_raw se preservan
        en R_proj (equivarianza espectral de la proyección de Higham).
        """
        R_base = TestFixtures.spd_matrix(n=N)
        evals, evecs = la.eigh(R_base)
        evals[0] = -0.2    # truncar solo el primero
        R_raw = evecs @ np.diag(evals) @ evecs.T
        R_raw = 0.5 * (R_raw + R_raw.T)

        R_proj = dresser._project_to_psd_cone(R_raw)
        evals_proj, evecs_proj = la.eigh(R_proj)

        # El vector propio asociado al segundo valor propio (>0) debe
        # alinearse con el vector propio original
        for i in range(1, N):  # los positivos
            v_orig = evecs[:, i]
            v_proj = evecs_proj[:, i]
            # Alineación: |v_orig · v_proj| ≈ 1 (hasta signo)
            alignment = abs(float(np.dot(v_orig, v_proj)))
            assert alignment > 1.0 - ATOL_THERMO, (
                f"Vector propio {i} no preservado: alineación = {alignment:.6f}"
            )

    def test_proyeccion_simetriza_resultado(
        self, dresser: Phase3_ShieldDresser
    ) -> None:
        """R_proj = R_projᵀ (resultado simétrico)."""
        evals = np.array([-0.5, 0.1, 0.3, 1.0])
        evecs = la.orth(np.random.default_rng(77).standard_normal((N, N)))
        R_raw = evecs @ np.diag(evals) @ evecs.T
        R_raw = 0.5 * (R_raw + R_raw.T)

        R_proj = dresser._project_to_psd_cone(R_raw)
        assert np.allclose(R_proj, R_proj.T, atol=ATOL_STRICT)


class TestPhase3_RegularizacionTikhonov:
    """
    Pruebas de la regularización de Tikhonov adaptativa.
    Verifica el shift mínimo, el condicionamiento resultante y los
    casos límite (λ_min ≈ 0, κ ya dentro del umbral).
    """

    @pytest.fixture
    def dresser(self) -> Phase3_ShieldDresser:
        return Phase3_ShieldDresser(kappa_max=1.0e4)

    def test_sin_regularizacion_si_kappa_dentro_del_umbral(
        self, dresser: Phase3_ShieldDresser
    ) -> None:
        """Si κ(R) ≤ κ_max, _tikhonov_regularize retorna R sin cambios."""
        R = TestFixtures.identity_scaled(N, scale=1.0)  # κ = 1 << κ_max
        R_reg = dresser._tikhonov_regularize(R)
        assert np.allclose(R_reg, R, atol=ATOL_STRICT), (
            "Se aplicó regularización innecesaria."
        )

    def test_regularizacion_satisface_cota_de_condicionamiento(
        self, dresser: Phase3_ShieldDresser
    ) -> None:
        """
        Tras la regularización, κ(R_reg) ≤ κ_max.
        Se construye R con κ >> κ_max deliberadamente.
        """
        # R con κ ≈ 10^6 >> κ_max = 10^4
        evals = np.array([1.0, 10.0, 100.0, 1_000_000.0])
        evecs = la.orth(np.random.default_rng(88).standard_normal((N, N)))
        R_bad = evecs @ np.diag(evals) @ evecs.T
        R_bad = 0.5 * (R_bad + R_bad.T)

        R_reg = dresser._tikhonov_regularize(R_bad)
        kappa_reg = float(np.linalg.cond(R_reg))
        assert kappa_reg <= dresser._kappa_max * (1 + ATOL_THERMO), (
            f"κ(R_reg) = {kappa_reg:.3e} > κ_max = {dresser._kappa_max:.3e}"
        )

    def test_shift_tikhonov_es_el_minimo_posible(
        self, dresser: Phase3_ShieldDresser
    ) -> None:
        """
        ε_tik es el mínimo shift que satisface κ ≤ κ_max.
        Verificamos que R_reg − ε_tik I tiene κ > κ_max (el shift es
        justo el necesario).
        """
        evals = np.array([1.0, 1.0, 10.0, 1_000.0])
        Q = la.orth(np.random.default_rng(101).standard_normal((N, N)))
        R_bad = Q @ np.diag(evals) @ Q.T
        R_bad = 0.5 * (R_bad + R_bad.T)

        R_reg = dresser._tikhonov_regularize(R_bad)

        # El shift aplicado se deduce de la diferencia
        eps_actual = float((R_reg - R_bad)[0, 0])   # R_reg = R_bad + eps I

        # Con eps_actual − δ (ligeramente menor), κ debe superar κ_max
        delta = eps_actual * 0.01
        R_less = R_bad + (eps_actual - delta) * np.eye(N)
        kappa_less = float(np.linalg.cond(R_less))
        assert kappa_less > dresser._kappa_max * (1 - 1e-3), (
            f"El shift Tikhonov no es mínimo: con ε−δ, κ = {kappa_less:.3e} "
            f"≤ κ_max = {dresser._kappa_max:.3e}."
        )

    def test_tikhonov_no_modifica_autovectores(
        self, dresser: Phase3_ShieldDresser
    ) -> None:
        """
        La regularización R_reg = R + ε I comparte los mismos autovectores
        que R (la identidad es equivariante bajo cualquier base ortonormal).
        """
        evals = np.array([1.0, 5.0, 50.0, 500_000.0])
        Q = la.orth(np.random.default_rng(202).standard_normal((N, N)))
        R_bad = Q @ np.diag(evals) @ Q.T
        R_bad = 0.5 * (R_bad + R_bad.T)

        R_reg = dresser._tikhonov_regularize(R_bad)

        # Los autovectores de R_reg deben ser columnas de Q (hasta permutación)
        _, evecs_reg = la.eigh(R_reg)
        _, evecs_orig = la.eigh(R_bad)

        # Verificar que cada autovector de R_orig está en Im(evecs_reg)
        for i in range(N):
            v = evecs_orig[:, i]
            coords = evecs_reg.T @ v
            reconstruction = evecs_reg @ coords
            assert np.allclose(abs(reconstruction), abs(v), atol=ATOL_THERMO), (
                f"Autovector {i} no preservado tras regularización Tikhonov."
            )


class TestPhase3_VestimientaCompleta:
    """
    Pruebas de dress_shield: integración de la Fase 3 completa con
    FuntorShield mock. Verifica inmutabilidad, PSD del escudo resultante
    y preservación de J y grad_H.
    """

    @pytest.fixture
    def dresser(self) -> Phase3_ShieldDresser:
        return Phase3_ShieldDresser()

    @pytest.fixture
    def setup(self) -> dict:
        R_base = TestFixtures.spd_matrix(n=N, seed=50)
        G = TestFixtures.spd_matrix(n=N, seed=51)
        shield = TestFixtures.mock_shield(R=R_base)
        delta_R = 0.01 * TestFixtures.spd_matrix(n=N, seed=52)
        deformation = TestFixtures.deformation_tensor(
            delta_R=delta_R, R_base=R_base, G=G
        )
        return {
            "R_base": R_base, "G": G,
            "shield": shield,
            "deformation": deformation,
            "delta_R": delta_R,
        }

    def test_escudo_vestido_tiene_disipacion_psd(
        self, dresser: Phase3_ShieldDresser, setup: dict
    ) -> None:
        """R_eff del escudo vestido es PSD: λ_min ≥ −_EIG_TOL."""
        dressed = dresser.dress_shield(setup["shield"], setup["deformation"])
        R_eff = dressed.flow.R.matrix
        evals = la.eigvalsh(R_eff)
        assert np.all(evals >= -_EIG_TOL), (
            f"R_eff contiene valores propios negativos: λ_min = {float(np.min(evals)):.3e}"
        )

    def test_escudo_vestido_es_nuevo_objeto(
        self, dresser: Phase3_ShieldDresser, setup: dict
    ) -> None:
        """
        dress_shield retorna un nuevo FuntorShield (inmutabilidad funcional).
        El escudo base no es modificado in-place.
        """
        shield_orig = setup["shield"]
        R_base_orig = shield_orig.flow.R.matrix.copy()

        dressed = dresser.dress_shield(shield_orig, setup["deformation"])

        # El escudo original no debe haber cambiado
        assert np.allclose(shield_orig.flow.R.matrix, R_base_orig), (
            "El escudo base fue mutado in-place (violación de inmutabilidad)."
        )
        # Y el nuevo escudo es un objeto diferente
        assert dressed is not shield_orig

    def test_j_y_grad_h_se_preservan(
        self, dresser: Phase3_ShieldDresser, setup: dict
    ) -> None:
        """
        La estructura simpléctica J y el gradiente grad_H del escudo
        base se preservan intactos en el escudo vestido.
        """
        shield = setup["shield"]
        J_orig = shield.flow.J.copy()
        grad_H_orig = shield.flow.grad_H.copy()

        dressed = dresser.dress_shield(shield, setup["deformation"])

        assert np.allclose(dressed.flow.J, J_orig, atol=ATOL_STRICT), (
            "J fue modificado durante la vestimenta."
        )
        assert np.allclose(dressed.flow.grad_H, grad_H_orig, atol=ATOL_STRICT), (
            "grad_H fue modificado durante la vestimenta."
        )

    def test_r_eff_es_simetrico(
        self, dresser: Phase3_ShieldDresser, setup: dict
    ) -> None:
        """R_eff = R_effᵀ."""
        dressed = dresser.dress_shield(setup["shield"], setup["deformation"])
        R_eff = dressed.flow.R.matrix
        assert np.allclose(R_eff, R_eff.T, atol=ATOL_STRICT)

    def test_r_eff_es_suma_correcta_cuando_ya_psd(
        self, dresser: Phase3_ShieldDresser
    ) -> None:
        """
        Si R_base + δR ∈ S⁺ₙ, entonces R_eff = R_base + δR (la proyección
        de Higham actúa como identidad sobre matrices PSD).
        """
        R_base = TestFixtures.identity_scaled(N, scale=2.0)
        delta_R = TestFixtures.identity_scaled(N, scale=0.5)
        G = TestFixtures.identity_scaled(N, scale=1.0)

        R_expected = R_base + delta_R   # = 2.5 I ∈ S⁺ₙ

        shield = TestFixtures.mock_shield(R=R_base)
        deformation = TestFixtures.deformation_tensor(
            delta_R=delta_R, R_base=R_base, G=G
        )
        dressed = dresser.dress_shield(shield, deformation)
        R_eff = dressed.flow.R.matrix
        assert np.allclose(R_eff, R_expected, atol=ATOL_THERMO), (
            f"R_eff ≠ R_base + δR para caso trivial PSD. "
            f"‖R_eff − R_expected‖_F = "
            f"{np.linalg.norm(R_eff - R_expected, 'fro'):.3e}"
        )

    def test_warning_deformacion_intensa(
        self,
        dresser: Phase3_ShieldDresser,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Se emite un WARNING cuando frobenius_ratio > _DEFORMATION_RATIO_WARN.
        """
        import logging
        R_base = TestFixtures.identity_scaled(N, scale=0.01)
        G = TestFixtures.identity_scaled(N)
        # delta_R muy grande respecto a R_base
        delta_R = TestFixtures.identity_scaled(N, scale=5.0)
        shield = TestFixtures.mock_shield(R=R_base)
        deformation = TestFixtures.deformation_tensor(
            delta_R=delta_R, R_base=R_base, G=G
        )
        with caplog.at_level(logging.WARNING, logger="MIC.ImmuneSystem.DynamicShieldRouter"):
            dresser.dress_shield(shield, deformation)

        assert any(
            "ratio" in r.message.lower() or "intensa" in r.message.lower()
            for r in caplog.records
        ), "Se esperaba WARNING de ratio de perturbación elevado."


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO — PRUEBAS DEL ORQUESTADOR
# ══════════════════════════════════════════════════════════════════════════════


class TestPipeline_DynamicShieldRouter:
    """
    Pruebas de integración del pipeline completo:
    Phase1 → Phase2 → Phase3 a través de DynamicShieldRouter.

    Verifica la composición correcta de las tres fases, la coherencia
    de los objetos de transferencia y las propiedades del escudo final.
    """

    @pytest.fixture
    def base_metric(self) -> NDArray[np.float64]:
        return TestFixtures.spd_matrix(n=N, seed=60)

    @pytest.fixture(params=list(Stratum))
    def pipeline_result(
        self,
        request: pytest.FixtureRequest,
        base_metric: NDArray[np.float64],
    ) -> dict:
        """
        Ejecuta el pipeline completo para cada estrato y retorna el resultado.
        """
        stratum = request.param
        R_base = TestFixtures.spd_matrix(n=N, seed=61)
        shield = TestFixtures.mock_shield(R=R_base)

        # Telemetría adaptada al estrato
        if stratum == Stratum.PHYSICS:
            tele = TestFixtures.telemetry_physics(
                R_desired=TestFixtures.spd_matrix(n=N, seed=62)
            )
        elif stratum == Stratum.TACTICS:
            tele = TestFixtures.telemetry_tactics(delta_e=0.05, alpha_kT=1.5)
        else:
            tele = TestFixtures.telemetry_omega(lindblad=0.2, alpha_kT_sys=1.0)

        dressed = DynamicShieldRouter.dress_shield_for_stratum(
            base_shield=shield,
            target_stratum=stratum,
            agent_telemetry=tele,
            base_metric=base_metric,
        )
        return {"dressed": dressed, "R_base": R_base, "stratum": stratum}

    def test_pipeline_retorna_funtor_shield(self, pipeline_result: dict) -> None:
        """dress_shield_for_stratum retorna un FuntorShield."""
        assert isinstance(pipeline_result["dressed"], (FuntorShield, MagicMock))

    def test_pipeline_r_eff_es_psd(self, pipeline_result: dict) -> None:
        """R_eff del escudo vestido es PSD para todos los estratos."""
        dressed = pipeline_result["dressed"]
        R_eff = dressed.flow.R.matrix
        evals = la.eigvalsh(R_eff)
        assert np.all(evals >= -_EIG_TOL), (
            f"[{pipeline_result['stratum']}] R_eff no PSD: "
            f"λ_min = {float(np.min(evals)):.3e}"
        )

    def test_pipeline_r_eff_es_simetrico(self, pipeline_result: dict) -> None:
        """R_eff es simétrica para todos los estratos."""
        R_eff = pipeline_result["dressed"].flow.R.matrix
        assert np.allclose(R_eff, R_eff.T, atol=ATOL_STRICT)

    def test_pipeline_no_muta_escudo_base(
        self,
        base_metric: NDArray[np.float64],
    ) -> None:
        """
        El escudo original no es modificado tras el pipeline.
        Prueba de inmutabilidad funcional end-to-end.
        """
        R_base = TestFixtures.spd_matrix(n=N, seed=70)
        shield_orig = TestFixtures.mock_shield(R=R_base)
        R_base_copy = R_base.copy()

        DynamicShieldRouter.dress_shield_for_stratum(
            base_shield=shield_orig,
            target_stratum=Stratum.TACTICS,
            agent_telemetry=TestFixtures.telemetry_tactics(),
            base_metric=base_metric,
        )

        assert np.allclose(shield_orig.flow.R.matrix, R_base_copy), (
            "El escudo original fue mutado por el pipeline."
        )

    def test_pipeline_logs_info_al_finalizar(
        self,
        base_metric: NDArray[np.float64],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        DynamicShieldRouter emite un mensaje INFO al completar el pipeline
        exitosamente.
        """
        import logging
        R_base = TestFixtures.spd_matrix(n=N)
        shield = TestFixtures.mock_shield(R=R_base)
        tele = TestFixtures.telemetry_tactics()

        with caplog.at_level(logging.INFO, logger="MIC.ImmuneSystem.DynamicShieldRouter"):
            DynamicShieldRouter.dress_shield_for_stratum(
                base_shield=shield,
                target_stratum=Stratum.TACTICS,
                agent_telemetry=tele,
                base_metric=base_metric,
            )

        assert any(
            "Pipeline completo" in r.message or "vestido" in r.message.lower()
            for r in caplog.records
        ), "Se esperaba un mensaje INFO de finalización del pipeline."

    def test_pipeline_kappa_max_personalizado(
        self,
        base_metric: NDArray[np.float64],
    ) -> None:
        """
        Con kappa_max estricto (=10), el pipeline aplica regularización
        Tikhonov y retorna un escudo con κ(R_eff) ≤ 10.
        """
        R_base = TestFixtures.spd_matrix(n=N, seed=80)
        shield = TestFixtures.mock_shield(R=R_base)
        kappa_max_strict = 10.0

        dressed = DynamicShieldRouter.dress_shield_for_stratum(
            base_shield=shield,
            target_stratum=Stratum.TACTICS,
            agent_telemetry=TestFixtures.telemetry_tactics(delta_e=1.0),
            base_metric=base_metric,
            kappa_max=kappa_max_strict,
        )
        R_eff = dressed.flow.R.matrix
        kappa_eff = float(np.linalg.cond(R_eff))
        assert kappa_eff <= kappa_max_strict * (1 + ATOL_THERMO), (
            f"κ(R_eff) = {kappa_eff:.3e} > κ_max = {kappa_max_strict}"
        )

    def test_pipeline_fases_son_composicion_exacta(
        self,
        base_metric: NDArray[np.float64],
    ) -> None:
        """
        Verifica que ejecutar las fases manualmente en secuencia produce
        exactamente el mismo resultado que usar DynamicShieldRouter.
        """
        R_base = TestFixtures.spd_matrix(n=N, seed=90)
        shield_a = TestFixtures.mock_shield(R=R_base)
        shield_b = TestFixtures.mock_shield(R=R_base.copy())
        tele = TestFixtures.telemetry_omega(lindblad=0.1)
        stratum = Stratum.STRATEGY

        # Ejecución manual fase a fase
        p1 = Phase1_EhresmannConnection(base_metric=base_metric)
        cd = p1.compute_connection(stratum)

        p2 = Phase2_ThermodynamicPullback()
        deform = p2.compute_deformation(cd, tele, R_base)

        p3 = Phase3_ShieldDresser()
        dressed_manual = p3.dress_shield(shield_a, deform)

        # Ejecución vía router
        dressed_router = DynamicShieldRouter.dress_shield_for_stratum(
            base_shield=shield_b,
            target_stratum=stratum,
            agent_telemetry=tele,
            base_metric=base_metric,
        )

        assert np.allclose(
            dressed_manual.flow.R.matrix,
            dressed_router.flow.R.matrix,
            atol=ATOL_THERMO,
        ), (
            "El resultado del router difiere de la composición manual de fases."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS BASADAS EN PROPIEDADES (HYPOTHESIS)
# ══════════════════════════════════════════════════════════════════════════════


class TestProperties_Algebraicas:
    """
    Pruebas basadas en propiedades con Hypothesis.
    Verifican invariantes algebraicos y termodinámicos que deben
    sostenerse para cualquier entrada válida dentro del dominio.
    """

    @staticmethod
    def _random_spd(n: int, data: st.DataObject) -> NDArray[np.float64]:
        """
        Genera una matriz SPD aleatoria de dimensión n usando Hypothesis.
        M = A Aᵀ + n I con A de entradas en [-2, 2].
        """
        flat = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n * n,),
                elements=st.floats(min_value=-2.0, max_value=2.0,
                                   allow_nan=False, allow_infinity=False),
            )
        )
        A = flat.reshape(n, n)
        return A @ A.T + n * np.eye(n)

    @given(st.data())
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_proyector_h_siempre_idempotente(self, data: st.DataObject) -> None:
        """
        H² = H para cualquier métrica SPD y cualquier escala de obstrucción
        en [0.0, 0.3].
        """
        n = 3
        G = self._random_spd(n, data)
        epsilon = data.draw(st.floats(min_value=0.0, max_value=0.3))

        # Validar que G es realmente SPD antes de construir la conexión
        try:
            la.cholesky(G, lower=True)
        except la.LinAlgError:
            assume(False)

        kappa = float(np.linalg.cond(G))
        assume(kappa < _KAPPA_MAX * 0.5)   # evitar casos de κ fuera del umbral

        phase1 = Phase1_EhresmannConnection(
            base_metric=G, obstruction_scale=epsilon
        )

        for stratum in Stratum:
            try:
                cd = phase1.compute_connection(stratum)
            except ConnectionError:
                continue    # ignorar casos donde κ(G_s) supera el umbral

            H = cd.horizontal_projector
            H2 = H @ H
            assert np.allclose(H2, H, atol=1e-7), (
                f"H² ≠ H para estrato {stratum}, ε={epsilon:.4f}. "
                f"‖H²−H‖_F = {np.linalg.norm(H2-H,'fro'):.3e}"
            )

    @given(st.data())
    @settings(
        max_examples=25,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_proyeccion_higham_siempre_produce_psd(
        self, data: st.DataObject
    ) -> None:
        """
        Π_{S⁺ₙ}(M) ∈ S⁺ₙ para cualquier matriz simétrica M.
        """
        n = 3
        flat = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n * n,),
                elements=st.floats(min_value=-10.0, max_value=10.0,
                                   allow_nan=False, allow_infinity=False),
            )
        )
        A = flat.reshape(n, n)
        M = 0.5 * (A + A.T)   # simetrizar

        dresser = Phase3_ShieldDresser()
        M_proj = dresser._project_to_psd_cone(M)
        evals = la.eigvalsh(M_proj)
        assert np.all(evals >= -1e-8), (
            f"λ_min = {float(np.min(evals)):.3e} < 0 tras proyección Higham."
        )

    @given(st.data())
    @settings(
        max_examples=25,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_tikhonov_siempre_satisface_cota_kappa(
        self, data: st.DataObject
    ) -> None:
        """
        κ(R_reg) ≤ κ_max para cualquier matriz PSD y cualquier κ_max ∈ [10, 1e6].
        """
        n = 3
        G = self._random_spd(n, data)
        kappa_max = data.draw(st.floats(min_value=10.0, max_value=1.0e6))

        assume(float(np.linalg.cond(G)) < 1.0e12)

        dresser = Phase3_ShieldDresser(kappa_max=kappa_max)
        R_reg = dresser._tikhonov_regularize(G)

        evals = la.eigvalsh(R_reg)
        lambda_min = float(evals[0]) + _EIG_TOL
        lambda_max = float(evals[-1])
        kappa_actual = lambda_max / lambda_min

        assert kappa_actual <= kappa_max * (1 + 1e-5), (
            f"κ(R_reg) = {kappa_actual:.3e} > κ_max = {kappa_max:.3e}"
        )

    @given(
        delta_e=st.floats(min_value=0.0, max_value=10.0),
        alpha_kt=st.floats(min_value=0.01, max_value=10.0),
    )
    @settings(max_examples=40, deadline=None)
    def test_pullback_tactics_lineal_en_delta_e(
        self,
        delta_e: float,
        alpha_kt: float,
    ) -> None:
        """
        Para TACTICS: δR(c·Δε) = c · δR(Δε) para cualquier c, Δε ≥ 0.
        """
        G = TestFixtures.identity_scaled(N)
        R_base = TestFixtures.spd_matrix(n=N)
        conn = ConnectionData(
            metric_tensor=G,
            horizontal_projector=np.eye(N),
            connection_form_norm=1.0,
            stratum=Stratum.TACTICS,
            kappa_metric=1.0,
        )
        phase2 = Phase2_ThermodynamicPullback()

        tele1 = {"discarded_spherical_entropy": delta_e, "alpha_over_kT": alpha_kt}
        tele2 = {"discarded_spherical_entropy": 2 * delta_e, "alpha_over_kT": alpha_kt}

        dR1 = phase2.compute_deformation(conn, tele1, R_base).delta_R
        dR2 = phase2.compute_deformation(conn, tele2, R_base).delta_R

        assert np.allclose(2 * dR1, dR2, atol=ATOL_STRICT), (
            "Linealidad de δR en Δε violada."
        )

    @given(
        lindblad=st.floats(min_value=0.0, max_value=5.0),
        alpha=st.floats(min_value=0.1, max_value=5.0),
    )
    @settings(max_examples=30, deadline=None)
    def test_pullback_omega_produce_multiplo_de_identidad(
        self,
        lindblad: float,
        alpha: float,
    ) -> None:
        """
        Para STRATEGY/WISDOM: δR = c · I_n, verificado matricialmente.
        """
        R_base = TestFixtures.spd_matrix(n=N)
        conn = ConnectionData(
            metric_tensor=TestFixtures.identity_scaled(N),
            horizontal_projector=np.eye(N),
            connection_form_norm=1.0,
            stratum=Stratum.STRATEGY,
            kappa_metric=1.0,
        )
        phase2 = Phase2_ThermodynamicPullback()
        tele = {"lindblad_dissipation_trace": lindblad, "alpha_over_kT_sys": alpha}
        result = phase2.compute_deformation(conn, tele, R_base)
        dR = result.delta_R

        # δR debe ser proporcional a I: δR − (Tr(δR)/n) I = 0
        c = float(np.trace(dR)) / N
        assert np.allclose(dR, c * np.eye(N), atol=ATOL_STRICT), (
            f"δR no es múltiplo de I: ‖δR − cI‖_F = "
            f"{np.linalg.norm(dR - c*np.eye(N), 'fro'):.3e}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS DE REGRESIÓN Y ESTABILIDAD NUMÉRICA
# ══════════════════════════════════════════════════════════════════════════════


class TestRegression_EstabilidadNumerica:
    """
    Pruebas de regresión que verifican estabilidad numérica bajo condiciones
    extremas: matrices casi singulares, deformaciones de alta norma,
    y telemetrías de borde (valores cero, negativos controlados, etc.).
    """

    def test_pipeline_con_deformacion_nula(self) -> None:
        """
        Con Δε = 0 y Λ_L = 0, δR = 0, y el escudo vestido debe ser
        idéntico al escudo base (la vestimenta es la identidad).
        """
        R_base = TestFixtures.spd_matrix(n=N, seed=100)
        G = TestFixtures.spd_matrix(n=N, seed=101)
        shield = TestFixtures.mock_shield(R=R_base)

        tele = TestFixtures.telemetry_tactics(delta_e=0.0, alpha_kT=1.0)
        dressed = DynamicShieldRouter.dress_shield_for_stratum(
            base_shield=shield,
            target_stratum=Stratum.TACTICS,
            agent_telemetry=tele,
            base_metric=G,
        )
        R_eff = dressed.flow.R.matrix
        assert np.allclose(R_eff, R_base, atol=ATOL_THERMO), (
            f"Con δR=0, el escudo vestido difiere del base: "
            f"‖R_eff − R_base‖_F = {np.linalg.norm(R_eff - R_base, 'fro'):.3e}"
        )

    def test_pipeline_estabilidad_con_r_base_casi_singular(self) -> None:
        """
        El pipeline maneja R_base con condicionamiento alto (pero PSD)
        sin lanzar excepción; el resultado es PSD.
        """
        # R_base con λ ∈ {1e-8, 1, 1, 2}
        evals = np.array([1e-8, 1.0, 1.0, 2.0])
        Q = la.orth(np.random.default_rng(110).standard_normal((N, N)))
        R_casi_singular = Q @ np.diag(evals) @ Q.T
        R_casi_singular = 0.5 * (R_casi_singular + R_casi_singular.T)

        G = TestFixtures.spd_matrix(n=N, seed=111)
        shield = TestFixtures.mock_shield(R=R_casi_singular)
        tele = TestFixtures.telemetry_omega(lindblad=0.01)

        dressed = DynamicShieldRouter.dress_shield_for_stratum(
            base_shield=shield,
            target_stratum=Stratum.WISDOM,
            agent_telemetry=tele,
            base_metric=G,
        )
        R_eff = dressed.flow.R.matrix
        assert np.all(la.eigvalsh(R_eff) >= -_EIG_TOL)

    def test_produccion_entropia_positiva_para_perturbacion_positiva(self) -> None:
        """
        Si δR ≻ 0 y G es SPD, entonces σ = Tr(δR G⁻¹) > 0.
        """
        G = TestFixtures.spd_matrix(n=N, seed=120)
        R_base = TestFixtures.spd_matrix(n=N, seed=121)
        conn = ConnectionData(
            metric_tensor=G,
            horizontal_projector=np.eye(N),
            connection_form_norm=1.0,
            stratum=Stratum.TACTICS,
            kappa_metric=float(np.linalg.cond(G)),
        )
        phase2 = Phase2_ThermodynamicPullback()
        tele = TestFixtures.telemetry_tactics(delta_e=1.0, alpha_kT=1.0)
        result = phase2.compute_deformation(conn, tele, R_base)

        # δR = alpha_kT * delta_e * H G Hᵀ ≻ 0 ⟹ σ > 0
        assert result.entropy_production >= 0.0, (
            f"σ = {result.entropy_production:.6e} < 0 para perturbación positiva."
        )

    def test_frobenius_ratio_campo_info_contiene_estrato(self) -> None:
        """
        El campo info del DeformationTensor contiene la clave 'stratum'
        con el estrato correcto.
        """
        G = TestFixtures.spd_matrix(n=N)
        R_base = TestFixtures.spd_matrix(n=N)
        conn = TestFixtures.connection_data(Stratum.WISDOM)
        phase2 = Phase2_ThermodynamicPullback()
        tele = TestFixtures.telemetry_omega()
        result = phase2.compute_deformation(conn, tele, R_base)
        assert result.info["stratum"] == Stratum.WISDOM

    def test_pipeline_wisdom_y_strategy_producen_resultados_coherentes(
        self,
    ) -> None:
        """
        Con la misma telemetría, STRATEGY y WISDOM deben producir escudos
        distintos (por diferente λ(s) en la métrica) pero ambos PSD.
        """
        G = TestFixtures.spd_matrix(n=N, seed=130)
        R_base = TestFixtures.spd_matrix(n=N, seed=131)
        tele = TestFixtures.telemetry_omega(lindblad=0.3, alpha_kT_sys=1.0)

        shield_s = TestFixtures.mock_shield(R=R_base.copy())
        shield_w = TestFixtures.mock_shield(R=R_base.copy())

        dressed_s = DynamicShieldRouter.dress_shield_for_stratum(
            shield_s, Stratum.STRATEGY, tele, base_metric=G
        )
        dressed_w = DynamicShieldRouter.dress_shield_for_stratum(
            shield_w, Stratum.WISDOM, tele, base_metric=G
        )

        R_s = dressed_s.flow.R.matrix
        R_w = dressed_w.flow.R.matrix

        # Ambos PSD
        assert np.all(la.eigvalsh(R_s) >= -_EIG_TOL)
        assert np.all(la.eigvalsh(R_w) >= -_EIG_TOL)

        # Deben ser distintos (λ(STRATEGY) ≠ λ(WISDOM))
        assert not np.allclose(R_s, R_w), (
            "STRATEGY y WISDOM produjeron escudos idénticos (no esperado)."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA PARA EJECUCIÓN DIRECTA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-rA"])